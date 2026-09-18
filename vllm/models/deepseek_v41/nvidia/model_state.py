# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import DPMetadata
from vllm.logger import init_logger
from vllm.models.deepseek_v41.nvidia.model import ReplayBatch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    compute_mm_prefix_ranges,
)
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model

logger = init_logger(__name__)


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


@triton.jit
def _pad_replayed_slots_kernel(
    slot_mappings_ptr,  # [NUM_GROUPS, num_tokens_padded]
    group_stride,
    query_start_loc_ptr,  # [num_reqs + 1]
    positions_ptr,  # [num_tokens]
    replay_start_ptr,  # [num_reqs]
    window,
    pad_slot_id,
    CACHEABLE_GROUPS: tl.constexpr,  # bit g set: group g is prefix-cacheable
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(replay_start_ptr + req)
    if start == 0:
        return
    begin = tl.load(query_start_loc_ptr + req)
    end = tl.load(query_start_loc_ptr + req + 1)
    for tok in range(begin, end, BLOCK):
        offs = tok + tl.arange(0, BLOCK)
        pos = tl.load(positions_ptr + offs, mask=offs < end, other=0)
        replayed = (offs < end) & (pos >= start) & (pos < start + window)
        for g in tl.static_range(NUM_GROUPS):
            if (CACHEABLE_GROUPS >> g) & 1:
                tl.store(
                    slot_mappings_ptr + g * group_stride + offs,
                    pad_slot_id,
                    mask=replayed,
                )


@triton.jit
def _gather_replay_batch_kernel(
    query_start_loc_ptr,  # [num_reqs + 1] the batch's device boundaries
    dropped_before_ptr,  # [num_reqs + 1] rows trimmed before each boundary
    positions_ptr,
    is_padding_ptr,
    slot_mappings_ptr,  # [num_groups, num_tokens_padded]
    slot_mappings_stride,
    rows_ptr,  # out: [num_tokens] batch row of every kept row
    replay_query_start_loc_ptr,  # out: [num_reqs + 1]
    replay_positions_ptr,  # out: [num_tokens]
    replay_is_padding_ptr,  # out: [num_tokens]
    replay_slot_mappings_ptr,  # out: [num_groups, num_tokens]
    replay_slot_mappings_stride,
    pad_slot_id,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    end = tl.load(query_start_loc_ptr + req + 1)
    kept_begin = tl.load(query_start_loc_ptr + req) - tl.load(dropped_before_ptr + req)
    kept_end = end - tl.load(dropped_before_ptr + req + 1)
    if req == 0:
        tl.store(replay_query_start_loc_ptr, kept_begin)
    tl.store(replay_query_start_loc_ptr + req + 1, kept_end)
    # Kept row r of the request is its window start + (r - kept_begin).
    window_start = end - (kept_end - kept_begin)
    for tok in range(kept_begin, kept_end, BLOCK):
        offs = tok + tl.arange(0, BLOCK)
        mask = offs < kept_end
        row = window_start + (offs - kept_begin)
        tl.store(rows_ptr + offs, row.to(tl.int64), mask=mask)
        tl.store(
            replay_positions_ptr + offs,
            tl.load(positions_ptr + row, mask=mask, other=0),
            mask=mask,
        )
        tl.store(
            replay_is_padding_ptr + offs,
            tl.load(is_padding_ptr + row, mask=mask, other=0),
            mask=mask,
        )
        for g in tl.static_range(NUM_GROUPS):
            tl.store(
                replay_slot_mappings_ptr + g * replay_slot_mappings_stride + offs,
                tl.load(
                    slot_mappings_ptr + g * slot_mappings_stride + row,
                    mask=mask,
                    other=pad_slot_id,
                ),
                mask=mask,
            )


class ReplayAttnMetadata(ModelSpecificAttnMetadata):
    """Hands the batch's replay starts to the sliding-window builders."""

    def __init__(self, replay_start: torch.Tensor) -> None:
        self.replay_start = replay_start

    def get_extra_attn_kwargs(
        self, attn_metadata_builder: Any, num_reqs: int
    ) -> dict[str, Any]:
        if isinstance(attn_metadata_builder, DeepseekSparseSWAMetadataBuilder):
            return {"replay_start": self.replay_start}
        return {}


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window and SWA bounded replay.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.

    SWA bounded replay: ``prepare_attn`` gathers the batch's replay starts (the
    position from which each request holds window KV, set by the scheduler at
    a prefix hit), pads the replayed tokens' slots in the prefix-cacheable
    groups so the cached KV stays as is, and hands the starts to the
    sliding-window builders. With the decoder side on, it also builds the
    replay layers' batch when a request trims to its window: the kept rows,
    and the replay layers' attention metadata from builders of their own (a
    builder's buffers hold one batch's metadata, and the batch's own build is
    still read after this one). Under data parallelism the ranks agree first:
    the replay layers' MoE collectives need every rank in the replay together,
    so all replay when any rank trims, on every rank's own replay token count.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

        # SWA bounded replay: the replay start of every request, by state
        # index, and the batch's, gathered in prepare_attn.
        self._req_replay_start = np.zeros(self.max_num_reqs, dtype=np.int32)
        self._batch_replay_start = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._replay_start_staging = UvaBufferPool(self.max_num_reqs, torch.int32)

        self.decoder_replay: DeepseekV4Model | None = getattr(
            model, "decoder_replay_model", None
        )
        if self.decoder_replay is None:
            return
        window = self.decoder_replay.decoder_replay_window
        assert window is not None
        self._replay_window = window
        logger.info_once(
            "Decoder SWA bounded replay: layers past the last KV source prefill "
            "only each request's last %d tokens.",
            window,
        )
        self._replay_builders: dict[int, AttentionMetadataBuilder] = {}
        # Requests whose every prompt row is read (prompt logprobs) never trim.
        self._req_keeps_rows = np.zeros(self.max_num_reqs, dtype=np.bool_)
        # The replay layers' batch, in buffers refilled in stream order every
        # replaying step: its rows of the step's batch, its inputs, its slot
        # mappings ([num_kv_cache_groups, max_num_tokens], sized on first use)
        # and the position from which each request holds replay-layer window KV.
        self._replay_rows = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._replay_inputs = InputBuffers(
            self.max_num_reqs, self.max_num_tokens, device
        )
        self._replay_slot_mappings: torch.Tensor | None = None
        self._replay_kv_start = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # One pool per staged array: a pool rotates its buffers, and a buffer
        # must not be reused while its GPU readers are in flight.
        self._dropped_staging = UvaBufferPool(self.max_num_reqs + 1, torch.int32)
        self._replay_kv_start_staging = UvaBufferPool(self.max_num_reqs, torch.int32)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self._req_replay_start[req_index] = new_req_data.replay_start
        if self.decoder_replay is not None:
            params = new_req_data.sampling_params
            self._req_keeps_rows[req_index] = (
                params is not None and params.prompt_logprobs is not None
            )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
        window = max(spec.prefix_replay_tokens for spec in specs)
        replay_start_np = None
        if window:
            num_reqs = input_batch.num_reqs
            # Decode rows sit above the hit, so only prefills carry a replay
            # start; dummy batches (captures, profiling) carry none.
            replay_start_np = np.where(
                input_batch.is_prefilling_np[:num_reqs],
                self._req_replay_start[input_batch.idx_mapping_np[:num_reqs]],
                0,
            ).astype(np.int32)
            replay_start = self._replay_start_staging.copy_to_gpu(
                replay_start_np, out=self._batch_replay_start[:num_reqs]
            )
            if replay_start_np.any():
                # The replayed tokens rebuild window KV only: their slots in the
                # prefix-cacheable groups are padded so the cached KV stays as is.
                _pad_replayed_slots_kernel[(num_reqs,)](
                    slot_mappings,
                    slot_mappings.stride(0),
                    input_batch.query_start_loc,
                    input_batch.positions,
                    replay_start,
                    window,
                    PAD_SLOT_ID,
                    CACHEABLE_GROUPS=sum(
                        1 << i for i, spec in enumerate(specs) if spec.prefix_cacheable
                    ),
                    NUM_GROUPS=len(specs),
                    BLOCK=1024,
                )
            assert model_specific_attn_metadata is None
            model_specific_attn_metadata = ReplayAttnMetadata(replay_start)
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        if self.decoder_replay is not None:
            assert replay_start_np is not None  # the decoder side replays too
            self.decoder_replay.replay_batch = self._prepare_replay_batch(
                input_batch,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                replay_start_np,
            )
        return attn_metadata

    def _prepare_replay_batch(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        replay_start_np: np.ndarray,
    ) -> ReplayBatch | None:
        """The replay layers' batch for this forward, or None when they run on
        the whole batch with the batch's own metadata: nothing trims on any
        rank (also every dummy/capture and FULL-graph batch).
        ``replay_start_np`` is the batch's encoder-side replay start."""
        assert self.decoder_replay is not None
        window = self._replay_window
        num_reqs = input_batch.num_reqs
        query_start_loc = input_batch.query_start_loc_np[: num_reqs + 1]
        lens = np.diff(query_start_loc)
        # Only prefills run past the window; dummy batches (captures, an idle DP
        # rank's) are not prefills and keep their rows unless a rank trims.
        keep = np.where(
            self._req_keeps_rows[input_batch.idx_mapping_np[:num_reqs]],
            lens,
            np.minimum(lens, window),
        )
        trims = bool((keep < lens)[input_batch.is_prefilling_np[:num_reqs]].any())
        trims, counts = self._agree_across_dp(trims, int(keep.sum()))
        if not trims:
            return None
        replay_query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(keep, out=replay_query_start_loc[1:])
        num_tokens = int(replay_query_start_loc[-1])

        rows, replay_slot_mappings = self._gather_replay_rows(
            input_batch,
            slot_mappings,
            query_start_loc - replay_query_start_loc,
            num_tokens,
        )
        # No replay-layer window KV exists below a trimmed request's window, on
        # top of what the encoder-side replay excludes.
        seq_lens = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
        replay_start = self._replay_kv_start_staging.copy_to_gpu(
            np.maximum(
                replay_start_np, np.where(keep < lens, seq_lens - window, 0)
            ).astype(np.int32),
            out=self._replay_kv_start[:num_reqs],
        )
        max_query_len = int(keep.max())
        if input_batch.max_query_len is not None:
            # Adaptive verification's bound on the decodes' device-side lengths.
            max_query_len = max(max_query_len, input_batch.max_query_len)
        inputs = self._replay_inputs
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            req_doc_ranges = compute_mm_prefix_ranges(
                req_ids=input_batch.req_ids,
                mm_features=self.encoder_cache.mm_features,
                sliding_window=self.model_config.get_sliding_window(),
            )
        # Keep this argument list in sync with DefaultModelState.prepare_attn's
        # build_attn_metadata call: the replay layers' builders need the same
        # fields, computed for the replay batch. Reduced (kept rows only):
        # num_tokens, query_start_loc, positions, slot_mappings, max_query_len.
        # Unchanged (per-request): seq_lens, block_tables, dcp/rswa/req_idx.
        attn_metadata = build_attn_metadata(
            attn_groups=self._replay_attn_groups(attn_groups),
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=inputs.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=torch.from_numpy(replay_query_start_loc),
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=int(seq_lens.max()),
            block_tables=block_tables,
            slot_mappings=replay_slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            dcp_local_seq_lens_cpu_upper_bound=(
                input_batch.dcp_local_seq_lens_cpu_upper_bound
            ),
            positions=inputs.positions[:num_tokens],
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
            mm_req_doc_ranges=req_doc_ranges,
            rswa_prefix_lens=input_batch.prompt_lens,
            req_idx=input_batch.idx_mapping_np,
            model_specific_attn_metadata=ReplayAttnMetadata(replay_start),
        )
        return ReplayBatch(
            rows=rows,
            trims=trims,
            attn_metadata=attn_metadata,
            slot_mapping=build_slot_mappings_by_layer(
                replay_slot_mappings, kv_cache_config
            ),
            is_padding=inputs.is_padding[:num_tokens],
            dp_metadata=self._replay_dp_metadata(num_tokens, counts),
        )

    def _agree_across_dp(
        self, trims: bool, num_replay_tokens: int
    ) -> tuple[bool, torch.Tensor | None]:
        """Whether any rank trims and, then, every rank's replay token count:
        the replay layers' MoE collectives need all of them, and every rank
        calls prepare_attn every step."""
        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        if dp_size == 1:
            return trims, None
        agreed = torch.zeros(dp_size, 2, dtype=torch.int32)
        agreed[parallel_config.data_parallel_rank] = torch.tensor(
            [trims, num_replay_tokens], dtype=torch.int32
        )
        if not should_skip_dp_coordination():
            dist.all_reduce(agreed, group=get_dp_group().cpu_group)
        trims = bool(agreed[:, 0].any())
        return trims, agreed[:, 1] if trims else None

    def _replay_dp_metadata(
        self, num_tokens: int, counts: torch.Tensor | None
    ) -> DPMetadata | None:
        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        if dp_size == 1:
            return None
        assert counts is not None  # DP replays only when a rank trims
        return DPMetadata.make(parallel_config, num_tokens, counts)

    def _gather_replay_rows(
        self,
        input_batch: InputBatch,
        slot_mappings: torch.Tensor,
        dropped_before: np.ndarray,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill the replay batch's inputs and slot mappings from the batch's;
        returns its rows of the batch and its slot mappings."""
        # The kept rows follow the device boundaries minus the rows trimmed
        # before them: adaptive verification resizes the decodes on the GPU
        # alone.
        dropped_before_uva = self._dropped_staging.copy_to_uva(dropped_before)
        if self._replay_slot_mappings is None:
            self._replay_slot_mappings = torch.zeros(
                slot_mappings.shape[0],
                self.max_num_tokens,
                dtype=slot_mappings.dtype,
                device=self.device,
            )
        inputs = self._replay_inputs
        _gather_replay_batch_kernel[(input_batch.num_reqs,)](
            input_batch.query_start_loc,
            dropped_before_uva,
            input_batch.positions,
            input_batch.is_padding,
            slot_mappings,
            slot_mappings.stride(0),
            self._replay_rows,
            inputs.query_start_loc,
            inputs.positions,
            inputs.is_padding,
            self._replay_slot_mappings,
            self._replay_slot_mappings.stride(0),
            PAD_SLOT_ID,
            NUM_GROUPS=slot_mappings.shape[0],
            BLOCK=1024,
        )
        return self._replay_rows[:num_tokens], self._replay_slot_mappings[
            :, :num_tokens
        ]

    def _replay_attn_groups(
        self, attn_groups: list[list[AttentionGroup]]
    ) -> list[list[AttentionGroup]]:
        """The step's attention groups with metadata builders of their own.

        A builder writes its metadata into buffers it owns (fixed addresses
        for CUDA graphs), so it can hold one batch's metadata at a time; the
        batch's own build -- same groups, and the prefix layers in them are
        read later this step -- must not be clobbered by the replay build.
        """

        def builder(group: AttentionGroup) -> AttentionMetadataBuilder:
            if id(group) not in self._replay_builders:
                self._replay_builders[id(group)] = group.make_metadata_builder(
                    self.vllm_config,
                    self.device,
                    group.metadata_builders[0].kernel_block_size,
                )
            return self._replay_builders[id(group)]

        return [
            [replace(group, metadata_builders=[builder(group)]) for group in groups]
            for groups in attn_groups
        ]
