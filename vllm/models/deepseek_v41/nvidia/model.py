# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    is_breakable_cudagraph_enabled,
)
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_engram_dp_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    DPMetadata,
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MoE as DeepseekV4MoEBase,
)
from vllm.models.deepseek_v4.nvidia.model import (
    MegaGateRoutingMetadata,
    make_deepseek_v4_expert_params_mapping,
    prepare_mega_gate_routing_metadata,
)
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
    DeepseekV4MegaAttnAttention,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import weak_ref_tensor
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from ..common.engram import EngramLayout, NgramHashState
from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID, image_sentinel_mask
from .engram import Engram, gather_engram_hashes
from .ops.mega_mhc import mhc_shifted_post_pre

if typing.TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)


class DeepseekV4MoE(DeepseekV4MoEBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ):
        config = vllm_config.model_config.hf_config
        n_routed_experts = config.n_routed_experts
        n_activated_experts = config.num_experts_per_tok
        if extract_layer_index(prefix) >= config.num_hidden_layers:
            n_routed_experts = (
                getattr(config, "dspark_n_routed_experts", 0) or n_routed_experts
            )
            n_activated_experts = (
                getattr(config, "dspark_num_experts_per_tok", 0) or n_activated_experts
            )
        super().__init__(
            vllm_config,
            prefix=prefix,
            use_sequence_parallel=use_sequence_parallel,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            num_hash_layers=0,
            image_sentinel_lo=IMAGE_SENTINEL_BASE_ID,
        )


def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    """Pick the CUDA sparse-MLA attention class for the configured backend.

    The generic CUDA backend selector does not instantiate DSv4 layers directly,
    so map generic sparse-MLA choices to the DSv4-specialized attention class.
    Without an explicit backend: SM12 takes FlashInfer, SM100 takes mega
    attention where the topology allows it, and everything else keeps the
    FlashMLA path.
    """
    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):
        raise ValueError(
            f"{backend.name} is not a DeepSeek V4.1 attention backend. "
            "Use FLASHINFER_MLA_SPARSE_DSV41 for DeepSeek V4.1 FlashInfer "
            "sparse MLA."
        )
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41,
    ):
        if device_capability is not None and device_capability.major == 12:
            return DeepseekV4FlashInferSM120Attention
        return DeepseekV4FlashInferMLAAttention
    if backend is AttentionBackendEnum.FLASHMLA_MEGA_ATTN_DSV41:
        return DeepseekV4MegaAttnAttention
    if backend in (
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV41,
    ):
        return DeepseekV4FlashMLAAttention

    if device_capability is not None and device_capability.major == 12:
        return DeepseekV4FlashInferSM120Attention
    # Mega attention is the SM100 default: it fuses Q RoPE, sparse attention,
    # the output's inverse RoPE and its FP8 cast into one launch, and brings
    # the 288 B NVFP4 compressed record -- the format the reference
    # implementation itself stores. It declines topologies it cannot serve
    # (non-SM100, TP that leaves fewer than WV_GROUP_SIZE heads per wo_a
    # group, a build without the kernel), which then fall through to FlashMLA.
    if DeepseekV4MegaAttnAttention.is_available_for(vllm_config):
        return DeepseekV4MegaAttnAttention
    return DeepseekV4FlashMLAAttention


def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    use_mega_moe = vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
    return (
        parallel_config.pipeline_parallel_size == 1
        and parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and (use_mega_moe or parallel_config.data_parallel_size > 1)
    )


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)

        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = Engram(
                    config,
                    vllm_config.quant_config,
                    engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    use_sequence_parallel=self.use_sequence_parallel,
                    prefix=f"{prefix}.engram",
                )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        if self.use_sequence_parallel:
            self.attn.wo_b.reduce_results = False
        self.ffn = DeepseekV4MoE(
            vllm_config,
            prefix=f"{prefix}.ffn",
            use_sequence_parallel=self.use_sequence_parallel,
        )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        if vllm_config.kernel_config.enable_jit_warmup and current_platform.is_cuda():
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                mhc_fused_post_pre_splits,
            )
            from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            if self.use_sequence_parallel:
                tp_size = vllm_config.parallel_config.tensor_parallel_size
                max_tokens = (max_tokens + tp_size - 1) // tp_size
            # The epilogue compiles per projection width and per pre-mix mode.
            # The first layer projects the broadcast embedding, so it reads one
            # hidden_size-wide row and selects stream zero; every later sublayer
            # projects the full hc stream, and collapses it with the pre-mix the
            # previous sublayer carried in. Those later shapes also arrive
            # through the fused post + pre-norm GEMM, which picks split-k
            # factors the token sweep below never produces.
            broadcast_embedding = {
                "rms_numel": self.hidden_size,
                "use_pre_mix_in": False,
                "extra_splits": (),
            }
            hc_stream = {
                "rms_numel": self.hc_mult * self.hidden_size,
                "extra_splits": mhc_fused_post_pre_splits(
                    self.hidden_size, self.hc_mult
                ),
            }
            variants = [
                broadcast_embedding,
                {**hc_stream, "use_pre_mix_in": False},
                {**hc_stream, "use_pre_mix_in": True},
            ]
            if vllm_config.speculative_config is not None:
                # Draft setups read aux hidden states out of the same collapse,
                # which is its own specialization of the kernel.
                variants.append(
                    {**hc_stream, "use_pre_mix_in": True, "write_aux": True}
                )
            for variant in variants:
                MHC_PRE_NORM_KERNEL.register_warmup(
                    max_tokens=max_tokens,
                    hidden_size=self.hidden_size,
                    rms_eps=self.rms_norm_eps,
                    hc_pre_eps=self.hc_eps,
                    hc_sinkhorn_eps=self.hc_eps,
                    hc_post_mult_value=self.hc_post_alpha,
                    sinkhorn_repeat=self.hc_sinkhorn_iters,
                    norm_eps=self.rms_norm_eps,
                    hc_mult=self.hc_mult,
                    **variant,
                )
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        *,
        capture_previous_aux: bool = False,
        mega_gate_metadata: MegaGateRoutingMetadata | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        previous_aux: torch.Tensor | None = None
        # The reference collapses each sublayer's input with the *previous*
        # sublayer's pre-mix: attention uses the pre-mix carried in (identity
        # for the first layer), the FFN uses this layer's attention pre-mix.
        if residual is None:
            if x.dim() == 2:
                # First layer: the stream is the embedding broadcast to hc
                # copies and the identity pre-mix selects copy 0.
                assert self.hc_attn_fn_broadcast is not None
                residual = x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
                post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                    residual,
                    self.hc_attn_fn_broadcast,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    x=x,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                )
            else:
                residual = x
                post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=pre_mix,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                )
        elif self.engram is not None and engram_hashes is not None:
            # Engram injection happens between the previous sublayer's post
            # and this block's pre, on the full hc stream, so the mix
            # coefficients see the injected stream. The injection also keeps
            # the post out of the pre-norm GEMM's fused prologue.
            previous_post = mhc_post_tilelang(x, residual, post_mix, res_mix)
            if capture_previous_aux:
                previous_aux = previous_post.mean(dim=1)
            residual = self.engram(
                previous_post,
                engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
            )
            post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                residual,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
            )
        else:
            # The collapse already reads the post-mapped streams, so the mean
            # aux consumers want comes out of the same kernel.
            residual, post_mix, res_mix, x, attn_pre, aux = mhc_shifted_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
                capture_aux=capture_previous_aux,
            )
            if capture_previous_aux:
                previous_aux = aux

        if self.use_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.attn(positions, x, None)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)

        residual, post_mix, res_mix, x, ffn_pre, _ = mhc_shifted_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=attn_pre,
            norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon,
        )
        x = self.ffn(x, input_ids, mega_gate_metadata)
        return x, residual, post_mix, res_mix, ffn_pre, previous_aux


@dataclass
class ReplayBatch:
    """The rows the decoder replay layers run on, and what their forward
    context replaces in the batch's. Built per step by the model state
    (``DeepseekV41ModelState``); None when the layers run on the whole batch."""

    rows: torch.Tensor  # [num_tokens] rows of the full batch, in batch order
    # [num_batch_tokens] int32: the replay row each full-batch row maps to,
    # -1 for dropped rows (padding rows hold stale values; readers mask on
    # ``< num_tokens``).
    inv_rows: torch.Tensor
    trims: bool  # whether any DP rank cut a request down to its window
    attn_metadata: dict[str, typing.Any]
    slot_mapping: dict[str, torch.Tensor]
    is_padding: torch.Tensor
    dp_metadata: DPMetadata | None


# The replay layers' inputs: hidden_states, positions, input_ids, pre_mix,
# post_mix, res_mix, residual. Only input_ids can be absent (non-first PP
# ranks): the prefix layers have run by the replay boundary.
ReplayStates = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _map_optional(
    t: torch.Tensor | None, fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor | None:
    return None if t is None else fn(t)


def _weak_ref_states(states: ReplayStates) -> ReplayStates:
    """Non-owning aliases the model graph's eager break records; the graph
    owns the memory and recomputes it in place before the break replays."""
    return (
        weak_ref_tensor(states[0]),
        weak_ref_tensor(states[1]),
        _map_optional(states[2], weak_ref_tensor),
        weak_ref_tensor(states[3]),
        weak_ref_tensor(states[4]),
        weak_ref_tensor(states[5]),
        weak_ref_tensor(states[6]),
    )


@triton.jit
def _copy_row(src_ptr, dst_ptr, src_row, dst_row, width, keep, BLOCK: tl.constexpr):
    """Copy row ``src_row`` of a row-major tensor to row ``dst_row`` of
    another, writing zeros when not ``keep``. Width 0 marks an absent slot."""
    for off in range(0, width, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < width
        val = tl.load(src_ptr + src_row * width + cols, mask=mask & keep, other=0)
        tl.store(dst_ptr + dst_row * width + cols, val, mask=mask)


@triton.jit
def _replay_gather_kernel(
    rows_ptr,  # [num_rows] int64: the full-batch row of every replay row
    # (src, dst, width) per slot, states contiguous along the trailing dims;
    # width 0 marks an absent slot (no input_ids on non-first PP ranks, no
    # indexer buffers to realign when nothing trims).
    src0_ptr,
    dst0_ptr,
    w0,  # hidden_states
    src1_ptr,
    dst1_ptr,
    w1,  # positions
    src2_ptr,
    dst2_ptr,
    w2,  # input_ids
    src3_ptr,
    dst3_ptr,
    w3,  # pre_mix
    src4_ptr,
    dst4_ptr,
    w4,  # post_mix
    src5_ptr,
    dst5_ptr,
    w5,  # res_mix
    src6_ptr,
    dst6_ptr,
    w6,  # residual
    src7_ptr,
    dst7_ptr,
    w7,  # indexer topk indices, into scratch
    src8_ptr,
    dst8_ptr,
    w8,  # indexer candidate blocks, into scratch
    BLOCK: tl.constexpr,
):
    # One program per replay row: copy that row of every tensor. The row
    # buffers go to scratch (see the caller): gathering them in place would
    # race, since a kept row can sit below a later replay row.
    j = tl.program_id(0).to(tl.int64)
    row = tl.load(rows_ptr + j)
    _copy_row(src0_ptr, dst0_ptr, row, j, w0, True, BLOCK)
    _copy_row(src1_ptr, dst1_ptr, row, j, w1, True, BLOCK)
    _copy_row(src2_ptr, dst2_ptr, row, j, w2, True, BLOCK)
    _copy_row(src3_ptr, dst3_ptr, row, j, w3, True, BLOCK)
    _copy_row(src4_ptr, dst4_ptr, row, j, w4, True, BLOCK)
    _copy_row(src5_ptr, dst5_ptr, row, j, w5, True, BLOCK)
    _copy_row(src6_ptr, dst6_ptr, row, j, w6, True, BLOCK)
    _copy_row(src7_ptr, dst7_ptr, row, j, w7, True, BLOCK)
    _copy_row(src8_ptr, dst8_ptr, row, j, w8, True, BLOCK)


@triton.jit(do_not_specialize=["num_src"])
def _replay_scatter_kernel(
    inv_rows_ptr,  # [num_dst] int32: the src row of each dst row
    num_src,
    # (src, dst, width) per slot; width 0 marks an absent slot.
    src0_ptr,
    dst0_ptr,
    w0,  # hidden_states
    src1_ptr,
    dst1_ptr,
    w1,  # pre_mix
    src2_ptr,
    dst2_ptr,
    w2,  # aux hidden states (up to 4 per launch)
    src3_ptr,
    dst3_ptr,
    w3,
    src4_ptr,
    dst4_ptr,
    w4,
    src5_ptr,
    dst5_ptr,
    w5,
    USE_INV: tl.constexpr,  # False: plain copy, dst row i = src row i
    BLOCK: tl.constexpr,
):
    # One program per destination row.
    i = tl.program_id(0).to(tl.int64)
    if USE_INV:
        idx = tl.load(inv_rows_ptr + i)
        # Stale entries past the batch's rows must not read out of bounds.
        keep = (idx >= 0) & (idx < num_src)
        src_row = tl.maximum(idx, 0).to(tl.int64)
    else:
        keep = True
        src_row = i
    _copy_row(src0_ptr, dst0_ptr, src_row, i, w0, keep, BLOCK)
    _copy_row(src1_ptr, dst1_ptr, src_row, i, w1, keep, BLOCK)
    _copy_row(src2_ptr, dst2_ptr, src_row, i, w2, keep, BLOCK)
    _copy_row(src3_ptr, dst3_ptr, src_row, i, w3, keep, BLOCK)
    _copy_row(src4_ptr, dst4_ptr, src_row, i, w4, keep, BLOCK)
    _copy_row(src5_ptr, dst5_ptr, src_row, i, w5, keep, BLOCK)


def _gather_replay_states(
    states: ReplayStates,
    rows: torch.Tensor,
    row_buffers: list[torch.Tensor],
    row_scratches: list[torch.Tensor],
) -> ReplayStates:
    """Gather the replay rows of every state in one kernel launch, realigning
    the source's indexer row buffers into scratch on the way (they are copied
    back by the caller once the gather has finished reading them)."""
    num_rows = rows.shape[0]
    hidden, positions, input_ids, pre_mix, post_mix, res_mix, residual = states
    gathered: list[torch.Tensor | None] = []
    slots: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
    for src in (hidden, positions, input_ids, pre_mix, post_mix, res_mix, residual):
        dst = (
            torch.empty((num_rows, *src.shape[1:]), dtype=src.dtype, device=src.device)
            if src is not None
            else None
        )
        gathered.append(dst)
        slots.append((src, dst))
    slots += list(zip(row_buffers, row_scratches))
    assert len(slots) <= 9
    args: list = []
    for src, dst in slots + [(None, None)] * (9 - len(slots)):
        if src is None or dst is None:
            args += [rows, rows, 0]  # dummy slot, never read (width 0)
        else:
            assert src.is_contiguous() and dst.is_contiguous()
            # The per-row width comes from the source: scratch destinations
            # keep their full-size row count.
            args += [src, dst, src.numel() // src.shape[0]]
    _replay_gather_kernel[(num_rows,)](rows, *args, BLOCK=1024)
    return (
        gathered[0],  # type: ignore[return-value]
        gathered[1],
        gathered[2],
        gathered[3],
        gathered[4],
        gathered[5],
        gathered[6],
    )


def _scatter_replay_outputs(
    srcs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    dsts: tuple[torch.Tensor, ...] | list[torch.Tensor],
    inv_rows: torch.Tensor | None,
) -> None:
    """Scatter replay outputs to their full-batch rows in one kernel launch
    per six tensors (zeroing rows without a replay row), or copy them whole
    when ``inv_rows`` is None."""
    num_dst = dsts[0].shape[0]
    num_src = srcs[0].shape[0]
    for off in range(0, len(srcs), 6):
        chunk = list(zip(srcs[off : off + 6], dsts[off : off + 6]))
        args: list = []
        for src, dst in chunk + [(None, None)] * (6 - len(chunk)):
            if src is None or dst is None:
                args += [dsts[0], dsts[0], 0]  # dummy slot, never read
            else:
                assert src.is_contiguous() and dst.is_contiguous()
                args += [src, dst, dst.numel() // num_dst]
        _replay_scatter_kernel[(num_dst,)](
            inv_rows if inv_rows is not None else srcs[0],
            num_src,
            *args,
            USE_INV=inv_rows is not None,
            BLOCK=1024,
        )


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_mega_moe = vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4Attention._run_parallel_input_projections
        # (compressor kv_score, indexer.weights_proj). fused_wqa_wkv stays on
        # the default stream.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        # Two-level candidate filtering: the indexer at
        # candidate_source_layer_id publishes the top candidate blocks of
        # compressed positions here; later ratio-1 indexers (24/28/32/36)
        # mask their scores with it.
        candidate_source_layer = getattr(config, "candidate_source_layer_id", -1)
        candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        if candidate_source_layer >= 0 and candidate_topk_blocks > 0:
            self.candidate_block_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                candidate_topk_blocks,
                dtype=torch.int32,
            )
        else:
            self.candidate_block_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.engram_layout = EngramLayout.from_config(config)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                candidate_block_buffer=self.candidate_block_buffer,
                engram_layout=self.engram_layout,
            ),
            prefix=f"{prefix}.layers",
        )

        # Decoder-side SWA bounded replay: layers past the last KV source
        # prefill only each request's trailing window (see _replay_forward).
        self.decoder_replay_start = self.end_layer
        self.decoder_replay_window: int | None = None
        # Set by the model state for every forward that trims a request down
        # to its window (on any DP rank); None when the replay layers run on
        # the whole batch with the batch's own attention metadata.
        self.replay_batch: ReplayBatch | None = None
        self._replay_row_buffers: list[torch.Tensor] = []
        self._replay_row_scratch: list[torch.Tensor] | None = None
        self._replay_static_outputs = False
        self._replay_max_output_rows = 0
        self._replay_outputs: list[torch.Tensor] | None = None
        cut = max(config.kv_source_layer_ids)
        if (
            cut < self.end_layer - 1
            and self._decoder_replay_supported(vllm_config, cut)
            and self.layers[cut].attn.swa_cache_layer.bounded_replay
        ):
            self.decoder_replay_start = cut + 1
            self.decoder_replay_window = config.sliding_window
            # Per-row indexer outputs the source publishes for the layers
            # after it.
            source_attn = self.layers[cut].attn
            bufs = (source_attn.topk_indices_buffer, source_attn.candidate_block_buffer)
            self._replay_row_buffers = [buf for buf in bufs if buf is not None]
            # Under breakable piecewise graphs the replay runs as an eager
            # break of the model graph; the graph's post-break segments read
            # the replay's outputs at fixed addresses: static buffers, sized
            # on the first (eager) forward to the largest capture.
            compilation_config = vllm_config.compilation_config
            capture_sizes = compilation_config.cudagraph_capture_sizes
            self._replay_static_outputs = (
                compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
                and is_breakable_cudagraph_enabled()
                and bool(capture_sizes)
            )
            self._replay_max_output_rows = max(capture_sizes) if capture_sizes else 0

        # The n-gram hash needs a slot-keyed rolling store of compressed ids
        # (chunked prefill / decode lookback); key it off the first local
        # layer's sliding-window KV cache. Only PP ranks owning an engram
        # layer need it.
        self.engram_hash: NgramHashState | None = None
        self.engram_dp_shared_memory = bool(
            vllm_config.engram_config and vllm_config.engram_config.dp_shared_memory
        )
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None:
            local_engram = any(
                isinstance(layer, DeepseekV4DecoderLayer) and layer.engram is not None
                for layer in islice(self.layers, self.start_layer, self.end_layer)
            )
            if local_engram:
                first_layer = next(
                    iter(islice(self.layers, self.start_layer, self.end_layer))
                )
                swa_cache_module = first_layer.attn.swa_cache_layer
                self.engram_hash = NgramHashState(
                    vllm_config, self.engram_layout, swa_cache_module
                )
                self.engram_swa_prefix = swa_cache_module.prefix

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until the final hc collapse — plus the
        # (num_tokens, hc_mult) pre-mix the next rank's first layer needs
        # for its attention collapse.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        # Engram n-gram hashes for the whole (flattened) batch, computed once
        # on the full token stream — before any sequence-parallel sharding —
        # and consumed by the engram layers (1 and 14) below. Skipped on
        # profile runs (KV cache unbound).
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[dbo_current_ubatch_id()]
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = typing.cast(
                    "DeepseekSparseSWAMetadata", attn_metadata[self.engram_swa_prefix]
                )
                # Image-span tokens are dead: they break n-grams (hash op
                # takes True=dead) and their gate is zeroed (Engram.forward
                # takes True=keep).
                image_mask = image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    if not self.engram_hash.use_slot_cache:
                        raise NotImplementedError(
                            "engram needs `lookback_token_ids` from the model "
                            "runner (the DBO/ubatch wrapper drops model kwargs)"
                        )
                    num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
                    lookback_token_ids = input_ids.new_full(
                        (num_reqs, self.engram_hash.lookback_depth), -1
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                    swa_metadata.slot_mapping,
                    swa_metadata.block_table,
                )
            elif not self.engram_dp_shared_memory and get_engram_dp_size() > 1:
                # DP-sharded lookups are collective, so a replica skipping the
                # hash still has to reach them.
                engram_hashes, engram_mask = self.engram_hash.dummy_hashes(input_ids)
            if engram_hashes is not None:
                # Gather all Engram rows before entering the decoder layers.
                # One gather feeds every layer sharing the DP-split table.
                gathered_hashes = gather_engram_hashes(
                    engram_hashes, dp_shared_memory=self.engram_dp_shared_memory
                )
                for layer in islice(self.layers, self.start_layer, self.end_layer):
                    engram = getattr(layer, "engram", None)
                    if engram is not None:
                        engram.prepare_embeddings(
                            gathered_hashes[:, engram.layer_hash_index]
                        )

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        mega_gate_metadata = self._mega_gate_metadata(input_ids)

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
            range(self.start_layer, self.decoder_replay_start),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_by_layer,
            engram_hashes,
            engram_mask,
            mega_gate_metadata,
        )
        if self.decoder_replay_start < self.end_layer:
            # The drafter's aux hidden state layers are all inside the replay
            # range (checked in _decoder_replay_supported), so the prefix
            # layers capture none and the replay returns all of them.
            hidden_states, pre_mix, *aux_hidden_states = self._replay_forward(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
        else:
            hidden_states = self._collapse(
                hidden_states,
                residual,
                post_mix,
                res_mix,
                aux_hidden_by_layer,
                full_num_tokens,
            )
            aux_hidden_states = [
                aux_hidden_by_layer[layer_id]
                for layer_id in self.aux_hidden_state_layers
                if layer_id in aux_hidden_by_layer
            ]

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )

        # MTP needs full HC states; otherwise collapse and normalize locally
        # before gathering to reduce communication.
        if self._mtp_hidden_buffer is not None:
            if self.use_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference applies via
        # ``last_layer.hc_pre(h, pre_mix)`` (v4.1 has no learned hc_head).
        assert pre_mix is not None
        hidden_states = hc_collapse_triton(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if self.use_sequence_parallel and self._mtp_hidden_buffer is None:
            # Without MTP, gather only the collapsed and normalized hidden states.
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _run_layers(
        self,
        layer_ids: range,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        mega_gate_metadata: MegaGateRoutingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Every layer's post runs inside the next layer's fused pre, so aux
        # hidden states are read back from there instead of recomputed.
        full_num_tokens = positions.shape[0]
        for idx in layer_ids:
            hidden_states, residual, post_mix, res_mix, pre_mix, previous_aux = (
                self.layers[idx](
                    hidden_states,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                    engram_hashes,
                    engram_mask,
                    capture_previous_aux=idx in self.aux_hidden_state_layers,
                    mega_gate_metadata=mega_gate_metadata,
                )
            )
            if previous_aux is not None:
                # idx is the one-based id of the layer whose post this is.
                if self.use_sequence_parallel:
                    previous_aux = sp_all_gather(previous_aux)[:full_num_tokens]
                aux_hidden_by_layer[idx] = previous_aux
        return hidden_states, residual, post_mix, res_mix, pre_mix

    def _collapse(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        full_num_tokens: int,
    ) -> torch.Tensor:
        # The last layer has no successor to fold its post into.
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        if self.end_layer in self.aux_hidden_state_layers:
            final_aux = hidden_states.mean(dim=1)
            if self.use_sequence_parallel:
                final_aux = sp_all_gather(final_aux)[:full_num_tokens]
            aux_hidden_by_layer[self.end_layer] = final_aux
        return hidden_states

    def _run_decoder_replay_layers(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
        mega_gate_metadata: MegaGateRoutingMetadata | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Layers past the last KV source, on whatever rows they are given."""
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(
            range(self.decoder_replay_start, self.end_layer),
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_by_layer,
            mega_gate_metadata=mega_gate_metadata,
        )
        hidden_states = self._collapse(
            hidden_states,
            residual,
            post_mix,
            res_mix,
            aux_hidden_by_layer,
            positions.shape[0],
        )
        aux_hidden = (
            aux_hidden_by_layer[layer_id]
            for layer_id in self.aux_hidden_state_layers
            if layer_id in aux_hidden_by_layer
        )
        return hidden_states, pre_mix, *aux_hidden

    def _replay_forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the layers past the last KV source: on the step's replay batch
        when the model state trimmed requests to their trailing window, else
        on the whole batch.

        Under breakable piecewise CUDA graphs this region is an eager break of
        the model graph: it runs eagerly at every replay, on whatever rows the
        step's replay batch holds, and writes its outputs into fixed buffers
        that the model graph's post-break segments read by address. Running
        eagerly -- rather than in a graph of the replay's own -- keeps the
        replay batch shape-dynamic: no padding to a capture size, and no
        nested capture.
        """
        states: ReplayStates = (
            hidden_states,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
        )
        outer = BreakableCUDAGraphCapture.current()
        if outer is not None and outer.capturing:
            # The model graph's eager break: the segments after it read the
            # replay's outputs at fixed addresses.
            assert self._replay_outputs is not None
            outputs = tuple(
                out[: hidden_states.shape[0]] for out in self._replay_outputs
            )
            weak_states = _weak_ref_states(states)
            outer.add_eager(lambda: self._replay_run(weak_states, outputs))
            return outputs

        outputs = self._replay_run(states)
        if self._replay_static_outputs and self._replay_outputs is None:
            self._replay_outputs = [
                out.new_zeros((self._replay_max_output_rows, *out.shape[1:]))
                for out in outputs
            ]
        return outputs

    def _mega_gate_metadata(
        self, input_ids: torch.Tensor | None
    ) -> MegaGateRoutingMetadata | None:
        """The MoE gate routing metadata for a batch's (or a replay batch's)
        input ids; None where mega MoE or the ids are absent (non-first PP
        ranks)."""
        if input_ids is None or not self.use_mega_moe:
            return None
        sentinel_id = None
        if getattr(self.config, "vision_n_layers", 0) > 0:
            sentinel_id = IMAGE_SENTINEL_BASE_ID
        return prepare_mega_gate_routing_metadata(
            input_ids, has_hash_routing=False, image_sentinel_base_id=sentinel_id
        )

    def _replay_run(
        self,
        states: ReplayStates,
        outputs: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the replay layers and scatter the results to full-batch rows,
        into ``outputs`` when given (the model graph's static buffers)."""
        batch = self.replay_batch
        if batch is None:
            # A graph replay's step with nothing to trim: the whole batch,
            # under the batch's own forward context.
            replay_outputs = self._run_decoder_replay_layers(
                *states, self._mega_gate_metadata(states[2])
            )
            if outputs is None:
                return replay_outputs
            _scatter_replay_outputs(replay_outputs, outputs, inv_rows=None)
            return outputs
        num_tokens = batch.rows.shape[0]

        # Realign the source's per-row indexer outputs to the replay rows.
        # The gather cannot write them in place (a kept row can sit below a
        # later replay row), so it lands in scratch and is copied back.
        scratches: list[torch.Tensor] = []
        if batch.trims and self._replay_row_buffers:
            if self._replay_row_scratch is None:
                self._replay_row_scratch = [
                    torch.empty_like(buf) for buf in self._replay_row_buffers
                ]
            scratches = self._replay_row_scratch
        gathered = _gather_replay_states(
            states, batch.rows, self._replay_row_buffers if scratches else [], scratches
        )
        if scratches:
            _scatter_replay_outputs(
                [s[:num_tokens] for s in scratches],
                [buf[:num_tokens] for buf in self._replay_row_buffers],
                inv_rows=None,
            )
        forward_context = get_forward_context()
        replay_context = replace(
            forward_context,
            attn_metadata=batch.attn_metadata,
            slot_mapping=batch.slot_mapping,
            is_padding=batch.is_padding,
            dp_metadata=batch.dp_metadata or forward_context.dp_metadata,
        )
        with override_forward_context(replay_context):
            replay_outputs = self._run_decoder_replay_layers(
                *gathered, self._mega_gate_metadata(gathered[2])
            )

        if outputs is None:
            if not batch.trims:
                return replay_outputs
            num_batch_tokens = states[0].shape[0]
            outputs = tuple(
                t.new_empty((num_batch_tokens, *t.shape[1:])) for t in replay_outputs
            )
        _scatter_replay_outputs(replay_outputs, outputs, batch.inv_rows)
        return outputs

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers and self.decoder_replay_start < self.end_layer:
            # Decoder replay only produces aux hidden states for layers in
            # the replay range; _decoder_replay_supported rejects drafter
            # configs needing earlier ones.
            assert min(layers) >= self.decoder_replay_start, (
                "Decoder SWA bounded replay requires the drafter's aux hidden "
                "state layers to be past the last KV source layer "
                f"({self.decoder_replay_start - 1}), got {layers}"
            )
        super()._set_aux_hidden_state_layers(layers)

    def _decoder_replay_supported(self, vllm_config: VllmConfig, cut: int) -> bool:
        """Whether this rank may trim the layers after ``cut``; warns when not."""
        parallel_config = vllm_config.parallel_config
        spec_config = vllm_config.speculative_config
        draft_config = spec_config.draft_model_config if spec_config else None
        draft_hf_config = getattr(draft_config, "hf_config", None)
        draft_window = getattr(draft_hf_config, "sliding_window", None)
        draft_layer_types = getattr(draft_hf_config, "layer_types", None) or ()
        window = self.config.sliding_window
        if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:
            reason = (
                "the pipeline stage holding the last KV source layer must also "
                "hold every layer after it"
            )
        elif (
            self.use_sequence_parallel
            or parallel_config.prefill_context_parallel_size > 1
            or parallel_config.use_ubatching
        ):
            reason = (
                "the replay-layer batch shrinks per rank, which sequence and "
                "prefill-context parallelism and microbatching cannot follow"
            )
        elif any(i > cut for i in getattr(self.config, "engram_layer_ids", ())):
            reason = "an Engram layer sits after the last KV source layer"
        elif draft_config is not None and (
            draft_window is None
            or draft_window > window
            or any(t != "sliding_attention" for t in draft_layer_types)
        ):
            reason = (
                f"the drafter (sliding window {draft_window}) reads hidden states "
                f"outside the target's {window}-token window"
            )
        elif (aux_layers := self._drafter_aux_layers(spec_config)) is not None and (
            min(aux_layers) <= cut
        ):
            reason = (
                "the drafter reads aux hidden states captured at or before the "
                "last KV source layer"
            )
        else:
            return True
        logger.warning_once("Decoder SWA bounded replay is off: %s.", reason)
        return False

    def _drafter_aux_layers(
        self, spec_config: SpeculativeConfig | None
    ) -> tuple[int, ...] | None:
        """The drafter's effective aux hidden state layers, or None when the
        spec method reads no aux hidden states.

        Mirrors the runner's setup (``set_eagle3_aux_hidden_state_layers``,
        whose aux-using method list this method's list must stay in sync with):
        the layers come from the draft config, else the EAGLE3 default.
        """
        if spec_config is None or spec_config.method not in (
            "eagle3",
            "dflash",
            "dspark",
            "extract_hidden_states",
        ):
            return None
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        aux_layers = get_eagle3_aux_layers_from_config(spec_config)
        if aux_layers:
            return aux_layers
        num_layers = self.config.num_hidden_layers
        return (2, num_layers // 2, num_layers - 3)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Block-FP8 shared experts: pad the intermediate up to the TP-uniform
        # block count so the standard loaders below slice it evenly (trailing
        # ranks land on the zero pad). SP / unquantized ones need no padding.
        pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not self.use_sequence_parallel
        )

        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    head, _, leaf = name.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        name = suffixed
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at
                    # ``<head>.base_layer.<leaf>``; the checkpoint is plain.
                    if name not in params_dict:
                        head, _, leaf = name.rpartition(".")
                        suffixed = f"{head}.base_layer.{leaf}"
                        if suffixed in params_dict:
                            name = suffixed
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    @staticmethod
    def _pad_shared_expert_weight(
        quant_config: QuantizationConfig | None,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-pad a block-FP8 shared-expert weight/scale on its intermediate
        axis so the standard TP loaders split it into even, block-aligned shards
        (trailing ranks get the zero pad). gate (w1)/up (w3) [I, H] pad dim 0;
        down (w2 -> down_proj) [H, I] pads dim 1.
        """
        block_size = getattr(quant_config, "weight_block_size", None)
        assert block_size is not None
        # Round the intermediate axis up to a whole number of TP shards. The axis
        # is in elements for weights (step = block) and in blocks for scales.
        step = (
            1 if name.endswith(("weight_scale_inv", "weight_scale")) else block_size[0]
        )
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        if first_layer.ffn.use_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.ffn.finalize_mega_moe_weights()

    def finalize_mega_attn_weights(self) -> None:
        """Permute wq_b / wo_a into FlashMLA's mega-attention layouts.

        A no-op for every other attention layer, and idempotent, so a second
        post-load pass cannot permute twice.
        """
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            finalize = getattr(layer.attn, "finalize_loaded_weights", None)
            if finalize is not None:
                finalize()

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, DeepseekV4DecoderLayer):
            broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )
            if layer.hc_attn_fn_broadcast is None:
                layer.hc_attn_fn_broadcast = broadcast
            else:
                layer.hc_attn_fn_broadcast.copy_(broadcast)


def _linear_scale_param_name(vllm_config: VllmConfig, expert_dtype: str) -> str:
    """Parameter name the linear quant method registers for weight scales.

    Native MXFP8 checkpoints ([32, 32] blocks with MXFP4 experts) route linear
    layers through ModelOptLinearMethod, which registers ``weight_scale``;
    block-FP8 linear layers register ``weight_scale_inv``.
    """
    use_mxfp8 = (
        getattr(vllm_config.quant_config, "weight_block_size", None) == [32, 32]
        and expert_dtype == "fp4"
    )
    return "weight_scale" if use_mxfp8 else "weight_scale_inv"


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, linear_scale_name: str = "weight_scale_inv"
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). Linear scales use the
        # parameter name registered by their quantization method.
        scale_regex = {
            # ``.base_layer.``-namespace variant (LoRA-wrapped experts).
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale",
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # The ``embed.weight`` -> ``embed_tokens.weight`` suffix rule
            # renames the engram fp8 table but not its scale; route the
            # scale explicitly to the same module.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            # ``.base_layer.``-namespace variant of the above.
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale_inv",
            # Same engram reroute as the fp4 branch above.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            "mtp.": None,
            # The v4.1 checkpoint declares the VL arch even for text-only
            # use; the text backbone drops the vision-tower weights.
            "vision.": None,
            "aligner.": None,
            "image_": None,
        },
    )


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list["DeepseekV4MoE"]

    def extract_moe_parameters(self, example_moe: "DeepseekV4MoE | None") -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekV41LLMForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsEagle3,
    SupportsLoRA,
    DeepseekV4MixtureOfExperts,
):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # The MTP draft head is not LoRA-adapted.
    lora_skip_prefixes = ["mtp."]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    @property
    def decoder_replay_model(self) -> DeepseekV4Model | None:
        """The inner text model when decoder-side SWA bounded replay is on.

        The model state sets its ``replay_batch`` every step (None when the
        replay layers run on the whole batch)."""
        if self.model.decoder_replay_window is None:
            return None
        return self.model

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe: DeepseekV4MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV4DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV4MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the model runner
        passes them as `lookback_token_ids`."""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-collapse residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.process_weights_after_loading()
        return loaded_params

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()
        self.model.finalize_mega_attn_weights()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
