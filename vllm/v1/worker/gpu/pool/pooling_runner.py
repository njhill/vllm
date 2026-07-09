# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.pool.metadata import PoolingStates as HiddenStatesCache
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pool.late_interaction import LateInteractionStates
from vllm.v1.worker.gpu.pool.native import PoolingBatch, build_native_poolers
from vllm.v1.worker.gpu.pool.states import _POOLING_TASK_IDS, PoolingStates
from vllm.v1.worker.gpu.states import RequestState


class PoolingRunner:
    """Model Runner V2 pooling.

    Recognized pooler structures (CLS/LAST/MEAN/ALL/STEP with the standard
    heads, BOS/EOS filtering, BGE-M3) run through the V2-native path: pooling
    is driven directly by ``query_start_loc`` + numpy mirrors and the
    tensorized per-slot params in ``PoolingStates`` (see ``native.py``) — no
    ``PoolingMetadata``/``PoolingCursor`` and no per-request Python loops.
    Unrecognized poolers (plugins, custom fns/subclasses) and mixed-task
    batches fall back to the shared ``model.pooler()`` with metadata assembled
    from the same sync-free CPU mirrors.
    """

    def __init__(
        self,
        model: nn.Module,
        max_num_reqs: int,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.model = cast(VllmModelForPooling, model)
        self.pooler = self.model.pooler
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device

        self.states = PoolingStates(max_num_reqs)
        self.late_interaction = LateInteractionStates(max_num_reqs)
        self.native_poolers = build_native_poolers(self.pooler)
        self.native_poolers_by_id = {
            _POOLING_TASK_IDS[task]: pooler
            for task, pooler in self.native_poolers.items()
        }
        # Cross-encoder token-type bits are only decoded by BERT-family
        # embeddings; never write them for other models.
        self.model_decodes_token_types = any(
            getattr(m, "token_type_embeddings", None) is not None
            for m in model.modules()
        )

    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        if not is_pooling_model(model):
            return []
        return list(model.pooler.get_supported_tasks())

    def add_request(
        self,
        req_idx: int,
        pooling_params: PoolingParams,
        prompt_token_ids: list[int] | None,
    ) -> None:
        task = pooling_params.task
        if task is not None:
            # Apply pooler-declared updates (e.g. requires_token_ids) once, at
            # add time; task is None only for warmup dummy requests.
            self.pooler.get_pooling_updates(task).apply(pooling_params)
        self.states.add_request(req_idx, pooling_params, prompt_token_ids)
        self.late_interaction.add_request(req_idx, pooling_params)

    def remove_request(self, req_idx: int) -> None:
        self.states.remove_request(req_idx)
        self.late_interaction.remove_request(req_idx)

    def apply_staged_writes(self) -> None:
        self.states.apply_staged_writes()

    def encode_token_type_ids(self, input_batch: InputBatch) -> None:
        """Write cross-encoder token-type bits in-band into this step's
        ``input_ids`` (BERT-family embeddings decode and strip them; see
        ``_decode_token_type_ids`` in ``models/bert.py``). Fully on-GPU from
        the slot-order UVA boundaries: no sync, no per-request loops, and
        cudagraph-safe since it mutates the persistent input buffer before
        launch/replay."""
        if self.states.num_token_type_reqs == 0 or not self.model_decodes_token_types:
            return
        # Deferred: keeps this module importable without the model-file chain.
        from vllm.model_executor.models.bert import TOKEN_TYPE_SHIFT

        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        boundaries = self.states.token_type_boundary.gpu[input_batch.idx_mapping]
        query_start_loc = input_batch.query_start_loc[: num_reqs + 1]
        per_token_boundary = torch.repeat_interleave(
            boundaries, torch.diff(query_start_loc), output_size=num_tokens
        )
        token_types = (input_batch.positions[:num_tokens] >= per_token_boundary).to(
            torch.int32
        )
        input_batch.input_ids[:num_tokens].bitwise_or_(token_types << TOKEN_TYPE_SHIFT)

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[PoolerOutput, list[bool]]:
        batch = self._build_batch(hidden_states, input_batch, req_states)

        task_ids = self.states.task_id.np[batch.idx_mapping_np]
        task_id = int(task_ids[0])
        native = self.native_poolers_by_id.get(task_id)
        if native is not None and (task_ids == task_id).all():
            output = native.pool(batch, self.states)
        else:
            output = self._pool_shared(batch, input_batch, req_states)

        output = self.late_interaction.postprocess_pooler_output(output, batch)
        return output, batch.finished_np.tolist()

    def _build_batch(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> PoolingBatch:
        # All-CPU, all sync-free: reuse the mirrors prepared in prepare_inputs.
        num_reqs = input_batch.num_reqs
        idx_mapping_np = input_batch.idx_mapping_np
        prompt_lens_np = req_states.prompt_len.np[idx_mapping_np]
        seq_lens_np = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
        return PoolingBatch(
            num_reqs=num_reqs,
            # Drop cudagraph padding so poolers only see real tokens.
            hidden_states=hidden_states[: input_batch.num_tokens],
            query_start_loc=input_batch.query_start_loc[: num_reqs + 1],
            idx_mapping=input_batch.idx_mapping,
            idx_mapping_np=idx_mapping_np,
            num_scheduled_tokens_np=input_batch.num_scheduled_tokens,
            prompt_lens_np=prompt_lens_np,
            seq_lens_np=seq_lens_np,
            # A request only emits an output once its whole prompt is pooled;
            # unfinished chunked-prefill requests emit None.
            finished_np=seq_lens_np == prompt_lens_np,
            dimensions_np=self.states.dimensions.np[idx_mapping_np],
            use_activation_np=self.states.use_activation.np[idx_mapping_np],
            dimensions_uva=self.states.dimensions.gpu,
            use_activation_uva=self.states.use_activation.gpu,
        )

    def _pool_shared(
        self,
        batch: PoolingBatch,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> PoolerOutput:
        """Fallback: dispatch to the shared ``model.pooler()`` via
        ``PoolingMetadata`` built from the batch's CPU mirrors (no sync)."""
        idx_mapping_np = batch.idx_mapping_np
        pooling_params = self.states.get_params(idx_mapping_np)
        pooling_states = self.states.get_hidden_caches(idx_mapping_np)

        prompt_token_ids = None
        prompt_token_ids_cpu = None
        if any(p.requires_token_ids for p in pooling_params):
            prompt_token_ids, prompt_token_ids_cpu = self._gather_prompt_token_ids(
                req_states, idx_mapping_np, batch.prompt_lens_np
            )

        pooling_metadata = PoolingMetadata(
            prompt_lens=torch.from_numpy(batch.prompt_lens_np),
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_cpu=prompt_token_ids_cpu,
            pooling_params=pooling_params,
            pooling_states=pooling_states,
        )
        pooling_metadata.build_pooling_cursor(
            batch.num_scheduled_tokens_np,
            torch.from_numpy(batch.seq_lens_np),
            device=batch.device,
            query_start_loc_gpu=batch.query_start_loc,
        )
        return self.pooler(
            hidden_states=batch.hidden_states, pooling_metadata=pooling_metadata
        )

    def _gather_prompt_token_ids(
        self,
        req_states: RequestState,
        idx_mapping_np: np.ndarray,
        prompt_lens_np: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Only reached for step/BOS-EOS poolers (requires_token_ids). Gather the
        # per-request prompt token ids into a [num_reqs, max_prompt_len] tensor;
        # poolers slice each row to its own prompt_len, so trailing positions are
        # left as-is.
        max_prompt_len = int(prompt_lens_np.max())
        idx = torch.from_numpy(idx_mapping_np.astype(np.int64)).to(self.device)
        prompt_token_ids = req_states.all_token_ids.gpu.index_select(0, idx)[
            :, :max_prompt_len
        ].contiguous()
        return prompt_token_ids, prompt_token_ids.to("cpu")

    @torch.inference_mode()
    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> PoolerOutput:
        mm_config = self.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM-encoder-only models do not run the pooler.
            return torch.tensor([])

        supported_tasks = self.get_supported_tasks(self.model)
        if not supported_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support any pooling "
                "tasks. See https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        # Run every task to make sure none of them OOM, then re-run the one with
        # the largest output so downstream buffers are sized for the worst case.
        output_size: dict[PoolingTask, float] = {}
        for task in supported_tasks:
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output

        max_task = max(output_size.items(), key=lambda kv: kv[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def _dummy_pooler_run_task(
        self, hidden_states: torch.Tensor, task: PoolingTask
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        num_reqs = min(num_tokens, self.scheduler_config.max_num_seqs)
        num_scheduled_tokens_np = np.full(
            num_reqs, num_tokens // num_reqs, dtype=np.int32
        )
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs

        dummy_params = PoolingParams(task=task)
        dummy_params.verify(self.model_config)
        self.pooler.get_pooling_updates(task).apply(dummy_params)

        try:
            native = self.native_poolers.get(task)
            if native is not None:
                batch = self._build_dummy_batch(hidden_states, num_scheduled_tokens_np)
                # All dummy requests finish in one step, so no per-slot state
                # beyond the (empty) defaults is touched.
                return native.pool(batch, self.states)
            return self._dummy_pool_shared(
                hidden_states, num_scheduled_tokens_np, dummy_params
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    f"CUDA out of memory occurred when warming up pooler ({task=}) "
                    f"with {num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when initializing "
                    "the engine."
                ) from e
            raise

    def _build_dummy_batch(
        self, hidden_states: torch.Tensor, num_scheduled_tokens_np: np.ndarray
    ) -> PoolingBatch:
        num_reqs = len(num_scheduled_tokens_np)
        query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1:])
        return PoolingBatch(
            num_reqs=num_reqs,
            hidden_states=hidden_states,
            query_start_loc=torch.from_numpy(query_start_loc_np).to(self.device),
            idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=self.device),
            idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            prompt_lens_np=num_scheduled_tokens_np,
            seq_lens_np=num_scheduled_tokens_np,
            finished_np=np.ones(num_reqs, dtype=bool),
            dimensions_np=np.full(num_reqs, -1, dtype=np.int32),
            use_activation_np=np.ones(num_reqs, dtype=bool),
            dimensions_uva=self.states.dimensions.gpu,
            use_activation_uva=self.states.use_activation.gpu,
        )

    def _dummy_pool_shared(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens_np: np.ndarray,
        dummy_params: PoolingParams,
    ) -> PoolerOutput:
        num_reqs = len(num_scheduled_tokens_np)
        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, int(num_scheduled_tokens_np.max())),
            dtype=torch.int32,
            device=self.device,
        )
        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            prompt_token_ids_cpu=dummy_token_ids.cpu(),
            pooling_params=[dummy_params] * num_reqs,
            pooling_states=[HiddenStatesCache() for _ in range(num_reqs)],
        )
        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )
        return self.pooler(hidden_states=hidden_states, pooling_metadata=dummy_metadata)
