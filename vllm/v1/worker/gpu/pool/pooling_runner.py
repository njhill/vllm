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
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner
from vllm.v1.worker.gpu.pool.states import PoolingStates
from vllm.v1.worker.gpu.states import RequestState


class PoolingRunner:
    """Model Runner V2 pooling: builds ``PoolingMetadata`` from GPU-side batch
    state and dispatches to the shared ``model.pooler()`` for full task parity.

    Metadata is assembled entirely from existing CPU mirrors
    (``prompt_len.np``, ``num_computed_tokens_np``, ``num_scheduled_tokens``) and
    the already-on-GPU ``query_start_loc``, so no GPU->CPU sync is introduced.
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
        self.late_interaction = LateInteractionRunner()

    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        if not is_pooling_model(model):
            return []
        return list(model.pooler.get_supported_tasks())

    def add_request(
        self, req_idx: int, req_id: str, pooling_params: PoolingParams
    ) -> None:
        task = pooling_params.task
        if task is not None:
            # Apply pooler-declared updates (e.g. requires_token_ids) once, at
            # add time; task is None only for warmup dummy requests.
            self.pooler.get_pooling_updates(task).apply(pooling_params)
        self.states.add_request(req_idx, pooling_params)
        self.late_interaction.register_request(req_id, pooling_params)

    def remove_request(self, req_idx: int, req_id: str) -> None:
        self.states.remove_request(req_idx)
        self.late_interaction.on_requests_finished((req_id,))

    def apply_staged_writes(self) -> None:
        self.states.apply_staged_writes()

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[PoolerOutput, list[bool]]:
        num_reqs = input_batch.num_reqs
        idx_mapping_np = input_batch.idx_mapping_np

        # Drop cudagraph padding so token-wise poolers only see real tokens.
        hidden_states = hidden_states[: input_batch.num_tokens]

        # All-CPU, all sync-free: reuse the mirrors prepared in prepare_inputs.
        num_scheduled_tokens_np = input_batch.num_scheduled_tokens
        prompt_lens_np = req_states.prompt_len.np[idx_mapping_np]
        seq_lens_np = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
        prompt_lens = torch.from_numpy(prompt_lens_np)
        seq_lens_cpu = torch.from_numpy(seq_lens_np)

        pooling_params = self.states.get_params(idx_mapping_np)
        pooling_states = self.states.get_hidden_caches(idx_mapping_np)

        prompt_token_ids = None
        prompt_token_ids_cpu = None
        if any(p.requires_token_ids for p in pooling_params):
            prompt_token_ids, prompt_token_ids_cpu = self._gather_prompt_token_ids(
                req_states, idx_mapping_np, prompt_lens_np
            )

        pooling_metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_cpu=prompt_token_ids_cpu,
            pooling_params=pooling_params,
            pooling_states=pooling_states,
        )
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=hidden_states.device,
            query_start_loc_gpu=input_batch.query_start_loc[: num_reqs + 1],
        )

        raw_pooler_output: PoolerOutput = self.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        # A request only emits an output once its whole prompt is pooled;
        # unfinished chunked-prefill requests emit None. Computed CPU-side
        # from the mirrors above (no sync).
        finished_mask = (seq_lens_np == prompt_lens_np).tolist()

        raw_pooler_output = self.late_interaction.postprocess_pooler_output(
            raw_pooler_output=raw_pooler_output,
            pooling_params=pooling_params,
            req_ids=input_batch.req_ids,
            finished_mask=finished_mask,
        )
        return raw_pooler_output, finished_mask

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
        num_scheduled_tokens_np = np.full(num_reqs, num_tokens // num_reqs)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        dummy_params = PoolingParams(task=task)
        dummy_params.verify(self.model_config)
        self.pooler.get_pooling_updates(task).apply(dummy_params)

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

        try:
            return self.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
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
