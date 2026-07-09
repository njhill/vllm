# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.tasks import POOLING_TASKS, PoolingTask
from vllm.v1.pool.metadata import PoolingStates as HiddenStatesCache
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

# Stable integer id per pooling task, for the tensorized params consumed by
# the V2-native fast path. -1 == unset (e.g. warmup).
_POOLING_TASK_IDS: dict[PoolingTask, int] = {
    task: i for i, task in enumerate(POOLING_TASKS)
}

# Sentinel for "no token-type boundary": every position compares below it.
NO_TOKEN_TYPES = torch.iinfo(torch.int32).max


class PoolingStates:
    """Runner-side, per-request pooling state for Model Runner V2.

    Everything derivable from the request's params or prompt token ids is
    computed once here, at add time, into slot-indexed storage: scalar params
    live in ``UvaBackedTensor``s (gathered GPU-side via ``idx_mapping``),
    prompt-derived data (step-tag positions, first/last token ids) in numpy /
    small per-slot arrays. The full ``PoolingParams`` objects and the
    chunked-prefill hidden-state caches are kept for the shared
    ``model.pooler()`` fallback path (which needs the non-tensorizable fields).
    """

    def __init__(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs

        # Tensorized scalar params (per persistent slot; gathered by
        # idx_mapping into batch order at pooling time).
        self.task_id = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.use_activation = UvaBackedTensor(max_num_reqs, dtype=torch.bool)
        self.dimensions = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        # First token index with token_type_id == 1 (cross-encoder second
        # segment); NO_TOKEN_TYPES when the request has none.
        self.token_type_boundary = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.task_id.np.fill(-1)
        self.task_id.copy_to_uva()
        self.dimensions.np.fill(-1)
        self.dimensions.copy_to_uva()
        self.token_type_boundary.np.fill(NO_TOKEN_TYPES)
        self.token_type_boundary.copy_to_uva()
        self.num_token_type_reqs = 0

        # First/last prompt token ids, for BOS/EOS output filtering.
        self.first_token_np = np.full(max_num_reqs, -1, dtype=np.int64)
        self.last_token_np = np.full(max_num_reqs, -1, dtype=np.int64)

        # Prompt positions matching step_tag_id / label-column selections,
        # precomputed at add time for STEP pooling. Slot-keyed; absent when
        # the request doesn't use them.
        self.step_indices: dict[int, np.ndarray] = {}
        self.returned_token_ids: dict[int, np.ndarray] = {}

        # Full params + chunked-prefill caches, keyed by persistent slot index.
        self.params: dict[int, PoolingParams] = {}
        self.hidden_caches: dict[int, HiddenStatesCache] = {}

    def add_request(
        self,
        req_idx: int,
        params: PoolingParams,
        prompt_token_ids: list[int] | None,
    ) -> None:
        self.params[req_idx] = params
        self.hidden_caches[req_idx] = HiddenStatesCache()

        task = params.task
        self.task_id.np[req_idx] = -1 if task is None else _POOLING_TASK_IDS[task]
        self.use_activation.np[req_idx] = bool(params.use_activation)
        self.dimensions.np[req_idx] = (
            params.dimensions if params.dimensions is not None else -1
        )

        boundary = NO_TOKEN_TYPES
        if params.extra_kwargs is not None:
            compressed = params.extra_kwargs.get("compressed_token_type_ids")
            if compressed is not None:
                boundary = int(compressed)
        if self.token_type_boundary.np[req_idx] != NO_TOKEN_TYPES:
            self.num_token_type_reqs -= 1
        self.token_type_boundary.np[req_idx] = boundary
        if boundary != NO_TOKEN_TYPES:
            self.num_token_type_reqs += 1

        self.step_indices.pop(req_idx, None)
        self.returned_token_ids.pop(req_idx, None)
        if prompt_token_ids is not None:
            self.first_token_np[req_idx] = prompt_token_ids[0]
            self.last_token_np[req_idx] = prompt_token_ids[-1]
            if params.step_tag_id is not None:
                prompt_np = np.asarray(prompt_token_ids, dtype=np.int64)
                self.step_indices[req_idx] = np.nonzero(
                    prompt_np == params.step_tag_id
                )[0].astype(np.int64)
        else:
            self.first_token_np[req_idx] = -1
            self.last_token_np[req_idx] = -1
        if params.returned_token_ids:
            self.returned_token_ids[req_idx] = np.asarray(
                params.returned_token_ids, dtype=np.int64
            )

    def remove_request(self, req_idx: int) -> None:
        self.params.pop(req_idx, None)
        self.hidden_caches.pop(req_idx, None)
        self.step_indices.pop(req_idx, None)
        self.returned_token_ids.pop(req_idx, None)
        if self.token_type_boundary.np[req_idx] != NO_TOKEN_TYPES:
            self.num_token_type_reqs -= 1
            self.token_type_boundary.np[req_idx] = NO_TOKEN_TYPES

    def apply_staged_writes(self) -> None:
        self.task_id.copy_to_uva()
        self.use_activation.copy_to_uva()
        self.dimensions.copy_to_uva()
        self.token_type_boundary.copy_to_uva()

    def get_params(self, idx_mapping_np: np.ndarray) -> list[PoolingParams]:
        return [self.params[i] for i in idx_mapping_np]

    def get_hidden_caches(self, idx_mapping_np: np.ndarray) -> list[HiddenStatesCache]:
        return [self.hidden_caches[i] for i in idx_mapping_np]
