# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.tasks import POOLING_TASKS, PoolingTask
from vllm.v1.pool.metadata import PoolingStates as HiddenStatesCache
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

# Stable integer id per pooling task, for the tensorized (Bucket-1) params
# consumed by the V2-native seq-level fast path. -1 == unset (e.g. warmup).
_POOLING_TASK_IDS: dict[PoolingTask, int] = {
    task: i for i, task in enumerate(POOLING_TASKS)
}


class PoolingStates:
    """Runner-side, per-request pooling state for Model Runner V2.

    Mirrors ``SamplingStates``: the "Bucket-1" scalar pooling params live in
    per-request ``UvaBackedTensor``s (gathered GPU-side via ``idx_mapping``) so
    the V2-native seq-level fast path can consume them without Python loops,
    while the full ``PoolingParams`` objects and the chunked-prefill hidden-state
    caches are kept in Python dicts keyed by the persistent request slot for the
    shared ``model.pooler()`` path (which needs the non-tensorizable fields).
    """

    def __init__(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs

        # Tensorized Bucket-1 params (per persistent slot; gathered by
        # idx_mapping into batch order at pooling time).
        self.task_id = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.use_activation = UvaBackedTensor(max_num_reqs, dtype=torch.bool)
        self.dimensions = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.task_id.np.fill(-1)
        self.task_id.copy_to_uva()
        self.dimensions.np.fill(-1)
        self.dimensions.copy_to_uva()

        # Full params + chunked-prefill caches, keyed by persistent slot index.
        self.params: dict[int, PoolingParams] = {}
        self.hidden_caches: dict[int, HiddenStatesCache] = {}

    def add_request(self, req_idx: int, params: PoolingParams) -> None:
        self.params[req_idx] = params
        self.hidden_caches[req_idx] = HiddenStatesCache()

        task = params.task
        self.task_id.np[req_idx] = -1 if task is None else _POOLING_TASK_IDS[task]
        self.use_activation.np[req_idx] = bool(params.use_activation)
        self.dimensions.np[req_idx] = (
            params.dimensions if params.dimensions is not None else -1
        )

    def remove_request(self, req_idx: int) -> None:
        self.params.pop(req_idx, None)
        self.hidden_caches.pop(req_idx, None)

    def apply_staged_writes(self) -> None:
        self.task_id.copy_to_uva()
        self.use_activation.copy_to_uva()
        self.dimensions.copy_to_uva()

    def get_params(self, idx_mapping_np: np.ndarray) -> list[PoolingParams]:
        return [self.params[i] for i in idx_mapping_np]

    def get_hidden_caches(self, idx_mapping_np: np.ndarray) -> list[HiddenStatesCache]:
        return [self.hidden_caches[i] for i in idx_mapping_np]
