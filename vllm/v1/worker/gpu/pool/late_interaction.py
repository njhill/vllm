# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2-native late-interaction (ColBERT-style MaxSim) scoring.

Params are parsed and validated once at ``add_request`` into slot-indexed
state; per-step work is numpy mask selection of finished query/doc requests
plus one ragged-MaxSim triton launch for all doc scores (no padded bmm, no
per-pair copy loops). Only the string-keyed query cache is Python dicts.
"""

import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.late_interaction import (
    LATE_INTERACTION_MODE_CACHE_QUERY,
    LATE_INTERACTION_MODE_SCORE_DOC,
)
from vllm.v1.worker.gpu.pool.kernels import ragged_maxsim
from vllm.v1.worker.gpu.pool.native import PoolingBatch

_LI_MODE_NONE = 0
_LI_MODE_QUERY = 1
_LI_MODE_DOC = 2


class LateInteractionStates:
    def __init__(self, max_num_reqs: int):
        # Persistent-slot mode; nonzero only for active late-interaction reqs.
        self.mode_np = np.zeros(max_num_reqs, dtype=np.int8)
        self.num_active = 0
        self.slot_query_keys: dict[int, str] = {}
        # Slot -> declared query_uses (query slots only).
        self.slot_query_uses: dict[int, int] = {}
        # query_key -> cached query token embeddings / remaining doc uses.
        self.query_cache: dict[str, torch.Tensor] = {}
        self.query_uses: dict[str, int] = {}

    def add_request(self, req_idx: int, params: PoolingParams | None) -> None:
        self._clear_slot(req_idx)
        li_params = params.late_interaction_params if params is not None else None
        if li_params is None:
            return

        query_key = li_params.query_key
        if not isinstance(query_key, str) or not query_key:
            raise ValueError(
                "late-interaction request is missing a valid query key in "
                "pooling_params.late_interaction_params."
            )

        mode = li_params.mode
        if mode == LATE_INTERACTION_MODE_CACHE_QUERY:
            query_uses_raw = li_params.query_uses
            try:
                query_uses = max(
                    1, 1 if query_uses_raw is None else int(query_uses_raw)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "late-interaction query uses must be an integer value."
                ) from exc
            self.mode_np[req_idx] = _LI_MODE_QUERY
            self.slot_query_uses[req_idx] = query_uses
        elif mode == LATE_INTERACTION_MODE_SCORE_DOC:
            self.mode_np[req_idx] = _LI_MODE_DOC
        else:
            raise ValueError(f"Unsupported late-interaction mode: {mode!r}")

        self.slot_query_keys[req_idx] = query_key
        self.num_active += 1

    def remove_request(self, req_idx: int) -> None:
        self._clear_slot(req_idx)

    def _clear_slot(self, req_idx: int) -> None:
        mode = self.mode_np[req_idx]
        if mode == _LI_MODE_NONE:
            return
        key = self._consume_slot(req_idx)
        # A doc that was removed before being scored releases its query use.
        if mode == _LI_MODE_DOC and key is not None:
            self._release_query_use(key)

    def _consume_slot(self, req_idx: int) -> str | None:
        self.mode_np[req_idx] = _LI_MODE_NONE
        self.num_active -= 1
        self.slot_query_uses.pop(req_idx, None)
        return self.slot_query_keys.pop(req_idx, None)

    def _release_query_use(self, query_key: str) -> None:
        remaining = self.query_uses.get(query_key, 1) - 1
        if remaining <= 0:
            self.query_uses.pop(query_key, None)
            self.query_cache.pop(query_key, None)
        else:
            self.query_uses[query_key] = remaining

    def postprocess_pooler_output(
        self, outputs: PoolerOutput, batch: PoolingBatch
    ) -> PoolerOutput:
        if self.num_active == 0:
            return outputs

        mode_batch = self.mode_np[batch.idx_mapping_np]
        active = (mode_batch != _LI_MODE_NONE) & batch.finished_np
        if not active.any() or not isinstance(outputs, list):
            # Late interaction requires ragged token-level outputs.
            return outputs

        for i in np.nonzero(active & (mode_batch == _LI_MODE_QUERY))[0].tolist():
            slot = int(batch.idx_mapping_np[i])
            output = outputs[i]
            if output is None:
                continue
            query_uses = self.slot_query_uses[slot]
            key = self._consume_slot(slot)
            assert key is not None
            # Clone: the output may be a view into this step's hidden-states
            # buffer and must survive across scheduling steps.
            self.query_cache[key] = output.clone()
            self.query_uses[key] = query_uses
            outputs[i] = torch.zeros((), device=output.device, dtype=torch.float32)

        doc_indices = [
            i
            for i in np.nonzero(active & (mode_batch == _LI_MODE_DOC))[0].tolist()
            if outputs[i] is not None
        ]
        if doc_indices:
            self._score_docs(outputs, batch, doc_indices)
        return outputs

    def _score_docs(
        self,
        outputs: list[torch.Tensor | None],
        batch: PoolingBatch,
        doc_indices: list[int],
    ) -> None:
        q_embs: list[torch.Tensor] = []
        d_embs: list[torch.Tensor] = []
        for i in doc_indices:
            key = self.slot_query_keys[int(batch.idx_mapping_np[i])]
            query_emb = self.query_cache.get(key)
            if query_emb is None:
                raise ValueError(
                    "late-interaction query cache miss for key "
                    f"{key!r}. Ensure query requests are executed "
                    "before their paired document requests."
                )
            doc_emb = outputs[i]
            assert doc_emb is not None
            if query_emb.shape[-1] != doc_emb.shape[-1]:
                raise ValueError("Query and document embeddings must have same dim")
            q_embs.append(query_emb)
            d_embs.append(doc_emb)

        num_pairs = len(doc_indices)
        start_locs = np.zeros((2, num_pairs + 1), dtype=np.int32)
        q_lens = np.array([q.shape[0] for q in q_embs], dtype=np.int32)
        np.cumsum(q_lens, out=start_locs[0, 1:])
        np.cumsum([d.shape[0] for d in d_embs], out=start_locs[1, 1:])
        start_locs_gpu = torch.from_numpy(start_locs).to(batch.device)

        scores = ragged_maxsim(
            torch.cat(q_embs),
            torch.cat(d_embs),
            start_locs_gpu[0],
            start_locs_gpu[1],
            int(q_lens.max()),
        )
        for i, score in zip(doc_indices, scores.unbind(0)):
            outputs[i] = score
            consumed_key = self._consume_slot(int(batch.idx_mapping_np[i]))
            assert consumed_key is not None
            self._release_query_use(consumed_key)
