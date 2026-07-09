# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2-native pooling: executes the standard pooler structures directly from
GPU batch state (flat hidden states + ``query_start_loc``) and numpy mirrors,
with no ``PoolingMetadata``/``PoolingCursor`` and no per-request Python loops
on the hot path. Non-standard poolers fall back to the shared ``model.pooler``
path in ``PoolingRunner``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.seqwise.heads import (
    ClassifierPoolerHead,
    EmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.seqwise.methods import (
    CLSPool,
    LastPool,
    MeanPool,
)
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import (
    BgeM3Pooler,
    BOSEOSFilter,
    DispatchPooler,
)
from vllm.model_executor.layers.pooler.tokwise.heads import (
    TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool, StepPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.tasks import PoolingTask
from vllm.v1.worker.gpu.pool.kernels import segment_mean

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pool.states import PoolingStates


@dataclass
class PoolingBatch:
    """Sync-free view of one step's pooling inputs.

    All numpy arrays are in batch order (already gathered via ``idx_mapping``);
    the ``*_uva`` tensors are in persistent-slot order and gathered lazily on
    the GPU only when a batch is non-uniform.
    """

    num_reqs: int
    # [num_tokens, hidden_size], padding rows already dropped.
    hidden_states: torch.Tensor
    # [num_reqs + 1] on GPU.
    query_start_loc: torch.Tensor
    # [num_reqs] batch idx -> persistent slot, on GPU / CPU.
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    num_scheduled_tokens_np: np.ndarray
    prompt_lens_np: np.ndarray
    seq_lens_np: np.ndarray
    # [num_reqs] == (seq_lens_np == prompt_lens_np)
    finished_np: np.ndarray
    # [num_reqs] matryoshka dims (-1 = unset) / activation flags, batch order.
    dimensions_np: np.ndarray
    use_activation_np: np.ndarray
    # [max_num_reqs] slot-order UVA views of the same params.
    dimensions_uva: torch.Tensor
    use_activation_uva: torch.Tensor

    _dimensions_gpu: torch.Tensor | None = field(default=None, init=False)
    _use_activation_gpu: torch.Tensor | None = field(default=None, init=False)

    @property
    def device(self) -> torch.device:
        return self.hidden_states.device

    def dimensions_gpu(self) -> torch.Tensor:
        if self._dimensions_gpu is None:
            self._dimensions_gpu = self.dimensions_uva[self.idx_mapping]
        return self._dimensions_gpu

    def use_activation_gpu(self) -> torch.Tensor:
        if self._use_activation_gpu is None:
            self._use_activation_gpu = self.use_activation_uva[self.idx_mapping]
        return self._use_activation_gpu


def _slice_or_mask_dims(
    embeddings: torch.Tensor,
    dimensions_np: np.ndarray,
    dimensions_gpu: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    """Apply matryoshka dimensions to stacked embeddings.

    Uniform dims become a single slice. Ragged dims zero out each row beyond
    its dim (L2-normalizing a zero-padded row equals normalizing the slice),
    and the caller slices rows to their final ragged lengths at the end.
    Returns (embeddings, ragged).
    """
    if not (dimensions_np >= 0).any():
        return embeddings, False
    if dimensions_np.min() == dimensions_np.max():
        return embeddings[..., : int(dimensions_np[0])], False

    embed_dim = embeddings.shape[-1]
    assert dimensions_gpu is not None
    dims = torch.where(dimensions_gpu < 0, embed_dim, dimensions_gpu)
    mask = torch.arange(embed_dim, device=embeddings.device) < dims.unsqueeze(-1)
    return embeddings.masked_fill(~mask, 0), True


def _apply_activation(
    data: torch.Tensor,
    activation,
    use_activation_np: np.ndarray,
    use_activation_gpu: torch.Tensor | None,
) -> torch.Tensor:
    if activation is None or not use_activation_np.any():
        return data
    if use_activation_np.all():
        return activation(data)
    assert use_activation_gpu is not None
    return torch.where(use_activation_gpu.unsqueeze(-1), activation(data), data)


def _ragged_rows(
    stacked: torch.Tensor, dimensions_np: np.ndarray
) -> list[torch.Tensor]:
    return [
        row if d < 0 else row[..., :d]
        for row, d in zip(stacked.unbind(0), dimensions_np.tolist())
    ]


class NativeSeqPooler:
    """CLS/LAST/MEAN pooling + embed/classify head, fully batched."""

    def __init__(
        self,
        method: str,  # "cls" | "last" | "mean"
        head: EmbeddingPoolerHead | ClassifierPoolerHead,
    ):
        self.method = method
        self.head = head

    def pool(
        self, batch: PoolingBatch, states: "PoolingStates | None" = None
    ) -> torch.Tensor | list[torch.Tensor]:
        if (
            self.method != "last"
            and (batch.num_scheduled_tokens_np != batch.prompt_lens_np).any()
        ):
            raise RuntimeError(
                f"partial prefill is not supported with {self.method.upper()} pooling"
            )

        qsl = batch.query_start_loc
        hidden_states = batch.hidden_states
        if self.method == "cls":
            pooled = hidden_states[qsl[:-1]]
        elif self.method == "last":
            pooled = hidden_states[qsl[1:] - 1]
        else:
            pooled = segment_mean(hidden_states, qsl, batch.num_reqs)

        head = self.head
        if head.head_dtype is not None:
            pooled = pooled.to(head.head_dtype)

        if isinstance(head, EmbeddingPoolerHead):
            return self._embed_head(pooled, batch)
        return self._classify_head(pooled, batch)

    def _embed_head(
        self, pooled: torch.Tensor, batch: PoolingBatch
    ) -> torch.Tensor | list[torch.Tensor]:
        head = self.head
        assert isinstance(head, EmbeddingPoolerHead)
        embeddings = head.projector(pooled) if head.projector is not None else pooled

        dims_np = batch.dimensions_np
        needs_gpu_dims = (dims_np >= 0).any() and dims_np.min() != dims_np.max()
        embeddings, ragged = _slice_or_mask_dims(
            embeddings, dims_np, batch.dimensions_gpu() if needs_gpu_dims else None
        )
        needs_gpu_flags = (
            head.activation is not None
            and batch.use_activation_np.any()
            and not batch.use_activation_np.all()
        )
        embeddings = _apply_activation(
            embeddings,
            head.activation,
            batch.use_activation_np,
            batch.use_activation_gpu() if needs_gpu_flags else None,
        )
        if ragged:
            return _ragged_rows(embeddings, dims_np)
        return embeddings

    def _classify_head(self, pooled: torch.Tensor, batch: PoolingBatch) -> torch.Tensor:
        head = self.head
        assert isinstance(head, ClassifierPoolerHead)
        logits = head.classifier(pooled) if head.classifier is not None else pooled
        if head.logit_mean is not None:
            logits = logits - head.logit_mean
        if head.logit_sigma is not None:
            logits = logits / head.logit_sigma

        needs_gpu_flags = (
            head.activation is not None
            and batch.use_activation_np.any()
            and not batch.use_activation_np.all()
        )
        return _apply_activation(
            logits,
            head.activation,
            batch.use_activation_np,
            batch.use_activation_gpu() if needs_gpu_flags else None,
        )


class NativeTokenPooler:
    """ALL pooling + token embed/classify head.

    The head is applied once to the flat token tensor (or, for finishing
    chunked prefills, once to the finishing requests' concatenated prompts);
    raggedization into per-request views happens only at the output boundary.
    Chunked-prefill caches hold raw (pre-head) hidden chunks in the same
    per-slot ``HiddenStatesCache`` objects used by the shared-pooler fallback,
    so a request may move between the two paths across steps.
    """

    def __init__(
        self, head: TokenEmbeddingPoolerHead | TokenClassifierPoolerHead | None
    ):
        self.head = head

    def pool(
        self, batch: PoolingBatch, states: "PoolingStates"
    ) -> list[torch.Tensor | None]:
        lens_np = batch.num_scheduled_tokens_np
        # Every request computed its whole sequence this step iff no request
        # has earlier cached chunks and all finish now: the common case.
        if np.array_equal(lens_np, batch.seq_lens_np) and batch.finished_np.all():
            return self._finalize(batch.hidden_states, lens_np, batch, states, None)

        return self._pool_chunked(batch, states)

    @staticmethod
    def _slice_ragged_dims(
        outputs: list[torch.Tensor | None], dims_np: np.ndarray
    ) -> list[torch.Tensor | None]:
        """Per-request output-boundary views for ragged matryoshka dims (the
        stacked math zero-masks beyond each row's dim, so slicing here yields
        the same values as per-request slicing before normalization)."""
        if not (dims_np >= 0).any() or dims_np.min() == dims_np.max():
            return outputs
        return [
            out if out is None or d < 0 else out[..., :d]
            for out, d in zip(outputs, dims_np.tolist())
        ]

    def _pool_chunked(
        self, batch: PoolingBatch, states: "PoolingStates"
    ) -> list[torch.Tensor | None]:
        # Only reached on chunked-prefill steps; loops are per-request
        # bookkeeping of cached chunks, never per-token.
        chunk_caches = states.hidden_caches
        splits = torch.split(
            batch.hidden_states, batch.num_scheduled_tokens_np.tolist()
        )
        finished_np = batch.finished_np
        idx_mapping_np = batch.idx_mapping_np

        outputs: list[torch.Tensor | None] = [None] * batch.num_reqs
        finished_idx: list[int] = []
        finished_parts: list[torch.Tensor] = []
        finished_lens: list[int] = []
        for i in np.nonzero(finished_np)[0].tolist():
            cache = chunk_caches[int(idx_mapping_np[i])].hidden_states_cache
            cache.append(splits[i])
            finished_idx.append(i)
            finished_parts.extend(cache)
            finished_lens.append(sum(c.shape[0] for c in cache))
            cache.clear()
        for i in np.nonzero(~finished_np)[0].tolist():
            # Clone: the chunk must outlive this step's hidden-states buffer.
            chunk_caches[int(idx_mapping_np[i])].hidden_states_cache.append(
                splits[i].clone()
            )

        if finished_idx:
            flat = (
                torch.cat(finished_parts)
                if len(finished_parts) > 1
                else finished_parts[0]
            )
            finished_outputs = self._finalize(
                flat,
                np.array(finished_lens),
                batch,
                states,
                np.array(finished_idx),
            )
            for i, out in zip(finished_idx, finished_outputs):
                outputs[i] = out
        return outputs

    def _finalize(
        self,
        tokens: torch.Tensor,
        lens_np: np.ndarray,
        batch: PoolingBatch,
        states: "PoolingStates",
        req_indices: np.ndarray | None,
    ) -> list[torch.Tensor | None]:
        """Head + raggedization for a flat [n_tokens, H] tensor whose rows are
        the full prompts of requests ``req_indices`` (batch order, all requests
        if None), ``lens_np`` tokens each."""
        out = self._apply_head(tokens, lens_np, batch, req_indices)
        splits: list[torch.Tensor | None] = list(torch.split(out, lens_np.tolist()))
        dims_np = batch.dimensions_np
        if req_indices is not None:
            dims_np = dims_np[req_indices]
        return self._slice_ragged_dims(splits, dims_np)

    def _apply_head(
        self,
        tokens: torch.Tensor,
        lens_np: np.ndarray,
        batch: PoolingBatch,
        req_indices: np.ndarray | None = None,
    ) -> torch.Tensor:
        head = self.head
        if head is None:
            return tokens

        if head.head_dtype is not None:
            tokens = tokens.to(head.head_dtype)

        dims_np = batch.dimensions_np
        ua_np = batch.use_activation_np
        if req_indices is not None:
            dims_np = dims_np[req_indices]
            ua_np = ua_np[req_indices]

        def expand(per_req: torch.Tensor) -> torch.Tensor:
            repeats = torch.from_numpy(np.ascontiguousarray(lens_np)).to(
                batch.device, non_blocking=True
            )
            return torch.repeat_interleave(
                per_req, repeats, output_size=tokens.shape[0]
            )

        def gather_per_req(uva: torch.Tensor) -> torch.Tensor:
            idx = batch.idx_mapping
            if req_indices is not None:
                idx = idx[torch.from_numpy(req_indices).to(batch.device)]
            return uva[idx]

        if isinstance(head, TokenEmbeddingPoolerHead):
            out = head.projector(tokens) if head.projector is not None else tokens
            needs_gpu_dims = (dims_np >= 0).any() and (dims_np.min() != dims_np.max())
            dims_gpu = (
                expand(gather_per_req(batch.dimensions_uva)) if needs_gpu_dims else None
            )
            out, _ = _slice_or_mask_dims(out, dims_np, dims_gpu)
        else:
            out = head.classifier(tokens) if head.classifier is not None else tokens
            if head.logit_mean is not None:
                out = out - head.logit_mean
            if head.logit_sigma is not None:
                out = out / head.logit_sigma

        needs_gpu_flags = (
            head.activation is not None and ua_np.any() and not ua_np.all()
        )
        ua_gpu = (
            expand(gather_per_req(batch.use_activation_uva))
            if needs_gpu_flags
            else None
        )
        return _apply_activation(out, head.activation, ua_np, ua_gpu)


class NativeStepPooler(NativeTokenPooler):
    """STEP pooling: ALL pooling that keeps only step_tag_id-matching rows and
    optionally selects label columns before the head. Row positions were
    precomputed per slot at add time from the prompt token ids
    (``PoolingStates.step_indices``), so no token ids are needed here."""

    def _finalize(
        self,
        tokens: torch.Tensor,
        lens_np: np.ndarray,
        batch: PoolingBatch,
        states: "PoolingStates",
        req_indices: np.ndarray | None,
    ) -> list[torch.Tensor | None]:
        slots = batch.idx_mapping_np
        if req_indices is not None:
            slots = slots[req_indices]
        slot_list = slots.tolist()

        step_lists = [states.step_indices.get(slot) for slot in slot_list]
        if any(sl is not None for sl in step_lists):
            offsets = np.zeros(len(slot_list) + 1, dtype=np.int64)
            np.cumsum(lens_np, out=offsets[1:])
            parts = [
                (np.arange(n, dtype=np.int64) if sl is None else sl) + off
                for sl, off, n in zip(step_lists, offsets[:-1], lens_np.tolist())
            ]
            row_idx = np.concatenate(parts)
            tokens = tokens.index_select(0, torch.from_numpy(row_idx).to(batch.device))
            lens_np = np.array([len(p) for p in parts])

        cols = [states.returned_token_ids.get(slot) for slot in slot_list]
        first_cols = cols[0]
        uniform_cols = all(
            (c is None) == (first_cols is None)
            and (c is None or np.array_equal(c, first_cols))
            for c in cols
        )
        if uniform_cols:
            if first_cols is not None:
                tokens = tokens.index_select(
                    1, torch.from_numpy(first_cols).to(batch.device)
                )
            out = self._apply_head(tokens, lens_np, batch, req_indices)
            splits: list[torch.Tensor | None] = list(torch.split(out, lens_np.tolist()))
            return splits

        # Rare: per-request label columns; matches the shared per-chunk path.
        outputs: list[torch.Tensor | None] = []
        for j, chunk in enumerate(torch.split(tokens, lens_np.tolist())):
            if cols[j] is not None:
                chunk = chunk.index_select(
                    1, torch.from_numpy(cols[j]).to(batch.device)
                )
            orig = np.array([j if req_indices is None else req_indices[j]])
            outputs.append(self._apply_head(chunk, lens_np[j : j + 1], batch, orig))
        return outputs


class NativeBOSEOSFilter:
    """Trims BOS/EOS rows from a token-level pooler's per-request outputs.
    First/last prompt token ids were recorded per slot at add time, so the
    trim decisions are numpy compares; the trims themselves are output-boundary
    views."""

    def __init__(self, inner: NativeTokenPooler, bos_token_id: int, eos_token_id: int):
        self.inner = inner
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def pool(
        self, batch: PoolingBatch, states: "PoolingStates"
    ) -> list[torch.Tensor | None]:
        outputs = self.inner.pool(batch, states)
        slots = batch.idx_mapping_np
        trim_start = states.first_token_np[slots] == self.bos_token_id
        trim_end = states.last_token_np[slots] == self.eos_token_id
        for i in np.nonzero(batch.finished_np)[0].tolist():
            out = outputs[i]
            if out is None:
                continue
            start = 1 if trim_start[i] else 0
            end = out.shape[0] - (1 if trim_end[i] else 0)
            outputs[i] = out[start:end].squeeze(-1)
        return outputs


class NativeBgeM3Pooler:
    """embed&token_classify: dense CLS embedding concatenated with per-token
    sparse logits, flattened per request at the output boundary."""

    def __init__(
        self,
        token_classify: "NativeTokenPooler | NativeBOSEOSFilter",
        embed: NativeSeqPooler,
    ):
        self.token_classify = token_classify
        self.embed = embed

    def pool(
        self, batch: PoolingBatch, states: "PoolingStates"
    ) -> list[torch.Tensor | None]:
        # embed (CLS) rejects partial prefill before token_classify touches
        # the chunk caches.
        embed_out = self.embed.pool(batch, states)
        token_out = self.token_classify.pool(batch, states)
        outputs: list[torch.Tensor | None] = []
        for dense, sparse in zip(embed_out, token_out):
            if dense is None or sparse is None:
                outputs.append(None)
            else:
                outputs.append(torch.cat([dense.view(-1), sparse.view(-1)]))
        return outputs


NativePooler = (
    NativeSeqPooler | NativeTokenPooler | NativeBOSEOSFilter | NativeBgeM3Pooler
)

_SEQ_METHODS: dict[type, str] = {CLSPool: "cls", LastPool: "last", MeanPool: "mean"}


def _extract_native(pooler: Pooler) -> NativePooler | None:
    # Exact type checks throughout: subclasses and plain callables may change
    # behavior the native paths do not reproduce.
    if type(pooler) is SequencePooler:
        method = _SEQ_METHODS.get(type(pooler.pooling))
        if method is not None and type(pooler.head) in (
            EmbeddingPoolerHead,
            ClassifierPoolerHead,
        ):
            return NativeSeqPooler(
                method, cast(EmbeddingPoolerHead | ClassifierPoolerHead, pooler.head)
            )
    elif type(pooler) is TokenPooler:
        pooling_cls = type(pooler.pooling)
        if pooling_cls in (AllPool, StepPool) and (
            pooler.head is None
            or type(pooler.head)
            in (TokenEmbeddingPoolerHead, TokenClassifierPoolerHead)
        ):
            head = cast(
                TokenEmbeddingPoolerHead | TokenClassifierPoolerHead | None,
                pooler.head,
            )
            if pooling_cls is StepPool:
                return NativeStepPooler(head)
            return NativeTokenPooler(head)
    elif type(pooler) is BOSEOSFilter:
        inner = _extract_native(pooler.pooler)
        if isinstance(inner, NativeTokenPooler):
            return NativeBOSEOSFilter(inner, pooler.bos_token_id, pooler.eos_token_id)
    elif type(pooler) is BgeM3Pooler:
        embed = _extract_native(pooler.embed_pooler)
        token_classify = _extract_native(pooler.token_classify_pooler)
        if isinstance(embed, NativeSeqPooler) and isinstance(
            token_classify, (NativeTokenPooler, NativeBOSEOSFilter)
        ):
            return NativeBgeM3Pooler(token_classify, embed)
    return None


def build_native_poolers(pooler: Pooler) -> dict[PoolingTask, NativePooler]:
    """Map each pooling task to a V2-native implementation where the model's
    pooler has a recognized standard structure. Unmapped tasks are served by
    the shared-pooler fallback."""
    if isinstance(pooler, DispatchPooler):
        items = list(pooler.poolers_by_task.items())
    else:
        items = [(task, pooler) for task in pooler.get_supported_tasks()]

    native: dict[PoolingTask, NativePooler] = {}
    for task, sub_pooler in items:
        sub_native = _extract_native(sub_pooler)
        if sub_native is not None:
            native[task] = sub_native
    return native
