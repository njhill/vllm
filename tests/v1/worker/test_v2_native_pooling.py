# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2 native pooling: triton kernels, parity of the native
seq/token pooler paths against the shared `model.pooler` implementations, and
the slot-indexed late-interaction state machine."""

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for V2 native pooling tests", allow_module_level=True)

from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score
from vllm.model_executor.layers.pooler.activations import (
    PoolerClassify,
    PoolerNormalize,
)
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
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.late_interaction import (
    build_late_interaction_doc_params,
    build_late_interaction_query_params,
)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.pool.metadata import PoolingStates as HiddenStatesCache
from vllm.v1.worker.gpu.pool.kernels import ragged_maxsim, segment_mean
from vllm.v1.worker.gpu.pool.late_interaction import LateInteractionStates
from vllm.v1.worker.gpu.pool.native import (
    NativeBgeM3Pooler,
    NativeBOSEOSFilter,
    NativeSeqPooler,
    NativeStepPooler,
    NativeTokenPooler,
    PoolingBatch,
    build_native_poolers,
)
from vllm.v1.worker.gpu.pool.states import PoolingStates

DEVICE = "cuda"


def make_states(num_reqs: int = 8) -> PoolingStates:
    states = PoolingStates(max_num_reqs=num_reqs)
    for i in range(num_reqs):
        states.hidden_caches[i] = HiddenStatesCache()
    return states


def _cumsum0(lens: list[int]) -> np.ndarray:
    out = np.zeros(len(lens) + 1, dtype=np.int32)
    np.cumsum(lens, out=out[1:])
    return out


def make_batch(
    hidden_states: torch.Tensor,
    lens: list[int],
    prompt_lens: list[int] | None = None,
    computed_before: list[int] | None = None,
    dims: list[int] | None = None,
    use_activation: list[bool] | None = None,
) -> PoolingBatch:
    num_reqs = len(lens)
    lens_np = np.array(lens, dtype=np.int32)
    computed_np = np.array(computed_before or [0] * num_reqs, dtype=np.int32)
    seq_lens_np = computed_np + lens_np
    prompt_lens_np = (
        seq_lens_np if prompt_lens is None else np.array(prompt_lens, np.int32)
    )
    dims_np = np.array(dims or [-1] * num_reqs, dtype=np.int32)
    ua_np = np.array(
        [True] * num_reqs if use_activation is None else use_activation, dtype=bool
    )
    return PoolingBatch(
        num_reqs=num_reqs,
        hidden_states=hidden_states,
        query_start_loc=torch.from_numpy(_cumsum0(lens)).to(DEVICE),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=DEVICE),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens_np=lens_np,
        prompt_lens_np=prompt_lens_np,
        seq_lens_np=seq_lens_np,
        finished_np=seq_lens_np == prompt_lens_np,
        dimensions_np=dims_np,
        use_activation_np=ua_np,
        dimensions_uva=torch.from_numpy(dims_np).to(DEVICE),
        use_activation_uva=torch.from_numpy(ua_np).to(DEVICE),
    )


def make_metadata(
    batch: PoolingBatch,
    task: str,
    hidden_caches: list[HiddenStatesCache] | None = None,
    prompt_token_ids: list[list[int]] | None = None,
    step_tag_id: int | None = None,
    returned_token_ids: list[int] | None = None,
) -> PoolingMetadata:
    params = []
    for i in range(batch.num_reqs):
        p = PoolingParams(task=task)
        d = int(batch.dimensions_np[i])
        p.dimensions = d if d >= 0 else None
        p.use_activation = bool(batch.use_activation_np[i])
        p.step_tag_id = step_tag_id
        p.returned_token_ids = returned_token_ids
        params.append(p)
    token_ids_gpu = token_ids_cpu = None
    if prompt_token_ids is not None:
        max_len = max(len(t) for t in prompt_token_ids)
        token_ids_cpu = torch.zeros((len(prompt_token_ids), max_len), dtype=torch.int32)
        for i, t in enumerate(prompt_token_ids):
            token_ids_cpu[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
        token_ids_gpu = token_ids_cpu.to(DEVICE)
    metadata = PoolingMetadata(
        prompt_lens=torch.from_numpy(batch.prompt_lens_np),
        prompt_token_ids=token_ids_gpu,
        prompt_token_ids_cpu=token_ids_cpu,
        pooling_params=params,
        pooling_states=hidden_caches
        or [HiddenStatesCache() for _ in range(batch.num_reqs)],
    )
    metadata.build_pooling_cursor(
        batch.num_scheduled_tokens_np,
        torch.from_numpy(batch.seq_lens_np),
        device=batch.query_start_loc.device,
        query_start_loc_gpu=batch.query_start_loc,
    )
    return metadata


def assert_outputs_match(native_out, shared_out, num_reqs: int):
    for i in range(num_reqs):
        n, s = native_out[i], shared_out[i]
        if s is None:
            assert n is None
            continue
        assert n.shape == s.shape, f"req {i}: {n.shape} != {s.shape}"
        torch.testing.assert_close(n, s, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_segment_mean_matches_reference(dtype):
    torch.manual_seed(0)
    lens = [1, 5, 33, 200, 7]
    qsl = torch.from_numpy(_cumsum0(lens)).to(DEVICE)
    hidden = torch.randn(sum(lens), 300, dtype=dtype, device=DEVICE)
    out = segment_mean(hidden, qsl, len(lens))
    ref = torch.stack([chunk.float().mean(0) for chunk in torch.split(hidden, lens)])
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_ragged_maxsim_matches_reference():
    torch.manual_seed(0)
    q_lens, d_lens, dim = [3, 17, 1, 40], [9, 2, 100, 33], 129
    q = torch.randn(sum(q_lens), dim, device=DEVICE)
    d = torch.randn(sum(d_lens), dim, device=DEVICE)
    out = ragged_maxsim(
        q,
        d,
        torch.from_numpy(_cumsum0(q_lens)).to(DEVICE),
        torch.from_numpy(_cumsum0(d_lens)).to(DEVICE),
        max(q_lens),
    )
    ref = torch.stack(
        [
            compute_maxsim_score(qc, dc)
            for qc, dc in zip(torch.split(q, q_lens), torch.split(d, d_lens))
        ]
    )
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize(
    "method,pooling", [("cls", CLSPool()), ("last", LastPool()), ("mean", MeanPool())]
)
@pytest.mark.parametrize("with_projector", [False, True])
def test_native_seq_embed_parity(method, pooling, with_projector):
    torch.manual_seed(0)
    lens, hidden_size = [3, 7, 1, 12], 64
    hidden = torch.randn(sum(lens), hidden_size, device=DEVICE)
    projector = torch.nn.Linear(hidden_size, 32).to(DEVICE) if with_projector else None
    head = EmbeddingPoolerHead(
        projector=projector, head_dtype=torch.float32, activation=PoolerNormalize()
    )
    shared = SequencePooler(pooling=pooling, head=head)
    native = NativeSeqPooler(method, head)

    # Ragged matryoshka dims and mixed activation flags in one batch.
    batch = make_batch(
        hidden, lens, dims=[-1, 16, -1, 8], use_activation=[True, False, True, True]
    )
    native_out = native.pool(batch)
    shared_out = shared(hidden, make_metadata(batch, "embed"))
    assert_outputs_match(native_out, shared_out, batch.num_reqs)

    # Uniform batch keeps the stacked-tensor output.
    batch = make_batch(hidden, lens)
    native_out = native.pool(batch)
    assert isinstance(native_out, torch.Tensor)
    shared_out = shared(hidden, make_metadata(batch, "embed"))
    assert_outputs_match(native_out, shared_out, batch.num_reqs)


def test_native_seq_classify_parity():
    torch.manual_seed(0)
    lens, hidden_size, num_labels = [4, 2, 9], 32, 5
    hidden = torch.randn(sum(lens), hidden_size, device=DEVICE)
    head = ClassifierPoolerHead(
        classifier=torch.nn.Linear(hidden_size, num_labels).to(DEVICE),
        logit_mean=0.1,
        logit_sigma=2.0,
        head_dtype=torch.float32,
        activation=PoolerClassify(num_labels=num_labels),
    )
    shared = SequencePooler(pooling=LastPool(), head=head)
    native = NativeSeqPooler("last", head)

    batch = make_batch(hidden, lens, use_activation=[True, False, True])
    native_out = native.pool(batch)
    shared_out = shared(hidden, make_metadata(batch, "classify"))
    assert_outputs_match(native_out, shared_out, batch.num_reqs)


def test_native_seq_partial_prefill_raises():
    hidden = torch.randn(10, 8, device=DEVICE)
    head = EmbeddingPoolerHead(head_dtype=torch.float32)
    native = NativeSeqPooler("mean", head)
    batch = make_batch(hidden, [4, 6], prompt_lens=[4, 9])
    with pytest.raises(RuntimeError, match="partial prefill"):
        native.pool(batch)


def test_native_token_embed_parity_single_step(default_vllm_config):
    torch.manual_seed(0)
    lens, hidden_size = [5, 1, 8], 48
    hidden = torch.randn(sum(lens), hidden_size, device=DEVICE)
    head = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32, projector=None, activation=PoolerNormalize()
    )
    shared = TokenPooler(pooling=AllPool(), head=head)
    native = NativeTokenPooler(head)

    batch = make_batch(
        hidden, lens, dims=[16, -1, 8], use_activation=[True, False, True]
    )
    states = make_states()
    native_out = native.pool(batch, states)
    shared_out = shared(hidden, make_metadata(batch, "token_embed"))
    assert_outputs_match(native_out, shared_out, batch.num_reqs)
    assert all(not c.hidden_states_cache for c in states.hidden_caches.values())


def test_native_token_embed_parity_chunked(default_vllm_config):
    torch.manual_seed(0)
    hidden_size = 48
    head = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32, projector=None, activation=PoolerNormalize()
    )
    shared = TokenPooler(pooling=AllPool(), head=head)
    native = NativeTokenPooler(head)

    # Request 0's 9-token prompt is prefilled over two steps; request 1
    # completes in step 1, request 2 in step 2.
    prompts = [torch.randn(n, hidden_size, device=DEVICE) for n in (9, 3, 2)]

    states = make_states()
    step1 = torch.cat([prompts[0][:4], prompts[1]])
    batch1 = make_batch(step1, lens=[4, 3], prompt_lens=[9, 3])
    out1 = native.pool(batch1, states)
    assert out1[0] is None

    step2 = torch.cat([prompts[0][4:], prompts[2]])
    batch2 = make_batch(step2, lens=[5, 2], computed_before=[4, 0])
    out2 = native.pool(batch2, states)

    shared_caches = [HiddenStatesCache() for _ in range(3)]
    ref1 = shared(step1, make_metadata(batch1, "token_embed", shared_caches[:2]))
    ref2 = shared(
        step2,
        make_metadata(batch2, "token_embed", [shared_caches[0], shared_caches[2]]),
    )
    assert_outputs_match(out1, ref1, 2)
    assert_outputs_match(out2, ref2, 2)
    # Both requests' full prompts were pooled.
    assert out2[0].shape[0] == 9 and out2[1].shape[0] == 2


def test_build_native_poolers_recognizes_standard_structures(default_vllm_config):
    head = TokenEmbeddingPoolerHead(head_dtype=torch.float32)
    pooler = DispatchPooler(
        {
            "embed": SequencePooler(
                pooling=MeanPool(), head=EmbeddingPoolerHead(head_dtype=torch.float32)
            ),
            "token_embed": TokenPooler(pooling=AllPool(), head=head),
        }
    )
    native = build_native_poolers(pooler)
    assert isinstance(native["embed"], NativeSeqPooler)
    assert native["embed"].method == "mean"
    assert isinstance(native["token_embed"], NativeTokenPooler)

    step_pooler = DispatchPooler(
        {"token_embed": TokenPooler(pooling=StepPool(), head=head)}
    )
    assert isinstance(
        build_native_poolers(step_pooler)["token_embed"], NativeStepPooler
    )


def _li_batch(slots: list[int], finished: list[bool] | None = None) -> PoolingBatch:
    num_reqs = len(slots)
    batch = make_batch(
        torch.empty(num_reqs, 1, device=DEVICE),
        [1] * num_reqs,
        prompt_lens=None if finished is None else [1 if f else 2 for f in finished],
    )
    batch.idx_mapping_np = np.array(slots, dtype=np.int32)
    return batch


def _query_params(key: str, uses: int) -> PoolingParams:
    return PoolingParams(
        task="token_embed",
        late_interaction_params=build_late_interaction_query_params(key, uses),
    )


def _doc_params(key: str) -> PoolingParams:
    return PoolingParams(
        task="token_embed",
        late_interaction_params=build_late_interaction_doc_params(key),
    )


def test_late_interaction_scores_and_releases():
    states = LateInteractionStates(max_num_reqs=8)
    query_emb = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=DEVICE
    )
    doc_embs = [
        torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32, device=DEVICE),
        torch.tensor(
            [[0.0, 1.0], [0.3, 0.7], [1.0, 0.0]], dtype=torch.float32, device=DEVICE
        ),
    ]

    states.add_request(0, _query_params("q", uses=2))
    out = states.postprocess_pooler_output([query_emb], _li_batch([0]))
    assert out[0].shape == torch.Size([])
    assert states.num_active == 0

    states.add_request(0, _doc_params("q"))
    states.add_request(1, _doc_params("q"))
    out = states.postprocess_pooler_output(list(doc_embs), _li_batch([0, 1]))
    for score, doc_emb in zip(out, doc_embs):
        torch.testing.assert_close(
            score, compute_maxsim_score(query_emb, doc_emb), atol=1e-5, rtol=1e-5
        )

    # Both uses consumed: the cache entry is gone.
    states.add_request(2, _doc_params("q"))
    with pytest.raises(ValueError, match="query cache miss"):
        states.postprocess_pooler_output([doc_embs[0]], _li_batch([2]))


def test_late_interaction_unfinished_requests_wait():
    states = LateInteractionStates(max_num_reqs=8)
    states.add_request(0, _query_params("q", uses=1))
    out = states.postprocess_pooler_output([None], _li_batch([0], finished=[False]))
    assert out == [None]
    assert "q" not in states.query_cache


def test_late_interaction_removed_doc_releases_use():
    states = LateInteractionStates(max_num_reqs=8)
    emb = torch.ones(2, 2, device=DEVICE)
    states.add_request(0, _query_params("q", uses=1))
    states.postprocess_pooler_output([emb], _li_batch([0]))

    # Doc aborted before scoring: its use is released and the cache dropped.
    states.add_request(1, _doc_params("q"))
    states.remove_request(1)
    assert "q" not in states.query_cache

    states.add_request(2, _doc_params("q"))
    with pytest.raises(ValueError, match="query cache miss"):
        states.postprocess_pooler_output([emb], _li_batch([2]))


def test_late_interaction_invalid_query_uses_raises():
    states = LateInteractionStates(max_num_reqs=8)
    params = _query_params("q", uses=1)
    params.late_interaction_params.query_uses = "bad-int"
    with pytest.raises(ValueError, match="must be an integer value"):
        states.add_request(0, params)


def _classify_token_head() -> TokenClassifierPoolerHead:
    return TokenClassifierPoolerHead(
        classifier=torch.nn.Linear(32, 2).to(DEVICE),
        head_dtype=torch.float32,
        activation=PoolerClassify(num_labels=2),
    )


def test_native_step_pool_parity(default_vllm_config):
    torch.manual_seed(0)
    step_tag = 99
    token_ids = [
        [1, 2, step_tag, 4, step_tag],
        [step_tag, 6, 7],
        [8, 9],  # no step tags: empty output
    ]
    lens = [len(t) for t in token_ids]
    hidden = torch.randn(sum(lens), 32, device=DEVICE)
    head = _classify_token_head()
    shared = TokenPooler(pooling=StepPool(), head=head)
    native = NativeStepPooler(head)

    states = make_states()
    for i, tokens in enumerate(token_ids):
        params = PoolingParams(task="token_classify", step_tag_id=step_tag)
        states.add_request(i, params, tokens)

    batch = make_batch(hidden, lens)
    native_out = native.pool(batch, states)
    shared_out = shared(
        hidden,
        make_metadata(
            batch, "token_classify", prompt_token_ids=token_ids, step_tag_id=step_tag
        ),
    )
    assert_outputs_match(native_out, shared_out, batch.num_reqs)
    assert native_out[0].shape[0] == 2
    assert native_out[2].shape[0] == 0


def test_native_step_pool_returned_token_ids_parity(default_vllm_config):
    torch.manual_seed(0)
    token_ids = [[1, 2, 3], [4, 5]]
    lens = [3, 2]
    hidden = torch.randn(sum(lens), 32, device=DEVICE)
    head = TokenClassifierPoolerHead(
        classifier=None, head_dtype=torch.float32, activation=PoolerClassify()
    )
    shared = TokenPooler(pooling=StepPool(), head=head)
    native = NativeStepPooler(head)

    returned = [3, 7, 11]
    states = make_states()
    for i, tokens in enumerate(token_ids):
        params = PoolingParams(task="token_classify", returned_token_ids=returned)
        states.add_request(i, params, tokens)

    batch = make_batch(hidden, lens)
    native_out = native.pool(batch, states)
    shared_out = shared(
        hidden,
        make_metadata(
            batch,
            "token_classify",
            prompt_token_ids=token_ids,
            returned_token_ids=returned,
        ),
    )
    assert_outputs_match(native_out, shared_out, batch.num_reqs)
    assert native_out[0].shape == (3, 3)


def test_native_step_pool_chunked(default_vllm_config):
    torch.manual_seed(0)
    step_tag = 42
    token_ids = [7, step_tag, 8, 9, step_tag, step_tag, 10]
    prompt = torch.randn(len(token_ids), 32, device=DEVICE)
    head = _classify_token_head()
    shared = TokenPooler(pooling=StepPool(), head=head)
    native = NativeStepPooler(head)

    states = make_states()
    states.add_request(
        0, PoolingParams(task="token_classify", step_tag_id=step_tag), token_ids
    )

    batch1 = make_batch(prompt[:3], lens=[3], prompt_lens=[len(token_ids)])
    out1 = native.pool(batch1, states)
    assert out1 == [None]
    batch2 = make_batch(prompt[3:], lens=[4], computed_before=[3])
    out2 = native.pool(batch2, states)

    full_batch = make_batch(prompt, lens=[len(token_ids)])
    ref = shared(
        prompt,
        make_metadata(
            full_batch,
            "token_classify",
            prompt_token_ids=[token_ids],
            step_tag_id=step_tag,
        ),
    )
    assert_outputs_match(out2, ref, 1)
    assert out2[0].shape[0] == 3


def test_native_boseos_filter_parity(default_vllm_config):
    torch.manual_seed(0)
    bos, eos = 101, 102
    token_ids = [
        [bos, 1, 2, eos],  # trim both
        [1, 2, 3],  # trim neither
        [bos, 5],  # trim bos only
    ]
    lens = [len(t) for t in token_ids]
    hidden = torch.randn(sum(lens), 32, device=DEVICE)
    head = TokenClassifierPoolerHead(
        classifier=torch.nn.Linear(32, 1).to(DEVICE),
        head_dtype=torch.float32,
        activation=None,
    )
    shared = BOSEOSFilter(TokenPooler(pooling=AllPool(), head=head), bos, eos)
    native = NativeBOSEOSFilter(NativeTokenPooler(head), bos, eos)

    states = make_states()
    for i, tokens in enumerate(token_ids):
        states.add_request(i, PoolingParams(task="token_classify"), tokens)

    batch = make_batch(hidden, lens)
    native_out = native.pool(batch, states)
    shared_out = shared(
        hidden,
        make_metadata(batch, "token_classify", prompt_token_ids=token_ids),
    )
    assert_outputs_match(native_out, shared_out, batch.num_reqs)
    assert native_out[0].shape == (2,)  # trimmed + squeezed
    assert native_out[1].shape == (3,)


def test_native_bge_m3_parity(default_vllm_config):
    torch.manual_seed(0)
    bos, eos = 101, 102
    token_ids = [[bos, 1, 2, eos], [bos, 3, 4, 5, eos]]
    lens = [len(t) for t in token_ids]
    hidden_size = 32
    hidden = torch.randn(sum(lens), hidden_size, device=DEVICE)

    sparse_linear = torch.nn.Linear(hidden_size, 1).to(DEVICE)
    tclass_head = TokenClassifierPoolerHead(
        classifier=sparse_linear, head_dtype=torch.float32, activation=torch.relu
    )
    embed_head = EmbeddingPoolerHead(
        head_dtype=torch.float32, activation=PoolerNormalize()
    )
    shared_tclass = BOSEOSFilter(
        TokenPooler(pooling=AllPool(), head=tclass_head), bos, eos
    )
    shared_embed = SequencePooler(pooling=CLSPool(), head=embed_head)
    shared = BgeM3Pooler(shared_tclass, shared_embed)

    native = NativeBgeM3Pooler(
        NativeBOSEOSFilter(NativeTokenPooler(tclass_head), bos, eos),
        NativeSeqPooler("cls", embed_head),
    )

    states = make_states()
    for i, tokens in enumerate(token_ids):
        states.add_request(i, PoolingParams(task="embed&token_classify"), tokens)

    batch = make_batch(hidden, lens)
    native_out = native.pool(batch, states)
    shared_out = shared(
        hidden,
        make_metadata(batch, "embed&token_classify", prompt_token_ids=token_ids),
    )
    assert_outputs_match(native_out, shared_out, batch.num_reqs)
    # dense embedding + one sparse logit per non-special token
    assert native_out[0].shape == (hidden_size + 2,)


def test_build_native_poolers_boseos_and_bgem3(default_vllm_config):
    head = TokenClassifierPoolerHead(
        classifier=torch.nn.Linear(8, 1).to(DEVICE), head_dtype=torch.float32
    )
    embed = SequencePooler(
        pooling=CLSPool(), head=EmbeddingPoolerHead(head_dtype=torch.float32)
    )
    boseos = BOSEOSFilter(TokenPooler(pooling=AllPool(), head=head), 1, 2)
    pooler = DispatchPooler(
        {
            "token_classify": boseos,
            "embed": embed,
            "embed&token_classify": BgeM3Pooler(boseos, embed),
        }
    )
    native = build_native_poolers(pooler)
    assert isinstance(native["token_classify"], NativeBOSEOSFilter)
    assert isinstance(native["embed&token_classify"], NativeBgeM3Pooler)


def test_encode_token_type_ids():
    from types import SimpleNamespace

    # Needs the attention-extension import chain via models/bert.py.
    bert = pytest.importorskip("vllm.model_executor.models.bert")
    _decode_token_type_ids = bert._decode_token_type_ids
    from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner

    lens = [4, 3, 5]
    boundaries = [2, None, 1]  # second-segment start per request
    states = make_states()
    for i, b in enumerate(boundaries):
        params = PoolingParams(task="classify")
        if b is not None:
            params.extra_kwargs = {"compressed_token_type_ids": b}
        states.add_request(i, params, [1] * lens[i])
    states.apply_staged_writes()

    num_tokens = sum(lens)
    input_ids = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=DEVICE)
    original = input_ids.clone()
    positions = torch.cat(
        [torch.arange(n, dtype=torch.int64, device=DEVICE) for n in lens]
    )
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=num_tokens,
        idx_mapping=torch.arange(3, dtype=torch.int32, device=DEVICE),
        query_start_loc=torch.from_numpy(_cumsum0(lens)).to(DEVICE),
        positions=positions,
        input_ids=input_ids,
    )

    runner = PoolingRunner.__new__(PoolingRunner)
    runner.states = states
    runner.model_decodes_token_types = True
    runner.encode_token_type_ids(input_batch)

    token_types = _decode_token_type_ids(input_ids)
    torch.testing.assert_close(input_ids, original)  # bits stripped by decode
    expected = torch.cat(
        [
            (torch.arange(n, device=DEVICE) >= (b if b is not None else n)).int()
            for n, b in zip(lens, boundaries)
        ]
    )
    torch.testing.assert_close(token_types, expected)
