# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV4Model._replay_forward/_replay_run run the replay layers on the
forward's replay batch."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model, ReplayBatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")
NUM_TOKENS = 401
GRAPH_SIZE = 512
# decode (1 token), trimmed prefill (300 of 300), untrimmed prefill (100 of 300)
WINDOW = 128
REPLAY_ROWS = [0, *range(301 - WINDOW, 301), *range(301, 401)]


def _replay_model(run_layers, topk_buffer=None, candidate_buffer=None, graphs=False):
    """DeepseekV4Model's replay forward, bound to just the state it uses.

    graphs: size static output buffers so the model graph's post-break
    segments can read them (breakable piecewise capture).
    """
    model = SimpleNamespace(
        replay_batch=None,
        use_mega_moe=False,
        _replay_row_buffers=[
            buf for buf in (topk_buffer, candidate_buffer) if buf is not None
        ],
        _replay_row_scratch=None,
        _replay_static_outputs=graphs,
        _replay_max_output_rows=GRAPH_SIZE,
        _replay_outputs=None,
    )
    model._run_decoder_replay_layers = run_layers
    model._replay_forward = MethodType(DeepseekV4Model._replay_forward, model)
    model._replay_run = MethodType(DeepseekV4Model._replay_run, model)
    model._mega_gate_metadata = MethodType(DeepseekV4Model._mega_gate_metadata, model)
    return model


def _context(metadata, cudagraph_mode=CUDAGraphMode.NONE):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=metadata,
        slot_mapping={},
        is_padding=torch.zeros(NUM_TOKENS, dtype=torch.bool, device=DEVICE),
        cudagraph_runtime_mode=cudagraph_mode,
    )


def _replay_batch(rows, metadata, trims=True):
    rows = torch.tensor(rows, device=DEVICE)
    inv_rows = torch.full((NUM_TOKENS,), -1, dtype=torch.int32, device=DEVICE)
    inv_rows[rows] = torch.arange(len(rows), dtype=torch.int32, device=DEVICE)
    return ReplayBatch(
        rows=rows,
        inv_rows=inv_rows,
        trims=trims,
        attn_metadata=metadata,
        slot_mapping={},
        is_padding=torch.zeros(len(rows), dtype=torch.bool, device=DEVICE),
        dp_metadata=None,
    )


def test_run_gathers_states_and_realigns_shared_indexer_buffers():
    topk = torch.arange(NUM_TOKENS * 4, device=DEVICE).view(NUM_TOKENS, 4).int()
    candidates = torch.arange(NUM_TOKENS * 3, device=DEVICE).view(NUM_TOKENS, 3).int()
    topk_before, candidates_before = topk.clone(), candidates.clone()
    seen = {}

    def run_layers(hidden_states, *rest, **kwargs):
        seen["hidden_states"] = hidden_states.clone()
        replay_context = get_forward_context()
        seen["attn_metadata"] = replay_context.attn_metadata
        seen["batch_descriptor"] = replay_context.batch_descriptor
        return (hidden_states, rest[2])  # pre_mix

    model = _replay_model(run_layers, topk, candidates)
    hidden = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.float32)[:, None]
    states = (hidden, hidden.long(), None, hidden, hidden, hidden, hidden)
    full, replay = object(), object()
    model.replay_batch = _replay_batch(REPLAY_ROWS, replay)
    context = _context(full)
    with override_forward_context(context):
        outputs = model._replay_forward(*states)
        assert get_forward_context() is context

    rows = torch.tensor(REPLAY_ROWS, device=DEVICE)
    assert torch.equal(seen["hidden_states"], hidden[rows])
    assert seen["attn_metadata"] is replay
    # The replay runs eagerly, so the batch descriptor dispatch is the batch's.
    assert seen["batch_descriptor"] is None
    assert torch.equal(topk[: len(REPLAY_ROWS)], topk_before[rows])
    assert torch.equal(candidates[: len(REPLAY_ROWS)], candidates_before[rows])
    assert outputs[0].shape[0] == NUM_TOKENS
    assert torch.equal(outputs[0][rows], hidden[rows])
    assert outputs[0][1:173].abs().sum() == 0


def test_no_replay_batch_runs_the_whole_batch():
    seen = {}

    def run_layers(hidden_states, *rest, **kwargs):
        seen["attn_metadata"] = get_forward_context().attn_metadata
        return (hidden_states, rest[2])

    model = _replay_model(run_layers)
    hidden = torch.zeros(NUM_TOKENS, 1, device=DEVICE)
    states = (hidden, hidden.long(), None, hidden, hidden, hidden, hidden)
    full = object()
    with override_forward_context(_context(full)):
        outputs = model._replay_forward(*states)
    assert outputs[0] is hidden and seen["attn_metadata"] is full


class _Metadata:
    """Persistent buffers the fake kernels read, refilled per step."""

    def __init__(self):
        self.slot_mapping = torch.zeros(GRAPH_SIZE, dtype=torch.int64, device=DEVICE)
        self.token_to_req_indices = torch.zeros(
            GRAPH_SIZE, dtype=torch.int32, device=DEVICE
        )
        self.num_tokens = 0

    def fill(self, rows: list[int]) -> "_Metadata":
        rows_t = torch.tensor(rows, device=DEVICE)
        self.slot_mapping[: len(rows)] = rows_t * 7
        self.token_to_req_indices[: len(rows)] = (rows_t // 100).int()
        self.num_tokens = len(rows)
        return self


def _fake_attention(x: torch.Tensor, out: torch.Tensor) -> None:
    """Reads the current metadata eagerly, like the real attention kernels
    running in the model graph's replay break."""
    md = get_forward_context().attn_metadata
    n = md.num_tokens
    out[:n] = x[:n] + md.token_to_req_indices[:n].to(x.dtype)[:, None]


def _fake_replay_layers(
    hidden, positions, _, pre_mix, post_mix, res_mix, residual, *args, **kwargs
):
    # Like the window KV insert, this op takes the metadata's slot mapping by
    # address.
    slots = get_forward_context().attn_metadata.slot_mapping
    x = hidden * 2 + slots[: hidden.shape[0], None].to(hidden.dtype)
    out = torch.empty_like(x)
    _fake_attention(x, out)
    return (out + positions[:, None].to(out.dtype), pre_mix + residual)


def _states(seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    hidden = torch.randn(NUM_TOKENS, 3, device=DEVICE, generator=g)
    return (
        hidden,
        torch.arange(NUM_TOKENS, device=DEVICE),
        None,
        hidden + 1,
        hidden + 2,
        hidden + 3,
        hidden + 4,
    )


def test_replay_break_matches_eager():
    """The replay runs as an eager break of the model graph: captured with a
    whole (untrimmed) batch, a replay with a trimming batch runs the replay
    layers eagerly on the replay rows and writes the static output buffers the
    graph's post-break segments read. Matches the eager path on the same
    batch."""
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import _current_stream_tls

    eager = _replay_model(_fake_replay_layers)
    graphed = _replay_model(_fake_replay_layers, graphs=True)
    metadata = _Metadata()
    states = _states(0)
    all_rows = list(range(NUM_TOKENS))

    prev_stream = getattr(_current_stream_tls, "value", None)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            with override_forward_context(_context(metadata.fill(all_rows))):
                graphed._replay_forward(*states)  # sizes the static buffers
            with override_forward_context(
                _context(metadata.fill(all_rows), CUDAGraphMode.PIECEWISE)
            ):
                outer = BreakableCUDAGraphCapture(
                    current_platform.get_global_graph_pool()
                )
                with outer:
                    hidden_out, pre_mix_out = graphed._replay_forward(*states)
            assert outer.num_eager_breaks == 1
            for dst, src in zip(states, _states(1)):
                if dst is not None:
                    dst.copy_(src)
            graphed.replay_batch = _replay_batch(
                REPLAY_ROWS, metadata.fill(REPLAY_ROWS)
            )
            with override_forward_context(_context(None, CUDAGraphMode.PIECEWISE)):
                outer.replay()
                torch.accelerator.synchronize()
                outputs = graphed._replay_forward(*states)
            eager.replay_batch = _replay_batch(REPLAY_ROWS, metadata.fill(REPLAY_ROWS))
            with override_forward_context(_context(None)):
                expected = eager._replay_forward(*states)
    finally:
        torch.cuda.current_stream().wait_stream(stream)
        _current_stream_tls.value = prev_stream
    assert torch.equal(hidden_out, expected[0])
    assert torch.equal(pre_mix_out, expected[1])
    assert torch.equal(outputs[0], expected[0])
    assert torch.equal(outputs[1], expected[1])
