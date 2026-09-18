# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decoder-side SWA bounded replay batch DeepseekV41ModelState prepares."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.models.deepseek_v41.nvidia.model_state as model_state_module
from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the replay buffers live on the GPU"
)

WINDOW = 128
DEVICE = torch.device("cuda")
GROUPS = SimpleNamespace(
    kv_cache_groups=[
        SimpleNamespace(
            layer_names=["swa"],
            kv_cache_spec=SimpleNamespace(
                prefix_cacheable=False, prefix_replay_tokens=WINDOW
            ),
        ),
        SimpleNamespace(
            layer_names=["mla"],
            kv_cache_spec=SimpleNamespace(
                prefix_cacheable=True, prefix_replay_tokens=0
            ),
        ),
    ]
)
BLOCK_TABLES = (torch.zeros(4, 4, device=DEVICE),) * 2

# decode (1 token), trimmed prefill (300 of 300), untrimmed prefill (100 of 300)
QUERY_LENS = [1, 300, 100]
SEQ_LENS = [500, 300, 300]
PREFILLING = [False, True, True]
REPLAY_ROWS = [0, *range(301 - WINDOW, 301), *range(301, 401)]


@pytest.fixture
def state(monkeypatch):
    """A DeepseekV41ModelState whose attention builds are recorded, not run.

    The batch's own build goes through DefaultModelState.prepare_attn (faked);
    the replay layers' build calls build_attn_metadata directly.
    """
    cfg = MagicMock()
    cfg.model_config.enable_prompt_embeds = False
    cfg.model_config.uses_mrope = False
    cfg.model_config.is_multimodal_model = False
    cfg.scheduler_config.max_num_seqs = 16
    cfg.scheduler_config.max_num_batched_tokens = 1024
    cfg.compilation_config.cudagraph_capture_sizes = [256, 512]
    cfg.parallel_config.data_parallel_size = 1
    # The model the state replays into: decoder_replay_window set, and its
    # replay_batch attribute set per step.
    replay_model = SimpleNamespace(decoder_replay_window=WINDOW, replay_batch=None)
    model = SimpleNamespace(token_lookback_depth=0, decoder_replay_model=replay_model)
    builds: list = []
    replay_builds: list = []

    def prepare_attn(self, input_batch, cg_mode, block_tables, slot_mappings, *a, **kw):
        builds.append(
            SimpleNamespace(
                batch=input_batch,
                slot_mappings=slot_mappings.clone(),
                replay_start=kw["model_specific_attn_metadata"].replay_start,
            )
        )
        return {"swa": object()}

    def record_build_attn_metadata(**kwargs):
        replay_builds.append(SimpleNamespace(**kwargs))
        return {"swa": object()}

    monkeypatch.setattr(DefaultModelState, "prepare_attn", prepare_attn)
    monkeypatch.setattr(
        model_state_module, "build_attn_metadata", record_build_attn_metadata
    )
    state = DeepseekV41ModelState(cfg, model, None, DEVICE)
    state.builds = builds  # type: ignore[attr-defined]
    state.replay_builds = replay_builds  # type: ignore[attr-defined]
    return state


def _input_batch(
    query_lens: list[int],
    seq_lens: list[int],
    is_prefilling: list[bool],
    device_query_lens: list[int] | None = None,
    num_tokens_after_padding: int | None = None,
    max_query_len: int | None = None,
) -> InputBatch:
    num_reqs, num_tokens = len(query_lens), sum(query_lens)
    num_padded = num_tokens_after_padding or num_tokens
    batch = InputBatch.make_dummy(num_reqs, num_tokens, InputBuffers(16, 1024, DEVICE))
    return replace(
        batch,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_padded,
        query_start_loc=torch.tensor(
            [0, *np.cumsum(device_query_lens or query_lens)],
            dtype=torch.int32,
            device=DEVICE,
        ),
        query_start_loc_np=np.array([0, *np.cumsum(query_lens)], dtype=np.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        is_prefilling_np=np.array(is_prefilling, dtype=np.bool_),
        positions=torch.cat(
            [torch.arange(s - q, s) for q, s in zip(query_lens, seq_lens)]
            + [torch.zeros(num_padded - num_tokens, dtype=torch.int64)]
        ).to(DEVICE),
        is_padding=torch.arange(num_padded, device=DEVICE) >= num_tokens,
        max_query_len=max_query_len,
    )


def _slot_mappings(num_tokens: int) -> torch.Tensor:
    return torch.stack([torch.arange(num_tokens), torch.arange(num_tokens) * 10]).to(
        DEVICE
    )


def _prepare(state, batch, cg_mode=CUDAGraphMode.NONE):
    """Runs prepare_attn; returns the replay batch and the replay build's inputs."""
    state.prepare_attn(
        batch,
        cg_mode,
        BLOCK_TABLES,
        _slot_mappings(batch.num_tokens_after_padding),
        [],
        GROUPS,
    )
    replay = state.decoder_replay.replay_batch
    return replay, (state.replay_builds[-1] if replay is not None else None)


def test_replay_batch_keeps_each_request_window(state):
    state._req_replay_start[2] = 50  # the encoder-side replay start of request 2
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING)
    replay, build = _prepare(state, batch)
    assert replay is not None and replay.trims
    rows = torch.tensor(REPLAY_ROWS, device=DEVICE)
    assert torch.equal(replay.rows, rows)
    assert build.num_reqs == 3 and build.num_tokens == 229
    assert build.query_start_loc_cpu.tolist() == [0, 1, 129, 229]
    assert build.query_start_loc_gpu.tolist() == [0, 1, 129, 229]
    assert build.max_query_len == WINDOW
    # The trimmed request's window starts at its replay window; a higher
    # encoder-side replay start stands.
    assert build.model_specific_attn_metadata.replay_start.tolist() == [
        0,
        300 - WINDOW,
        50,
    ]
    assert torch.equal(build.positions, batch.positions[rows])
    assert torch.equal(build.slot_mappings, _slot_mappings(401)[:, rows])
    assert torch.equal(replay.slot_mapping["mla"], _slot_mappings(401)[1, rows])
    assert not replay.is_padding.any() and replay.dp_metadata is None


def test_replay_batch_keeps_device_decode_boundaries(state):
    """Adaptive verification resizes decodes on the GPU (CPU [2, 2], device
    [1, 3]): their rows stay and the boundaries shift only by trimmed rows."""
    batch = _input_batch(
        [2, 2, 300], [10, 10, 300], PREFILLING, device_query_lens=[1, 3, 300]
    )
    replay, build = _prepare(state, batch)
    assert replay is not None
    assert replay.rows.tolist() == [0, 1, 2, 3, *range(304 - WINDOW, 304)]
    assert build.query_start_loc_gpu.tolist() == [0, 1, 4, 4 + WINDOW]


def test_prompt_logprobs_requests_keep_their_rows(state):
    """Every prompt row of a prompt-logprobs request is read, so it never
    trims; the other prefills still do."""
    state._req_keeps_rows[1] = True
    batch = _input_batch([1, 300, 300], [500, 300, 300], PREFILLING)
    replay, build = _prepare(state, batch)
    assert replay is not None and replay.trims
    assert replay.rows.tolist() == [0, *range(1, 301), *range(601 - WINDOW, 601)]
    assert build.max_query_len == 300


def test_replay_batch_keeps_adaptive_verification_query_bound(state):
    """Adaptive verification bounds the decodes' device-side query lengths
    above their CPU lengths; the replay batch keeps that bound."""
    batch = _input_batch(QUERY_LENS, SEQ_LENS, PREFILLING, max_query_len=200)
    replay, build = _prepare(state, batch)
    assert replay is not None and build.max_query_len == 200


@pytest.mark.parametrize(
    "cg_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE]
)
def test_whole_batch_forwards_get_no_replay_batch(state, cg_mode):
    """Nothing trims (short prefills, a dummy/capture batch, a FULL-graph
    batch): the replay layers run on the whole batch, whatever the mode."""
    short = _input_batch([100, 100], [100, 100], [True, True])
    assert _prepare(state, short, cg_mode) == (None, None)
    dummy = _input_batch([512], [512], [False], num_tokens_after_padding=512)
    assert _prepare(state, dummy, cg_mode) == (None, None)
    assert len(state.builds) == 2  # the batch's own metadata only
    assert not state.replay_builds


@pytest.fixture
def dp_state(state, monkeypatch):
    """`state` on DP rank 0 of 2; `state.other` sets what rank 1 reports
    (trims, replay tokens)."""
    state.vllm_config.parallel_config.data_parallel_size = 2
    state.vllm_config.parallel_config.data_parallel_rank = 0
    state.vllm_config.parallel_config.is_moe_model = True
    state.other = (False, 0)  # type: ignore[attr-defined]

    def all_reduce(tensor, group):
        tensor[1] = torch.tensor(state.other, dtype=torch.int32)

    monkeypatch.setattr(model_state_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        model_state_module, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    return state


@pytest.mark.parametrize(
    "cg_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE]
)
def test_dp_ranks_replay_together(dp_state, cg_mode):
    """The replay layers' MoE collectives need every rank in the replay
    together: no rank trimming means no replay on any rank and in any mode;
    a trimming rank makes the others replay too, on their own counts."""
    short = _input_batch(
        [100, 100], [100, 100], [True, True], num_tokens_after_padding=512
    )
    dp_state.other = (False, 200)
    assert _prepare(dp_state, short, cg_mode) == (None, None)

    dp_state.other = (True, 300)  # rank 1 trims to 300 rows
    replay, _ = _prepare(dp_state, short, cg_mode)
    assert replay is not None and replay.trims
    # This rank keeps its 200 rows (nothing to trim) and replays on the
    # agreed counts: its own 200 and rank 1's 300.
    assert replay.rows.shape[0] == 200
    assert replay.dp_metadata.num_tokens_across_dp_cpu.tolist() == [200, 300]


def test_idle_dp_rank_dummy_trims_to_the_agreed_replay(dp_state):
    dummy = _input_batch([512], [512], [False])
    dp_state.other = (True, 129)
    replay, _ = _prepare(dp_state, dummy, CUDAGraphMode.PIECEWISE)
    assert replay is not None and replay.trims
    assert replay.rows.shape[0] == WINDOW
    assert replay.dp_metadata.num_tokens_across_dp_cpu.tolist() == [WINDOW, 129]
