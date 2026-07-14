# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-step prefill attention-budget scheduling.

Covers the per-step accumulator cap (chunk-size inversion, depth sensitivity,
chunk rounding, the forward-progress/no-starvation guard) and the scheduler
admission behavior (deep chunks capped, per-step packing across requests).
"""

from tests.v1.core.utils import create_requests, create_scheduler


def _enable_budget(scheduler, budget_ms, b, s, k1, k2, chunk_round=1):
    """Turn on the attention budget on an already-built scheduler (schedule()
    reads these attributes live, so no need to thread them through the
    create_scheduler helper). ``chunk_round`` defaults to 1 so cap sizes in the
    unit tests are exact unless rounding is what is under test."""
    scheduler.latency_budget_ms = budget_ms
    scheduler._lat_b = b
    scheduler._lat_s = s
    scheduler._lat_k1 = k1
    scheduler._lat_k2 = k2
    scheduler._chunk_round = chunk_round
    scheduler._step_lat = b  # fresh step (schedule() resets this to _lat_b)


def test_attn_budget_cap_inverts_cost():
    # Linear regime (k2=0): remaining budget / per-token cost.
    sched = create_scheduler()
    _enable_budget(sched, budget_ms=140.0, b=90.0, s=0.01, k1=0.0, k2=0.0)
    # remaining = 140 - 90 = 50 ms; 50 / 0.01 = 5000 tokens.
    assert sched._attn_budget_token_cap(prefix_p=0, num_new=100_000) == 5000
    # Never returns more than the request has left.
    assert sched._attn_budget_token_cap(prefix_p=0, num_new=1234) == 1234


def test_attn_budget_cap_shrinks_with_depth():
    # A quadratic k2 term makes deeper prefixes admit fewer tokens per step.
    sched = create_scheduler()
    _enable_budget(sched, budget_ms=200.0, b=90.0, s=0.01, k1=0.0, k2=1e-6)
    caps = [sched._attn_budget_token_cap(p, 100_000) for p in (0, 8192, 65536)]
    assert caps[0] > caps[1] > caps[2] > 0


def test_attn_budget_cap_charges_kv_read_up_front():
    # The fixed k1*P KV-read cost is subtracted before solving for C, so a
    # deeper prefix leaves less budget for query tokens.
    sched = create_scheduler()
    _enable_budget(sched, budget_ms=200.0, b=90.0, s=0.01, k1=5e-4, k2=0.0)
    # P=0: remaining = 110 ms -> 11000 tok. P=100k: 110 - 0.5e-3*1e5 = 60 -> 6000.
    assert sched._attn_budget_token_cap(prefix_p=0, num_new=100_000) == 11000
    assert sched._attn_budget_token_cap(prefix_p=100_000, num_new=100_000) == 6000


def test_no_starvation_guard_admits_minimal_chunk():
    # Budget below the base step cost b: a fresh step (nothing admitted yet)
    # still lets one page-rounded minimal chunk through so a deep request can
    # never starve.
    sched = create_scheduler()
    _enable_budget(
        sched, budget_ms=50.0, b=90.0, s=0.01, k1=0.0, k2=0.0, chunk_round=64
    )
    assert sched._attn_budget_token_cap(prefix_p=0, num_new=100_000) == 64
    # Once the step already holds work (step_lat past b), an over-budget request
    # is deferred instead (returns 0).
    sched._step_lat = 200.0
    assert sched._attn_budget_token_cap(prefix_p=0, num_new=100_000) == 0


def test_amortization_floor_lifts_deep_cap():
    # With k1>0 the deep-prefix cap is floored at C* = sqrt(2*k1*P/k2) so the
    # fixed KV read is amortized over enough tokens instead of re-paid on an
    # over-small chunk. No-op when k1=0 (see the other tests).
    sched = create_scheduler(max_num_batched_tokens=32768)
    _enable_budget(
        sched, budget_ms=130.0, b=90.0, s=0.01, k1=6e-4, k2=4e-7, chunk_round=1
    )
    # Normal path (remaining>0 but small): raw cap ~332 tok, floored up to C*.
    p = 50_000
    c_opt = int((2.0 * (6e-4 * p) / 4e-7) ** 0.5)
    assert sched._attn_budget_token_cap(prefix_p=p, num_new=100_000) == c_opt
    # Guard path (k1*P alone exceeds the budget): floored to C* rather than a
    # tiny minimal chunk, so a deep request still amortizes its read.
    p2 = 200_000
    c_opt2 = min(int((2.0 * (6e-4 * p2) / 4e-7) ** 0.5), 32768)
    assert sched._attn_budget_token_cap(prefix_p=p2, num_new=100_000) == c_opt2


def test_cap_rounds_down_to_page_multiple():
    sched = create_scheduler()
    _enable_budget(
        sched, budget_ms=140.0, b=90.0, s=0.01, k1=0.0, k2=0.0, chunk_round=256
    )
    # raw cap 5000 -> floor to a multiple of 256 -> 4864 (= 19 * 256).
    assert sched._attn_budget_token_cap(prefix_p=0, num_new=100_000) == 4864


def test_charge_step_lat_accumulates():
    sched = create_scheduler()
    _enable_budget(sched, budget_ms=140.0, b=90.0, s=0.01, k1=5e-4, k2=4e-7)
    sched._step_lat = 90.0
    sched._charge_step_lat(num_new=1000, prefix_p=8192)
    expected = 90.0 + 5e-4 * 8192 + 0.01 * 1000 + 4e-7 * (1000 * 8192 + 1000 * 1000 / 2)
    assert sched._step_lat == expected


def test_set_latency_coeffs_installs_coeffs():
    sched = create_scheduler()
    sched.set_latency_coeffs(M_floor=140.0, b=33.0, s=0.012, k1=2.3e-3, k2=5.9e-7)
    assert sched._lat_floor == 140.0
    assert sched._lat_b == 33.0
    assert sched._lat_s == 0.012
    assert sched._lat_k1 == 2.3e-3
    assert sched._lat_k2 == 5.9e-7


def test_schedule_caps_deep_prefill_chunk():
    # Prompt (6000) exceeds the buffer (2048); the budget caps the first chunk
    # below the buffer: fresh cap = (105-90)/0.01 = 1500 tokens.
    sched = create_scheduler(max_num_batched_tokens=2048)
    _enable_budget(sched, budget_ms=105.0, b=90.0, s=0.01, k1=0.0, k2=0.0)
    req = create_requests(num_requests=1, num_tokens=6000)[0]
    sched.add_request(req)
    output = sched.schedule()
    assert output.num_scheduled_tokens[req.request_id] == 1500


def test_schedule_packs_requests_until_budget_exhausted():
    # The first request's 1500-token chunk consumes the budget; the second is
    # deferred to a later step.
    sched = create_scheduler(max_num_batched_tokens=2048)
    _enable_budget(sched, budget_ms=105.0, b=90.0, s=0.01, k1=0.0, k2=0.0)
    reqs = create_requests(num_requests=2, num_tokens=6000)
    for req in reqs:
        sched.add_request(req)
    output = sched.schedule()
    assert output.num_scheduled_tokens[reqs[0].request_id] == 1500
    assert reqs[1].request_id not in output.num_scheduled_tokens


def test_schedule_progresses_when_budget_below_base_cost():
    # Even with budget < b, the no-starvation guard admits one minimal chunk so
    # the request is never stuck.
    sched = create_scheduler(max_num_batched_tokens=2048)
    _enable_budget(
        sched, budget_ms=50.0, b=90.0, s=0.01, k1=0.0, k2=0.0, chunk_round=64
    )
    req = create_requests(num_requests=1, num_tokens=6000)[0]
    sched.add_request(req)
    output = sched.schedule()
    assert output.num_scheduled_tokens[req.request_id] == 64


def test_disabled_budget_is_a_noop():
    # With the budget off, the first chunk fills to the buffer (2048) as usual.
    sched = create_scheduler(max_num_batched_tokens=2048)
    req = create_requests(num_requests=1, num_tokens=6000)[0]
    sched.add_request(req)
    output = sched.schedule()
    assert output.num_scheduled_tokens[req.request_id] == 2048
