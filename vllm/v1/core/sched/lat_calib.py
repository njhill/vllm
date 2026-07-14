# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Startup self-calibration of the prefill latency-budget coefficients.

The budget model is  T = max(M_floor, b + s*N + k1*sum(P_i) + k2*A)  with
  N  = sum_i C_i                        (query tokens)
  P_i= per-chunk KV-prefix length       (k1 = KV-read cost, ~independent of C)
  A  = sum_i (C_i*P_i + C_i*(C_i-1)/2)   (k2 = compute cost per query,key pair)

The coefficients depend on the full deployment (model, quant, GPU, and the
TP/PP/DP/EP layout), so they should be measured *for the running config*. This
runs a handful of synthetic prefill steps through the executor's normal
execute_model path -- which already orchestrates TP all-reduce, PP stage
send/recv, and DP sync -- so it works for any parallelism, and times the true
per-step (microbatch) latency by wall clock. Runs in a few tens of seconds.
"""

from __future__ import annotations

import time

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

_CALIB_REQ_PREFIX = "_latcalib_"


def _build_probe_cases(max_model_len: int, max_num_batched_tokens: int):
    """Probe set scaled to the deployment's limits:
    * fresh single chunks -> memory-bound floor + MLP-per-token slope `s`
    * deep continuation chunks at TWO chunk sizes, each at shallow & deep P
      -> separates the KV-read term k1 (∝P, C-independent) from the compute
      term k2 (∝C·P), since dLatency/dP = k1 + k2*C.
    """
    cap = max(256, min(max_num_batched_tokens, max_model_len - 1))
    fresh = sorted({n for n in (1024, 4096, 8192, 16384) if n <= cap})
    if len(fresh) < 2:
        fresh = sorted({max(256, cap // 4), cap})
    c1 = min(2048, cap)
    c2 = min(16384, cap)
    cases = [("fresh", [(0, n)]) for n in fresh]
    for c in sorted({c1, c2}):
        p_deep = max(0, max_model_len - c - 1)
        for p in sorted({0, p_deep}):
            cases.append((f"deep_c{c}", [(p, c)]))
    return cases


def _make_scheduler_output(batch, block_size, total_blocks, cursor):
    """Build a synthetic SchedulerOutput (engine-side) for `batch` = list of
    (prefix_P, chunk_C). KV-cache *contents* are irrelevant for latency, so block
    ids are handed out from a rolling cursor (wrapping the pool -> always valid).
    """
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

    def cdiv(a, b):
        return (a + b - 1) // b

    def alloc(n_tokens):
        nblk = cdiv(n_tokens, block_size)
        usable = max(total_blocks - 1, 1)
        ids = [1 + (cursor[0] + i) % usable for i in range(nblk)]
        cursor[0] = (cursor[0] + nblk) % usable
        return (ids,)  # single KV-cache group (MLA)

    new_reqs, num_sched, total, req_ids = [], {}, 0, []
    for i, (p, c) in enumerate(batch):
        rid = f"{_CALIB_REQ_PREFIX}{i}"
        ids = [0] * (p + c)
        new_reqs.append(
            NewRequestData(
                req_id=rid,
                prompt_token_ids=ids,
                mm_features=[],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                pooling_params=None,
                block_ids=alloc(p + c),
                num_computed_tokens=p,
                lora_request=None,
                prompt_embeds=None,
                prompt_is_token_ids=[True] * (p + c),
                prefill_token_ids=ids,
            )
        )
        num_sched[rid] = c
        total += c
        req_ids.append(rid)
    so = SchedulerOutput.make_empty()
    so.scheduled_new_reqs = new_reqs
    so.num_scheduled_tokens = num_sched
    so.total_num_scheduled_tokens = total
    so.num_common_prefix_blocks = [0]
    return so, req_ids


def _time_step(executor, so) -> float:
    """Run one synthetic step through the coordinated path and return its
    wall-clock latency (ms). sample_tokens' output copy syncs the GPU, so the
    measured time covers the full forward (incl. PP stage traversal)."""
    t0 = time.perf_counter()
    executor.execute_model(so, non_block=False)
    executor.sample_tokens(None, non_block=False)
    return (time.perf_counter() - t0) * 1000.0


def fit_coeffs(samples):
    """Fit (M_floor, b, s, k1, k2) from probe samples. Two-term attention:
    k1 (KV-read, ∝P) + k2 (compute, ∝C·P). Returns None if too few points."""
    if not samples or len(samples) < 4:
        return None
    fresh = [r for r in samples if r["kind"] == "fresh"]
    deep = [r for r in samples if r["kind"].startswith("deep")]
    M = min(r["fwd_ms"] for r in fresh) if fresh else min(r["fwd_ms"] for r in samples)

    by_c: dict[int, list] = {}
    for r in deep:
        by_c.setdefault(r["C"], []).append(r)
    slopes = {}
    for c, rs in by_c.items():
        rs = sorted(rs, key=lambda r: r["P"])
        dP = rs[-1]["P"] - rs[0]["P"]
        if dP > 0:
            slopes[c] = (rs[-1]["fwd_ms"] - rs[0]["fwd_ms"]) / dP

    k1, k2 = None, None
    if len(slopes) >= 2:
        cs = sorted(slopes)
        c_lo, c_hi = cs[0], cs[-1]
        k2 = max(0.0, (slopes[c_hi] - slopes[c_lo]) / (c_hi - c_lo))
        k1 = max(0.0, slopes[c_lo] - k2 * c_lo)
    elif len(slopes) == 1:
        c, sl = next(iter(slopes.items()))
        k1, k2 = 0.0, max(0.0, sl / c)

    f = sorted(fresh, key=lambda r: r["N"])
    if len(f) >= 2 and k2 is not None:
        cb = [r for r in f if r["fwd_ms"] > M * 1.02] or f[-2:]
        X = np.array([[1.0, r["N"]] for r in cb])
        y = np.array([r["fwd_ms"] - k2 * (r["N"] * (r["N"] - 1) / 2.0) for r in cb])
        b, s = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        return None
    return {
        "M_floor": float(M),
        "b": float(b),
        "s": max(0.0, float(s)),
        "k1": float(k1 or 0.0),
        "k2": float(k2 or 0.0),
    }


def calibrate(
    model_executor,
    kv_cache_config,
    max_model_len: int,
    max_num_batched_tokens: int,
    n_warmup: int = 2,
    n_iters: int = 3,
):
    """Probe + fit the latency coefficients for the running deployment. Returns
    a coeffs dict or None. Parallelism-agnostic (uses executor.execute_model)."""
    t0 = time.perf_counter()
    try:
        group = kv_cache_config.kv_cache_groups[0]
        block_size = group.kv_cache_spec.block_size
        total_blocks = kv_cache_config.num_blocks
        # Don't probe deeper than the KV pool can index for a single sequence.
        max_model_len = min(max_model_len, total_blocks * block_size - 1)
    except Exception as e:  # noqa: BLE001
        logger.warning("Latency calibration: cannot read KV geometry: %s", e)
        return None

    cursor = [1]
    samples = []
    all_req_ids: set[str] = set()
    try:
        for kind, batch in _build_probe_cases(max_model_len, max_num_batched_tokens):
            ms = []
            try:
                for it in range(n_warmup + n_iters):
                    so, rids = _make_scheduler_output(
                        batch, block_size, total_blocks, cursor
                    )
                    all_req_ids.update(rids)
                    t = _time_step(model_executor, so)
                    if it >= n_warmup:
                        ms.append(t)
            except Exception as e:  # noqa: BLE001
                logger.warning("Latency calibration probe %s failed: %s", kind, e)
                continue
            if ms:
                ms.sort()
                p, c = batch[0]
                samples.append(
                    {
                        "kind": kind,
                        "N": c,
                        "C": c,
                        "P": p,
                        "A": c * p + c * (c - 1) / 2,
                        "fwd_ms": ms[len(ms) // 2],
                    }
                )
    finally:
        # Free the synthetic requests from worker state.
        if all_req_ids:
            try:
                from vllm.v1.core.sched.output import SchedulerOutput

                cleanup = SchedulerOutput.make_empty()
                cleanup.finished_req_ids = set(all_req_ids)
                model_executor.execute_model(cleanup, non_block=False)
            except Exception:  # noqa: BLE001, S110
                pass

    coeffs = fit_coeffs(samples)
    dt = time.perf_counter() - t0
    if coeffs:
        logger.info(
            "Prefill latency auto-calibration (%.1fs): M_floor=%.1fms b=%.1fms "
            "s=%.4f ms/tok k1=%.3e ms/ctx-tok k2=%.3e ms/pair (%d probes)",
            dt,
            coeffs["M_floor"],
            coeffs["b"],
            coeffs["s"],
            coeffs["k1"],
            coeffs["k2"],
            len(samples),
        )
    else:
        logger.warning(
            "Latency auto-calibration produced no fit (%.1fs, %d probes).",
            dt,
            len(samples),
        )
    return coeffs
