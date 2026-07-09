# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _segment_mean_kernel(
    hidden_states_ptr,
    hidden_states_stride,
    query_start_loc_ptr,
    out_ptr,
    out_stride,
    hidden_size,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seg_idx = tl.program_id(0)
    start = tl.load(query_start_loc_ptr + seg_idx).to(tl.int64)
    end = tl.load(query_start_loc_ptr + seg_idx + 1).to(tl.int64)

    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h < hidden_size

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for t0 in range(start, end, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        tile = tl.load(
            hidden_states_ptr + t[:, None] * hidden_states_stride + h[None, :],
            mask=(t[:, None] < end) & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(tile, axis=0)

    num_tokens = (end - start).to(tl.float32)
    tl.store(out_ptr + seg_idx * out_stride + h, acc / num_tokens, mask=h_mask)


def segment_mean(
    hidden_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Per-request mean of ``hidden_states`` rows, segmented by
    ``query_start_loc`` ([num_segments + 1]). Returns [num_segments, H] fp32."""
    hidden_size = hidden_states.shape[-1]
    out = torch.empty(
        (num_segments, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    if num_segments == 0:
        return out
    BLOCK_T = 32
    BLOCK_H = 128
    _segment_mean_kernel[(num_segments, triton.cdiv(hidden_size, BLOCK_H))](
        hidden_states,
        hidden_states.stride(0),
        query_start_loc,
        out,
        out.stride(0),
        hidden_size,
        BLOCK_T=BLOCK_T,
        BLOCK_H=BLOCK_H,
    )
    return out


@triton.jit
def _ragged_maxsim_kernel(
    q_ptr,
    q_stride,
    d_ptr,
    d_stride,
    q_start_loc_ptr,
    d_start_loc_ptr,
    out_ptr,
    dim,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    q_start = tl.load(q_start_loc_ptr + pair_idx).to(tl.int64)
    q_end = tl.load(q_start_loc_ptr + pair_idx + 1).to(tl.int64)
    q0 = q_start + tl.program_id(1).to(tl.int64) * BLOCK_Q
    if q0 >= q_end:
        return
    d_start = tl.load(d_start_loc_ptr + pair_idx).to(tl.int64)
    d_end = tl.load(d_start_loc_ptr + pair_idx + 1).to(tl.int64)
    if d_start == d_end:
        return

    q_rows = q0 + tl.arange(0, BLOCK_Q)
    q_mask = q_rows < q_end

    running_max = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    for d0 in range(d_start, d_end, BLOCK_D):
        d_rows = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_rows < d_end

        scores = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
        for k0 in range(0, dim, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < dim
            q_tile = tl.load(
                q_ptr + q_rows[:, None] * q_stride + k[None, :],
                mask=q_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            d_tile = tl.load(
                d_ptr + d_rows[:, None] * d_stride + k[None, :],
                mask=d_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q_tile, tl.trans(d_tile), input_precision="ieee")

        scores = tl.where(d_mask[None, :], scores, float("-inf"))
        running_max = tl.maximum(running_max, tl.max(scores, axis=1))

    contribution = tl.sum(tl.where(q_mask, running_max, 0.0))
    tl.atomic_add(out_ptr + pair_idx, contribution)


def ragged_maxsim(
    q_embs: torch.Tensor,
    d_embs: torch.Tensor,
    q_start_loc: torch.Tensor,
    d_start_loc: torch.Tensor,
    max_q_len: int,
) -> torch.Tensor:
    """MaxSim scores for ragged query/doc pairs.

    ``q_embs`` [sum_q, dim] and ``d_embs`` [sum_d, dim] hold the pairs'
    token embeddings back to back, delimited by ``q_start_loc`` /
    ``d_start_loc`` ([num_pairs + 1], on GPU). Returns [num_pairs] fp32
    scores: sum over query tokens of the max dot product over doc tokens.
    """
    num_pairs = q_start_loc.shape[0] - 1
    out = torch.zeros(num_pairs, dtype=torch.float32, device=q_embs.device)
    if num_pairs == 0 or max_q_len == 0:
        return out
    dim = q_embs.shape[-1]
    BLOCK_Q = 16
    BLOCK_D = 64
    BLOCK_K = max(16, min(triton.next_power_of_2(dim), 128))
    _ragged_maxsim_kernel[(num_pairs, triton.cdiv(max_q_len, BLOCK_Q))](
        q_embs,
        q_embs.stride(0),
        d_embs,
        d_embs.stride(0),
        q_start_loc,
        d_start_loc,
        out,
        dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )
    return out
