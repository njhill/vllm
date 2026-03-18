# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for flash sampling via Gumbel-Max trick.

Fuses temperature scaling, min_p thresholding, Gumbel noise generation,
and per-tile argmax into a single tiled pass over logits. This avoids
materializing the full [B, V] logits tensor to HBM and enables
lightweight TP reduction via Gumbel-Max instead of all-gathering logits.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _flash_sample_max_kernel(
    max_vals_ptr,
    max_vals_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Pre-pass: compute per-tile max of temperature-scaled logits.

    Used for min_p thresholding. Each tile stores its local max to
    max_vals_ptr[token_idx, tile_idx].
    """
    token_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    # Apply temperature scaling.
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0 and temp != 1.0:
        logits = logits / temp

    tile_max = tl.max(logits, axis=0)
    tl.store(max_vals_ptr + token_idx * max_vals_stride + tile_idx, tile_max)


@triton.jit
def _flash_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    min_p_ptr,
    global_max_ptr,
    seeds_ptr,
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_MIN_P: tl.constexpr,
):
    """Main kernel: temperature + min_p + Gumbel noise + per-tile argmax.

    Grid: (num_tokens, num_tiles).
    Each tile produces a (max_gumbel_value, token_index) candidate pair.
    """
    token_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    # Apply temperature scaling.
    temp = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0 and temp != 1.0:
        logits = logits / temp

    # Apply min_p thresholding.
    if HAS_MIN_P:
        min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
        if min_p > 0.0:
            g_max = tl.load(global_max_ptr + token_idx)
            threshold = g_max + tl.log(min_p)
            logits = tl.where(logits < threshold, float("-inf"), logits)

    if temp != 0.0:
        # Generate Gumbel noise for sampling.
        seed = tl.load(seeds_ptr + req_state_idx)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)
        u = tl.rand(gumbel_seed, block)
        u = tl.maximum(u, 1e-7)
        gumbel_noise = -tl.log(-tl.log(u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    # Per-tile argmax.
    value, idx = tl.max(logits, axis=0, return_indices=True)
    token_id = tile_idx * BLOCK_SIZE + idx
    tl.store(
        local_argmax_ptr + token_idx * local_argmax_stride + tile_idx,
        token_id,
    )
    tl.store(
        local_max_ptr + token_idx * local_max_stride + tile_idx,
        value,
    )


def flash_sample_kernel(
    logits: torch.Tensor,  # [num_tokens, V_local] (FP16/BF16)
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    min_p: torch.Tensor,  # [max_num_reqs]
    seeds: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    has_min_p: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused flash sampling kernel.

    Returns:
        (max_gumbel_values, token_indices) each of shape [num_tokens].
    """
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_tiles = triton.cdiv(vocab_size, BLOCK_SIZE)

    local_argmax = logits.new_empty(num_tokens, num_tiles, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_tiles, dtype=torch.float32)

    # If min_p is needed, run a pre-pass to get per-token global max.
    global_max = None
    if has_min_p:
        tile_maxes = logits.new_empty(num_tokens, num_tiles, dtype=torch.float32)
        _flash_sample_max_kernel[(num_tokens, num_tiles)](
            tile_maxes,
            tile_maxes.stride(0),
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            temperature,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        global_max = tile_maxes.max(dim=-1).values  # [num_tokens]

    _flash_sample_kernel[(num_tokens, num_tiles)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        min_p,
        global_max,
        seeds,
        pos,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_MIN_P=has_min_p,
    )

    # Reduce across tiles to find global winner per token.
    max_tile_idx = local_max.argmax(dim=-1, keepdim=True)
    max_values = local_max.gather(dim=-1, index=max_tile_idx).squeeze(-1)
    token_ids = local_argmax.gather(dim=-1, index=max_tile_idx).squeeze(-1)
    return max_values, token_ids
