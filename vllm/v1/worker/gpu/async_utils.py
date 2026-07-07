# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool = False,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)
        self._has_fault: torch.Tensor | None = None

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return self.model_runner_output


def _copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    """Move finished requests' pooled outputs to CPU (non-blocking).

    Handles both a stacked ``[num_reqs, ...]`` tensor (seq-level poolers) and a
    ragged ``list[Tensor | None]`` (token-level poolers). Unfinished
    chunked-prefill requests map to ``None``. Caller must synchronize the copy
    stream before reading the returned tensors.
    """
    num_reqs = len(finished_mask)
    if isinstance(raw_pooler_output, torch.Tensor):
        if raw_pooler_output.shape[0] != num_reqs:
            raise ValueError(
                "pooler output batch size does not match num_reqs: "
                f"{raw_pooler_output.shape[0]} != {num_reqs}."
            )
        if all(finished_mask):
            return list(raw_pooler_output.to("cpu", non_blocking=True).unbind(dim=0))

        finished_indices = [i for i, finished in enumerate(finished_mask) if finished]
        index_tensor = torch.tensor(
            finished_indices, device=raw_pooler_output.device, dtype=torch.long
        )
        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(
            "cpu", non_blocking=True
        )
        output: list[torch.Tensor | None] = [None] * num_reqs
        for i, out in zip(finished_indices, finished_outputs):
            output[i] = out
        return output

    if len(raw_pooler_output) != num_reqs:
        raise ValueError(
            "pooler output length does not match num_reqs: "
            f"{len(raw_pooler_output)} != {num_reqs}."
        )
    output = [None] * num_reqs
    for i, (out, finished) in enumerate(zip(raw_pooler_output, finished_mask)):
        if finished and out is not None:
            output[i] = out.to("cpu", non_blocking=True)
    return output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: PoolerOutput,
        finished_mask: list[bool],
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        # Retain a reference to the GPU output until the copy completes.
        self.pooler_output = pooler_output
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            self.model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                pooler_output, finished_mask
            )
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
