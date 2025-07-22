import inspect
from functools import partial
from typing import Callable, Optional

import torch

from logits_process import LogitsProcessor as V0LogitsProcessor
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality
from vllm import SamplingParams

from vllm.v1.sample.logits_processor import LogitsProcessor

# This is the class that the user will have to provide.
# Perhaps change this to an ABC with its own is_argmax_invariant method or use a metaclass
RequestLogitsProcessor = Callable[[SamplingParams], Optional[V0LogitsProcessor]]


class AdapterLogitsProcessor(LogitsProcessor):

    def __init__(self, req_logits_processor: RequestLogitsProcessor):
        self.req_logits_processor = req_logits_processor
        self.requests: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}

        # TODO
        #self.argmax_invariant = req_logits_processor.is_argmax_invariant()
        self.argmax_invariant = False

    def is_argmax_invariant(self) -> bool:
        return self.argmax_invariant

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update:
            for i in batch_update.removed:
                self.requests.pop(i, None)

            for i, j, mdir in batch_update.moved:
                i_lp = self.requests.pop(i, None)
                j_lp = self.requests.pop(j, None)
                if i_lp is not None:
                    self.requests[j] = i_lp
                if mdir == MoveDirectionality.SWAP and j_lp is not None:
                    self.requests[i] = j_lp

            for index, sampling_params, input_ids in batch_update.added:
                prompt_ids = []  # TODO
                v0_lp = self.req_logits_processor(sampling_params)
                if v0_lp is not None:
                    takes_prompt_ids = len(
                        inspect.signature(v0_lp).parameters) == 3
                    args = [prompt_ids, input_ids] if (
                        takes_prompt_ids) else [input_ids]
                    self.requests[index] = partial(v0_lp, *args)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.requests:
            for index, lp in self.requests.items():
                req_logits = logits[index]
                new_logits = lp(req_logits)
                if new_logits is not req_logits:
                    logits[index] = new_logits
        return logits
