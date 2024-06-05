from typing import Dict, List, Optional, Tuple

import torch

from vllm.model_executor import SamplingMetadata
from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.proposer_worker_base import NonLLMProposerWorkerBase
from vllm.worker.model_runner import ModelInput


class MLPSpeculatorWorker(NonLLMProposerWorkerBase, MultiStepWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prev_request_context_lengths: Dict[str, int] = {}

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)

        seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        (input_tokens, input_positions, seq_lens,
         query_lens) = self.prepare_input_tensors(seq_group_metadata_list)

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            query_lens,  #TODO these are expected to be List[int]
            self.device,
            self.model_runner.pin_memory)

        model_outputs = self.model_runner.model.generate_proposals(
            input_ids=input_tokens,
            sample_len=sample_len,
            sampling_metadata=sampling_metadata)

        assert len(model_outputs) == sample_len

        return model_outputs, True

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ):
        if not seq_group_metadata_list:
            return ModelInput.empty(self.device)

        input_tokens: List[int] = []
        input_positions: List[int] = []

        seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        accepted_lengths_list: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            is_prompt = seq_group_metadata.is_prompt

            for seq_id in seq_ids:

                seq_data = seq_group_metadata.seq_data[seq_id]
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                else:
                    # get_num_computed_tokens is incorrect for spec decoding.
                    # So, we should have a special logic here.
                    # TODO(sang): Fix it.
                    context_len = seq_data.get_len() - 1

                seq_len = min(
                    seq_data.get_len(),
                    context_len + seq_group_metadata.token_chunk_size)
                if is_prompt:
                    tokens = seq_data.get_token_ids()[context_len:seq_len]
                else:
                    # Optimization. get_token_ids requires the entire copy of
                    # tokens.
                    tokens = [seq_data.get_last_token_id()]

                seq_lens.append(seq_len)
                context_lens.append(context_len)
                query_len = seq_len - context_len
                query_lens.append(query_len)
                input_tokens.extend(tokens)
                input_positions.extend(list(range(context_len, seq_len)))

                if seq_group_metadata.request_id in (
                        self.prev_request_context_lengths):
                    prev_context_length = self.prev_request_context_lengths[
                        seq_group_metadata.request_id]
                    accepted_length = context_len - prev_context_length
                    accepted_lengths_list.append(accepted_length)
                self.prev_request_context_lengths[
                    seq_group_metadata.request_id] = context_len

        if not self.model_runner.model.first_decode_step:
            self.model_runner.model.accepted_token_lengths = torch.tensor(
                accepted_lengths_list, device=self.device, dtype=torch.long)

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        input_positions_tensor = torch.tensor(input_positions,
                                              dtype=torch.long,
                                              device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        query_lens_tensor = torch.tensor(query_lens,
                                         dtype=torch.long,
                                         device=self.device)
        return (input_tokens_tensor, input_positions_tensor, seq_lens_tensor,
                query_lens_tensor)