# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from unittest.mock import patch

import pytest
import torch

from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def test_basic_lifecycle():
    """Test lifecycle of a Remote Decode request."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        max_tokens=1,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1): Prefill.
    # (1a): schedule()
    scheduler_output = scheduler.schedule()
    assert len(scheduler.requests) == 1
    assert len(scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1

    # (1b): execute_model()
    model_runner_output = create_model_runner_output(reqs=[request])

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )

    # Ensure the request is finished after 1 token.
    assert request.is_finished()
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    output = engine_core_outputs[0].outputs[0]
    assert output.finish_reason == FinishReason.LENGTH
    assert output.kv_transfer_params is not None
    # The prefill stopped short of the prompt, so its token is not an output.
    assert output.new_token_ids == [] and output.new_logprobs is None

    # Request freed in Scheduler and in Persistent Batch ...
    assert request_id in scheduler.finished_req_ids
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0

    # ... but blocks should not be freed.
    assert len(scheduler.requests) == 1
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks[request_id]
    for block in blocks:
        assert block.ref_cnt == 1

    # STEP (2): Send Finished to PB.
    # (2a): schedule() - pass finished request to PB.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.requests) == 1
    assert len(scheduler.running) == 0
    assert len(scheduler_output.finished_req_ids) == 1
    assert request_id in scheduler_output.finished_req_ids
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler.finished_req_ids) == 0

    # (2b): execute_model()
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT

    # (2c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP (3): Finished sending.
    # (3a): schedule() - pass finished request to PB.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.requests) == 1
    assert len(scheduler.running) == 0
    assert len(scheduler_output.finished_req_ids) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler.finished_req_ids) == 0

    # (3b): execute_model()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={request_id}
    )

    # (3c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm we do not have any memory leaks after req lifecycle.
    assert_scheduler_empty(scheduler)


def test_short_prompt_lifecycle():
    """Test lifecycle of a Remote Decode request with short prompt."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Not enough tokens for full block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = BLOCK_SIZE // 2
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        max_tokens=1,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )

    scheduler.add_request(request)

    # STEP (1): Prefill.
    # (1a): schedule()
    scheduler_output = scheduler.schedule()
    assert len(scheduler.requests) == 1
    assert len(scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1

    # (1b): execute_model()
    model_runner_output = create_model_runner_output(reqs=[request])

    # (1c): update_from_output()
    # Even though tokens < block_size, there will be kv xfer for partial block.
    eco = scheduler.update_from_output(scheduler_output, model_runner_output)
    kv_transfer_params = eco[0].outputs[0].kv_transfer_params

    assert len(kv_transfer_params["remote_block_ids"]) == 1

    # Confirm we do not have any memory leaks after req lifecycle.
    # We need to mark sending finish to clear data for persistent batch.
    scheduler_output = scheduler.schedule()
    # Use create_model_runner_output to pass kv_connector_output along
    model_runner_output = create_model_runner_output(
        reqs=[request], finished_sending={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


def test_prompt_logprobs_prefill_stops_short():
    """A prefill with prompt logprobs (which skips reading the prefix cache)
    still stops at N-1, where it has every prompt logprob, so a decoder without
    them loads the same prefix; it returns no token."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = create_request(
        request_id=1,
        block_size=vllm_config.cache_config.block_size,
        num_tokens=24,
        do_remote_decode=True,
    )
    request.sampling_params.prompt_logprobs = 0
    request.sampling_params.skip_reading_prefix_cache = True
    request.max_tokens = 16

    scheduler.add_request(request)
    assert request.num_prompt_tokens == 24
    assert request.prefill_stop == 23
    assert request.max_tokens == 1
    scheduler_output = scheduler.schedule()
    assert scheduler_output.num_scheduled_tokens[request.request_id] == 23
    # The step reaching the stop returns all N-1 prompt logprobs, samples none.
    prompt_logprobs = LogprobsTensors(
        logprob_token_ids=torch.zeros(23, 1, dtype=torch.int32),
        logprobs=torch.zeros(23, 1),
        selected_token_ranks=torch.zeros(23, dtype=torch.int32),
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[]],
        prompt_logprobs_dict={request.request_id: prompt_logprobs},
    )
    eco = scheduler.update_from_output(scheduler_output, model_runner_output)
    output = eco[0].outputs[0]
    assert output.finish_reason == FinishReason.LENGTH
    assert output.new_token_ids == []
    assert output.new_prompt_logprobs_tensors is prompt_logprobs
    assert output.kv_transfer_params is not None

    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request], finished_sending={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


def test_prefix_cache_lifecycle():
    """Test that remote decode params still work with a prefix cache hit."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Prime the KVCache.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 3
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request_normal = create_request(
        request_id=1, block_size=BLOCK_SIZE, num_tokens=NUM_TOKENS
    )

    scheduler.add_request(request_normal)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_normal], use_eos=True
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    #####################
    # Actual Test: confirm we send all blocks.

    # Step (1): Send the KV Transfer.
    NUM_EXTERNAL_FULL_BLOCKS -= 1
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request_remote = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )

    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote])
    eco = scheduler.update_from_output(scheduler_output, model_runner_output)
    kv_transfer_params = eco[0].outputs[0].kv_transfer_params

    # Ensure we send all block ids, including the partial blocks,
    # even if there is a cache hit.
    # remote_block_ids is BlockIds (tuple of lists); sum block counts across groups.
    num_remote_blocks = sum(len(g) for g in kv_transfer_params["remote_block_ids"])
    assert num_remote_blocks == (NUM_EXTERNAL_FULL_BLOCKS + 1)

    # STEP (2): Ensure it is freed.
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={request_remote.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


def test_prefix_hit_leaves_a_token_before_prefill_stop():
    """A prefill that stops at N-1 on a block boundary must still compute a
    token: its prefix-cache hit stops one short of the stop, as a normal
    request's does of its prompt."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = 2 * BLOCK_SIZE + 1  # The prefill stops at 2 full blocks.

    requests = []
    for request_id, num_cached_blocks in ((1, 0), (2, 1)):
        request = create_request(
            request_id=request_id,
            block_size=BLOCK_SIZE,
            num_tokens=NUM_TOKENS,
            common_prefix_len=NUM_TOKENS,
            do_remote_decode=True,
        )
        requests.append(request)
        scheduler.add_request(request)
        assert request.prefill_stop == 2 * BLOCK_SIZE
        scheduler_output = scheduler.schedule()
        # The second request hits one block only, then computes the other.
        assert scheduler_output.num_scheduled_tokens[request.request_id] == (
            (2 - num_cached_blocks) * BLOCK_SIZE
        )
        # Stopping short of the prompt, the step is a partial prefill.
        model_runner_output = ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
        )
        eco = scheduler.update_from_output(scheduler_output, model_runner_output)
        (output,) = eco[0].outputs
        assert output.finish_reason == FinishReason.LENGTH
        assert output.new_token_ids == []

    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={request.request_id for request in requests}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


def test_external_hit_leaves_a_token_before_prefill_stop():
    """A connector on the prefiller that caps its hit at the prompt (as
    offloading does) can report one reaching the stop; the scheduler trims it
    so the prefill still computes a token."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = 2 * BLOCK_SIZE + 1
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )
    scheduler.add_request(request)
    assert request.prefill_stop == NUM_TOKENS - 1

    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        return_value=(NUM_TOKENS - 1, False),
    ):
        scheduler_output = scheduler.schedule()
    assert scheduler_output.num_scheduled_tokens[request.request_id] == 1
    assert request.num_computed_tokens == request.prefill_stop


def test_abort_during_kv_transfer():
    """Test aborting request does not release blocks for remote decode."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Prime the KVCache.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    # Request removed from PB but blocks should not be freed.
    assert len(scheduler.requests) == 1

    # Abort the request, and check the blocks are still not freed
    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
    assert len(scheduler.requests) == 1

    # Simulate a finished sending notification
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending=[request.request_id]
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


@pytest.mark.parametrize("lookahead", [0, 1, 2, 3])
@pytest.mark.parametrize("parameter", ["prompt_logprobs", "prompt_logprob_token_ids"])
def test_prompt_logprobs_rejected_when_prefill_stops_further_short(
    lookahead, parameter
):
    """A P/D prefill stops short of the drafter's lookahead window less one
    (multi-module MTP); past one token it can't compute the last prompt
    logprobs, so the frontend rejects such requests rather than let the
    prefiller and decoder disagree on where the KV ends."""
    from types import SimpleNamespace

    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.input_processor import InputProcessor

    processor = SimpleNamespace(
        vllm_config=SimpleNamespace(num_prefill_lookahead_tokens=lookahead)
    )

    def validate(params: SamplingParams) -> None:
        InputProcessor._validate_kv_transfer_params(processor, params)

    def make_params(kv_transfer_params) -> SamplingParams:
        params = SamplingParams(extra_args={"kv_transfer_params": kv_transfer_params})
        setattr(params, parameter, 1 if parameter == "prompt_logprobs" else [[1]])
        return params

    prefill = make_params({"do_remote_decode": True})
    if lookahead > 2:
        with pytest.raises(VLLMValidationError, match=parameter):
            validate(prefill)
    else:
        validate(prefill)
    # Only the prefill leg of a P/D request is affected.
    validate(make_params({"do_remote_prefill": True}))
    validate(make_params(None))
