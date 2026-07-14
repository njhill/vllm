# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E501
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
)
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

# Importing utils registers TestExampleConnector with the factory
from .utils import create_vllm_config


def _make_empty_scheduler_output():
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=ExampleConnectorMetadata(),
    )


def _initialize_connector(vllm_config):
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )
    # Initialize the global connector instance.
    # kv_transfer init now syncs engine_id across TP, so unit tests need
    # a minimal mocked TP group.
    mock_tp_group = MagicMock()
    mock_tp_group.broadcast_object.side_effect = lambda value, src=0: value

    with patch(
        "vllm.distributed.parallel_state.get_tp_group",
        return_value=mock_tp_group,
    ):
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)


def test_kv_connector_mixin_clears_metadata():
    vllm_config = create_vllm_config(
        kv_connector="TestExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"name": "unit"},
    )
    _initialize_connector(vllm_config)

    try:
        # Minimal scheduler output with empty metadata; mixin should still
        # bind/clear metadata even if no loads happen
        scheduler_output = _make_empty_scheduler_output()

        # Invoke the no-forward path which uses the mixin context manager
        KVConnectorModelRunnerMixin().kv_connector_no_forward(
            scheduler_output, vllm_config
        )

        # Verify clear_connector_metadata was called on the connector
        connector = get_kv_transfer_group()
        assert connector._connector_metadata is None
        # Test connector wrapper records method calls
        assert connector.call_record.get("bind_connector_metadata", 0) == 1
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        # Ensure we clean up the global connector between tests
        ensure_kv_transfer_shutdown()


def test_async_loads_start_after_forward():
    """On steps without sync loads, the mixin starts the connector's loads
    after the forward body, not before; sync loads feed the forward so
    start pre-forward."""
    vllm_config = create_vllm_config(
        kv_connector="TestExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"name": "unit"},
    )
    _initialize_connector(vllm_config)

    try:
        connector = get_kv_transfer_group()
        mixin = KVConnectorModelRunnerMixin()
        with (
            set_forward_context(None, vllm_config),
            mixin._get_kv_connector_output(_make_empty_scheduler_output()),
        ):
            # Within the forward body, loads have not been started.
            assert connector.call_record.get("start_load_kv", 0) == 0
        assert connector.call_record.get("start_load_kv", 0) == 1

        # With sync loads, loads start before the forward.
        scheduler_output = _make_empty_scheduler_output()
        scheduler_output.has_sync_kv_loads = True
        with (
            set_forward_context(None, vllm_config),
            mixin._get_kv_connector_output(scheduler_output),
        ):
            assert connector.call_record.get("start_load_kv", 0) == 2
    finally:
        ensure_kv_transfer_shutdown()
