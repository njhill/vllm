# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.v1.worker.utils import (
    KVBlockZeroer,
    KVConnectorLoadGate,
    _zero_kv_blocks_kernel,
)


def _make_zeroer(
    storages: list[torch.Tensor],
    seg_page_sizes: list[int],
    has_external_block_writers: bool = False,
) -> KVBlockZeroer:
    # Build the minimal zeroer state directly so tests can focus on behavior
    # without constructing model attention groups.
    device = storages[0].device
    blk_size = min(min(largest_power_of_2_divisor(ps) for ps in seg_page_sizes), 1024)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer.has_external_block_writers = has_external_block_writers
    zeroer._meta = (
        torch.tensor(
            [s.data_ptr() for s in storages], dtype=torch.uint64, device=device
        ),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        max(seg_page_sizes) // blk_size,
        blk_size,
        len(storages),
    )
    zeroer._main_stream = torch.cuda.current_stream(device)
    zeroer._zero_stream = (
        torch.cuda.Stream(device=device) if has_external_block_writers else None
    )
    return zeroer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)
    zeroer = _make_zeroer([storage], [page_size_el])

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = _make_zeroer([storage_a, storage_b], [page_size_a, page_size_b])

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_every_n_blocks_specialization():
    """After warmup, no launch should trigger a first-request JIT compile.

    ``n_blocks`` is ``do_not_specialize``, so a single warmup launch must
    cover every block count.
    """
    device = torch.device("cuda")
    num_blocks = 64
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)
    zeroer = _make_zeroer([storage], [page_size_el])

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids(list(range(n_blocks)))
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)
    zeroer = _make_zeroer([storage], [page_size_el])

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zeroing_event_gates_external_writers():
    """With external block writers (consumer KV connector RDMA pulls), the
    zeroing runs on a dedicated stream and the returned event marks its
    completion, without waiting for compute-stream backlog. A writer gated
    on that event (as KVConnectorLoadGate defers the connector's loads) must
    not be wiped by the zeroing.
    """
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)
    zeroer = _make_zeroer([storage], [page_size_el], has_external_block_writers=True)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Backlog the ambient (compute) stream; the zeroing must not queue
        # behind it.
        torch.cuda._sleep(100_000_000)
        event = zeroer.zero_block_ids([1])

    assert event is not None
    # Gate on the event as the connector does before posting a READ. This
    # must complete without waiting for the ambient-stream backlog.
    event.synchronize()

    # Emulate the out-of-stream writer (RDMA landing) on the idle default
    # stream: if the zeroing were still queued behind the sleeping stream,
    # it would execute after this write and wipe it.
    storage[1].fill_(7)
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    # The external write must not have been wiped by the zeroing.
    assert torch.all(storage[1] == 7)
    assert torch.all(storage[2] == 1)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zeroing_events_are_not_reused():
    """Each zeroing must return a fresh event: KVConnectorLoadGate retains
    events across steps, and re-recording a shared event would make an
    earlier deferred step wait on later zeroing."""
    storage = torch.ones((4, 4), dtype=torch.int32, device="cuda")
    zeroer = _make_zeroer([storage], [4], has_external_block_writers=True)

    event1 = zeroer.zero_block_ids([1])
    event2 = zeroer.zero_block_ids([2])
    assert event1 is not event2
    event2.synchronize()
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)


class _FakeEvent:
    """Completion event on a FIFO stream: synchronizing it also implies any
    earlier events on the stream have fired."""

    def __init__(self, fired: bool, earlier: list["_FakeEvent"] | None = None):
        self.fired = fired
        self.synchronized = False
        self._earlier = earlier or []

    def query(self) -> bool:
        return self.fired

    def synchronize(self) -> None:
        self.synchronized = True
        self.fired = True
        for event in self._earlier:
            event.fired = True


class _FakeConnector:
    def __init__(self):
        self.bound = None
        # Metadata bound at the time of each start_load_kv call.
        self.started = []

    def bind_connector_metadata(self, metadata) -> None:
        self.bound = metadata

    def start_load_kv(self, _forward_context) -> None:
        self.started.append(self.bound)


def _step(metadata: str, has_sync_kv_loads: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=metadata, has_sync_kv_loads=has_sync_kv_loads
    )


@pytest.fixture
def load_gate(monkeypatch) -> KVConnectorLoadGate:
    monkeypatch.setattr("vllm.v1.worker.utils.get_forward_context", lambda: None)
    return KVConnectorLoadGate()


def _run_step(load_gate, connector, step) -> None:
    # As the model runners do: bind this step's metadata, then let the gate
    # start the loads.
    connector.bind_connector_metadata(step.kv_connector_metadata)
    load_gate.start_loads(connector, step)


def test_gate_starts_loads_when_no_zeroing_pending(load_gate):
    connector = _FakeConnector()
    _run_step(load_gate, connector, _step("m1"))
    load_gate.set_zeroing_event(_FakeEvent(fired=True))
    _run_step(load_gate, connector, _step("m2"))
    assert connector.started == ["m1", "m2"]


def test_gate_defers_loads_until_zeroing_completes(load_gate):
    """Loads must not start while zeroing of their target blocks is in
    flight (an out-of-stream RDMA write could be wiped), but the step's
    metadata must still be bound for the rest of the step (KV saves)."""
    connector = _FakeConnector()
    event = _FakeEvent(fired=False)
    load_gate.set_zeroing_event(event)
    _run_step(load_gate, connector, _step("m1"))
    assert connector.started == []
    assert connector.bound == "m1"

    # A subsequent step with no pending zeroing must not start its loads
    # ahead of the still-deferred earlier step.
    _run_step(load_gate, connector, _step("m2"))
    assert connector.started == []
    assert connector.bound == "m2"

    event.fired = True
    _run_step(load_gate, connector, _step("m3"))
    assert connector.started == ["m1", "m2", "m3"]


def test_gate_drains_completed_steps_independently(load_gate):
    """Each step's zeroing has its own event, so an earlier step whose
    zeroing has completed starts its loads even while a later step's
    zeroing is still in flight."""
    connector = _FakeConnector()
    event1 = _FakeEvent(fired=False)
    load_gate.set_zeroing_event(event1)
    _run_step(load_gate, connector, _step("m1"))
    event2 = _FakeEvent(fired=False)
    load_gate.set_zeroing_event(event2)
    _run_step(load_gate, connector, _step("m2"))
    assert connector.started == []

    event1.fired = True
    _run_step(load_gate, connector, _step("m3"))
    assert connector.started == ["m1"]
    assert connector.bound == "m3"

    event2.fired = True
    _run_step(load_gate, connector, _step("m4"))
    assert connector.started == ["m1", "m2", "m3", "m4"]


def test_gate_blocks_for_sync_loads(load_gate):
    """A step whose forward consumes synchronously loaded KV cannot defer:
    the gate must wait for the zeroing and start all pending loads."""
    connector = _FakeConnector()
    earlier_event = _FakeEvent(fired=False)
    load_gate.set_zeroing_event(earlier_event)
    _run_step(load_gate, connector, _step("m1"))
    assert connector.started == []

    sync_event = _FakeEvent(fired=False, earlier=[earlier_event])
    load_gate.set_zeroing_event(sync_event)
    _run_step(load_gate, connector, _step("m2", has_sync_kv_loads=True))
    assert sync_event.synchronized
    assert connector.started == ["m1", "m2"]
    assert connector.bound == "m2"
