# SPDX-License-Identifier: Apache-2.0
import sys
import time
from typing import Optional

import msgspec.msgpack
import zmq
from config import ParallelConfig

from vllm.utils import make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.utils import get_engine_zmq_addresses, wait_for_engine_startup

NONE_TUPLE = (None, None)


class EngineRouter:

    def __init__(self, parallel_config: ParallelConfig, api_server_count: int):

        self.back_input_address, self.back_output_address = (
            get_engine_zmq_addresses(parallel_config, False))

    def get_engine_zmq_addresses(self):
        return self.back_input_address, self.back_output_address

    def run_engine_router_proc(self):
        pass  # TODO wip


# Sockets:
#   - Publish for balancing (rcv for new requests)
#   -


class EngineState:

    def __init__(self):
        # waiting, running
        self.request_counts = [0, 0]


class RouterProc:

    def __init__(self, parallel_config: ParallelConfig):

        self.ctx = zmq.Context()
        self.parallel_config = parallel_config

        engine_count = parallel_config.data_parallel_size
        local_engine_count = parallel_config.data_parallel_size_local

        self.output_address = ""  #TODO

        # front_input_addr = get_open_zmq_ipc_path()
        # front_output_addr = get_open_zmq_ipc_path()

        back_input_address, back_output_address = get_engine_zmq_addresses(
            parallel_config, False)

        self.engines = [EngineState() for _ in range(engine_count)]

        self.current_wave = 0
        self.engines_running = False
        self.stats_changed = False

    def process_input_socket(self, front_address: str, back_address: str,
                             inproc_path: str):
        """Input socket IO thread."""

        front_publish_address, front_input_address = front_address
        back_publish_address, back_output_address = back_address

        decoder = MsgpackDecoder(EngineCoreOutputs)
        encoder = MsgpackEncoder()

        with make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
        ) as publish_front, make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
        ) as output_back, make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
        ) as publish_back:

            wait_for_engine_startup(input_socket=input_back,
                                    output_address=self.output_address,
                                    core_engines=self.engines,
                                    parallel_config=self.parallel_config,
                                    proc_manager=None)  #TODO

            # TODO signal ready here
            poller = zmq.Poller()
            poller.register(publish_front)
            poller.register(output_back)
            last_publish = 0
            while True:
                since = time.time() - last_publish
                wait_for = 200 if
                timeout = max(0, int((next_publish - time.time()) * 1000))
                events = poller.poll(timeout=timeout)
                if not events:
                    engine_list = self._get_engine_list()
                    to_publish = (engine_list, self.current_wave,
                                  self.engines_running)
                    publish_front.send(encoder.encode(to_publish))
                    next_publish += 0.2 if self.engines_running else 3
                    continue

                if publish_front in events:
                    buffer = publish_front.recv()
                    engine_index, wave = msgspec.msgpack.decode(buffer)
                    if wave < self.current_wave:
                        engine_index = None
                    if not self.engines_running:
                        self.engines_running = True
                        self._send_start_wave(publish_back, self.current_wave,
                                              engine_index)

                if output_back in events:
                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert outputs.outputs is None
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    if outputs.scheduler_stats:
                        stats = self.engines[eng_index].request_counts
                        stats[0] = outputs.scheduler_stats.num_waiting_reqs
                        stats[1] = outputs.scheduler_stats.num_running_reqs
                        self.stats_changed = True

                        #TODO record prometheus metrics here?

                    if outputs.wave_complete is not None:
                        if self.current_wave <= wave:
                            self.current_wave = wave + 1
                            self.engines_running = False
                    elif outputs.start_wave is not None and (
                            wave > self.current_wave or
                        (wave == self.current_wave
                         and not self.engines_running)):
                        # Engine received request for a non-current wave so
                        # we must ensure that other engines progress to the
                        # next wave.
                        self.current_wave = wave
                        self.engines_running = True
                        self._send_start_wave(publish_back, wave, eng_index)

    @staticmethod
    def _send_start_wave(socket: zmq.Socket, wave: int,
                         exclude_engine_index: Optional[int]):
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart(
            (EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    def _get_engine_list(self) -> Optional[list[int]]:
        shortlist = []
        min_counts = [sys.maxsize, sys.maxsize]
        for i, e in enumerate(self.engines):
            if e.request_counts <= min_counts:
                if e.request_counts < min_counts:
                    shortlist.clear()
                shortlist.append(i)
        return None if len(shortlist) == len(self.engines) else shortlist
