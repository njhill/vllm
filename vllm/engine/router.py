# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict

import msgspec.msgpack
import zmq
from zmq import Frame

from config import ParallelConfig
from vllm.v1.utils import CoreEngine, wait_for_engine_startup, get_engine_zmq_addresses
from vllm.utils import make_zmq_socket
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

NONE_TUPLE = (None, None)


class EngineRouter:

    def __init__(self,
                 parallel_config: ParallelConfig,
                 api_server_count: int):

        self.back_input_address, self.back_output_address = (
            get_engine_zmq_addresses(parallel_config, False))

    def get_engine_zmq_addresses(self):
        return self.back_input_address, self.back_output_address


    def run_engine_router_proc(self):



class RouterProc:

    def __init__(self,
                 parallel_config: ParallelConfig,
                 api_server_count: int):

        self.ctx = zmq.Context()
        self.parallel_config = parallel_config

        engine_count = parallel_config.data_parallel_size
        local_engine_count = parallel_config.data_parallel_size_local

        self.output_address = ""  #TODO

        # front_input_addr = get_open_zmq_ipc_path()
        # front_output_addr = get_open_zmq_ipc_path()

        back_input_address, back_output_address = get_engine_zmq_addresses(
            parallel_config, False)

        self.api_ids = [
            i.to_bytes(length=2, byteorder="little")
            for i in range(api_server_count)
        ]

        self.engines = [
            CoreEngine(index=i, local=(i < local_engine_count))
            for i in range(engine_count)
        ]

        # call_id -> api_id
        self.utility_pending: dict[int, bytes] = {}

        # req_id -> (api_id, eng_id)
        self.reqs_in_flight: dict[str, tuple[bytes, bytes]] = {}

        self.current_wave = 0
        self.engines_running = False




    def process_input_socket(self, front_address: str, back_address: str,
                             inproc_path: str):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()
        encoder = MsgpackEncoder()

        with make_zmq_socket(
            path=front_address,
            ctx=self.ctx,
            socket_type=zmq.ROUTER,
            bind=True,
        ) as input_front, make_zmq_socket(
            path=back_address,
            ctx=self.ctx,
            socket_type=zmq.ROUTER,
            bind=True,
        ) as input_back, self.ctx.socket(zmq.PAIR) as inproc_socket:

            wait_for_engine_startup(input_socket=input_back,
                                    output_address=self.output_address,
                                    core_engines=self.engines,
                                    parallel_config=self.parallel_config,
                                    proc_manager=None) #TODO

            # TODO signal ready here

            inproc_socket.bind(inproc_path)
            poller = zmq.Poller()
            poller.register(inproc_socket)
            poller.register(input_front)
            while True:
                socks = poller.poll()
                if len(socks) == 2 or socks[0][0] == inproc_socket:
                    data = inproc_socket.recv()
                    wave = int.from_bytes(data[1:5], byteorder="little")
                    if data[0] == b'c':
                        if self.current_wave <= wave:
                            self.current_wave = wave + 1
                            self.engines_running = False
                    elif wave is not None and (wave > self.current_wave or
                                               (wave == self.current_wave
                                                and not self.engines_running)):
                        # Engine received request for a non-current wave so
                        # we must ensure that other engines progress to the
                        # next wave.
                        self.current_wave = wave
                        self.engines_running = True
                        from_eng_id = data[5:7]
                        self._send_start_wave(input_back, wave, from_eng_id)
                    continue

                api_frame, type_frame, data_frame = input_front.recv_multipart(
                    copy=False)

                request_type = EngineCoreRequestType(bytes(type_frame.buffer))
                api_id = api_frame.buffer

                if request_type == EngineCoreRequestType.ADD:
                    request = add_request_decoder.decode(data_frame.buffer)
                    eng_id = self.get_eng_id_for_request()
                    self.reqs_in_flight[request.request_id] = (api_id, eng_id)
                    input_back.send_multipart((eng_id, type_frame, data_frame),
                                              copy=False)
                    if not self.engines_running:
                        # Send dp start loop control message to all other
                        # engines.
                        self.engines_running = True
                        self._send_start_wave(input_back, self.current_wave,
                                              eng_id)

                elif request_type == EngineCoreRequestType.UTILITY:
                    call_id, _, _ = generic_decoder.decode(data_frame.buffer)
                    self.utility_pending[call_id] = api_id
                    for eng in self.engines:
                        input_back.send_multipart(
                            (eng.identity, type_frame, data_frame), copy=False)

                elif request_type == EngineCoreRequestType.ABORT:
                    self.handle_abort_request(input_back, type_frame,
                                              data_frame, generic_decoder,
                                              encoder)

    def get_eng_id_for_request(self) -> bytes:
        return min(self.engines, key=lambda e: e.request_counts[1]).identity

    def handle_abort_request(self, input_back: zmq.Socket, type_frame: Frame,
                             data_frame: Frame, decoder: MsgpackDecoder,
                             encoder: MsgpackEncoder):
        request_ids = decoder.decode(data_frame)
        if len(request_ids) == 1:
            # Fast-path common case.
            _, eng_id = self.reqs_in_flight.get(request_ids[0], NONE_TUPLE)
            if eng_id is not None:
                input_back.send_multipart((eng_id, type_frame, data_frame),
                                          copy=False)
            return

        by_engine: dict[bytes, list[str]] = defaultdict(lambda: [])
        for req_id in request_ids:
            _, eng_id = self.reqs_in_flight.get(req_id, NONE_TUPLE)
            if eng_id is not None:
                by_engine[eng_id].append(req_id)

        for eng_id, req_ids in by_engine.items():
            encoded = encoder.encode(req_ids)
            input_back.send_multipart((eng_id, type_frame, encoded),
                                      copy=False)

    def _send_start_wave(self, input_back: zmq.Socket, wave: int,
                         excude_eng_id: bytes):
        wave_encoded = msgspec.msgpack.encode(wave)
        for eng in self.engines:
            if eng.identity != excude_eng_id:
                input_back.send_multipart(
                    (eng.identity, EngineCoreRequestType.START_DP_WAVE.value,
                     wave_encoded),
                    copy=False)

    def process_outputs_socket(self, front_address: str, back_address: str,
                               inproc_path: str):
        decoder = MsgpackDecoder(EngineCoreOutputs)
        encoder = MsgpackEncoder()

        with make_zmq_socket(
            path=front_address,
            ctx=self.ctx,
            socket_type=zmq.ROUTER,
            bind=True,
        ) as output_front, make_zmq_socket(
            path=back_address,
            ctx=self.ctx,
            socket_type=zmq.PULL,
            bind=True,
        ) as output_back, self.ctx.socket(zmq.PAIR) as inproc_socket:

            inproc_socket.connect(inproc_path)
            api_outs = defaultdict(list)
            while True:
                frame = output_back.recv(copy=False)
                outputs: EngineCoreOutputs = decoder.decode(frame.buffer)

                if outputs.utility_output:
                    call_id = outputs.utility_output.call_id
                    api_id = self.utility_pending.pop(call_id, None)
                    if api_id is not None:
                        output_front.send_multipart((api_id, frame),
                                                    copy=False)
                    continue

                eng_index = outputs.engine_index
                if outputs.scheduler_stats:
                    stats = self.engines[eng_index].request_counts
                    stats[0] = outputs.scheduler_stats.num_waiting_reqs
                    stats[1] = outputs.scheduler_stats.num_running_reqs

                for out in outputs.outputs:
                    api_index, _ = self.reqs_in_flight.get(
                        out.request_id, NONE_TUPLE)
                    if api_index is not None:
                        api_outs.get(api_index).append(out)

                for req_id in outputs.finished_requests or ():
                    self.reqs_in_flight.pop(req_id, None)

                if outputs.wave_complete is not None or (outputs.start_wave
                                                         is not None):
                    # Notify input thread.
                    prefix = b'c' if outputs.wave_complete is not None else b's'
                    eng_id = self.engines[eng_index].identity
                    wave = outputs.wave_complete.to_bytes(length=8,
                                                          byteorder="little")
                    inproc_socket.send(prefix + wave + eng_id, copy=False)

                while api_outs:
                    api_index, out_list = api_outs.popitem()
                    new_outputs = EngineCoreOutputs(eng_index, out_list)
                    encoded = encoder.encode(new_outputs)
                    api_id = self.api_ids[api_index]
                    # TODO reuse buffer
                    output_front.send_multipart((api_id, encoded), copy=False)
