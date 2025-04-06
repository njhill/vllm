# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict

import msgspec.msgpack
import zmq
from zmq import Frame

from vllm.utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

NONE_TUPLE = (None, None)


class Router:

    def __init__(self):

        self.ctx = zmq.Context()

        api_count = 4
        engine_count = 4

        # front_input_addr = get_open_zmq_ipc_path()
        # front_output_addr = get_open_zmq_ipc_path()

        self.start_dp_msg = (EngineCoreRequestType.START_DP.value,
                             msgspec.msgpack.encode(None))

        self.api_ids = [
            i.to_bytes(length=2, byteorder="little") for i in range(api_count)
        ]
        self.eng_ids = [(i.to_bytes(2, byteorder="little"), [0, 0])
                        for i in range(engine_count)]

        # call_id -> api_id
        self.utility_pending: dict[int, bytes] = {}

        # req_id -> (api_id, eng_id)
        self.reqs_in_flight: dict[str, tuple[bytes, bytes]] = {}

        self.num_engines_running = 0

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

            inproc_socket.bind(inproc_path)
            poller = zmq.Poller()
            poller.register(inproc_socket)
            poller.register(input_front)
            while True:
                socks = poller.poll()
                if len(socks) == 2 or socks[0][0] == inproc_socket:
                    wave = inproc_socket.recv()
                    self.num_engines_running -= 1
                    if not self.num_engines_running and self.reqs_in_flight:
                        # If there are requests in flight here, they must have
                        # been sent after the engines paused. We must make
                        # sure to start the other engines:
                        self.num_engines_running = len(self.eng_ids)
                        for eng_id in self.eng_ids:
                            input_back.send_multipart(
                                (eng_id, ) + self.start_dp_msg, copy=False)
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

                elif request_type == EngineCoreRequestType.UTILITY:
                    call_id, _, _ = generic_decoder.decode(data_frame.buffer)
                    self.utility_pending[call_id] = api_id
                    for eng_id in self.eng_ids:
                        input_back.send_multipart(
                            (eng_id, type_frame, data_frame), copy=False)

                elif request_type == EngineCoreRequestType.ABORT:
                    self.handle_abort_request(input_back, type_frame,
                                              data_frame, generic_decoder,
                                              encoder)

    def get_eng_id_for_request(self) -> bytes:
        return min(self.eng_ids, key=lambda e: e[1])[0]

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
            api_outs = defaultdict(lambda: [])
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
                    _, stats = self.eng_ids[eng_index]
                    stats[0] = outputs.scheduler_stats.num_waiting_reqs
                    stats[1] = outputs.scheduler_stats.num_running_reqs

                for out in outputs.outputs:
                    api_index, _ = self.reqs_in_flight.get(
                        out.request_id, NONE_TUPLE)
                    if api_index is not None:
                        api_outs.get(api_index).append(out)

                for req_id in outputs.finished_requests or ():
                    self.reqs_in_flight.pop(req_id, None)

                if outputs.engine_paused:
                    # Notify input thread
                    inproc_socket.send(b'')  # TODO send wave number

                while api_outs:
                    api_index, out_list = api_outs.popitem()
                    new_outputs = EngineCoreOutputs(eng_index, out_list)
                    encoded = encoder.encode(new_outputs)
                    api_id = self.api_ids[api_index]
                    # TODO reuse buffer
                    output_front.send_multipart((api_id, encoded), copy=False)
