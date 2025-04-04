from collections import defaultdict

import msgspec.msgpack
import zmq
from zmq import Frame

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType
from vllm.utils import make_zmq_socket, get_open_zmq_ipc_path

NONE_TUPLE = (None, None)

class Router:

    def __init__(self):

        self.ctx = zmq.Context()

        self.api_count = 4

        input_path = get_open_zmq_ipc_path()
        output_path = get_open_zmq_ipc_path()

        self.input_front = make_zmq_socket(
            path=input_path,
            ctx=self.ctx,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self.input_back = make_zmq_socket(
            path=input_path,
            ctx=self.ctx,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self.output_front = make_zmq_socket(
            path=output_path,
            ctx=self.ctx,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self.output_back = make_zmq_socket(
            path=output_path,
            ctx=self.ctx,
            socket_type=zmq.PULL,
            bind=True,
        )

        self.start_dp_msg = (EngineCoreRequestType.START_DP.value,
                             msgspec.msgpack.encode(None))

        # call_id -> api_id
        self.utility_pending: dict[int, bytes] = {}

        # req_id -> (api_id, eng_id)
        self.reqs_in_flight: dict[str, tuple[bytes, bytes]] = {}

        self.api_ids = [i.to_bytes(length=2, byteorder="little")
                        for i in range(self.api_count)]

        self.num_engines_running = 0

        num_engines = 4

        self.eng_ids = [(i.to_bytes(2, byteorder="little"), [0, 0])
                        for i in range(num_engines)]

    def process_input_socket(self, input_socket: zmq.Socket, inproc_path: str):
        """Input socket IO thread."""

        inproc_socket = self.ctx.socket(zmq.PAIR)

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()
        encoder = MsgpackEncoder()

        inproc_socket.bind(inproc_path)
        poller = zmq.Poller()
        poller.register(inproc_socket)
        poller.register(input_socket)
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
                        self.input_back.send_multipart(
                            (eng_id,) + self.start_dp_msg, copy=False)
                continue

            api_frame, type_frame, data_frame = input_socket.recv_multipart(
                copy=False)

            request_type = EngineCoreRequestType(bytes(type_frame.buffer))
            api_id = api_frame.buffer

            if request_type == EngineCoreRequestType.ADD:
                request = add_request_decoder.decode(data_frame.buffer)
                eng_id = self.get_eng_id_for_request()
                self.reqs_in_flight[request.request_id] = (api_id, eng_id)
                self.input_back.send_multipart(
                    (eng_id, type_frame, data_frame), copy=False)

            elif request_type == EngineCoreRequestType.UTILITY:
                call_id, _, _ = generic_decoder.decode(data_frame.buffer)
                self.utility_pending[call_id] = api_id
                for eng_id in self.eng_ids:
                    self.input_back.send_multipart(
                        (eng_id, type_frame, data_frame),copy=False)

            elif request_type == EngineCoreRequestType.ABORT:
                self.handle_abort_request(type_frame, data_frame,
                                          generic_decoder, encoder)

    def get_eng_id_for_request(self) -> bytes:
        return min(self.eng_ids, key=lambda e: e[1])[0]

    def handle_abort_request(self,
            type_frame: Frame, data_frame: Frame,
            decoder: MsgpackDecoder, encoder: MsgpackEncoder):
        request_ids = decoder.decode(data_frame)
        if len(request_ids) == 1:
            # Fast-path common case.
            _, eng_id = self.reqs_in_flight.get(request_ids[0], NONE_TUPLE)
            if eng_id is not None:
                self.input_back.send_multipart(
                    (eng_id, type_frame, data_frame), copy=False)
            return

        by_engine: dict[bytes, list[str]] = defaultdict(lambda: [])
        for req_id in request_ids:
            _, eng_id = self.reqs_in_flight.get(req_id, NONE_TUPLE)
            if eng_id is not None:
                by_engine[eng_id].append(req_id)

        for eng_id, req_ids in by_engine.items():
            encoded = encoder.encode(req_ids)
            self.input_back.send_multipart(
                (eng_id, type_frame, encoded), copy=False)


    def process_outputs_socket(self, inproc_path: str):
        decoder = MsgpackDecoder(EngineCoreOutputs)
        encoder = MsgpackEncoder()
        out_socket = self.output_back

        inproc_socket = self.ctx.socket(zmq.PAIR)
        inproc_socket.bind(inproc_path)

        try:
            while True:
                api_outs = defaultdict(lambda: [])
                frame = out_socket.recv(copy=False)
                outputs: EngineCoreOutputs = decoder.decode(frame.buffer)

                if outputs.utility_output:
                    call_id = outputs.utility_output.call_id
                    api_id = self.utility_pending.pop(call_id, None)
                    if api_id is not None:
                        self.output_front.send_multipart(
                            (api_id, frame), copy=False)
                    continue

                if outputs.scheduler_stats:
                    _, stats = self.eng_ids[outputs.engine_index]
                    stats[0] = outputs.scheduler_stats.num_waiting_reqs
                    stats[1] = outputs.scheduler_stats.num_running_reqs

                for out in outputs.outputs:
                    api_index, _ = self.reqs_in_flight.get(
                        out.request_id, NONE_TUPLE)
                    if api_index is not None:
                        api_outs.get(api_index).append(out)

                for req_id in outputs.finished_requests or ():
                    self.reqs_in_flight.pop(req_id, None)
                    # engine.num_reqs_in_flight -= 1

                if outputs.engine_paused:
                    # Notify input thread
                    inproc_socket.send(b'')  # TODO send wave number

                while api_outs:
                    api_index, out_list = api_outs.popitem()
                    new_outputs = EngineCoreOutputs(
                        outputs.engine_index, out_list)
                    # TODO reuse buffer
                    self.output_front.send_multipart(
                        (self.api_ids[api_index],
                         encoder.encode(new_outputs)),
                        copy=False)
        finally:
            # Close socket.
            out_socket.close(linger=0)