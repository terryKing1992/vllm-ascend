"""Standard-library-only, best-effort local transport. No SDK, threads or retries."""

import json
import os
import socket
from dataclasses import asdict, dataclass, field

MAX_DATAGRAM_BYTES = 8192
MAX_RECORDS = 32
MAX_REQUESTS = 8


@dataclass
class Record:
    name: str
    start_ns: int
    end_ns: int = 0
    parent: int | None = None
    span_id: str | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Packet:
    contexts: tuple
    records: list[Record]
    metadata: dict = field(default_factory=dict)


class DatagramEmitter:
    def __init__(self, port):
        self.address = ("127.0.0.1", port)
        self.socket = None
        self.pid = os.getpid()
        self.dropped = 0
        self.sent = 0

    def submit(self, packet):
        if not packet.contexts:
            return
        # Only built-in values go over the wire, never pickle/custom class names.
        body = asdict(packet)
        body["metadata"] = dict(body["metadata"], source_pid=os.getpid())
        data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(data) > MAX_DATAGRAM_BYTES:
            self.dropped += 1
            return
        if self.pid != os.getpid():
            if self.socket is not None:
                self.socket.close()
            self.socket = None
            self.pid = os.getpid()
        if self.socket is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setblocking(False)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, MAX_DATAGRAM_BYTES)
            except Exception:
                sock.close()
                raise
            self.socket = sock
        try:
            self.socket.sendto(data, self.address)
            self.sent += 1  # Accepted by local kernel, not a delivery acknowledgement.
        except OSError:
            self.dropped += 1


def decode_packet(data):
    if len(data) > MAX_DATAGRAM_BYTES:
        raise ValueError("oversize datagram")
    body = json.loads(data)
    contexts, records = body["contexts"], body["records"]
    if not 0 < len(contexts) <= MAX_REQUESTS or not 0 < len(records) <= MAX_RECORDS:
        raise ValueError("invalid packet limits")
    for context in contexts:
        trace_id = context["trace_id"]
        if len(trace_id) != 32 or int(trace_id, 16) == 0:
            raise ValueError("invalid trace ID")
        parent = context.get("parent_span_id")
        if parent is not None and (len(parent) != 16 or int(parent, 16) == 0):
            raise ValueError("invalid parent ID")
    parsed = []
    for index, raw in enumerate(records):
        record = Record(**raw)
        if not isinstance(record.name, str) or len(record.name) > 256:
            raise ValueError("invalid span name")
        if not isinstance(record.start_ns, int) or not isinstance(record.end_ns, int):
            raise ValueError("invalid timestamps")
        if record.start_ns <= 0 or record.end_ns < record.start_ns:
            raise ValueError("invalid time range")
        if record.parent is not None and not 0 <= record.parent < index:
            raise ValueError("invalid parent index")
        if record.span_id is not None and (len(record.span_id) != 16 or int(record.span_id, 16) == 0):
            raise ValueError("invalid span ID")
        parsed.append(record)
    return Packet(tuple(contexts), parsed, body.get("metadata", {}))
