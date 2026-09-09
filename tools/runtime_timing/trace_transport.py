"""Standard-library-only, best-effort local transport. No SDK, threads or retries."""

import json
import os
import socket
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass, field

MAX_DATAGRAM_BYTES = 8192
MAX_RECORDS = 32
MAX_REQUESTS = 8
DEFAULT_DIAGNOSTIC_EVERY = 1000


def diagnostic_due(count, every):
    return count == 1 or count % every == 0


@dataclass
class Record:
    # Defaults keep these transport-only objects acceptable to torch.compile's
    # dataclass handling when a wrapped runner method is captured.
    name: str = ""
    start_ns: int = 0
    end_ns: int = 0
    parent: int | None = None
    span_id: str | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Packet:
    contexts: tuple = ()
    records: list[Record] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DatagramEmitter:
    def __init__(self, port, diagnostic_log=False, diagnostic_every=DEFAULT_DIAGNOSTIC_EVERY):
        self.address = ("127.0.0.1", port)
        self.diagnostic_log = diagnostic_log
        if diagnostic_every < 1:
            raise ValueError("diagnostic_every must be positive")
        self.diagnostic_every = diagnostic_every
        self.socket = None
        self.pid = os.getpid()
        self.dropped = 0
        self.sent = 0

    def _log(self, message):
        if self.diagnostic_log:
            with suppress(Exception):
                print(f"[timing-send] {message}", file=sys.stderr, flush=True)

    def submit(self, packet):
        if not packet.contexts:
            return
        # Only built-in values go over the wire, never pickle/custom class names.
        body = asdict(packet)
        body["metadata"] = dict(body["metadata"], source_pid=os.getpid())
        data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(data) > MAX_DATAGRAM_BYTES:
            self.dropped += 1
            if self.diagnostic_log and diagnostic_due(self.dropped, self.diagnostic_every):
                self._log(f"dropped=oversize total={self.dropped} bytes={len(data)} limit={MAX_DATAGRAM_BYTES}")
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
            if self.diagnostic_log and diagnostic_due(self.sent, self.diagnostic_every):
                names = ",".join(record.name for record in packet.records)
                self._log(
                    f"sent packet={self.sent} dropped={self.dropped} bytes={len(data)} port={self.address[1]} "
                    f"requests={len(packet.contexts)} records={len(packet.records)} names={names}"
                )
        except OSError as error:
            self.dropped += 1
            if self.diagnostic_log and diagnostic_due(self.dropped, self.diagnostic_every):
                self._log(f"dropped=socket total={self.dropped} error={type(error).__name__} port={self.address[1]}")


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
