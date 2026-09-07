"""Independent loopback UDP collector. Its lifecycle never controls model processes."""

import argparse
import importlib.metadata
import math
import os
import signal
import socket

from trace_export import BufferedExporter, CollectorConfig
from trace_transport import MAX_DATAGRAM_BYTES, decode_packet


class Collector:
    def __init__(self, port, exporter):
        self.exporter = exporter
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", port))
        self.socket.settimeout(0.5)
        self.stopped = False
        self.invalid = 0
        self.received = 0

    def run(self):
        try:
            while not self.stopped:
                try:
                    data, _ = self.socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
                except TimeoutError:
                    continue
                except OSError:
                    self.invalid += 1
                    continue
                try:
                    packet = decode_packet(data)
                    self.exporter.submit(packet)
                    self.received += 1
                except Exception:
                    self.invalid += 1
        finally:
            self.socket.close()
            self.exporter.close()

    def stop(self, *_):
        self.stopped = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--queue-size", type=int, default=256)
    parser.add_argument("--flush-at", type=int, default=256)
    parser.add_argument("--flush-interval", type=float, default=2)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.queue_size < 1 or not 1 <= args.flush_at <= 2048:
        parser.error("invalid port or queue/batch limits")
    if not math.isfinite(args.flush_interval) or args.flush_interval <= 0:
        parser.error("flush-interval must be positive and finite")
    try:
        if importlib.metadata.version("langfuse").split(".")[0] != "3":
            parser.error("collector requires Langfuse SDK v3")
    except importlib.metadata.PackageNotFoundError:
        parser.error("install requirements.txt in the collector environment")
    if not all(os.environ.get(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
        parser.error("configure Langfuse credentials in the collector environment only")
    if not (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")):
        parser.error("set LANGFUSE_BASE_URL to your server")
    exporter = BufferedExporter(CollectorConfig(args.queue_size, args.flush_at, args.flush_interval))
    collector = Collector(args.port, exporter)
    signal.signal(signal.SIGTERM, collector.stop)
    signal.signal(signal.SIGINT, collector.stop)
    print(f"Collector listening on 127.0.0.1:{args.port}", flush=True)
    collector.run()
    print(f"received={collector.received} invalid={collector.invalid} dropped={exporter.dropped}")


if __name__ == "__main__":
    main()
