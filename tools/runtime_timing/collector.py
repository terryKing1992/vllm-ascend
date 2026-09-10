"""Independent loopback UDP collector. Its lifecycle never controls model processes."""

import argparse
import importlib.metadata
import math
import os
import re
import signal
import socket
import sys

from trace_export import BufferedExporter, CollectorConfig, JsonLogSink, LangfuseSink
from trace_summary import RequestSummarySink, SummaryConfig
from trace_transport import DEFAULT_DIAGNOSTIC_EVERY, MAX_DATAGRAM_BYTES, decode_packet, diagnostic_due


class Collector:
    def __init__(self, port, exporter, diagnostic_log=False, diagnostic_every=DEFAULT_DIAGNOSTIC_EVERY):
        self.exporter = exporter
        self.diagnostic_log = diagnostic_log
        if diagnostic_every < 1:
            raise ValueError("diagnostic_every must be positive")
        self.diagnostic_every = diagnostic_every
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", port))
        self.socket.settimeout(0.5)
        self.stopped = False
        self.invalid = 0
        self.received = 0

    def log(self, message):
        if self.diagnostic_log:
            print(f"[timing-recv] {message}", file=sys.stderr, flush=True)

    def run(self):
        try:
            while not self.stopped:
                try:
                    data, _ = self.socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
                except TimeoutError:
                    continue
                except OSError as error:
                    self.invalid += 1
                    if self.diagnostic_log and diagnostic_due(self.invalid, self.diagnostic_every):
                        self.log(f"socket_error={type(error).__name__} invalid={self.invalid}")
                    continue
                try:
                    packet = decode_packet(data)
                    self.exporter.submit(packet)
                    self.received += 1
                    if self.diagnostic_log and diagnostic_due(self.received, self.diagnostic_every):
                        names = ",".join(record.name for record in packet.records)
                        self.log(
                            f"received packet={self.received} invalid={self.invalid} bytes={len(data)} "
                            f"source_pid={packet.metadata.get('source_pid')} requests={len(packet.contexts)} "
                            f"records={len(packet.records)} names={names}"
                        )
                except Exception as error:
                    self.invalid += 1
                    if self.diagnostic_log and diagnostic_due(self.invalid, self.diagnostic_every):
                        self.log(f"invalid error={type(error).__name__} bytes={len(data)} total={self.invalid}")
        finally:
            self.socket.close()
            self.exporter.close()

    def stop(self, *_):
        self.stopped = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", choices=("log", "langfuse"), default="log")
    parser.add_argument(
        "--report",
        choices=("request", "spans"),
        default="request",
        help="request: one summary per HTTP request; spans: export every observed stage",
    )
    parser.add_argument(
        "--summary-grace",
        type=float,
        default=SummaryConfig.settle_seconds,
        help="seconds to collect late stage packets after HTTP completion",
    )
    parser.add_argument(
        "--summary-ttl",
        type=float,
        default=SummaryConfig.ttl_seconds,
        help="maximum seconds retained for an unfinished summary",
    )
    parser.add_argument(
        "--summary-max-requests",
        type=int,
        default=SummaryConfig.max_requests,
        help="maximum retained summaries and recent-completion markers",
    )
    parser.add_argument(
        "--log-format",
        choices=("compact", "full"),
        default="compact",
        help="log output only: compact keeps trace IDs, duration and summary/batch metadata",
    )
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--queue-size", type=int, default=256)
    parser.add_argument("--flush-at", type=int, default=256)
    parser.add_argument("--flush-interval", type=float, default=2)
    parser.add_argument("--service-name", default=CollectorConfig.service_name, help="OpenTelemetry service.name")
    parser.add_argument("--environment", help="Langfuse deployment environment, e.g. production")
    parser.add_argument("--release", help="Langfuse release and OpenTelemetry service.version")
    parser.add_argument("--diagnostic-log", action="store_true", help="print received packet summaries to stderr")
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=DEFAULT_DIAGNOSTIC_EVERY,
        help="log first/every N packets and errors; 1 restores per-packet diagnostics",
    )
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.queue_size < 1 or not 1 <= args.flush_at <= 2048:
        parser.error("invalid port or queue/batch limits")
    if not math.isfinite(args.flush_interval) or args.flush_interval <= 0:
        parser.error("flush-interval must be positive and finite")
    if args.diagnostic_every < 1:
        parser.error("diagnostic-every must be positive")
    if not args.service_name.strip():
        parser.error("service-name must not be empty")
    if args.environment is not None and not re.fullmatch(r"(?!langfuse)[a-z0-9_-]+", args.environment):
        parser.error(
            "environment must use lowercase letters, digits, hyphens or underscores and not start with langfuse"
        )
    try:
        summary_config = SummaryConfig(args.summary_grace, args.summary_ttl, args.summary_max_requests)
    except ValueError as error:
        parser.error(str(error))
    if args.output == "langfuse":
        try:
            if importlib.metadata.version("langfuse").split(".")[0] != "3":
                parser.error("collector requires Langfuse SDK v3")
        except importlib.metadata.PackageNotFoundError:
            parser.error("install requirements.txt in the collector environment")
        if not all(os.environ.get(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
            parser.error("configure Langfuse credentials in the collector environment only")
        if not (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")):
            parser.error("set LANGFUSE_BASE_URL to your server")
    output_factory = JsonLogSink if args.output == "log" else LangfuseSink

    def sink_factory(config):
        sink = output_factory(config)
        return RequestSummarySink(summary_config, sink) if args.report == "request" else sink

    exporter = BufferedExporter(
        CollectorConfig(
            queue_size=args.queue_size,
            flush_at=args.flush_at,
            flush_interval=args.flush_interval,
            log_format=args.log_format,
            service_name=args.service_name,
            environment=args.environment,
            release=args.release,
        ),
        sink_factory,
    )
    collector = Collector(args.port, exporter, args.diagnostic_log, args.diagnostic_every)
    signal.signal(signal.SIGTERM, collector.stop)
    signal.signal(signal.SIGINT, collector.stop)
    print(
        f"Collector listening on 127.0.0.1:{args.port}, output={args.output}, report={args.report}",
        file=sys.stderr,
        flush=True,
    )
    collector.run()
    print(f"received={collector.received} invalid={collector.invalid} dropped={exporter.dropped}", file=sys.stderr)


if __name__ == "__main__":
    main()
