"""Collector-only Langfuse v3 export. Never imported by the model service."""

import atexit
import json
import os
import queue
import socket
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass


@dataclass(frozen=True)
class CollectorConfig:
    queue_size: int = 256
    flush_at: int = 256
    flush_interval: float = 2.0


class LangfuseSink:
    def __init__(self, config):
        # Lazy imports isolate SDK initialization in the export thread.
        from langfuse import Langfuse
        from opentelemetry import trace
        from opentelemetry.context import Context
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

        class ExplicitIds(RandomIdGenerator):
            trace_id = None
            span_id = None

            def generate_trace_id(self):
                return self.trace_id or super().generate_trace_id()

            def generate_span_id(self):
                return self.span_id or super().generate_span_id()

        self.trace = trace
        self.empty_context = Context
        self.ids = ExplicitIds()
        self.provider = TracerProvider(id_generator=self.ids)
        self.client = Langfuse(
            tracer_provider=self.provider,
            sample_rate=1.0,
            flush_at=config.flush_at,
            flush_interval=config.flush_interval,
            timeout=5,
        )
        self.tracer = self.provider.get_tracer("vllm-ascend-runtime")
        self.host = socket.gethostname()

    def emit(self, packet):
        for request in packet.contexts:
            span_ids = {}
            for index, record in enumerate(packet.records):
                parent = span_ids.get(record.parent, request.get("parent_span_id"))
                context = self.empty_context()
                self.ids.trace_id = int(request["trace_id"], 16)
                self.ids.span_id = int(record.span_id, 16) if record.span_id else None
                if parent:
                    parent_context = self.trace.SpanContext(
                        trace_id=self.ids.trace_id,
                        span_id=int(parent, 16),
                        is_remote=True,
                        trace_flags=self.trace.TraceFlags(self.trace.TraceFlags.SAMPLED),
                    )
                    context = self.trace.set_span_in_context(self.trace.NonRecordingSpan(parent_context), context)
                metadata = {
                    **packet.metadata,
                    **request.get("metadata", {}),
                    **record.metadata,
                    "host": self.host,
                    "pid": packet.metadata.get("source_pid"),
                    "request_id": request.get("request_id", ""),
                    "host_elapsed_ms": (record.end_ns - record.start_ns) / 1_000_000,
                    "timing_kind": "host_wall_inclusive",
                }
                attributes = {
                    "langfuse.observation.type": "span",
                    "langfuse.observation.metadata": json.dumps(metadata),
                }
                if record.error:
                    attributes["langfuse.observation.level"] = "ERROR"
                    attributes["langfuse.observation.status_message"] = record.error
                span = self.tracer.start_span(
                    record.name, context=context, start_time=record.start_ns, attributes=attributes
                )
                span_ids[index] = format(span.get_span_context().span_id, "016x")
                span.end(end_time=record.end_ns)

    def close(self):
        self.client.flush()
        self.provider.shutdown()


class BufferedExporter:
    def __init__(self, config, sink_factory=LangfuseSink):
        self.config = config
        self.sink_factory = sink_factory
        self.reset()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self.reset)
        atexit.register(self.close)

    def reset(self):
        self.queue = queue.Queue(maxsize=self.config.queue_size)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.dropped = 0
        self.failed = 0
        self.closed = False

    def submit(self, packet):
        if not packet.contexts or self.stop.is_set():
            return
        if self.thread is None:
            with self.lock:
                if self.thread is None:
                    self.thread = threading.Thread(target=self.work, name="langfuse-timing", daemon=True)
                    self.thread.start()
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.dropped += 1

    def work(self):
        sink = None
        reported = (0, 0)
        next_retry = 0.0
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    packet = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    if time.monotonic() < next_retry:
                        self.failed += 1
                        continue
                    if sink is None:
                        sink = self.sink_factory(self.config)
                    sink.emit(packet)
                except Exception as error:
                    self.failed += 1
                    next_retry = time.monotonic() + self.config.flush_interval
                    if self.failed == 1:
                        print(f"[timing] Export failed ({type(error).__name__}); inference continues.", file=sys.stderr)
                finally:
                    self.queue.task_done()
                counters = (self.dropped, self.failed)
                if counters != reported and sum(counters) & (sum(counters) - 1) == 0:
                    print(f"[timing] dropped_packets={self.dropped}, failed_packets={self.failed}", file=sys.stderr)
                    reported = counters
        finally:
            if sink is not None:
                with suppress(Exception):
                    sink.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive() or self.dropped or self.failed:
                print(
                    f"[timing] shutdown: pending={self.queue.qsize()}, dropped={self.dropped}, failed={self.failed}",
                    file=sys.stderr,
                )
