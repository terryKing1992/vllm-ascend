"""Collector-only JSON logging or Langfuse v3 export."""

import atexit
import json
import os
import queue
import socket
import sys
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass

SUMMARY_FILTER_FIELDS = (
    "request_ms",
    "max_queue_to_first_schedule_ms",
    "stage_data_status",
    "coverage",
    "step_interval",
    "history_evicted",
    "truncated_packets",
    "omitted_context_packets",
    "finish_reason",
    "received_stage_records",
)
HTTP_ATTRIBUTES = (
    ("http_method", "http.request.method"),
    ("http_route", "http.route"),
    ("http_status_code", "http.response.status_code"),
)


@dataclass(frozen=True)
class CollectorConfig:
    queue_size: int = 256
    flush_at: int = 256
    flush_interval: float = 2.0
    log_format: str = "full"
    service_name: str = "vllm-ascend-runtime"
    environment: str | None = None
    release: str | None = None


class JsonLogSink:
    def __init__(self, config, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.host = socket.gethostname()
        self.log_format = config.log_format

    def emit(self, packet):
        for request in packet.contexts:
            span_ids = {}
            for index, record in enumerate(packet.records):
                span_id = record.span_id or uuid.uuid4().hex[:16]
                parent_id = span_ids.get(record.parent, request.get("parent_span_id"))
                span_ids[index] = span_id
                event = {
                    "name": record.name,
                    "trace_id": request["trace_id"],
                    "span_id": span_id,
                    "parent_span_id": parent_id,
                    "request_id": request.get("request_id", ""),
                    "start_ns": record.start_ns,
                    "end_ns": record.end_ns,
                    "duration_ms": (record.end_ns - record.start_ns) / 1_000_000,
                    "timing_kind": "host_wall_inclusive",
                    "error": record.error,
                    "host": self.host,
                    "pid": packet.metadata.get("source_pid"),
                    "metadata": {**packet.metadata, **request.get("metadata", {}), **record.metadata},
                }
                if self.log_format == "compact":
                    for key in ("start_ns", "end_ns", "timing_kind"):
                        event.pop(key)
                    if event["error"] is None:
                        event.pop("error")
                    event["metadata"] = {
                        key: value
                        for key, value in event["metadata"].items()
                        if key
                        in (
                            "phase",
                            "step",
                            "rank",
                            "batch_id",
                            "shared_batch_time",
                            "batch_size",
                            "scheduled_tokens",
                            "omitted_sampled_requests",
                            "truncated_stage_calls",
                            "timing_summary",
                            "api_to_engine_ms",
                            "response_first_body_ms",
                            "reporting_mode",
                            "http_method",
                            "http_route",
                            "http_status_code",
                        )
                    }
                separators = (",", ":") if self.log_format == "compact" else None
                self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False, separators=separators) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.flush()


def langfuse_metadata(metadata):
    """Match Langfuse v3's per-key OTel mapping without expanding stage history."""
    return {
        f"langfuse.observation.metadata.{key}": (
            value
            if isinstance(value, (str, int))
            else json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        )
        for key, value in metadata.items()
        if value is not None
    }


@contextmanager
def sdk_batch_settings(config):
    # SDK v3.15 ignores explicit flush args when these SDK variables are absent.
    # Seed its existing settings only during this isolated collector's init.
    from langfuse._client.environment_variables import LANGFUSE_FLUSH_AT, LANGFUSE_FLUSH_INTERVAL

    values = {LANGFUSE_FLUSH_AT: str(config.flush_at), LANGFUSE_FLUSH_INTERVAL: str(config.flush_interval)}
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LangfuseSink:
    def __init__(self, config):
        # Lazy imports isolate SDK initialization in the export thread.
        from langfuse import Langfuse
        from langfuse._client.environment_variables import LANGFUSE_RELEASE, LANGFUSE_TRACING_ENVIRONMENT
        from opentelemetry import trace
        from opentelemetry.context import Context
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased

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
        self.host = socket.gethostname()
        self.environment = config.environment or os.environ.get(LANGFUSE_TRACING_ENVIRONMENT)
        self.release = config.release or os.environ.get(LANGFUSE_RELEASE)
        resource = {"service.name": config.service_name, "host.name": self.host}
        if self.environment:
            resource["deployment.environment.name"] = self.environment
        if self.release:
            resource["service.version"] = self.release
        self.provider = TracerProvider(
            id_generator=self.ids, sampler=ParentBased(ALWAYS_ON), resource=Resource.create(resource)
        )
        try:
            with sdk_batch_settings(config):
                self.client = Langfuse(
                    tracer_provider=self.provider,
                    sample_rate=1.0,
                    flush_at=config.flush_at,
                    flush_interval=config.flush_interval,
                    timeout=5,
                    environment=self.environment,
                    release=self.release,
                )
        except Exception:
            self.provider.shutdown()
            raise
        self.tracer = self.provider.get_tracer("vllm-ascend-runtime")
        self.closed = False

    def emit(self, packet):
        if self.closed:
            return
        for request in packet.contexts:
            if request.get("sampled") is False:
                continue
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
                        is_remote=record.parent not in span_ids,
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
                summary = metadata.get("timing_summary")
                if isinstance(summary, dict):
                    metadata.update({key: summary[key] for key in SUMMARY_FILTER_FIELDS if key in summary})
                attributes = {
                    "langfuse.observation.type": "span",
                    **langfuse_metadata(metadata),
                }
                is_http_request = record.name == "vllm.request"
                if is_http_request:
                    attributes.update({target: metadata[key] for key, target in HTTP_ATTRIBUTES if key in metadata})
                    if parent is None:
                        attributes["langfuse.trace.name"] = record.name
                if self.environment:
                    attributes["langfuse.environment"] = self.environment
                if self.release:
                    attributes["langfuse.release"] = self.release
                status_code = metadata.get("http_status_code") if is_http_request else None
                http_error = isinstance(status_code, int) and status_code >= 500
                if record.error:
                    attributes["langfuse.observation.level"] = "ERROR"
                    attributes["langfuse.observation.status_message"] = record.error
                    attributes["error.type"] = record.error
                elif http_error:
                    attributes["langfuse.observation.level"] = "ERROR"
                    attributes["error.type"] = str(status_code)
                elif isinstance(status_code, int) and status_code >= 400:
                    attributes["langfuse.observation.level"] = "WARNING"
                try:
                    span = self.tracer.start_span(
                        record.name,
                        context=context,
                        kind=self.trace.SpanKind.SERVER if is_http_request else self.trace.SpanKind.INTERNAL,
                        start_time=record.start_ns,
                        attributes=attributes,
                    )
                finally:
                    self.ids.trace_id = self.ids.span_id = None
                if record.error or http_error:
                    span.set_status(self.trace.Status(self.trace.StatusCode.ERROR, record.error))
                span_ids[index] = format(span.get_span_context().span_id, "016x")
                span.end(end_time=record.end_ns)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.client.shutdown()
        finally:
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
                    if sink is not None and callable(getattr(sink, "poll", None)):
                        try:
                            sink.poll()
                        except Exception as error:
                            self.failed += 1
                            if self.failed == 1:
                                print(f"[timing] Summary export failed ({type(error).__name__}).", file=sys.stderr)
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
