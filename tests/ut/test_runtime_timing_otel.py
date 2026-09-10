"""Langfuse v3 wire-format regressions, with every HTTP request intercepted."""

import base64
import gzip
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import unittest
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "runtime_timing"
sys.path.insert(0, str(TOOL_DIR))
if (TOOL_DIR / ".test-deps").is_dir():
    sys.path.insert(0, str(TOOL_DIR / ".test-deps"))

from trace_export import CollectorConfig, LangfuseSink  # noqa: E402
from trace_summary import RequestSummarySink, SummaryConfig  # noqa: E402
from trace_transport import Packet, Record  # noqa: E402

TRACE_ID = "12345678901234567890123456789012"
PARENT_ID = "1234567890123456"
REQUEST_ID = "aaaaaaaaaaaaaaaa"
METADATA_PREFIX = "langfuse.observation.metadata."


def attribute_value(value):
    """Decode OTLP AnyValue without converting nanosecond integers to floats."""
    kind = value.WhichOneof("value")
    if kind == "array_value":
        return [attribute_value(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return attributes(value.kvlist_value.values)
    return getattr(value, kind) if kind else None


def attributes(values):
    return {item.key: attribute_value(item.value) for item in values}


def exported_spans(calls):
    # The SDK is optional and must never be imported by the injection bundle.
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    spans = []
    for call in calls:
        payload = call["data"]
        compression = call["headers"].get("content-encoding", "")
        if compression == "gzip":
            payload = gzip.decompress(payload)
        elif compression == "deflate":
            payload = zlib.decompress(payload)
        elif compression:
            raise AssertionError(f"Unsupported test payload compression: {compression}")
        request = ExportTraceServiceRequest.FromString(payload)
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                spans.extend((span, attributes(resource.resource.attributes)) for span in scope.spans)
    return spans


@contextmanager
def wire_sink(config, environment=None):
    """Use the real SDK exporter, but capture bytes before the requests transport."""
    from requests import Response

    public_key = "pk-lf-test-" + uuid.uuid4().hex
    secret_key = "sk-lf-test-placeholder"
    env = os.environ.copy()
    for name in (
        "LANGFUSE_FLUSH_AT",
        "LANGFUSE_FLUSH_INTERVAL",
        "LANGFUSE_OTEL_TRACES_EXPORT_PATH",
        "LANGFUSE_ENVIRONMENT",
        "LANGFUSE_RELEASE",
        "OTEL_EXPORTER_OTLP_COMPRESSION",
        "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION",
    ):
        env.pop(name, None)
    env.update(
        LANGFUSE_PUBLIC_KEY=public_key,
        LANGFUSE_SECRET_KEY=secret_key,
        LANGFUSE_BASE_URL="http://127.0.0.1:9",
        LANGFUSE_TRACING_ENABLED="true",
        OTEL_TRACES_SAMPLER="always_off",
    )
    env.update(environment or {})
    calls = []
    sent = threading.Event()

    def intercept(session, url, **kwargs):
        headers = {key.lower(): value for key, value in session.headers.items()}
        headers.update({key.lower(): value for key, value in kwargs.get("headers", {}).items()})
        calls.append({"url": url, "headers": headers, "data": kwargs["data"]})
        sent.set()
        response = Response()
        response.status_code, response._content = 200, b""
        return response

    with (
        patch.dict(os.environ, env, clear=True),
        patch("requests.Session.post", autospec=True, side_effect=intercept),
        patch("requests.Session.send", side_effect=AssertionError("Unexpected real HTTP request")),
        patch("httpx.Client.send", side_effect=AssertionError("Unexpected real SDK API request")),
    ):
        sink = LangfuseSink(config)
        try:
            yield sink, calls, sent, public_key, secret_key
        finally:
            sink.close()


@unittest.skipUnless(importlib.util.find_spec("langfuse"), "optional Langfuse SDK v3 is not installed")
class TestLangfuseOTel(unittest.TestCase):
    def request(self, *, parent=PARENT_ID, sampled=True, name="vllm.request", metadata=None, error=None):
        start = time.time_ns() - 1_000_000_000
        context = {"trace_id": TRACE_ID, "sampled": sampled}
        if parent:
            context["parent_span_id"] = parent
        record = Record(name, start, start + 100_000_123, span_id=REQUEST_ID, metadata=metadata or {}, error=error)
        return Packet((context,), [record], {"source_pid": 123})

    def test_thousand_decode_steps_export_one_standard_otlp_summary(self):
        config = CollectorConfig(service_name="test-inference", environment="test", release="build-23")
        with wire_sink(config) as (sink, calls, _, public_key, secret_key):
            summary = RequestSummarySink(SummaryConfig(), sink)
            root = self.request(
                metadata={
                    "path": "/v1/chat/completions",
                    "response_first_body_ms": 2.5,
                    "custom": {"labels": ["推理", "decode"]},
                    "streaming": True,
                }
            )
            start = root.records[0].start_ns
            context = {
                "trace_id": TRACE_ID,
                "parent_span_id": REQUEST_ID,
                "request_id": "engine-request",
                "metadata": {"phase": "decode", "queue_to_first_schedule_ms": 3.0},
            }
            for step in range(1000):
                context["metadata"]["step"] = step
                summary.emit(
                    Packet(
                        (context,),
                        [Record("runner.execute_model", start, start + 1_000_000)],
                        {"rank": 0, "source_pid": 321, "every_n_steps": 1},
                    )
                )
            self.assertEqual(len(calls), 0, "decode observations must stay local until request completion")
            summary.emit(root)
            summary.close()
            exported = exported_spans(calls)
            self.assertEqual(len(exported), 1)
            span, resource = exported[0]
            self.assertEqual(span.name, "vllm.request")
            self.assertEqual(span.kind, 2)  # OTLP SpanKind.SERVER
            self.assertEqual(span.trace_id.hex(), TRACE_ID)
            self.assertEqual(span.span_id.hex(), REQUEST_ID)
            self.assertEqual(span.parent_span_id.hex(), PARENT_ID)
            self.assertEqual(span.start_time_unix_nano, start)
            self.assertEqual(span.end_time_unix_nano, root.records[0].end_ns)
            self.assertEqual(resource["service.name"], "test-inference")
            self.assertEqual(resource["host.name"], socket.gethostname())
            values = attributes(span.attributes)
            self.assertEqual(values["langfuse.environment"], "test")
            self.assertEqual(values["langfuse.release"], "build-23")
            self.assertEqual(values["langfuse.observation.type"], "span")
            self.assertNotIn("langfuse.trace.name", values, "an upstream trace name must be preserved")
            self.assertNotIn("langfuse.observation.metadata", values)
            self.assertEqual(values[METADATA_PREFIX + "path"], "/v1/chat/completions")
            # Langfuse v3 serializes non-str/int metadata scalars as JSON.
            self.assertEqual(json.loads(values[METADATA_PREFIX + "response_first_body_ms"]), 2.5)
            self.assertEqual(json.loads(values[METADATA_PREFIX + "custom"]), {"labels": ["推理", "decode"]})
            self.assertIs(values[METADATA_PREFIX + "streaming"], True)
            payload = values[METADATA_PREFIX + "timing_summary"]
            stats = json.loads(payload)
            self.assertEqual(payload, json.dumps(stats, ensure_ascii=False, separators=(",", ":")))
            self.assertEqual(stats["stages"][0]["calls"], 1000)
            self.assertEqual(stats["stages"][0]["mean_host_ms"], 1)
            self.assertEqual(stats["request_ids"], ["engine-request"])
            for key in (
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
            ):
                value = values[METADATA_PREFIX + key]
                if isinstance(stats[key], float):
                    value = json.loads(value)
                self.assertEqual(value, stats[key], key)
            for key in values:
                self.assertNotIn("completion_start_time", key)
                self.assertFalse(
                    key.startswith(("gen_ai.usage.", "langfuse.observation.usage", "langfuse.observation.cost"))
                )
            self.assertNotIn("gen_ai.request.model", values)
            self.assertNotIn("langfuse.observation.model.name", values)
            expected_auth = "Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
            for call in calls:
                self.assertEqual(call["url"], "http://127.0.0.1:9/api/public/otel/v1/traces")
                self.assertEqual(call["headers"]["content-type"], "application/x-protobuf")
                self.assertTrue(call["headers"]["authorization"] == expected_auth, "SDK must use project Basic Auth")
                self.assertTrue(call["headers"]["x-langfuse-sdk-version"].startswith("3."))
                self.assertNotIn("x-langfuse-ingestion-version", call["headers"])

    def test_sampler_and_parentage_preserve_recorded_requests_and_unsampled_flag(self):
        with wire_sink(CollectorConfig()) as (sink, calls, _, _, _):
            # Global OTEL_TRACES_SAMPLER=always_off must not discard this
            # independently recorded request. Explicit traceparent 00 still wins.
            sink.emit(self.request(sampled=False))
            root = self.request(parent=None)
            child_start = root.records[0].start_ns + 11
            root.records.append(Record("runner.sample_tokens", child_start, child_start + 19, parent=0))
            sink.emit(root)
            sink.close()
            exported = exported_spans(calls)
            self.assertEqual(len(exported), 2)
            spans = {span.name: span for span, _ in exported}
            self.assertEqual(set(spans), {"vllm.request", "runner.sample_tokens"})
            request, child = spans["vllm.request"], spans["runner.sample_tokens"]
            self.assertEqual(request.parent_span_id, b"")
            self.assertEqual(attributes(request.attributes)["langfuse.trace.name"], "vllm.request")
            self.assertEqual(child.kind, 1)  # OTLP SpanKind.INTERNAL
            self.assertEqual(child.trace_id, request.trace_id)
            self.assertEqual(child.parent_span_id, request.span_id)
            self.assertEqual((child.start_time_unix_nano, child.end_time_unix_nano), (child_start, child_start + 19))
            self.assertNotIn("langfuse.trace.name", attributes(child.attributes))
            self.assertEqual(attributes(request.attributes)[METADATA_PREFIX + "pid"], 123)

    def test_exceptions_and_http_server_failures_set_error_status(self):
        with wire_sink(CollectorConfig()) as (sink, calls, _, _, _):
            for index, (metadata, error) in enumerate(
                (({}, "RuntimeError"), ({"http_status_code": 503}, None), ({"http_status_code": 400}, None))
            ):
                packet = self.request(metadata=metadata, error=error)
                packet.records[0].span_id = f"{index + 1:016x}"
                sink.emit(packet)
            sink.close()
            spans = {span.span_id.hex(): span for span, _ in exported_spans(calls)}
            exception, unavailable, bad_request = (spans[f"{index:016x}"] for index in range(1, 4))
            self.assertEqual(exception.status.code, 2)  # OTLP StatusCode.ERROR
            self.assertEqual(unavailable.status.code, 2)
            self.assertEqual(bad_request.status.code, 0)  # Server-side 4xx is UNSET.
            values = attributes(exception.attributes)
            self.assertEqual(values["error.type"], "RuntimeError")
            self.assertEqual(values["langfuse.observation.level"], "ERROR")
            self.assertEqual(attributes(unavailable.attributes)["langfuse.observation.level"], "ERROR")

    def test_batch_size_and_interval_export_without_forced_flush_and_restore_environment(self):
        cases = (
            (CollectorConfig(flush_at=1, flush_interval=3), {}),
            (
                CollectorConfig(flush_at=128, flush_interval=0.05),
                {"LANGFUSE_FLUSH_AT": "2048", "LANGFUSE_FLUSH_INTERVAL": "60"},
            ),
        )
        for config, environment in cases:
            with (
                self.subTest(flush_at=config.flush_at, flush_interval=config.flush_interval),
                wire_sink(config, environment) as (sink, calls, sent, _, _),
            ):
                for name in ("LANGFUSE_FLUSH_AT", "LANGFUSE_FLUSH_INTERVAL"):
                    self.assertEqual(os.environ.get(name), environment.get(name))
                sink.emit(self.request())
                self.assertTrue(sent.wait(2), "configured batch size / interval must apply before close or flush")
                self.assertEqual(len(exported_spans(calls)), 1)

    def test_close_shuts_down_sdk_and_provider_once(self):
        with wire_sink(CollectorConfig()) as (sink, calls, _, _, _):
            sink.emit(self.request())
            with (
                patch.object(sink.client, "shutdown", wraps=sink.client.shutdown) as shutdown_client,
                patch.object(sink.provider, "shutdown", wraps=sink.provider.shutdown) as shutdown_provider,
            ):
                sink.close()
                self.assertTrue(shutdown_client.called)
                self.assertTrue(shutdown_provider.called)
                before = shutdown_client.call_count, shutdown_provider.call_count, len(calls)
                sink.close()
                self.assertEqual((shutdown_client.call_count, shutdown_provider.call_count, len(calls)), before)
                self.assertEqual(len(exported_spans(calls)), 1)


if __name__ == "__main__":
    unittest.main()
