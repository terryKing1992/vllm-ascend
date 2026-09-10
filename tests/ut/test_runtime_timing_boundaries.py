"""Boundary timing contracts without importing vLLM, torch or an NPU runtime."""

import asyncio
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "runtime_timing"
sys.path.insert(0, str(TOOL_DIR))

from timing_probe import ENQUEUE_CLOCK_ATTR, Config, RequestMiddleware, Runtime  # noqa: E402

TRACEPARENT = "00-12345678901234567890123456789012-1234567890123456-01"
NANOSECONDS_PER_MILLISECOND = 1_000_000


class TestBoundaryTiming(unittest.TestCase):
    def setUp(self):
        self.packets = []
        self.runtime = Runtime(Config(sample_rate=1, every_n_steps=1), SimpleNamespace(submit=self.packets.append))
        self.scope = {"type": "http", "path": "/v1/completions", "method": "POST"}

    def request(self):
        return SimpleNamespace(request_id="request-1", trace_headers={"traceparent": TRACEPARENT}, num_output_tokens=0)

    def test_api_and_first_body_boundaries_preserve_messages_and_results(self):
        clock = [1_000 * NANOSECONDS_PER_MILLISECOND]
        messages = [
            {"type": "http.response.start", "status": 200},
            {"type": "http.response.body", "body": b"", "more_body": True},
            {"type": "http.response.body", "body": b"first", "more_body": True},
            {"type": "http.response.body", "body": b"last"},
        ]
        sent, engine_headers = [], []
        result = object()

        async def original(owner, request_id, prompt, trace_headers=None):
            engine_headers.append(trace_headers)
            return result

        add_request = self.runtime.wrap_add_request(original)

        async def send(message):
            sent.append(message)
            return result

        async def app(scope, receive, send):
            self.assertIs(await send(messages[0]), result)
            self.assertIs(await send(messages[1]), result)
            clock[0] += 3 * NANOSECONDS_PER_MILLISECOND
            self.assertIs(await add_request(None, "a", {}), result)
            clock[0] += 2 * NANOSECONDS_PER_MILLISECOND
            self.assertIs(await add_request(None, "b", {}), result)
            clock[0] += 3 * NANOSECONDS_PER_MILLISECOND
            self.assertIs(await send(messages[2]), result)
            clock[0] += NANOSECONDS_PER_MILLISECOND
            self.assertIs(await send(messages[3]), result)
            return result

        with patch("timing_probe.time.perf_counter_ns", side_effect=lambda: clock[0]):
            actual = asyncio.run(RequestMiddleware(app, self.runtime)(self.scope, None, send))
        self.assertIs(actual, result)
        self.assertTrue(all(actual is original for actual, original in zip(sent, messages)))
        root = self.packets[0].records[0]
        self.assertEqual(
            root.metadata,
            {
                "api_to_engine_ms": 3.0,
                "response_first_body_ms": 8.0,
                "http_method": "POST",
                "http_route": "/v1/completions",
                "http_status_code": 200,
            },
        )
        self.assertEqual(root.end_ns - root.start_ns, 9 * NANOSECONDS_PER_MILLISECOND)
        self.assertEqual(engine_headers[0], engine_headers[1])
        self.assertIsNone(self.runtime.request.get())

    def test_manual_trace_context_without_boundary_fields_remains_supported(self):
        async def original(owner, request_id, prompt, trace_headers=None):
            return trace_headers

        token = self.runtime.request.set({"traceparent": TRACEPARENT})
        try:
            headers = asyncio.run(self.runtime.wrap_add_request(original)(None, "a", {}))
        finally:
            self.runtime.request.reset(token)
        self.assertEqual(headers, {"traceparent": TRACEPARENT})
        self.assertFalse(self.runtime.disabled)

    def test_first_body_observer_failure_preserves_send_and_app_result(self):
        message, result = {"type": "http.response.body", "body": b"content"}, object()
        sent = []

        async def send(value):
            sent.append(value)
            return result

        async def app(scope, receive, send):
            return await send(message)

        middleware = RequestMiddleware(app, self.runtime)
        with patch.object(middleware, "record_first_body", side_effect=RuntimeError("observer failed")):
            self.assertIs(asyncio.run(middleware(self.scope, None, send)), result)
        self.assertEqual(len(sent), 1)
        self.assertIs(sent[0], message)
        self.assertIsNone(self.runtime.request.get())

    def test_send_exception_is_not_swallowed_or_retried(self):
        error = RuntimeError("send failed")
        calls = []
        message = {"type": "http.response.body", "body": b"content"}

        async def send(value):
            calls.append(value)
            raise error

        async def app(scope, receive, send):
            await send(message)

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(RequestMiddleware(app, self.runtime)(self.scope, None, send))
        self.assertIs(raised.exception, error)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], message)
        self.assertEqual(self.packets[0].records[0].error, "RuntimeError")
        self.assertIsNone(self.runtime.request.get())

    def test_empty_response_has_no_first_body_boundary(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.body", "body": b""})

        async def send(message):
            return None

        asyncio.run(RequestMiddleware(app, self.runtime)(self.scope, None, send))
        self.assertNotIn("response_first_body_ms", self.packets[0].records[0].metadata)

    def test_http_status_without_body_preserves_response_and_normalizes_route(self):
        scope = {**self.scope, "path": "/proxy/v1/completions", "root_path": "/proxy", "method": "custom-method"}
        scope["headers"] = [(b"authorization", b"test-private-header")]
        for status in (200, 400, 503, None, True, 999):
            with self.subTest(status=status):
                message = {"type": "http.response.start", "status": status}
                sent = []

                async def send(value, sent=sent):
                    sent.append(value)

                async def app(scope, receive, send, message=message):
                    await send(message)

                asyncio.run(RequestMiddleware(app, self.runtime)(scope, None, send))
                self.assertEqual(len(sent), 1)
                self.assertIs(sent[0], message)
                record = self.packets[-1].records[0]
                expected = {"http_method": "_OTHER", "http_route": "/v1/completions"}
                if status in (200, 400, 503):
                    expected["http_status_code"] = status
                self.assertEqual(record.metadata, expected)
                self.assertIsNone(record.error)
        self.assertNotIn("api_to_engine_ms", self.packets[0].records[0].metadata)

    def test_first_schedule_wait_excludes_schedule_execution_and_zero_token_steps(self):
        request = self.request()
        scheduler = SimpleNamespace(requests={})
        clock = [1_000 * NANOSECONDS_PER_MILLISECOND]
        result = object()

        def add_request(owner, request):
            owner.requests[request.request_id] = request
            return result

        add = self.runtime.wrap_scheduler_add_request(add_request)

        def schedule(owner, num_tokens):
            clock[0] += 50 * NANOSECONDS_PER_MILLISECOND
            return SimpleNamespace(num_scheduled_tokens={request.request_id: num_tokens})

        scheduled = self.runtime.wrap_schedule(schedule)
        with patch("timing_probe.time.perf_counter_ns", side_effect=lambda: clock[0]):
            self.assertIs(add(scheduler, request=request), result)
            clock[0] += 10 * NANOSECONDS_PER_MILLISECOND
            empty = scheduled(scheduler, 0)
            self.assertEqual(empty._langfuse_runtime_packet["contexts"], [])
            clock[0] += 10 * NANOSECONDS_PER_MILLISECOND
            first = scheduled(scheduler, 1)
            second = scheduled(scheduler, 1)
        first_metadata = first._langfuse_runtime_packet["contexts"][0]["metadata"]
        self.assertEqual(first_metadata["step"], 0)
        self.assertEqual(first_metadata["queue_to_first_schedule_ms"], 70.0)
        self.assertNotIn("queue_to_first_schedule_ms", second._langfuse_runtime_packet["contexts"][0]["metadata"])
        schedule_record = self.packets[0].records[0]
        self.assertEqual(schedule_record.end_ns - schedule_record.start_ns, 50 * NANOSECONDS_PER_MILLISECOND)
        self.assertEqual(first._langfuse_runtime_packet["metadata"]["every_n_steps"], 1)
        self.assertEqual(first._langfuse_runtime_packet["metadata"]["sample_rate"], 1)

    def test_scheduler_repeated_registration_does_not_reset_enqueue(self):
        request = self.request()
        scheduler = SimpleNamespace(requests={})

        def add_request(owner, request):
            owner.requests.setdefault(request.request_id, request)

        add = self.runtime.wrap_scheduler_add_request(add_request)
        with patch("timing_probe.time.perf_counter_ns", return_value=100):
            add(scheduler, request)
        with patch("timing_probe.time.perf_counter_ns", return_value=900):
            add(scheduler, request)
            replacement = self.request()
            add(scheduler, replacement)
        self.assertEqual(getattr(request, ENQUEUE_CLOCK_ATTR), 100)
        self.assertFalse(hasattr(replacement, ENQUEUE_CLOCK_ATTR))

    def test_scheduler_registration_failure_preserves_original_exception(self):
        request = self.request()
        scheduler = SimpleNamespace(requests={})
        error = ValueError("request rejected")
        original = Mock(side_effect=error)
        with self.assertRaises(ValueError) as raised:
            self.runtime.wrap_scheduler_add_request(original)(scheduler, request)
        self.assertIs(raised.exception, error)
        original.assert_called_once_with(scheduler, request)
        self.assertFalse(hasattr(request, ENQUEUE_CLOCK_ATTR))

    def test_scheduler_observer_failure_does_not_change_registration(self):
        request = self.request()
        scheduler = SimpleNamespace(requests={})
        result = object()

        def original(owner, request):
            owner.requests[request.request_id] = request
            return result

        with patch.object(self.runtime, "mark_scheduler_enqueue", side_effect=RuntimeError("observer failed")):
            self.assertIs(self.runtime.wrap_scheduler_add_request(original)(scheduler, request), result)
        self.assertIs(scheduler.requests[request.request_id], request)

    def test_optional_add_request_signature_does_not_disable_schedule_patch(self):
        module = ModuleType("vllm.v1.core.sched.scheduler")
        exec(
            "class Scheduler:\n"
            "    def schedule(self):\n"
            "        return 'scheduled'\n"
            "    def add_request(self, unsupported):\n"
            "        return unsupported\n",
            module.__dict__,
        )
        self.runtime.safe_patch_module(module)
        self.assertTrue(module.Scheduler.schedule._runtime_timing_wrapped)
        self.assertFalse(getattr(module.Scheduler.add_request, "_runtime_timing_wrapped", False))
        self.assertFalse(self.runtime.disabled)


if __name__ == "__main__":
    unittest.main()
