"""Fault-injection tests, runnable without vLLM or NPU hardware."""

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import pickle
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "runtime_timing"
sys.path.insert(0, str(TOOL_DIR))
if (TOOL_DIR / ".test-deps").is_dir():
    sys.path.insert(0, str(TOOL_DIR / ".test-deps"))

from collector import Collector  # noqa: E402
from run import prepare  # noqa: E402
from timing_probe import Config, RequestMiddleware, Runtime, package_version, parse_traceparent, selected  # noqa: E402
from trace_export import BufferedExporter, CollectorConfig, JsonLogSink, LangfuseSink  # noqa: E402
from trace_transport import DatagramEmitter, Packet, Record, decode_packet  # noqa: E402

TRACE_ID = "12345678901234567890123456789012"
PARENT_ID = "1234567890123456"


class MemoryCollector:
    def __init__(self):
        self.packets = []

    def submit(self, packet):
        self.packets.append(packet)


class TestTracing(unittest.TestCase):
    def setUp(self):
        self.collector = MemoryCollector()
        self.runtime = Runtime(Config(sample_rate=1, every_n_steps=1), self.collector)

    def test_transport_dataclasses_have_no_required_fields(self):
        self.assertEqual(Record().name, "")
        self.assertEqual(Record().start_ns, 0)
        self.assertEqual(Packet().contexts, ())
        self.assertEqual(Packet().records, [])

    def test_package_version_failure_is_diagnostic_only(self):
        with patch("timing_probe.importlib.metadata.version", side_effect=RuntimeError("broken metadata")):
            self.assertEqual(package_version("vllm"), "unknown")

    def request(self, trace_id=TRACE_ID):
        return SimpleNamespace(trace_headers={"traceparent": f"00-{trace_id}-{PARENT_ID}-01"}, num_output_tokens=0)

    def carrier(self):
        return {"contexts": [parse_traceparent(f"00-{TRACE_ID}-{PARENT_ID}-01")], "metadata": {}}

    def test_json_log_keeps_trace_parentage_and_duration(self):
        stream = io.StringIO()
        sink = JsonLogSink(CollectorConfig(), stream)
        contexts = tuple(self.carrier()["contexts"])
        sink.emit(
            Packet(
                contexts,
                [Record("execute", 1000, 3000, span_id="a" * 16), Record("prepare", 1500, 2500, parent=0)],
                {"source_pid": 123},
            )
        )
        sink.close()
        root, child = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(root["trace_id"], TRACE_ID)
        self.assertEqual(root["parent_span_id"], PARENT_ID)
        self.assertEqual(child["parent_span_id"], root["span_id"])
        self.assertEqual(child["duration_ms"], 0.001)
        self.assertEqual(child["pid"], 123)

    def test_log_collector_starts_and_outputs_without_site_packages(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, "-S", str(TOOL_DIR / "collector.py"), "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        emitter = DatagramEmitter(port)
        try:
            self.assertIn("output=log", process.stderr.readline())
            emitter.submit(Packet(tuple(self.carrier()["contexts"]), [Record("execute", 1, 2)]))
            lines = []
            reader = threading.Thread(target=lambda: lines.append(process.stdout.readline()), daemon=True)
            reader.start()
            reader.join(timeout=5)
            self.assertFalse(reader.is_alive(), "collector must flush log packets")
            self.assertEqual(json.loads(lines[0])["trace_id"], TRACE_ID)
        finally:
            process.terminate()
            process.communicate(timeout=5)
            if emitter.socket is not None:
                emitter.socket.close()

    def test_trace_validation_and_sampling(self):
        self.assertIsNone(parse_traceparent("invalid"))
        self.assertIsNone(parse_traceparent(f"00-{'0' * 32}-{PARENT_ID}-01"))
        context = parse_traceparent(f"00-{TRACE_ID}-{PARENT_ID}-00")
        self.assertFalse(selected(context, 1))
        context["sampled"] = True
        self.assertFalse(selected(context, 0))
        self.assertTrue(selected(context, 1))

    def test_scheduler_ipc_runner_nesting_and_sampling_cleanup(self):
        scheduler = SimpleNamespace(requests={"a": self.request(), "b": self.request("2" * 32)})
        output = SimpleNamespace(num_scheduled_tokens={"a": 3, "b": 1}, total_num_scheduled_tokens=4)
        output = self.runtime.wrap_schedule(lambda owner: output)(scheduler)
        carrier = output._langfuse_runtime_packet
        self.assertIsInstance(carrier, dict)
        # A Python process with no observer module must deserialize the vLLM carrier.
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys,pickle; x=pickle.loads(sys.stdin.buffer.read()); assert isinstance(x,dict)",
            ],
            input=pickle.dumps(carrier),
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        stage = self.runtime.wrap_stage(lambda: "result", "prepare")
        runner = SimpleNamespace()
        execute = self.runtime.wrap_runner(lambda owner, scheduler_output: stage() and None)
        sample = self.runtime.wrap_runner(lambda owner: 7, sampling=True)
        self.assertIsNone(execute(runner, output))
        self.assertEqual(sample(runner), 7)
        sample(runner)
        self.assertEqual(len(self.collector.packets), 3)
        packet = self.collector.packets[1]
        self.assertEqual(len(packet.contexts), 2)
        self.assertEqual(packet.records[1].parent, 0)
        self.assertEqual(packet.contexts[0]["trace_id"], TRACE_ID)
        self.assertTrue(packet.metadata["shared_batch_time"])
        self.assertIsNone(self.runtime.stage.get())

    def test_sampling_and_inherited_scheduler(self):
        self.runtime.config = Config(sample_rate=1, every_n_steps=2)
        scheduler = SimpleNamespace(requests={"a": self.request()})
        base = self.runtime.wrap_schedule(lambda owner: SimpleNamespace(num_scheduled_tokens={"a": 1}))
        derived = self.runtime.wrap_schedule(lambda owner: base(owner))
        for _ in range(3):
            derived(scheduler)
        self.assertEqual(len(self.collector.packets), 2)
        self.assertEqual(self.collector.packets[-1].contexts[0]["metadata"]["step"], 2)

    def test_fault_before_business_calls_original_exactly_once(self):
        for method, wrapper in (
            ("begin_schedule", self.runtime.wrap_schedule),
            ("begin_runner", self.runtime.wrap_runner),
            ("begin_stage", lambda fn: self.runtime.wrap_stage(fn, "stage")),
        ):
            with self.subTest(method=method):
                self.runtime.disabled = False
                original = Mock(return_value=object())
                wrapped = wrapper(original)
                with patch.object(self.runtime, method, side_effect=RuntimeError("observation failed")):
                    self.assertIs(wrapped(SimpleNamespace()), original.return_value)
                original.assert_called_once()
                self.assertTrue(self.runtime.disabled)
                self.assertIs(wrapped(SimpleNamespace()), original.return_value)
                self.assertEqual(original.call_count, 2)

    def test_cleanup_fault_does_not_replace_business_exception(self):
        business_error = ValueError("business")
        original = Mock(side_effect=business_error)
        output = SimpleNamespace(_langfuse_runtime_packet=self.carrier())
        with (
            patch.object(self.runtime, "end_runner", side_effect=RuntimeError("cleanup failed")),
            self.assertRaises(ValueError) as caught,
        ):
            self.runtime.wrap_runner(original)(SimpleNamespace(), output)
        self.assertIs(caught.exception, business_error)
        original.assert_called_once()
        self.assertTrue(self.runtime.disabled)

    def test_clock_and_serialization_failure_leave_business_intact(self):
        for failure in ("clock", "serialization"):
            with self.subTest(failure=failure):
                runtime = Runtime(Config(sample_rate=1), DatagramEmitter(18765))
                original = Mock(return_value=11)
                output = SimpleNamespace(_langfuse_runtime_packet=self.carrier())
                target = "timing_probe.time.time_ns" if failure == "clock" else "trace_transport.json.dumps"
                with (
                    patch(target, side_effect=OSError("fault")),
                    patch("builtins.print", side_effect=OSError("stderr")),
                ):
                    self.assertEqual(runtime.wrap_runner(original)(SimpleNamespace(), output), 11)
                original.assert_called_once()
                self.assertTrue(runtime.disabled)

    def test_missing_or_bad_carrier_never_blocks_runner(self):
        for carrier in (None, {}, {"contexts": [None]}, {"contexts": [1], "metadata": None}):
            self.runtime.disabled = False
            output = SimpleNamespace(_langfuse_runtime_packet=carrier)
            original = Mock(return_value=23)
            self.assertEqual(self.runtime.wrap_runner(original)(SimpleNamespace(), output), 23)
            original.assert_called_once()

    def test_oversize_and_backpressure_drop_without_retry(self):
        emitter = DatagramEmitter(18765)
        sock = Mock()
        sock.sendto.side_effect = BlockingIOError()
        emitter.socket = sock
        packet = Packet(tuple(self.carrier()["contexts"]), [Record("stage", 1, 2)])
        for _ in range(3):
            emitter.submit(packet)
        self.assertEqual(sock.sendto.call_count, 3)
        self.assertEqual(emitter.dropped, 3)
        packet.metadata["oversize"] = "x" * 9000
        emitter.submit(packet)
        self.assertEqual(sock.sendto.call_count, 3)
        self.assertEqual(emitter.dropped, 4)

    def test_udp_socket_is_nonblocking_and_has_no_remote_resolution(self):
        sock = Mock()
        emitter = DatagramEmitter(18765)
        with patch("trace_transport.socket.socket", return_value=sock):
            emitter.submit(Packet(tuple(self.carrier()["contexts"]), [Record("stage", 1, 2)]))
        sock.setblocking.assert_called_once_with(False)
        self.assertEqual(sock.sendto.call_args.args[1], ("127.0.0.1", 18765))

    def test_sender_diagnostic_log_reports_packet(self):
        sock = Mock()
        emitter = DatagramEmitter(18765, diagnostic_log=True)
        emitter.socket = sock
        packet = Packet(tuple(self.carrier()["contexts"]), [Record("stage", 1, 2)])
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            emitter.submit(packet)
        self.assertIn("[timing-send] sent packet=1", stream.getvalue())
        self.assertIn("names=stage", stream.getvalue())

    def test_probe_diagnostic_log_reports_lifecycle(self):
        runtime = Runtime(Config(diagnostic_log=True), self.collector)
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            runtime.log("installed port=18765")
        self.assertIn("[timing-probe] installed port=18765", stream.getvalue())

    def test_receiver_diagnostic_log_is_separate_from_json_output(self):
        sink = Mock()
        collector = Collector(0, sink, diagnostic_log=True)
        stream = io.StringIO()
        try:
            with contextlib.redirect_stderr(stream):
                collector.log("received packet=1 names=stage")
        finally:
            collector.socket.close()
        self.assertIn("[timing-recv] received packet=1 names=stage", stream.getvalue())

    def test_caps_and_business_exception_preserved(self):
        self.runtime.config = Config(sample_rate=1, max_records=2)
        stage = self.runtime.wrap_stage(lambda: None, "stage")

        def execute(owner, output):
            for _ in range(10):
                stage()
            raise ValueError("unchanged")

        with self.assertRaisesRegex(ValueError, "unchanged"):
            self.runtime.wrap_runner(execute)(
                SimpleNamespace(), SimpleNamespace(_langfuse_runtime_packet=self.carrier())
            )
        packet = self.collector.packets[0]
        self.assertEqual(len(packet.records), 2)
        self.assertEqual(packet.metadata["truncated_stage_calls"], 9)
        self.assertEqual(packet.records[0].error, "ValueError")
        self.assertIsNone(self.runtime.stage.get())

    def test_asgi_streaming_context_and_original_messages(self):
        received, sent = [], []

        async def add_request(owner, request_id, prompt, trace_headers=None):
            received.append((request_id, trace_headers["traceparent"]))

        add = self.runtime.wrap_add_request(add_request)
        messages = [
            {"type": "http.response.start", "status": 200, "headers": []},
            {"type": "http.response.body", "body": b"chunk", "more_body": True},
            {"type": "http.response.body", "body": b"end"},
        ]

        async def app(scope, receive, send):
            await asyncio.sleep(0)
            await add(None, scope["request_id"], {})
            for message in messages:
                await send(message)

        async def send(message):
            sent.append(message)

        middleware = RequestMiddleware(app, self.runtime)

        async def run():
            await asyncio.gather(
                *(
                    middleware(
                        {
                            "type": "http",
                            "path": "/v1/chat/completions",
                            "request_id": key,
                            "headers": [(b"traceparent", f"00-{tid}-{PARENT_ID}-01".encode())],
                        },
                        None,
                        send,
                    )
                    for key, tid in (("a", TRACE_ID), ("b", "2" * 32))
                )
            )

        asyncio.run(run())
        self.assertEqual(len(self.collector.packets), 2)
        self.assertEqual(len(sent), 6)
        self.assertTrue(all(any(item is expected for expected in messages) for item in sent))
        root = next(p for p in self.collector.packets if p.contexts[0]["trace_id"] == TRACE_ID)
        self.assertEqual(parse_traceparent(dict(received)["a"])["parent_span_id"], root.records[0].span_id)
        self.assertIsNone(self.runtime.request.get())

    def test_asgi_and_add_request_instrumentation_failures(self):
        for point in ("begin", "end"):
            self.runtime.disabled = False
            calls = []

            async def app(scope, receive, send, calls=calls):
                calls.append(1)
                return 19

            middleware = RequestMiddleware(app, self.runtime)
            with patch.object(middleware, point, side_effect=RuntimeError("fault")):
                self.assertEqual(asyncio.run(middleware({"type": "http", "path": "/v1/completions"}, None, None)), 19)
            self.assertEqual(calls, [1])
        self.runtime.disabled = False

        async def original(value, trace_headers=None):
            return value, trace_headers

        with patch.object(self.runtime, "request_arguments", side_effect=RuntimeError("fault")):
            self.assertEqual(asyncio.run(self.runtime.wrap_add_request(original)(3)), (3, None))

    def test_add_request_logs_existing_trace_context_without_http_middleware(self):
        runtime = Runtime(Config(sample_rate=1, diagnostic_log=True), self.collector)

        async def original(owner, request_id, prompt, trace_headers=None):
            return trace_headers

        stream = io.StringIO()
        headers = {"traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01"}
        with contextlib.redirect_stderr(stream):
            result = asyncio.run(runtime.wrap_add_request(original)(None, "request-1", {}, headers))
        self.assertIs(result, headers)
        self.assertIn(
            f"[timing-probe] engine_request request_id=request-1 trace_id={TRACE_ID} sampled=true",
            stream.getvalue(),
        )

    def test_trace_headers_survive_when_vllm_tracing_is_disabled(self):
        runtime = Runtime(Config(sample_rate=1, diagnostic_log=True), self.collector)
        original = Mock(side_effect=AssertionError("valid trace context must not be discarded"))
        wrapped = runtime.wrap_trace_headers(original)
        headers = {
            "traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01",
            "tracestate": "vendor=value",
            "authorization": "secret",
        }
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            extracted = asyncio.run(wrapped(None, headers))
        self.assertEqual(
            extracted,
            {"traceparent": headers["traceparent"], "tracestate": "vendor=value"},
        )
        original.assert_not_called()
        self.assertIn(f"[timing-probe] trace_headers trace_id={TRACE_ID} sampled=true", stream.getvalue())

    def test_invalid_trace_headers_keep_vllm_behavior(self):
        async def original(owner, headers):
            return None

        wrapped = self.runtime.wrap_trace_headers(original)
        self.assertIsNone(asyncio.run(wrapped(None, {"traceparent": "invalid"})))

    def test_old_and_new_base_serving_trace_header_methods_are_patched(self):
        for module_name in (
            "vllm.entrypoints.openai.engine.serving",
            "vllm.entrypoints.serve.engine.serving",
            "vllm.entrypoints.generate.base.serving",
        ):
            with self.subTest(module=module_name):

                class BaseServing:
                    async def _get_trace_headers(self, headers):
                        return None

                BaseServing.__module__ = module_name
                module = SimpleNamespace(__name__=module_name, BaseServing=BaseServing)
                runtime = Runtime(Config(sample_rate=1), self.collector)
                runtime.patch_module(module)
                headers = {"traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01"}
                self.assertEqual(asyncio.run(BaseServing()._get_trace_headers(headers)), headers)

    def test_decoder_rejects_malformed_packets(self):
        packet = Packet(tuple(self.carrier()["contexts"]), [Record("stage", 1, 2)])
        data = json.dumps(asdict(packet)).encode()
        self.assertEqual(decode_packet(data).records[0].end_ns, 2)
        for invalid in (b"garbage", b"x" * 8193, b'{"contexts":[],"records":[]}'):
            with self.assertRaises((ValueError, KeyError)):
                decode_packet(invalid)

    def test_real_local_transport_and_collector_decode(self):
        received = threading.Event()

        class Sink:
            packet = None

            def submit(self, packet):
                self.packet = packet
                received.set()

            def close(self):
                pass

        sink = Sink()
        collector = Collector(0, sink)
        thread = threading.Thread(target=collector.run)
        thread.start()
        emitter = DatagramEmitter(collector.socket.getsockname()[1])
        try:
            emitter.submit(Packet(tuple(self.carrier()["contexts"]), [Record("stage", 1, 2)]))
            self.assertTrue(received.wait(timeout=5))
            self.assertEqual(sink.packet.contexts[0]["trace_id"], TRACE_ID)
            self.assertEqual(sink.packet.metadata["source_pid"], os.getpid())
        finally:
            collector.stop()
            thread.join(timeout=5)
            if emitter.socket is not None:
                emitter.socket.close()
        self.assertFalse(thread.is_alive())

    def test_preparer_exits_and_missing_observer_files_are_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = prepare(Path(directory) / "bundle", asdict(Config()))
            self.assertFalse((bundle / "trace_export.py").exists())
            env = os.environ.copy()
            env["PYTHONPATH"] = str(bundle)
            script = (
                "import sys; assert sys.getprofile() is None; "
                "assert 'langfuse' not in sys.modules; assert 'trace_export' not in sys.modules; print('service works')"
            )

            def check():
                result = subprocess.run(
                    [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=15
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("service works", result.stdout)

            check()
            (bundle / "config.json").write_text("invalid", encoding="utf-8")
            check()
            (bundle / "timing_probe.py").unlink()
            check()
            with self.assertRaises(FileExistsError):
                prepare(bundle, asdict(Config()))

    def test_real_import_hook_failure_does_not_fail_module_import(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = prepare(Path(directory) / "bundle", asdict(Config(sample_rate=1)))
            package = Path(directory) / "vllm" / "v1" / "engine"
            package.mkdir(parents=True)
            for root in (package, package.parent, package.parent.parent):
                (root / "__init__.py").touch()
            # Incompatible module: hook cannot find AsyncLLM, but the import remains usable.
            (package / "async_llm.py").write_text("BUSINESS_VALUE = 42\n", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(bundle) + os.pathsep + directory
            script = "from vllm.v1.engine.async_llm import BUSINESS_VALUE; assert BUSINESS_VALUE == 42"
            result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_import_hook_installs_without_sdk_or_extra_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = prepare(Path(directory) / "bundle", asdict(Config(sample_rate=1)))
            package = Path(directory) / "vllm" / "v1" / "engine"
            package.mkdir(parents=True)
            for root in (package, package.parent, package.parent.parent):
                (root / "__init__.py").touch()
            (package / "async_llm.py").write_text(
                "class AsyncLLM:\n"
                "    async def add_request(self, value, trace_headers=None):\n"
                "        return value, trace_headers\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(bundle) + os.pathsep + directory
            script = (
                "import asyncio,sys,threading; "
                "from vllm.v1.engine.async_llm import AsyncLLM; "
                "assert AsyncLLM.add_request._runtime_timing_wrapped; "
                "assert asyncio.run(AsyncLLM().add_request(7)) == (7,None); "
                "assert 'langfuse' not in sys.modules; assert len(threading.enumerate()) == 1"
            )
            result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_collector_queue_bound_and_failure_isolation(self):
        entered, release = threading.Event(), threading.Event()

        class SlowSink:
            def __init__(self, config):
                pass

            def emit(self, packet):
                entered.set()
                release.wait(timeout=5)
                raise RuntimeError("collector-side SDK failure")

            def close(self):
                pass

        exporter = BufferedExporter(CollectorConfig(queue_size=1), SlowSink)
        packet = Packet(({},), [])
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                exporter.submit(packet)
                self.assertTrue(entered.wait(timeout=5))
                for _ in range(5):
                    exporter.submit(packet)
                self.assertGreater(exporter.dropped, 0)
            finally:
                release.set()
                exporter.close()
        self.assertGreater(exporter.failed, 0)

    @unittest.skipUnless(importlib.util.find_spec("langfuse"), "optional Langfuse SDK is not installed")
    def test_real_sdk_timestamps_and_trace_ids_without_network(self):
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from requests import Response

        response = Response()
        response.status_code, response._content = 200, b""
        env = {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test-runtime",
            "LANGFUSE_SECRET_KEY": "sk-lf-test-runtime",
            "LANGFUSE_BASE_URL": "http://127.0.0.1:9",
            "LANGFUSE_TRACING_ENABLED": "true",
        }
        with patch.dict(os.environ, env), patch("requests.Session.post", return_value=response) as post:
            sink = LangfuseSink(CollectorConfig())
            memory = InMemorySpanExporter()
            sink.provider.add_span_processor(SimpleSpanProcessor(memory))
            start = time.time_ns() - 1_000_000_000
            records = [
                Record("request", start, start + 200, span_id="a" * 16),
                Record("execute", start + 50, start + 100, parent=0),
            ]
            sink.emit(Packet(tuple(self.carrier()["contexts"]), records, {"source_pid": 123}))
            sink.close()
            self.assertTrue(post.called)
            spans = memory.get_finished_spans()
            self.assertEqual(len(spans), 2)
            self.assertEqual(format(spans[0].context.trace_id, "032x"), TRACE_ID)
            self.assertEqual(format(spans[0].context.span_id, "016x"), "a" * 16)
            self.assertEqual(format(spans[0].parent.span_id, "016x"), PARENT_ID)
            self.assertEqual(spans[1].parent.span_id, spans[0].context.span_id)
            self.assertEqual(spans[1].start_time, start + 50)
            self.assertEqual(spans[1].end_time, start + 100)
            self.assertEqual(json.loads(spans[1].attributes["langfuse.observation.metadata"])["pid"], 123)

    def test_service_survives_missing_and_killed_collector(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            bundle = prepare(
                Path(directory) / "bundle", asdict(Config(sample_rate=1, every_n_steps=1, collector_port=port))
            )
            collector_script = Path(directory) / "collector_stub.py"
            collector_script.write_text(
                "import socket,sys\n"
                "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
                f"s.bind(('127.0.0.1',{port}))\n"
                "print('ready',flush=True)\n"
                "while True: s.recvfrom(9000)\n",
                encoding="utf-8",
            )
            service_script = Path(directory) / "service_stub.py"
            service_script.write_text(
                "import sys,time\n"
                "from types import SimpleNamespace as NS\n"
                "from timing_probe import Runtime,Config\n"
                f"r=Runtime(Config(sample_rate=1,every_n_steps=1,collector_port={port}))\n"
                f"req=NS(trace_headers={{'traceparent':'00-{TRACE_ID}-{PARENT_ID}-01'}},num_output_tokens=0)\n"
                "s=NS(requests={'a':req}); owner=NS()\n"
                "schedule=r.wrap_schedule(lambda _:NS(num_scheduled_tokens={'a':1}))\n"
                "execute=r.wrap_runner(lambda _,output:7)\n"
                "for i in range(12):\n"
                "    assert execute(owner,schedule(s))==7\n"
                "    print('ok',flush=True)\n"
                "    time.sleep(0.02)\n"
                "assert 'langfuse' not in sys.modules\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(bundle)
            # Missing collector.
            result = subprocess.run([sys.executable, str(service_script)], env=env, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            # Kill an independently launched collector while the service is running.
            collector = subprocess.Popen([sys.executable, str(collector_script)], stdout=subprocess.PIPE, text=True)
            service = None
            try:
                self.assertEqual(collector.stdout.readline().strip(), "ready")
                service = subprocess.Popen(
                    [sys.executable, str(service_script)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(service.stdout.readline().strip(), "ok")
                collector.kill()
                collector.wait(timeout=5)
                output, error = service.communicate(timeout=15)
                self.assertEqual(service.returncode, 0, error)
                self.assertEqual(output.count("ok"), 11)
            finally:
                if collector.poll() is None:
                    collector.kill()
                    collector.wait(timeout=5)
                collector.stdout.close()
                if service is not None and service.poll() is None:
                    service.kill()
                    service.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
