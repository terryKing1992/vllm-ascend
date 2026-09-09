"""API contract and process integration tests; no installed vLLM/NPU required.

Contracts checked against upstream tags v0.23.0, v0.24.0, v0.25.0, v0.26.0:
vllm/entrypoints/{openai/engine,generate/base}/serving.py,
vllm/entrypoints/launcher.py, vllm/v1/engine/async_llm.py,
vllm/v1/core/sched/output.py. These fixtures emulate vLLM, not device execution.
"""

import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "runtime_timing"
sys.path.insert(0, str(TOOL_DIR))

from collector import Collector  # noqa: E402
from run import prepare  # noqa: E402
from timing_probe import Config, RequestMiddleware, Runtime  # noqa: E402
from trace_export import CollectorConfig, JsonLogSink  # noqa: E402

TRACE_ID = "12345678901234567890123456789012"
PARENT_ID = "1234567890123456"
TRACEPARENT = f"00-{TRACE_ID}-{PARENT_ID}-01"
VERSION_CONTRACTS = (
    ("0.23.0", "openai.engine.serving", "OpenAIServing", "model_runner_v1", False),
    ("0.24.0", "openai.engine.serving", "OpenAIServing", "model_runner_v1", True),
    ("0.25.0", "generate.base.serving", "GenerateBaseServing", "v2.model_runner", False),
    ("0.26.0", "generate.base.serving", "GenerateBaseServing", "v2.model_runner", True),
)


def write_module(root, name, source):
    path = root.joinpath(*name.split(".")).with_suffix(".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in path.parents:
        if parent == root:
            break
        (parent / "__init__.py").touch()
    path.write_text(textwrap.dedent(source), encoding="utf-8")


class TestCompatibility(unittest.TestCase):
    def setUp(self):
        self.exporter = Mock()
        self.runtime = Runtime(Config(sample_rate=1, every_n_steps=1, diagnostic_log=True), self.exporter)

    def test_signature_drift_and_missing_class_do_not_disable_other_hooks(self):
        class AsyncLLM:
            def add_request(self, renamed_input):
                return renamed_input

        original = AsyncLLM.add_request
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.runtime.safe_patch_module(SimpleNamespace(__name__="vllm.v1.engine.async_llm", AsyncLLM=AsyncLLM))
            self.runtime.safe_patch_module(SimpleNamespace(__name__="vllm_ascend.worker.model_runner_v1"))
            self.runtime.safe_patch_module(SimpleNamespace(__name__="vllm.v1.engine.async_llm"))
        self.assertIs(AsyncLLM.add_request, original)
        self.assertEqual(AsyncLLM().add_request(7), 7)
        self.assertFalse(self.runtime.disabled)
        self.assertIn("reason=unsupported_signature", stream.getvalue())
        self.assertIn("reason=missing", stream.getvalue())
        self.assertEqual(self.runtime.wrap_stage(lambda: 9, "stage")(), 9)

    def test_failed_factory_does_not_prevent_next_patch(self):
        owner = SimpleNamespace(first=lambda: 1, second=lambda: 2)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(self.runtime.patch_method(owner, "first", Mock(side_effect=ValueError())))
            self.assertTrue(
                self.runtime.patch_method(owner, "second", lambda fn: self.runtime.wrap_stage(fn, "second"))
            )
        self.assertFalse(self.runtime.disabled)
        self.assertEqual(owner.first(), 1)
        self.assertEqual(owner.second(), 2)

    def test_middleware_install_is_idempotent_and_failure_is_local(self):
        app = SimpleNamespace(add_middleware=Mock())
        broken = SimpleNamespace(add_middleware=Mock(side_effect=RuntimeError("app already started")))
        with contextlib.redirect_stderr(io.StringIO()):
            self.runtime.attach_middleware(broken)
            self.runtime.attach_middleware(app)
            self.runtime.attach_middleware(app)
        app.add_middleware.assert_called_once_with(RequestMiddleware, runtime=self.runtime)
        self.assertFalse(self.runtime.disabled)

    def test_embedded_request_headers_and_prompt_are_preserved(self):
        async def add_request(owner, request_id, prompt, trace_headers=None):
            return prompt, trace_headers

        prompt = SimpleNamespace(trace_headers={"traceparent": TRACEPARENT, "tracestate": "vendor=value"})
        wrapped = self.runtime.wrap_add_request(add_request)
        with contextlib.redirect_stderr(io.StringIO()):
            result, headers = asyncio.run(wrapped(None, "a", prompt))
            self.assertIs(result, prompt)
            self.assertIsNone(headers)
            active = f"00-{TRACE_ID}-{'a' * 16}-01"
            token = self.runtime.request.set({"traceparent": active})
            try:
                result, headers = asyncio.run(wrapped(None, "a", prompt))
            finally:
                self.runtime.request.reset(token)
        self.assertIsNot(result, prompt)
        self.assertEqual(prompt.trace_headers["traceparent"], TRACEPARENT)
        self.assertEqual(headers, {"traceparent": active, "tracestate": "vendor=value"})
        self.assertEqual(result.trace_headers, headers)

    def test_disabled_header_wrapper_preserves_original_behavior(self):
        calls = []

        async def original(owner, headers):
            calls.append(headers)
            return None

        self.runtime.disabled = True
        headers = {"traceparent": TRACEPARENT}
        self.assertIsNone(asyncio.run(self.runtime.wrap_trace_headers(original)(None, headers)))
        self.assertEqual(calls, [headers])

    def test_header_parameter_binding_and_generated_trace(self):
        async def original(owner, option, *, headers):
            return None

        wrapped = self.runtime.wrap_trace_headers(original)

        async def app(scope, receive, send):
            result = await wrapped(None, True, headers={})
            self.assertEqual(result["traceparent"], self.runtime.request.get()["traceparent"])

        with contextlib.redirect_stderr(io.StringIO()):
            asyncio.run(RequestMiddleware(app, self.runtime)({"type": "http", "path": "/v1/completions"}, None, None))
        packet = self.exporter.submit.call_args.args[0]
        self.assertEqual(len(packet.contexts[0]["trace_id"]), 32)
        self.assertIsNone(packet.contexts[0]["parent_span_id"])

    def test_build_and_launcher_preserve_business_results_on_observer_failure(self):
        app = object()
        calls = Mock(return_value=app)

        def build():
            return calls()

        module = SimpleNamespace(__name__="vllm.entrypoints.openai.api_server", build_app=build)
        with contextlib.redirect_stderr(io.StringIO()):
            self.runtime.patch_module(module)
            with patch.object(self.runtime, "attach_middleware", side_effect=RuntimeError("observer failure")):
                self.assertIs(module.build_app(), app)
        calls.assert_called_once()

        async def original(app, sock):
            raise ValueError("business exception")

        self.runtime.disabled = False
        with (
            patch.object(self.runtime, "attach_middleware", side_effect=RuntimeError("observer failure")),
            self.assertRaisesRegex(ValueError, "business exception"),
        ):
            asyncio.run(self.runtime.wrap_serve_http(original)(app, None))

    def test_slot_output_keeps_scheduler_reporting(self):
        @dataclass(slots=True)
        class Output:
            num_scheduled_tokens: dict

        output = Output({"a": 1})
        scheduler = SimpleNamespace(
            requests={"a": SimpleNamespace(trace_headers={"traceparent": TRACEPARENT}, num_output_tokens=0)}
        )
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIs(self.runtime.wrap_schedule(lambda owner: output)(scheduler), output)
        self.assertFalse(self.runtime.disabled)
        self.assertEqual(self.exporter.submit.call_args.args[0].records[0].name, "scheduler.schedule")

    def test_root_path_routes_and_unsampled_context(self):
        async def app(scope, receive, send):
            self.assertIsNotNone(self.runtime.request.get())
            return 12

        middleware = RequestMiddleware(app, self.runtime)
        for path in ("/v1/completions", "/v1/chat/completions", "/v1/responses", "/v1/embeddings"):
            with self.subTest(path=path), contextlib.redirect_stderr(io.StringIO()):
                scope = {
                    "type": "http",
                    "root_path": "/proxy",
                    "path": "/proxy" + path,
                    "headers": [(b"traceparent", TRACEPARENT[:-2].encode() + b"00")],
                }
                self.assertEqual(asyncio.run(middleware(scope, None, None)), 12)
                self.assertIsNone(self.runtime.request.get())
        self.exporter.submit.assert_not_called()
        self.assertIsNone(middleware.begin({"type": "http", "path": "/health"}))

    def test_runtime_failure_explains_why_reporting_stopped(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.runtime.safe(Mock(side_effect=ValueError("must not log sensitive exception text")))
        self.assertTrue(self.runtime.disabled)
        self.assertIn("disabled operation=", stream.getvalue())
        self.assertIn("error=ValueError", stream.getvalue())
        self.assertNotIn("sensitive", stream.getvalue())

    def test_version_contracts_through_import_hook_worker_process_udp_and_json(self):
        for version, serving_module, serving_class, runner_module, module_launch in VERSION_CONTRACTS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                self.check_pipeline(Path(directory), serving_module, serving_class, runner_module, module_launch)

    def check_pipeline(self, root, serving_module, serving_class, runner_module, module_launch):
        # A real socket collector, a separately injected API process and a
        # separately injected worker. vLLM itself is represented by fixtures.
        stream = io.StringIO()
        sink = JsonLogSink(CollectorConfig(), stream)
        received = threading.Event()

        def submit(packet):
            sink.emit(packet)
            if packet.records[0].name == "vllm.request":
                received.set()

        collector = Collector(0, SimpleNamespace(submit=submit, close=lambda: None))
        bundle = prepare(
            root / "bundle",
            asdict(
                Config(
                    sample_rate=1,
                    every_n_steps=1,
                    collector_port=collector.socket.getsockname()[1],
                    diagnostic_log=True,
                )
            ),
        )
        package = root / "packages"
        write_module(
            package,
            "vllm.entrypoints." + serving_module,
            f"""
            class {serving_class}:
                async def _get_trace_headers(self, headers):
                    return None  # upstream tracing disabled
        """,
        )
        write_module(
            package,
            "vllm.v1.engine.async_llm",
            """
            class AsyncLLM:
                async def add_request(self, request_id, prompt, params, arrival_time=None,
                                      lora_request=None, tokenization_kwargs=None, trace_headers=None,
                                      priority=0, data_parallel_rank=None, prompt_text=None,
                                      reasoning_ended=None, reasoning_parser_kwargs=None):
                    return trace_headers
        """,
        )
        write_module(
            package,
            "vllm.v1.core.sched.scheduler",
            """
            from dataclasses import dataclass
            @dataclass
            class SchedulerOutput:
                num_scheduled_tokens: dict
            class Scheduler:
                def __init__(self, request):
                    self.requests = {'request-1': request}
                def schedule(self):
                    return SchedulerOutput({'request-1': 3})
        """,
        )
        write_module(
            package,
            "vllm_ascend.worker." + runner_module,
            """
            class NPUModelRunner:
                def _prepare_inputs(self):
                    return 7
                def execute_model(self, scheduler_output, intermediate_tensors=None):
                    assert self._prepare_inputs() == 7
                    return None
                def sample_tokens(self):
                    return 42
        """,
        )
        write_module(
            package,
            "vllm.entrypoints.launcher",
            """
            async def serve_http(app, sock, enable_ssl_refresh=False, **uvicorn_kwargs):
                return await app.run()
        """,
        )
        # Exercise both normal imports (build_app + launcher) and -m startup
        # (launcher only). Installing via both hooks must not duplicate spans.
        write_module(
            package,
            "vllm.entrypoints.openai.api_server",
            f"""
            import asyncio, pickle, subprocess, sys
            from types import SimpleNamespace
            from vllm.entrypoints.launcher import serve_http
            from vllm.entrypoints.{serving_module} import {serving_class}
            from vllm.v1.engine.async_llm import AsyncLLM
            from vllm.v1.core.sched.scheduler import Scheduler
            class App:
                middleware = None
                def add_middleware(self, cls, **kwargs):
                    assert self.middleware is None
                    self.middleware = cls(self.endpoint, **kwargs)
                async def run(self):
                    assert self.middleware is not None
                    return await self.middleware({{'type': 'http', 'path': '/proxy/v1/chat/completions',
                        'root_path': '/proxy', 'headers': [(b'traceparent', b'{TRACEPARENT}')] }}, None, None)
                async def endpoint(self, scope, receive, send):
                    headers = await {serving_class}()._get_trace_headers({{'traceparent': '{TRACEPARENT}'}})
                    headers = await AsyncLLM().add_request('request-1', {{}}, None, trace_headers=headers)
                    output = Scheduler(SimpleNamespace(trace_headers=headers, num_output_tokens=0)).schedule()
                    code = ('import pickle,sys; from vllm_ascend.worker.{runner_module} import NPUModelRunner; '
                            'r=NPUModelRunner(); output=pickle.loads(sys.stdin.buffer.read()); '
                            'assert r.execute_model(output) is None; '
                            'assert r.sample_tokens()==42; assert sys.getprofile() is None; '
                            'assert "langfuse" not in sys.modules')
                    result = subprocess.run([sys.executable, '-c', code], input=pickle.dumps(output),
                                            capture_output=True, timeout=15)
                    sys.stderr.write(result.stderr.decode())
                    assert result.returncode == 0, result.stderr
                    return 'response unchanged'
            def build_app():
                return App()
            def main():
                assert asyncio.run(serve_http(build_app(), None)) == 'response unchanged'
            if __name__ == '__main__':
                main()
        """,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(bundle) + os.pathsep + str(package)
        thread = threading.Thread(target=collector.run)
        thread.start()
        try:
            command = (
                ["-m", "vllm.entrypoints.openai.api_server"]
                if module_launch
                else ["-c", "from vllm.entrypoints.openai.api_server import main; main()"]
            )
            result = subprocess.run(
                [sys.executable, *command],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(received.wait(5), result.stderr)
        finally:
            collector.stop()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        by_name = {row["name"]: row for row in rows}
        self.assertEqual(
            set(by_name),
            {
                "vllm.request",
                "scheduler.schedule",
                "runner.execute_model",
                "runner._prepare_inputs",
                "runner.sample_tokens",
            },
        )
        self.assertEqual(len(rows), len(by_name))
        self.assertEqual({row["trace_id"] for row in rows}, {TRACE_ID})
        request = by_name["vllm.request"]
        self.assertEqual(request["parent_span_id"], PARENT_ID)
        for name in ("scheduler.schedule", "runner.execute_model", "runner.sample_tokens"):
            self.assertEqual(by_name[name]["parent_span_id"], request["span_id"])
            self.assertEqual(by_name[name]["request_id"], "request-1")
        self.assertEqual(
            by_name["runner._prepare_inputs"]["parent_span_id"], by_name["runner.execute_model"]["span_id"]
        )
        self.assertNotEqual(by_name["runner.execute_model"]["pid"], request["pid"])
        self.assertEqual(result.stderr.count("middleware_installed"), 1)
        self.assertIn("engine_request request_id=request-1", result.stderr)
        self.assertNotIn("disabled operation=", result.stderr)


if __name__ == "__main__":
    unittest.main()
