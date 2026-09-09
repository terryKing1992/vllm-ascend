"""Fail-open selective instrumentation. Worker dependencies: Python standard library only."""

import copy
import functools
import importlib.abc
import importlib.metadata
import inspect
import os
import re
import sys
import time
import uuid
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass

from trace_transport import (
    DEFAULT_DIAGNOSTIC_EVERY,
    MAX_RECORDS,
    MAX_REQUESTS,
    DatagramEmitter,
    Packet,
    Record,
    diagnostic_due,
)

TRACE_PACKET_ATTR = "_langfuse_runtime_packet"
PENDING_PACKET_ATTR = "_langfuse_runtime_pending"
STEP_ATTR = "_langfuse_runtime_step"
REQUEST_CONTEXT_ATTR = "_langfuse_runtime_context"
ENQUEUE_CLOCK_ATTR = "_langfuse_runtime_enqueue_clock"
MIDDLEWARE_ATTR = "_langfuse_runtime_middleware"
HTTP_PATHS = frozenset(("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/embeddings"))
STAGE_METHODS = (
    "_update_states",
    "_prepare_inputs",
    "prepare_inputs",
    "_build_attention_metadata",
    "_model_forward",
    "_sample",
    "propose_draft_token_ids",
    "postprocess",
    "postprocess_sampled",
)
SCHEDULER_MODULES = (
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.core.sched.async_scheduler",
    "vllm_ascend.core.recompute_scheduler",
    "vllm_ascend.core.scheduler_profiling_chunk",
    "vllm_ascend.core.batch_job_aware_scheduler",
    "vllm_ascend.patch.platform.patch_balance_schedule",
)
RUNNER_MODULES = ("vllm_ascend.worker.model_runner_v1", "vllm_ascend.worker.v2.model_runner")
TRACE_HEADER_MODULES = (
    "vllm.entrypoints.openai.engine.serving",
    "vllm.entrypoints.serve.engine.serving",
    "vllm.entrypoints.generate.base.serving",
    "vllm.entrypoints.pooling.base.serving",
)


@dataclass(frozen=True)
class Config:
    sample_rate: float = 0.01
    every_n_steps: int = 10
    max_records: int = 16
    max_requests: int = 4
    collector_port: int = 18765
    diagnostic_log: bool = False
    diagnostic_every: int = DEFAULT_DIAGNOSTIC_EVERY
    detail: str = "core"


def parse_traceparent(value):
    if not isinstance(value, str) or not re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", value):
        return None
    _, trace_id, parent_id, flags = value.split("-")
    if int(trace_id, 16) == 0 or int(parent_id, 16) == 0:
        return None
    return {"trace_id": trace_id, "parent_span_id": parent_id, "sampled": bool(int(flags, 16) & 1)}


def selected(context, rate):
    return bool(context and context["sampled"] and int(context["trace_id"], 16) < rate * (1 << 128))


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
    except Exception:
        return "unknown"


class Runtime:
    def __init__(self, config, exporter=None):
        self.config = config
        self.exporter = (
            exporter
            if exporter is not None
            else DatagramEmitter(config.collector_port, config.diagnostic_log, config.diagnostic_every)
        )
        self.request = ContextVar("runtime_request", default=None)
        self.stage = ContextVar("runtime_stage", default=None)
        self.scheduling = ContextVar("runtime_scheduling", default=False)
        self.disabled = False
        self.failures = 0
        self.diagnostics = set()
        self.event_counts = {}

    def log(self, message):
        if self.config.diagnostic_log:
            with suppress(Exception):
                print(f"[timing-probe] {message}", file=sys.stderr, flush=True)

    def log_once(self, key, message):
        if self.config.diagnostic_log and key not in self.diagnostics:
            self.diagnostics.add(key)
            self.log(f"{message} pid={os.getpid()}")

    def log_event(self, key, message):
        if not self.config.diagnostic_log:
            return
        count = self.event_counts.get(key, 0) + 1
        self.event_counts[key] = count
        if diagnostic_due(count, self.config.diagnostic_every):
            self.log(f"{message} event_count={count} pid={os.getpid()}")

    def safe(self, operation, *args):
        """Only instrumentation enters here; never catch/retry business execution."""
        try:
            return operation(*args)
        except Exception as error:
            self.disabled = True
            self.failures += 1
            self.log(f"disabled operation={getattr(operation, '__name__', 'unknown')} error={type(error).__name__}")
            return None

    def safe_patch_module(self, module):
        # An incompatible optional interface must not disable other capabilities.
        try:
            self.patch_module(module)
        except Exception as error:
            self.log(f"patch_skipped module={module.__name__} error={type(error).__name__}")

    def submit(self, packet):
        if not self.disabled:
            self.safe(self.exporter.submit, packet)

    def begin_schedule(self):
        if self.scheduling.get():
            return None
        origin, clock = time.time_ns(), time.perf_counter_ns()
        return self.scheduling.set(True), origin, clock

    def mark_scheduler_enqueue(self, signature, scheduler, args, kwargs):
        request = signature.bind(scheduler, *args, **kwargs).arguments["request"]
        if scheduler.requests.get(request.request_id) is request and not hasattr(request, ENQUEUE_CLOCK_ATTR):
            setattr(request, ENQUEUE_CLOCK_ATTR, time.perf_counter_ns())

    def wrap_scheduler_add_request(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def add_request(scheduler, *args, **kwargs):
            result = original(scheduler, *args, **kwargs)
            if not self.disabled:
                self.safe(self.mark_scheduler_enqueue, signature, scheduler, args, kwargs)
            return result

        return add_request

    def finish_schedule(self, state, scheduler, output):
        token, origin, clock = state
        try:
            if output is None or self.disabled:
                return
            end = origin + time.perf_counter_ns() - clock
            contexts, omitted = [], 0
            for req_id, num_tokens in output.num_scheduled_tokens.items():
                if num_tokens <= 0:
                    continue
                request = scheduler.requests.get(req_id)
                if request is None:
                    continue
                if not hasattr(request, REQUEST_CONTEXT_ATTR):
                    context = parse_traceparent((getattr(request, "trace_headers", None) or {}).get("traceparent"))
                    setattr(
                        request, REQUEST_CONTEXT_ATTR, context if selected(context, self.config.sample_rate) else None
                    )
                context = getattr(request, REQUEST_CONTEXT_ATTR)
                if context is None:
                    continue
                step = getattr(request, STEP_ATTR, 0)
                setattr(request, STEP_ATTR, step + 1)
                if step % self.config.every_n_steps:
                    continue
                if len(contexts) >= self.config.max_requests:
                    omitted += 1
                    continue
                request_metadata = {
                    "step": step,
                    "scheduled_tokens": num_tokens,
                    "phase": "prefill" if request.num_output_tokens == 0 else "decode",
                }
                enqueued_at = getattr(request, ENQUEUE_CLOCK_ATTR, None)
                if step == 0 and enqueued_at is not None and clock >= enqueued_at:
                    request_metadata["queue_to_first_schedule_ms"] = (clock - enqueued_at) / 1_000_000
                contexts.append(
                    dict(
                        context,
                        request_id=req_id[:256],
                        metadata=request_metadata,
                    )
                )
            metadata = {
                "batch_id": uuid.uuid4().hex if contexts else "",
                "batch_size": len(output.num_scheduled_tokens),
                "shared_batch_time": True,
                "omitted_sampled_requests": omitted,
                "every_n_steps": self.config.every_n_steps,
                "sample_rate": self.config.sample_rate,
            }
            # Optional built-in types only: uninstrumented workers can unpickle this.
            try:
                setattr(output, TRACE_PACKET_ATTR, {"contexts": contexts, "metadata": metadata})
            except (AttributeError, TypeError):
                self.log_once("carrier_unsupported", "carrier_unsupported worker_association=unavailable")
            self.log_once(
                "scheduler_active",
                f"scheduler_active batch_size={metadata['batch_size']} contexts={len(contexts)}",
            )
            if contexts:
                self.submit(Packet(tuple(contexts), [Record("scheduler.schedule", origin, end)], metadata))
        finally:
            self.scheduling.reset(token)

    def wrap_schedule(self, original):
        @functools.wraps(original)
        def schedule(scheduler, *args, **kwargs):
            if self.disabled:
                return original(scheduler, *args, **kwargs)
            state = self.safe(self.begin_schedule)
            output = None
            try:
                output = original(scheduler, *args, **kwargs)
                return output
            finally:
                if state is not None:
                    self.safe(self.finish_schedule, state, scheduler, output)

        return schedule

    def begin_stage(self, name):
        active = self.stage.get()
        if active is None:
            return None
        packet, parent, origin, clock = active
        if len(packet.records) >= self.config.max_records:
            packet.metadata["truncated_stage_calls"] = packet.metadata.get("truncated_stage_calls", 0) + 1
            return None
        record = Record(name, origin + time.perf_counter_ns() - clock, parent=parent)
        index = len(packet.records)
        packet.records.append(record)
        token = self.stage.set((packet, index, origin, clock))
        return token, packet, record, origin, clock

    def finish_stage(self, state, error, submit=False):
        token, packet, record, origin, clock = state
        try:
            if not self.disabled:
                record.end_ns = origin + time.perf_counter_ns() - clock
                record.error = error
                if submit:
                    self.submit(packet)
        finally:
            self.stage.reset(token)

    def wrap_stage(self, original, name):
        @functools.wraps(original)
        def stage(*args, **kwargs):
            if self.disabled:
                return original(*args, **kwargs)
            state = self.safe(self.begin_stage, name)
            error_name = None
            try:
                return original(*args, **kwargs)
            except BaseException as error:
                error_name = type(error).__name__
                raise
            finally:
                if state is not None:
                    self.safe(self.finish_stage, state, error_name)

        return stage

    def begin_runner(self, runner, sampling, args, kwargs):
        if sampling:
            carrier = getattr(runner, PENDING_PACKET_ATTR, None)
            setattr(runner, PENDING_PACKET_ATTR, None)
        else:
            output = kwargs.get("scheduler_output", args[0] if args else None)
            carrier = getattr(output, TRACE_PACKET_ATTR, None)
            setattr(runner, PENDING_PACKET_ATTR, carrier)
            self.log_once("runner_active", f"runner_active carrier={str(bool(carrier)).lower()}")
        if not isinstance(carrier, dict) or not carrier.get("contexts"):
            return None
        packet = Packet(tuple(carrier["contexts"][: self.config.max_requests]), [], dict(carrier["metadata"]))
        rank = getattr(getattr(runner, "parallel_config", None), "rank", None)
        if isinstance(rank, int):
            packet.metadata["rank"] = rank
        origin, clock = time.time_ns(), time.perf_counter_ns()
        record = Record("runner.sample_tokens" if sampling else "runner.execute_model", origin)
        packet.records.append(record)
        token = self.stage.set((packet, 0, origin, clock))
        return token, packet, record, origin, clock

    def end_runner(self, runner, sampling, result, state, error):
        try:
            if error is not None or (not sampling and result is not None):
                setattr(runner, PENDING_PACKET_ATTR, None)
        finally:
            if state is not None:
                self.finish_stage(state, error, submit=True)

    def wrap_runner(self, original, sampling=False):
        @functools.wraps(original)
        def run(runner, *args, **kwargs):
            if self.disabled:
                return original(runner, *args, **kwargs)
            state = self.safe(self.begin_runner, runner, sampling, args, kwargs)
            result, error_name = None, None
            try:
                result = original(runner, *args, **kwargs)
                return result
            except BaseException as error:
                error_name = type(error).__name__
                raise
            finally:
                self.safe(self.end_runner, runner, sampling, result, state, error_name)

        return run

    def request_arguments(self, signature, args, kwargs):
        bound = signature.bind(*args, **kwargs)
        prompt = bound.arguments.get("prompt")
        # EngineCoreRequest inputs carry their own headers; vLLM may ignore the
        # separate trace_headers argument for this input form.
        headers = dict(getattr(prompt, "trace_headers", None) or bound.arguments.get("trace_headers") or {})
        context = self.request.get()
        if context is not None:
            metadata = context.get("metadata")
            start_clock = context.get("start_clock")
            if metadata is not None and start_clock is not None and "api_to_engine_ms" not in metadata:
                metadata["api_to_engine_ms"] = (time.perf_counter_ns() - start_clock) / 1_000_000
            headers["traceparent"] = context["traceparent"]
            if "trace_headers" in signature.parameters:
                bound.arguments["trace_headers"] = headers
            if hasattr(prompt, "trace_headers"):
                prompt = copy.copy(prompt)
                prompt.trace_headers = headers
                bound.arguments["prompt"] = prompt
        trace_context = parse_traceparent(headers.get("traceparent"))
        if trace_context is not None:
            request_id = str(bound.arguments.get("request_id", ""))[:256]
            sampled = selected(trace_context, self.config.sample_rate)
            self.log_event(
                "engine_request",
                f"engine_request request_id={request_id} trace_id={trace_context['trace_id']} "
                f"sampled={str(sampled).lower()}",
            )
        else:
            self.log_once("engine_missing_trace", "engine_request trace_context=missing check=request_middleware")
        return bound.args, bound.kwargs

    def wrap_add_request(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        async def add_request(*args, **kwargs):
            if not self.disabled:
                replacement = self.safe(self.request_arguments, signature, args, kwargs)
                if replacement is not None:
                    args, kwargs = replacement
            return await original(*args, **kwargs)

        return add_request

    def wrap_trace_headers(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        async def trace_headers(*args, **kwargs):
            if self.disabled:
                return await original(*args, **kwargs)
            try:
                headers = signature.bind(*args, **kwargs).arguments["headers"]
                active = self.request.get()
                traceparent = active["traceparent"] if active is not None else headers.get("traceparent")
                context = parse_traceparent(traceparent)
            except Exception:
                context = None
            if context is None:
                return await original(*args, **kwargs)
            extracted = {"traceparent": traceparent}
            with suppress(Exception):
                tracestate = headers.get("tracestate")
                if tracestate:
                    extracted["tracestate"] = tracestate
            sampled = selected(context, self.config.sample_rate)
            self.log_event(
                "trace_headers", f"trace_headers trace_id={context['trace_id']} sampled={str(sampled).lower()}"
            )
            return extracted

        return trace_headers

    def patch_method(self, owner, name, factory, required=(), asynchronous=False):
        target = f"{getattr(owner, '__name__', 'unknown')}.{name}"
        try:
            original = getattr(owner, name, None)
            if original is None:
                self.log(f"patch_skipped target={target} reason=missing")
                return False
            if getattr(original, "_runtime_timing_wrapped", False):
                self.log(f"patch_existing target={target}")
                return False
            if (
                not callable(original)
                or inspect.iscoroutinefunction(original) != asynchronous
                or inspect.isgeneratorfunction(original)
                or inspect.isasyncgenfunction(original)
                or not set(required).issubset(inspect.signature(original).parameters)
            ):
                self.log(f"patch_skipped target={target} reason=unsupported_signature")
                return False
            wrapped = factory(original)
            wrapped._runtime_timing_wrapped = True
            setattr(owner, name, wrapped)
            return True
        except Exception as error:
            self.log(f"patch_skipped target={target} error={type(error).__name__}")
        return False

    def attach_middleware(self, app):
        try:
            if self.disabled or getattr(app, MIDDLEWARE_ATTR, False):
                return
            app.add_middleware(RequestMiddleware, runtime=self)
            setattr(app, MIDDLEWARE_ATTR, True)
            self.log(f"middleware_installed pid={os.getpid()}")
        except Exception as error:
            self.log_once("middleware_skipped", f"middleware_skipped error={type(error).__name__}")

    def wrap_serve_http(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        async def serve_http(*args, **kwargs):
            if not self.disabled:
                # The launcher also covers api_server executed as __main__, for
                # which Python does not call our import loader's exec_module.
                with suppress(Exception):
                    self.attach_middleware(signature.bind(*args, **kwargs).arguments["app"])
            return await original(*args, **kwargs)

        return serve_http

    def patch_module(self, module):
        if self.disabled:
            return
        name = module.__name__
        patched = []
        if name in SCHEDULER_MODULES:
            for obj in tuple(vars(module).values()):
                if inspect.isclass(obj) and obj.__module__ == name and "schedule" in vars(obj):
                    if self.patch_method(obj, "schedule", self.wrap_schedule):
                        patched.append(f"{obj.__name__}.schedule")
                    if self.patch_method(obj, "add_request", self.wrap_scheduler_add_request, required=("request",)):
                        patched.append(f"{obj.__name__}.add_request")
        elif name in RUNNER_MODULES:
            runner = getattr(module, "NPUModelRunner", None)
            if self.patch_method(runner, "execute_model", self.wrap_runner, required=("scheduler_output",)):
                patched.append("NPUModelRunner.execute_model")
            if self.patch_method(runner, "sample_tokens", lambda fn: self.wrap_runner(fn, sampling=True)):
                patched.append("NPUModelRunner.sample_tokens")
            for method in STAGE_METHODS if self.config.detail == "full" else ():
                if self.patch_method(runner, method, lambda fn, method=method: self.wrap_stage(fn, f"runner.{method}")):
                    patched.append(f"NPUModelRunner.{method}")
        elif name == "vllm.v1.engine.async_llm":
            if self.patch_method(
                getattr(module, "AsyncLLM", None),
                "add_request",
                self.wrap_add_request,
                required=("request_id", "prompt", "trace_headers"),
                asynchronous=True,
            ):
                patched.append("AsyncLLM.add_request")
        elif name in TRACE_HEADER_MODULES:
            for obj in tuple(vars(module).values()):
                if inspect.isclass(obj) and obj.__module__ == name and "_get_trace_headers" in vars(obj):
                    if self.patch_method(
                        obj,
                        "_get_trace_headers",
                        self.wrap_trace_headers,
                        required=("headers",),
                        asynchronous=True,
                    ):
                        patched.append(f"{obj.__name__}._get_trace_headers")
        elif name == "vllm.entrypoints.openai.api_server":

            def wrap_build(original):
                @functools.wraps(original)
                def build_app(*args, **kwargs):
                    app = original(*args, **kwargs)
                    self.safe(self.attach_middleware, app)
                    return app

                return build_app

            if self.patch_method(module, "build_app", wrap_build):
                patched.append("build_app")
        elif name == "vllm.entrypoints.launcher":
            if self.patch_method(module, "serve_http", self.wrap_serve_http, required=("app",), asynchronous=True):
                patched.append("serve_http")
        self.log(f"module={name} patched={','.join(patched) if patched else 'none'}")

    def install(self):
        if any(isinstance(finder, HookFinder) for finder in sys.meta_path):
            return
        modules = (
            *SCHEDULER_MODULES,
            *RUNNER_MODULES,
            *TRACE_HEADER_MODULES,
            "vllm.v1.engine.async_llm",
            "vllm.entrypoints.openai.api_server",
            "vllm.entrypoints.launcher",
        )
        sys.meta_path.insert(0, HookFinder(self, frozenset(modules)))
        self.log(
            f"installed port={self.config.collector_port} sample_rate={self.config.sample_rate} "
            f"every_n_steps={self.config.every_n_steps} vllm={package_version('vllm')} "
            f"vllm_ascend={package_version('vllm-ascend')} detail={self.config.detail} "
            f"diagnostic_every={self.config.diagnostic_every}"
        )
        for name in modules:
            if name in sys.modules:
                self.safe_patch_module(sys.modules[name])


class HookLoader:
    def __init__(self, loader, runtime):
        self.loader, self.runtime = loader, runtime

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def create_module(self, spec):
        return self.loader.create_module(spec) if hasattr(self.loader, "create_module") else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.runtime.safe_patch_module(module)


class HookFinder(importlib.abc.MetaPathFinder):
    def __init__(self, runtime, modules):
        self.runtime, self.modules = runtime, modules

    def find_spec(self, fullname, path=None, target=None):
        if self.runtime.disabled or fullname not in self.modules:
            return None
        for finder in tuple(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    wrapped = self.runtime.safe(HookLoader, spec.loader, self.runtime)
                    if wrapped is not None:
                        spec.loader = wrapped
                return spec
        return None


class RequestMiddleware:
    def __init__(self, app, runtime):
        self.app, self.runtime = app, runtime

    def begin(self, scope):
        if scope["type"] != "http":
            return None
        path = scope.get("path", "")
        root = scope.get("root_path", "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
        if path not in HTTP_PATHS:
            return None
        headers = dict(scope.get("headers", ()))
        incoming = parse_traceparent(headers.get(b"traceparent", b"").decode("ascii", errors="ignore"))
        context = incoming or {"trace_id": uuid.uuid4().hex, "parent_span_id": None, "sampled": True}
        sampled = selected(context, self.runtime.config.sample_rate)
        self.runtime.log_event("request", f"request trace_id={context['trace_id']} sampled={str(sampled).lower()}")
        span_id = uuid.uuid4().hex[:16]
        traceparent = f"00-{context['trace_id']}-{span_id}-{'01' if sampled else '00'}"
        origin, clock = time.time_ns(), time.perf_counter_ns()
        record = Record("vllm.request", origin, span_id=span_id)
        token = self.runtime.request.set(
            {"traceparent": traceparent, "start_clock": clock, "metadata": record.metadata}
        )
        return token, context, sampled, record, origin, clock

    def record_first_body(self, state, message):
        if state is None or self.runtime.disabled:
            return
        _, _, _, record, _, clock = state
        if (
            message.get("type") == "http.response.body"
            and message.get("body")
            and "response_first_body_ms" not in record.metadata
        ):
            # An ASGI response chunk may contain headers/events or an error;
            # this boundary is not a generated-token TTFT measurement.
            record.metadata["response_first_body_ms"] = (time.perf_counter_ns() - clock) / 1_000_000

    def end(self, state, error):
        token, context, sampled, record, origin, clock = state
        try:
            if sampled and not self.runtime.disabled:
                record.end_ns = origin + time.perf_counter_ns() - clock
                record.error = error
                self.runtime.submit(Packet((context,), [record]))
        finally:
            self.runtime.request.reset(token)

    async def __call__(self, scope, receive, send):
        if self.runtime.disabled:
            return await self.app(scope, receive, send)
        state = self.runtime.safe(self.begin, scope)

        async def observed_send(message):
            self.runtime.safe(self.record_first_body, state, message)
            return await send(message)

        error_name = None
        try:
            return await self.app(scope, receive, observed_send if state is not None else send)
        except BaseException as error:
            error_name = type(error).__name__
            raise
        finally:
            if state is not None:
                self.runtime.safe(self.end, state, error_name)


def install(config):
    config = Config(**config)
    if not 0 <= config.sample_rate <= 1 or config.every_n_steps < 1:
        raise ValueError("invalid sampling configuration")
    if not 1 <= config.max_records <= MAX_RECORDS or not 1 <= config.max_requests <= MAX_REQUESTS:
        raise ValueError("invalid packet limits")
    if not 1024 <= config.collector_port <= 65535:
        raise ValueError("invalid collector port")
    if config.detail not in ("core", "full") or config.diagnostic_every < 1:
        raise ValueError("invalid detail or diagnostic interval")
    if config.sample_rate:
        Runtime(config).install()
