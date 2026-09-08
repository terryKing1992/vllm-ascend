"""Fail-open selective instrumentation. Worker dependencies: Python standard library only."""

import copy
import functools
import importlib.abc
import inspect
import re
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass

from trace_transport import MAX_RECORDS, MAX_REQUESTS, DatagramEmitter, Packet, Record

TRACE_PACKET_ATTR = "_langfuse_runtime_packet"
PENDING_PACKET_ATTR = "_langfuse_runtime_pending"
STEP_ATTR = "_langfuse_runtime_step"
REQUEST_CONTEXT_ATTR = "_langfuse_runtime_context"
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


@dataclass(frozen=True)
class Config:
    sample_rate: float = 0.01
    every_n_steps: int = 10
    max_records: int = 16
    max_requests: int = 4
    collector_port: int = 18765


def parse_traceparent(value):
    if not isinstance(value, str) or not re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", value):
        return None
    _, trace_id, parent_id, flags = value.split("-")
    if int(trace_id, 16) == 0 or int(parent_id, 16) == 0:
        return None
    return {"trace_id": trace_id, "parent_span_id": parent_id, "sampled": bool(int(flags, 16) & 1)}


def selected(context, rate):
    return bool(context and context["sampled"] and int(context["trace_id"], 16) < rate * (1 << 128))


class Runtime:
    def __init__(self, config, exporter=None):
        self.config = config
        self.exporter = exporter if exporter is not None else DatagramEmitter(config.collector_port)
        self.request = ContextVar("runtime_request", default=None)
        self.stage = ContextVar("runtime_stage", default=None)
        self.scheduling = ContextVar("runtime_scheduling", default=False)
        self.disabled = False
        self.failures = 0

    def safe(self, operation, *args):
        """Only instrumentation enters here; never catch/retry business execution."""
        try:
            return operation(*args)
        except Exception:
            # No worker logging: stderr itself can block or be broken.
            self.disabled = True
            self.failures += 1
            return None

    def submit(self, packet):
        if not self.disabled:
            self.safe(self.exporter.submit, packet)

    def begin_schedule(self):
        if self.scheduling.get():
            return None
        origin, clock = time.time_ns(), time.perf_counter_ns()
        return self.scheduling.set(True), origin, clock

    def finish_schedule(self, state, scheduler, output):
        token, origin, clock = state
        try:
            if output is None or self.disabled:
                return
            end = origin + time.perf_counter_ns() - clock
            contexts, omitted = [], 0
            for req_id, num_tokens in output.num_scheduled_tokens.items():
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
                contexts.append(
                    dict(
                        context,
                        request_id=req_id[:256],
                        metadata={
                            "step": step,
                            "scheduled_tokens": num_tokens,
                            "phase": "prefill" if request.num_output_tokens == 0 else "decode",
                        },
                    )
                )
            metadata = {
                "batch_id": uuid.uuid4().hex if contexts else "",
                "batch_size": len(output.num_scheduled_tokens),
                "shared_batch_time": True,
                "omitted_sampled_requests": omitted,
            }
            # Optional built-in types only: uninstrumented workers can unpickle this.
            setattr(output, TRACE_PACKET_ATTR, {"contexts": contexts, "metadata": metadata})
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
        context = self.request.get()
        if context is None or "trace_headers" not in signature.parameters:
            return args, kwargs
        bound = signature.bind(*args, **kwargs)
        headers = dict(bound.arguments.get("trace_headers") or {})
        headers["traceparent"] = context["traceparent"]
        bound.arguments["trace_headers"] = headers
        prompt = bound.arguments.get("prompt")
        if hasattr(prompt, "trace_headers"):
            prompt = copy.copy(prompt)
            prompt.trace_headers = headers
            bound.arguments["prompt"] = prompt
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

    def patch_method(self, owner, name, factory):
        original = getattr(owner, name, None)
        if original is not None and not getattr(original, "_runtime_timing_wrapped", False):
            wrapped = factory(original)
            wrapped._runtime_timing_wrapped = True
            setattr(owner, name, wrapped)

    def patch_module(self, module):
        if self.disabled:
            return
        name = module.__name__
        if name in SCHEDULER_MODULES:
            for obj in tuple(vars(module).values()):
                if inspect.isclass(obj) and obj.__module__ == name and "schedule" in vars(obj):
                    self.patch_method(obj, "schedule", self.wrap_schedule)
        elif name in RUNNER_MODULES:
            runner = module.NPUModelRunner
            self.patch_method(runner, "execute_model", self.wrap_runner)
            self.patch_method(runner, "sample_tokens", lambda fn: self.wrap_runner(fn, sampling=True))
            for method in STAGE_METHODS:
                self.patch_method(runner, method, lambda fn, method=method: self.wrap_stage(fn, f"runner.{method}"))
        elif name == "vllm.v1.engine.async_llm":
            self.patch_method(module.AsyncLLM, "add_request", self.wrap_add_request)
        elif name == "vllm.entrypoints.openai.api_server":

            def wrap_build(original):
                @functools.wraps(original)
                def build_app(*args, **kwargs):
                    app = original(*args, **kwargs)
                    if not self.disabled:
                        self.safe(lambda: app.add_middleware(RequestMiddleware, runtime=self))
                    return app

                return build_app

            self.patch_method(module, "build_app", wrap_build)

    def install(self):
        if any(isinstance(finder, HookFinder) for finder in sys.meta_path):
            return
        modules = (
            *SCHEDULER_MODULES,
            *RUNNER_MODULES,
            "vllm.v1.engine.async_llm",
            "vllm.entrypoints.openai.api_server",
        )
        sys.meta_path.insert(0, HookFinder(self, frozenset(modules)))
        for name in modules:
            if name in sys.modules:
                self.safe(self.patch_module, sys.modules[name])


class HookLoader:
    def __init__(self, loader, runtime):
        self.loader, self.runtime = loader, runtime

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def create_module(self, spec):
        return self.loader.create_module(spec) if hasattr(self.loader, "create_module") else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.runtime.safe(self.runtime.patch_module, module)


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
        if scope["type"] != "http" or scope.get("path") not in ("/v1/chat/completions", "/v1/completions"):
            return None
        headers = dict(scope.get("headers", ()))
        incoming = parse_traceparent(headers.get(b"traceparent", b"").decode("ascii", errors="ignore"))
        context = incoming or {"trace_id": uuid.uuid4().hex, "parent_span_id": None, "sampled": True}
        sampled = selected(context, self.runtime.config.sample_rate)
        span_id = uuid.uuid4().hex[:16]
        traceparent = f"00-{context['trace_id']}-{span_id}-{'01' if sampled else '00'}"
        origin, clock = time.time_ns(), time.perf_counter_ns()
        record = Record("vllm.request", origin, span_id=span_id)
        token = self.runtime.request.set({"traceparent": traceparent})
        return token, context, sampled, record, origin, clock

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
        error_name = None
        try:
            return await self.app(scope, receive, send)
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
    if config.sample_rate:
        Runtime(config).install()
