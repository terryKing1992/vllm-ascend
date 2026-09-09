"""Collector-only, bounded request summaries. Never export per-decode spans."""

import math
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace

from trace_transport import Packet

STAGE_NAMES = frozenset(
    (
        "scheduler.schedule",
        "runner.execute_model",
        "runner.sample_tokens",
        "runner._update_states",
        "runner._prepare_inputs",
        "runner.prepare_inputs",
        "runner._build_attention_metadata",
        "runner._model_forward",
        "runner._sample",
        "runner.propose_draft_token_ids",
        "runner.postprocess",
        "runner.postprocess_sampled",
    )
)
MAX_REQUEST_IDS = 8
SWEEP_INTERVAL_SECONDS = 0.25
ROOT_METRICS = ("api_to_engine_ms", "response_first_body_ms")


@dataclass(frozen=True)
class SummaryConfig:
    settle_seconds: float = 1.0
    ttl_seconds: float = 300.0
    max_requests: int = 2048

    def __post_init__(self):
        if (
            not math.isfinite(self.settle_seconds)
            or self.settle_seconds < 0
            or not math.isfinite(self.ttl_seconds)
            or self.ttl_seconds <= self.settle_seconds
            or self.max_requests < 1
        ):
            raise ValueError("summary requires 0 <= grace < ttl and positive capacity")


@dataclass
class StageStats:
    calls: int = 0
    mean_ns: float = 0.0
    max_ns: int = -1
    slowest_step: int | None = None
    slowest_rank: int | None = None
    slowest_pid: int | None = None

    def add(self, record, step, rank, pid):
        elapsed = record.end_ns - record.start_ns
        self.calls += 1
        self.mean_ns += (elapsed - self.mean_ns) / self.calls
        if elapsed > self.max_ns:
            self.max_ns = elapsed
            self.slowest_step, self.slowest_rank, self.slowest_pid = step, rank, pid

    def result(self, name, phase):
        return {
            "name": name,
            "phase": phase,
            "calls": self.calls,
            "mean_host_ms": round(self.mean_ns / 1_000_000, 6),
            "max_host_ms": round(self.max_ns / 1_000_000, 6),
            "slowest_step": self.slowest_step,
            "slowest_rank": self.slowest_rank,
            "slowest_pid": self.slowest_pid,
        }


@dataclass
class RequestState:
    created: float
    root: object = None
    root_context: dict | None = None
    root_pid: int | None = None
    ready_at: float | None = None
    stages: dict = field(default_factory=dict)
    request_ids: set = field(default_factory=set)
    received_stage_records: int = 0
    truncated_packets: int = 0
    omitted_context_packets: int = 0
    max_queue_to_first_schedule_ms: float | None = None
    step_interval: int | str | None = None
    history_evicted: bool = False


def integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= (1 << 63) else None


class RequestSummarySink:
    """One original HTTP root span with fixed-size stage statistics in metadata.

    Calls are serialized by BufferedExporter. No per-step lists are retained.
    Different ranks contribute observations, never a summed request duration.
    """

    def __init__(self, config, sink, clock=time.monotonic):
        self.config, self.sink, self.clock = config, sink, clock
        self.pending = OrderedDict()
        self.closed = OrderedDict()
        self.lost = OrderedDict()
        self.next_sweep = clock() + SWEEP_INTERVAL_SECONDS
        self.stopped = False
        self.stats = {
            "emitted": 0,
            "orphan_dropped": 0,
            "late_packets": 0,
            "evicted": 0,
            "export_failed": 0,
            "ignored_stage_records": 0,
        }

    def _remember(self, cache, key, now):
        cache[key] = now + self.config.ttl_seconds
        cache.move_to_end(key)
        while len(cache) > self.config.max_requests:
            cache.popitem(last=False)

    def _state(self, key, now):
        state = self.pending.get(key)
        if state is not None:
            return state
        if len(self.pending) >= self.config.max_requests:
            oldest = next(iter(self.pending))
            self.stats["evicted"] += 1
            self._finish(oldest, "capacity", now)
        state = RequestState(now, history_evicted=key in self.lost)
        self.pending[key] = state
        return state

    def emit(self, packet):
        if self.stopped:
            return
        now = self.clock()
        if now >= self.next_sweep:
            self.poll()
        for context in packet.contexts:
            roots = [record for record in packet.records if record.name == "vllm.request" and record.span_id]
            if roots:
                for root in roots:
                    key = context["trace_id"], root.span_id
                    if key in self.closed:
                        self.stats["late_packets"] += 1
                        continue
                    state = self._state(key, now)
                    if state.root is None:
                        state.root = root
                        state.root_context = context
                        state.root_pid = integer(packet.metadata.get("source_pid"))
                        state.ready_at = now + self.config.settle_seconds
                continue
            parent = context.get("parent_span_id")
            if not parent:
                self.stats["orphan_dropped"] += 1
                continue
            key = context["trace_id"], parent
            if key in self.closed:
                self.stats["late_packets"] += 1
                continue
            state = self._state(key, now)
            request_id = context.get("request_id")
            if isinstance(request_id, str) and len(state.request_ids) < MAX_REQUEST_IDS:
                state.request_ids.add(request_id[:256])
            metadata = context.get("metadata", {})
            phase = metadata.get("phase")
            if phase not in ("prefill", "decode"):
                phase = "unknown"
            interval = integer(packet.metadata.get("every_n_steps"))
            if interval is not None:
                if state.step_interval is None:
                    state.step_interval = interval
                elif state.step_interval != interval:
                    state.step_interval = "mixed"
            # This is a maximum across engine requests in the HTTP request,
            # recorded by scheduler, not inferred from time left over.
            queued = metadata.get("queue_to_first_schedule_ms")
            if isinstance(queued, (int, float)) and math.isfinite(queued) and queued >= 0:
                state.max_queue_to_first_schedule_ms = max(state.max_queue_to_first_schedule_ms or 0, queued)
            state.truncated_packets += bool(packet.metadata.get("truncated_stage_calls"))
            state.omitted_context_packets += bool(packet.metadata.get("omitted_sampled_requests"))
            for record in packet.records:
                if record.name not in STAGE_NAMES:
                    self.stats["ignored_stage_records"] += 1
                    continue
                group = record.name, phase
                stats = state.stages.setdefault(group, StageStats())
                stats.add(
                    record,
                    integer(metadata.get("step")),
                    integer(packet.metadata.get("rank")),
                    integer(packet.metadata.get("source_pid")),
                )
                state.received_stage_records += 1

    def _finish(self, key, reason, now):
        state = self.pending.pop(key)
        if state.root is None:
            self.stats["orphan_dropped"] += 1
            self._remember(self.lost, key, now)
            return
        # Remember before export: no repeated exports if SDK/filesystem fails.
        self._remember(self.closed, key, now)
        summary = {
            "request_ms": (state.root.end_ns - state.root.start_ns) / 1_000_000,
            "stages": [stats.result(*group) for group, stats in sorted(state.stages.items())],
            "request_ids": sorted(state.request_ids),
            "received_stage_records": state.received_stage_records,
            "stage_data_status": "observed" if state.stages else "missing",
            "coverage": "best_effort_received_records",
            "step_interval": state.step_interval,
            "history_evicted": state.history_evicted,
            "truncated_packets": state.truncated_packets,
            "omitted_context_packets": state.omitted_context_packets,
            "finish_reason": reason,
            "max_queue_to_first_schedule_ms": state.max_queue_to_first_schedule_ms,
            "timing_kind": "host_wall_inclusive; calls across ranks; do not sum stages or ranks",
        }
        metadata = {name: state.root.metadata[name] for name in ROOT_METRICS if name in state.root.metadata}
        metadata["timing_summary"] = summary
        record = replace(state.root, metadata=metadata)
        context = {
            name: state.root_context[name]
            for name in ("trace_id", "parent_span_id", "sampled")
            if name in state.root_context
        }
        packet = Packet((context,), [record], {"source_pid": state.root_pid, "reporting_mode": "request"})
        try:
            self.sink.emit(packet)
            self.stats["emitted"] += 1
        except Exception:
            self.stats["export_failed"] += 1
            raise

    def poll(self):
        now = self.clock()
        self.next_sweep = now + SWEEP_INTERVAL_SECONDS
        for cache in (self.closed, self.lost):
            while cache and next(iter(cache.values())) <= now:
                cache.popitem(last=False)
        for key, state in tuple(self.pending.items()):
            if state.ready_at is not None and state.ready_at <= now:
                self._finish(key, "request_complete", now)
            elif state.created + self.config.ttl_seconds <= now:
                self._finish(key, "ttl", now)

    def close(self):
        if self.stopped:
            return
        self.stopped = True
        try:
            for key in tuple(self.pending):
                try:
                    self._finish(key, "shutdown", self.clock())
                except Exception:
                    continue
        finally:
            try:
                self.sink.close()
            finally:
                print(
                    "[timing-summary] " + " ".join(f"{key}={value}" for key, value in self.stats.items()),
                    file=sys.stderr,
                    flush=True,
                )
