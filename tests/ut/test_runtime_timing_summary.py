"""Request summary regressions, runnable directly without vLLM or NPU hardware."""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "runtime_timing"
sys.path.insert(0, str(TOOL_DIR))

from trace_export import CollectorConfig, JsonLogSink  # noqa: E402
from trace_summary import RequestSummarySink, SummaryConfig  # noqa: E402
from trace_transport import Packet, Record  # noqa: E402

TRACE_ID = "12345678901234567890123456789012"
UPSTREAM_PARENT = "1234567890123456"
ROOT_ID = "a" * 16
START_NS = 1_000_000_000
NS_PER_MS = 1_000_000


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class MemorySink:
    def __init__(self):
        self.packets = []
        self.closed = False

    def emit(self, packet):
        self.packets.append(packet)

    def close(self):
        self.closed = True


def root_packet(root_id=ROOT_ID, duration_ms=12_000, error=None):
    return Packet(
        ({"trace_id": TRACE_ID, "parent_span_id": UPSTREAM_PARENT, "sampled": True},),
        [
            Record(
                "vllm.request",
                START_NS,
                START_NS + duration_ms * NS_PER_MS,
                span_id=root_id,
                error=error,
                metadata={"path": "/v1/chat/completions"},
            )
        ],
        {"source_pid": 10},
    )


def stage_packet(
    root_id=ROOT_ID,
    name="runner.execute_model",
    phase="decode",
    duration_ms=1,
    step=0,
    rank=0,
    pid=100,
):
    return Packet(
        (
            {
                "trace_id": TRACE_ID,
                "parent_span_id": root_id,
                "request_id": f"request-{root_id}",
                "metadata": {"step": step, "phase": phase, "scheduled_tokens": 1},
            },
        ),
        [Record(name, START_NS, START_NS + duration_ms * NS_PER_MS)],
        {"source_pid": pid, "rank": rank, "shared_batch_time": True, "batch_size": 2},
    )


class TestRequestSummary(unittest.TestCase):
    def setUp(self):
        self.clock = ManualClock()
        self.sink = MemorySink()
        self.summary = RequestSummarySink(SummaryConfig(), self.sink, clock=self.clock)

    def finish(self):
        self.clock.advance(1.01)
        self.summary.poll()

    def payload(self, index=0):
        packet = self.sink.packets[index]
        self.assertEqual(len(packet.contexts), 1)
        self.assertEqual(len(packet.records), 1)
        self.assertEqual(packet.records[0].name, "vllm.request")
        return packet.records[0].metadata["timing_summary"]

    def stages(self, index=0):
        return {(stage["name"], stage["phase"]): stage for stage in self.payload(index)["stages"]}

    def test_thousand_decode_calls_emit_one_request_with_slowest_step(self):
        for step in range(1000):
            self.summary.emit(
                stage_packet(duration_ms=9 if step == 999 else 1, step=step, rank=step % 2, pid=100 + step % 2)
            )
        self.assertEqual(self.sink.packets, [])
        self.summary.emit(root_packet())
        self.assertEqual(self.sink.packets, [])
        self.finish()

        self.assertEqual(len(self.sink.packets), 1)
        payload = self.payload()
        self.assertEqual(payload["request_ms"], 12_000)
        self.assertEqual(payload["received_stage_records"], 1000)
        self.assertEqual(payload["stage_data_status"], "observed")
        stage = self.stages()["runner.execute_model", "decode"]
        self.assertEqual(stage["calls"], 1000)
        self.assertAlmostEqual(stage["mean_host_ms"], 1.008)
        self.assertEqual(stage["max_host_ms"], 9)
        self.assertEqual(stage["slowest_step"], 999)
        self.assertEqual(stage["slowest_rank"], 1)
        self.assertEqual(stage["slowest_pid"], 101)
        self.assertEqual(self.summary.stats["emitted"], 1)

    def test_prefill_decode_ranks_and_nested_calls_do_not_change_request_time(self):
        self.summary.emit(stage_packet(phase="prefill", duration_ms=40, step=0))
        self.summary.emit(stage_packet(duration_ms=20, step=1, rank=0))
        packet = stage_packet(duration_ms=30, step=1, rank=1)
        packet.records.append(Record("runner._prepare_inputs", START_NS, START_NS + 10 * NS_PER_MS, parent=0))
        self.summary.emit(packet)
        self.summary.emit(stage_packet(name="scheduler.schedule", duration_ms=2, step=1))
        self.summary.emit(root_packet(duration_ms=100))
        self.finish()

        self.assertEqual(self.payload()["request_ms"], 100)
        stages = self.stages()
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages["runner.execute_model", "prefill"]["max_host_ms"], 40)
        decode = stages["runner.execute_model", "decode"]
        self.assertEqual(decode["calls"], 2)
        self.assertEqual(decode["mean_host_ms"], 25)
        self.assertEqual(decode["max_host_ms"], 30)
        self.assertEqual(decode["slowest_rank"], 1)
        self.assertEqual(stages["runner._prepare_inputs", "decode"]["max_host_ms"], 10)

    def test_two_http_requests_under_same_trace_stay_separate(self):
        other_root = "b" * 16
        self.summary.emit(stage_packet(root_id=ROOT_ID, duration_ms=3))
        self.summary.emit(stage_packet(root_id=other_root, duration_ms=7))
        self.summary.emit(root_packet(root_id=ROOT_ID, duration_ms=100))
        self.summary.emit(root_packet(root_id=other_root, duration_ms=200))
        self.finish()

        self.assertEqual(len(self.sink.packets), 2)
        by_root = {packet.records[0].span_id: packet for packet in self.sink.packets}
        self.assertEqual(set(by_root), {ROOT_ID, other_root})
        for root_id, expected_ms in ((ROOT_ID, 3), (other_root, 7)):
            packet = by_root[root_id]
            self.assertEqual(packet.contexts[0]["trace_id"], TRACE_ID)
            self.assertEqual(packet.contexts[0]["parent_span_id"], UPSTREAM_PARENT)
            self.assertEqual(packet.records[0].metadata["timing_summary"]["stages"][0]["max_host_ms"], expected_ms)

    def test_one_batch_contributes_to_each_request_without_combining_roots(self):
        first = stage_packet(root_id=ROOT_ID, duration_ms=3)
        other_root = "b" * 16
        second = stage_packet(root_id=other_root, duration_ms=3)
        first.contexts += second.contexts
        self.summary.emit(first)
        self.summary.emit(root_packet(root_id=ROOT_ID))
        self.summary.emit(root_packet(root_id=other_root))
        self.finish()

        self.assertEqual(len(self.sink.packets), 2)
        for index in range(2):
            self.assertEqual(self.payload(index)["received_stage_records"], 1)
            self.assertEqual(self.stages(index)["runner.execute_model", "decode"]["calls"], 1)

    def test_root_grace_accepts_reordered_stages_without_extending_deadline(self):
        self.summary.emit(stage_packet(duration_ms=2))
        self.clock.advance(20)
        self.summary.emit(root_packet())
        self.clock.advance(0.9)
        self.summary.emit(stage_packet(duration_ms=6))
        self.summary.poll()
        self.assertEqual(self.sink.packets, [])
        self.clock.advance(0.2)
        self.summary.poll()

        self.assertEqual(len(self.sink.packets), 1)
        self.assertEqual(self.stages()["runner.execute_model", "decode"]["calls"], 2)
        self.assertEqual(self.stages()["runner.execute_model", "decode"]["mean_host_ms"], 4)

    def test_duplicate_roots_and_late_steps_cannot_emit_another_summary(self):
        self.summary.emit(root_packet())
        self.finish()
        self.summary.emit(stage_packet(duration_ms=20))
        self.summary.emit(root_packet())
        self.finish()
        self.summary.close()

        self.assertEqual(len(self.sink.packets), 1)
        self.assertEqual(self.summary.stats["late_packets"], 2)

    def test_missing_stage_data_still_exports_root_parent_error_and_metadata(self):
        packet = root_packet(error="CancelledError")
        self.summary.emit(packet)
        self.finish()

        exported = self.sink.packets[0]
        self.assertEqual(exported.contexts, packet.contexts)
        record = exported.records[0]
        original = packet.records[0]
        self.assertEqual(
            (record.start_ns, record.end_ns, record.span_id), (original.start_ns, original.end_ns, ROOT_ID)
        )
        self.assertEqual(record.error, "CancelledError")
        self.assertEqual(record.metadata["path"], "/v1/chat/completions")
        self.assertEqual(self.payload()["stage_data_status"], "missing")
        self.assertEqual(self.payload()["received_stage_records"], 0)
        self.assertEqual(self.payload()["stages"], [])
        self.assertTrue(self.payload()["finish_reason"])

    def test_expired_orphan_is_dropped_without_fabricating_a_request_span(self):
        self.summary = RequestSummarySink(SummaryConfig(ttl_seconds=2), self.sink, clock=self.clock)
        self.summary.emit(stage_packet())
        self.clock.advance(2.01)
        self.summary.poll()
        self.summary.close()

        self.assertEqual(self.sink.packets, [])
        self.assertEqual(self.summary.stats["orphan_dropped"], 1)

    def test_capacity_drops_orphans_and_exports_completed_requests(self):
        self.summary = RequestSummarySink(SummaryConfig(max_requests=2), self.sink, clock=self.clock)
        for root_id in ("a" * 16, "b" * 16):
            self.summary.emit(stage_packet(root_id=root_id))
        for root_id in ("c" * 16, "d" * 16, "e" * 16):
            self.summary.emit(root_packet(root_id=root_id))

        self.assertEqual(len(self.sink.packets), 1)
        self.assertEqual(self.sink.packets[0].records[0].span_id, "c" * 16)
        self.assertEqual(self.summary.stats["orphan_dropped"], 2)
        self.assertEqual(self.summary.stats["evicted"], 3)
        self.summary.close()
        self.assertEqual(len(self.sink.packets), 3)
        self.assertEqual({packet.records[0].span_id for packet in self.sink.packets}, {letter * 16 for letter in "cde"})

    def test_close_flushes_root_and_drops_orphan_without_waiting_for_grace(self):
        self.summary.emit(stage_packet(root_id="b" * 16))
        self.summary.emit(stage_packet())
        self.summary.emit(root_packet())
        self.summary.close()

        self.assertEqual(len(self.sink.packets), 1)
        self.assertEqual(self.payload()["stage_data_status"], "observed")
        self.assertEqual(self.summary.stats["orphan_dropped"], 1)
        self.assertTrue(self.sink.closed)

    def test_unknown_stage_names_cannot_grow_summary(self):
        for step in range(100):
            self.summary.emit(stage_packet(name=f"unknown.function_{step}", step=step))
        self.summary.emit(stage_packet(name="runner.execute_model"))
        self.summary.emit(root_packet())
        self.finish()

        self.assertEqual(set(self.stages()), {("runner.execute_model", "decode")})

    def test_fast_requests_are_all_exported_without_a_reporting_budget(self):
        for index in range(120):
            self.summary.emit(root_packet(root_id=f"{index + 1:016x}", duration_ms=1))
        self.finish()

        self.assertEqual(len(self.sink.packets), 120)
        self.assertTrue(all(self.payload(index)["request_ms"] == 1 for index in range(120)))

    def test_failed_export_does_not_discard_the_next_request_or_retry_failed_root(self):
        self.summary.emit(root_packet())
        self.clock.advance(2)
        other_root = "b" * 16
        with (
            patch.object(self.sink, "emit", side_effect=OSError("unavailable")),
            patch("builtins.print", side_effect=OSError("broken stderr")),
        ):
            self.summary.emit(root_packet(root_id=other_root))
        self.assertIn((TRACE_ID, other_root), self.summary.pending)
        self.assertEqual(self.summary.stats["export_failed"], 1)
        self.summary.emit(root_packet())
        self.finish()

        self.assertEqual(len(self.sink.packets), 1)
        self.assertEqual(self.sink.packets[0].records[0].span_id, other_root)

    def test_sampling_and_capacity_gaps_are_visible_in_summary(self):
        self.summary = RequestSummarySink(SummaryConfig(max_requests=1), self.sink, clock=self.clock)
        self.summary.emit(stage_packet())
        self.summary.emit(stage_packet(root_id="b" * 16))
        packet = stage_packet(step=10)
        packet.metadata.update(every_n_steps=10, truncated_stage_calls=2, omitted_sampled_requests=1)
        packet.contexts[0]["metadata"]["queue_to_first_schedule_ms"] = 7.5
        self.summary.emit(packet)
        self.summary.emit(root_packet())
        self.finish()

        payload = self.payload()
        self.assertTrue(payload["history_evicted"])
        self.assertEqual(payload["step_interval"], 10)
        self.assertEqual(payload["truncated_packets"], 1)
        self.assertEqual(payload["omitted_context_packets"], 1)
        self.assertEqual(payload["max_queue_to_first_schedule_ms"], 7.5)
        self.assertEqual(payload["received_stage_records"], 1)

    def test_compact_json_preserves_summary_and_trace_identity(self):
        stream = io.StringIO()
        log_sink = JsonLogSink(CollectorConfig(log_format="compact"), stream)
        summary = RequestSummarySink(SummaryConfig(), log_sink, clock=self.clock)
        summary.emit(stage_packet(duration_ms=7, step=42))
        summary.emit(root_packet())
        summary.close()

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0])
        self.assertEqual(event["trace_id"], TRACE_ID)
        self.assertEqual(event["span_id"], ROOT_ID)
        self.assertEqual(event["parent_span_id"], UPSTREAM_PARENT)
        payload = event["metadata"]["timing_summary"]
        self.assertEqual(payload["request_ms"], 12_000)
        self.assertEqual(payload["stages"][0]["slowest_step"], 42)
        self.assertEqual(payload["stages"][0]["max_host_ms"], 7)


if __name__ == "__main__":
    unittest.main()
