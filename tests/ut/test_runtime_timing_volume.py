"""Verify reduced output retains timing boundaries and trace parentage."""

import contextlib
import io
import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "runtime_timing"
sys.path.insert(0, str(TOOL_DIR))

from collector import Collector  # noqa: E402
from timing_probe import Config, Runtime, install  # noqa: E402
from trace_export import CollectorConfig, JsonLogSink  # noqa: E402
from trace_transport import DatagramEmitter, Packet, Record  # noqa: E402

TRACE_ID = "1" * 32
PARENT_ID = "2" * 16


def example_packet():
    return Packet(
        (
            {
                "trace_id": TRACE_ID,
                "parent_span_id": PARENT_ID,
                "request_id": "request-1",
                "metadata": {"phase": "decode", "step": 50},
            },
        ),
        [
            Record("runner.execute_model", 1000, 3000, span_id="3" * 16),
            Record("runner._prepare_inputs", 1100, 1500, parent=0, span_id="4" * 16),
        ],
        {"source_pid": 123, "rank": 0, "batch_id": "batch-1", "shared_batch_time": True},
    )


class TestVolume(unittest.TestCase):
    def test_core_and_full_modes_keep_business_and_outer_timing(self):
        for detail, expected_count in (("core", 1), ("full", 2)):
            with self.subTest(detail=detail):

                class NPUModelRunner:
                    def _prepare_inputs(self):
                        return 42

                    def execute_model(self, scheduler_output):
                        return self._prepare_inputs()

                exporter = Mock()
                runtime = Runtime(Config(detail=detail), exporter)
                runtime.patch_module(
                    SimpleNamespace(__name__="vllm_ascend.worker.model_runner_v1", NPUModelRunner=NPUModelRunner)
                )
                packet = example_packet()
                output = SimpleNamespace(
                    _langfuse_runtime_packet={"contexts": packet.contexts, "metadata": packet.metadata}
                )
                self.assertEqual(NPUModelRunner().execute_model(output), 42)
                actual = exporter.submit.call_args.args[0]
                self.assertEqual(len(actual.records), expected_count)
                self.assertEqual(actual.records[0].name, "runner.execute_model")
                self.assertEqual(actual.contexts[0]["trace_id"], TRACE_ID)
                self.assertEqual(actual.contexts[0]["parent_span_id"], PARENT_ID)
                if detail == "full":
                    self.assertEqual(actual.records[1].parent, 0)

    def test_diagnostic_throttling_never_drops_data(self):
        runtime = Runtime(Config(diagnostic_log=True, diagnostic_every=4), Mock())
        emitter = DatagramEmitter(18765, diagnostic_log=True, diagnostic_every=4)
        emitter.socket = Mock()
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            for _ in range(9):
                runtime.log_event("request", "request trace_id=test")
                emitter.submit(example_packet())
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 6)
        self.assertEqual(emitter.socket.sendto.call_count, 9)
        self.assertEqual(emitter.sent, 9)
        self.assertEqual(emitter.dropped, 0)
        for count in (1, 4, 8):
            self.assertIn(f"event_count={count} ", stream.getvalue())
            self.assertIn(f"sent packet={count} ", stream.getvalue())

    def test_socket_errors_are_throttled_and_counted(self):
        emitter = DatagramEmitter(18765, diagnostic_log=True, diagnostic_every=4)
        emitter.socket = Mock()
        emitter.socket.sendto.side_effect = BlockingIOError()
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            for _ in range(9):
                emitter.submit(example_packet())
        self.assertEqual(emitter.dropped, 9)
        self.assertEqual(len(stream.getvalue().splitlines()), 3)
        self.assertEqual(emitter.socket.sendto.call_count, 9)

    def test_disabled_diagnostics_do_not_format_packet_names(self):
        emitter = DatagramEmitter(18765)
        emitter.socket = Mock()
        with patch.object(emitter, "_log") as log:
            emitter.submit(example_packet())
        log.assert_not_called()
        self.assertEqual(emitter.sent, 1)

    def test_collector_throttles_logs_without_changing_export_counts(self):
        exporter = Mock()
        collector = Collector(0, exporter, diagnostic_log=True, diagnostic_every=4)
        collector.socket.close()
        collector.socket = Mock()
        data = json.dumps(asdict(example_packet())).encode()
        incoming = iter([data] * 9 + [b"invalid"] * 5)

        def recv(_):
            try:
                return next(incoming), ("127.0.0.1", 1234)
            except StopIteration:
                collector.stop()
                raise TimeoutError from None

        collector.socket.recvfrom.side_effect = recv
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            collector.run()
        self.assertEqual(collector.received, 9)
        self.assertEqual(collector.invalid, 5)
        self.assertEqual(exporter.submit.call_count, 9)
        self.assertEqual(len(stream.getvalue().splitlines()), 5)
        exporter.close.assert_called_once()

    def test_compact_json_retains_boundaries_and_parentage(self):
        packet = example_packet()
        outputs = {}
        for mode in ("compact", "full"):
            stream = io.StringIO()
            JsonLogSink(CollectorConfig(log_format=mode), stream).emit(packet)
            outputs[mode] = stream.getvalue()
        compact = [json.loads(line) for line in outputs["compact"].splitlines()]
        full = [json.loads(line) for line in outputs["full"].splitlines()]
        self.assertLess(len(outputs["compact"]), len(outputs["full"]))
        for short, long in zip(compact, full):
            for key in ("name", "trace_id", "span_id", "parent_span_id", "request_id", "duration_ms", "pid", "host"):
                self.assertEqual(short[key], long[key])
            self.assertNotIn("start_ns", short)
            self.assertNotIn("error", short)
            self.assertEqual(short["metadata"]["phase"], "decode")
            self.assertEqual(short["metadata"]["step"], 50)
            self.assertTrue(short["metadata"]["shared_batch_time"])
        self.assertEqual(compact[1]["parent_span_id"], compact[0]["span_id"])
        self.assertEqual(packet.records[0].start_ns, 1000)

    def test_compact_error_and_full_metadata_remain_available(self):
        packet = example_packet()
        packet.records[0].error = "ValueError"
        packet.metadata["custom_field"] = "custom_value"
        for mode in ("compact", "full"):
            with self.subTest(mode=mode):
                stream = io.StringIO()
                JsonLogSink(CollectorConfig(log_format=mode), stream).emit(packet)
                event = json.loads(stream.getvalue().splitlines()[0])
                self.assertEqual(event["error"], "ValueError")
                if mode == "full":
                    self.assertEqual(event["metadata"]["custom_field"], "custom_value")
                    self.assertEqual(event["start_ns"], 1000)
                    self.assertEqual(event["end_ns"], 3000)

    def test_step_sampling_retains_first_and_periodic_boundaries(self):
        exporter = Mock()
        runtime = Runtime(Config(sample_rate=1, every_n_steps=50), exporter)
        request = SimpleNamespace(trace_headers={"traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01"}, num_output_tokens=0)
        scheduler = SimpleNamespace(requests={"a": request})
        schedule = runtime.wrap_schedule(lambda owner: SimpleNamespace(num_scheduled_tokens={"a": 1}))
        for _ in range(120):
            schedule(scheduler)
            request.num_output_tokens += 1
        self.assertEqual(exporter.submit.call_count, 3)
        contexts = [call.args[0].contexts[0] for call in exporter.submit.call_args_list]
        self.assertEqual([context["metadata"]["step"] for context in contexts], [0, 50, 100])
        self.assertEqual([context["metadata"]["phase"] for context in contexts], ["prefill", "decode", "decode"])

    def test_invalid_detail_and_intervals_fail_validation(self):
        for config in ({"detail": "unknown"}, {"diagnostic_every": 0}, {"diagnostic_every": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                install(config)
        with self.assertRaises(ValueError):
            DatagramEmitter(18765, diagnostic_every=0)


if __name__ == "__main__":
    unittest.main()
