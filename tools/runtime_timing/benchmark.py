"""Synthetic CPU overhead per batch; not an NPU throughput benchmark."""

import argparse
import io
import time
from types import SimpleNamespace

from timing_probe import Config, Runtime
from trace_export import CollectorConfig, JsonLogSink
from trace_summary import RequestSummarySink, SummaryConfig
from trace_transport import Packet, Record


class CountingExporter:
    def __init__(self):
        self.packets = 0

    def submit(self, packet):
        self.packets += 1


def measure(iterations, batch_size, rate, every_n_steps, enabled=True):
    exporter = CountingExporter()
    runtime = Runtime(Config(sample_rate=rate, every_n_steps=every_n_steps), exporter)
    scheduler = SimpleNamespace(
        requests={
            str(index): SimpleNamespace(
                trace_headers={
                    "traceparent": f"00-{(index + 1) * ((1 << 128) // (batch_size + 1)):032x}-1234567890123456-01"
                },
                num_output_tokens=1,
            )
            for index in range(batch_size)
        }
    )
    output = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(scheduler.requests, 1))
    runner = SimpleNamespace()
    schedule = lambda owner: output
    stage = lambda: None
    if enabled:
        schedule = runtime.wrap_schedule(schedule)
        stage = runtime.wrap_stage(stage, "prepare")
    execute = lambda owner, scheduler_output: stage()
    sample = lambda owner: None
    if enabled:
        execute = runtime.wrap_runner(execute)
        sample = runtime.wrap_runner(sample, sampling=True)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        execute(runner, schedule(scheduler))
        sample(runner)
    microseconds = (time.perf_counter_ns() - start) / iterations / 1000
    label = f"sample_rate={rate:g}" if enabled else "baseline"
    print(
        f"{label} every_n_steps={every_n_steps} batch_size={batch_size} "
        f"host_us/batch={microseconds:.2f} packets={exporter.packets}"
    )


def measure_summary(steps):
    stream = io.StringIO()
    summary = RequestSummarySink(SummaryConfig(), JsonLogSink(CollectorConfig(log_format="compact"), stream))
    trace_id, root_id = "1" * 32, "2" * 16
    context = {"trace_id": trace_id, "parent_span_id": root_id, "metadata": {"phase": "decode"}}
    packet = Packet(
        (context,),
        [Record("runner.execute_model", 0, 1_000_000), Record("runner._prepare_inputs", 0, 200_000, parent=0)],
        {"every_n_steps": 1, "rank": 0, "source_pid": 1},
    )
    start = time.perf_counter_ns()
    for step in range(steps):
        context["metadata"]["step"] = step
        summary.emit(packet)
    microseconds = (time.perf_counter_ns() - start) / steps / 1000
    summary.emit(Packet(({"trace_id": trace_id},), [Record("vllm.request", 0, 1_000_000, span_id=root_id)]))
    summary.close()
    output = stream.getvalue()
    print(
        f"summary decode_steps={steps} host_us/step={microseconds:.2f} "
        f"jsonl_lines={len(output.splitlines())} json_bytes={len(output.encode('utf-8'))}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--every-n-steps", type=int, default=Config.every_n_steps)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0 or args.every_n_steps <= 0:
        parser.error("iterations, batch-size and every-n-steps must be positive")
    measure(args.iterations, args.batch_size, 0, args.every_n_steps, enabled=False)
    for rate in (0, 0.01, 1):
        measure(args.iterations, args.batch_size, rate, args.every_n_steps)
    # Isolate collector aggregation cost and output volume from model wrappers.
    for steps in sorted({10, args.iterations}):
        measure_summary(steps)


if __name__ == "__main__":
    main()
