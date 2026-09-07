"""Synthetic CPU overhead per batch; not an NPU throughput benchmark."""

import argparse
import time
from types import SimpleNamespace

from timing_probe import Config, Runtime


class CountingExporter:
    def __init__(self):
        self.packets = 0

    def submit(self, packet):
        self.packets += 1


def measure(iterations, batch_size, rate, enabled=True):
    exporter = CountingExporter()
    runtime = Runtime(Config(sample_rate=rate, every_n_steps=10), exporter)
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
    print(f"{label} batch_size={batch_size} host_us/batch={microseconds:.2f} packets={exporter.packets}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error("iterations and batch-size must be positive")
    measure(args.iterations, args.batch_size, 0, enabled=False)
    for rate in (0, 0.01, 1):
        measure(args.iterations, args.batch_size, rate)


if __name__ == "__main__":
    main()
