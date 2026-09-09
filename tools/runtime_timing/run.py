"""Prepare a persistent optional injection bundle; never launch or supervise vLLM."""

import argparse
import json
import math
import shutil
from pathlib import Path

from timing_probe import Config
from trace_transport import MAX_RECORDS, MAX_REQUESTS

STARTUP_DIAGNOSTICS = """
# Generated startup diagnostics; controlled when the bundle is prepared.
def _runtime_timing_boot_log(message):
    if not __RUNTIME_TIMING_DIAGNOSTIC__:
        return
    try:
        import os
        import sys
        print(f"[timing-bootstrap] {message} pid={os.getpid()}", file=sys.stderr, flush=True)
    except Exception:
        pass

_runtime_timing_boot_log(f"loading file={__file__}")
"""

BOOTSTRAP = """
# Generated optional observer bootstrap. Failure must not prevent Python startup.
def _install_optional_observer():
    phase = "config"
    try:
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parent
        if not (root / "enabled").is_file():
            _runtime_timing_boot_log("skipped reason=enabled_missing")
            return
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        phase = "probe_import"
        import timing_probe
        _runtime_timing_boot_log(f"probe_loaded file={timing_probe.__file__}")
        phase = "install"
        timing_probe.install(config)
        if config.get("sample_rate") == 0:
            _runtime_timing_boot_log("skipped reason=sample_rate_zero")
        else:
            _runtime_timing_boot_log("ready")
    except Exception as error:
        _runtime_timing_boot_log(f"failed phase={phase} error={type(error).__name__}")

try:
    _install_optional_observer()
except Exception:
    pass
"""

CHAIN_SITECUSTOMIZE = """
# Preserve the existing startup customization before installing observation.
def _runtime_timing_chain_sitecustomize():
    import sys
    from importlib.machinery import PathFinder
    from pathlib import Path
    own_dir = Path(__file__).resolve().parent
    paths = [p for p in sys.path if Path(p).resolve() != own_dir]
    while paths:
        spec = PathFinder.find_spec("sitecustomize", paths)
        if spec is None or spec.loader is None:
            return
        code = spec.loader.get_code("sitecustomize")
        if code is None:
            return
        # New and legacy generated bundles must not execute one another:
        # otherwise each finds the other again and recurses during startup.
        if "_install_optional_observer" in code.co_names and (
            "_runtime_timing_chain_sitecustomize" in code.co_names or "_own_dir" in code.co_names
        ):
            _runtime_timing_boot_log(f"skipped_previous_bundle file={spec.origin}")
            previous_dir = Path(spec.origin).resolve().parent
            remaining = [p for p in paths if Path(p).resolve() != previous_dir]
            if len(remaining) == len(paths):
                return
            paths = remaining
            continue
        exec(code, dict(__name__="sitecustomize", __file__=spec.origin))
        return

try:
    _runtime_timing_chain_sitecustomize()
except Exception as error:
    _runtime_timing_boot_log(f"existing_sitecustomize_failed error={type(error).__name__}")
    raise
"""


def prepare(directory, config):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("timing_probe.py", "trace_transport.py"):
        shutil.copyfile(Path(__file__).with_name(name), root / name)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # The bootstrap is installed last. Partial preparation never enables observation.
    diagnostics = STARTUP_DIAGNOSTICS.replace("__RUNTIME_TIMING_DIAGNOSTIC__", repr(bool(config.get("diagnostic_log"))))
    (root / "sitecustomize.py").write_text(diagnostics + CHAIN_SITECUSTOMIZE + BOOTSTRAP, encoding="utf-8")
    (root / "enabled").touch()
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=float, default=Config.sample_rate)
    parser.add_argument("--every-n-steps", type=int, default=Config.every_n_steps)
    parser.add_argument("--collector-port", type=int, default=Config.collector_port)
    parser.add_argument("--max-records", type=int, default=Config.max_records)
    parser.add_argument("--max-requests", type=int, default=Config.max_requests)
    parser.add_argument(
        "--detail",
        choices=("core", "full"),
        default=Config.detail,
        help="core: request/schedule/execute/sample boundaries; full: also instrument internal stages",
    )
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=Config.diagnostic_every,
        help="log the first event and every N events per category/process; 1 restores per-event logs",
    )
    parser.add_argument(
        "--diagnostic-log",
        action="store_true",
        help="print bootstrap, patch, request and sampled UDP diagnostics to the model process stderr",
    )
    args = parser.parse_args()
    config = vars(args).copy()
    directory = config.pop("output_dir")
    if not math.isfinite(args.sample_rate) or not 0 <= args.sample_rate <= 1:
        parser.error("sample-rate must be between 0 and 1")
    if args.every_n_steps < 1 or not 1024 <= args.collector_port <= 65535:
        parser.error("invalid step interval or collector port")
    if args.diagnostic_every < 1:
        parser.error("diagnostic-every must be positive")
    if not 1 <= args.max_records <= MAX_RECORDS or not 1 <= args.max_requests <= MAX_REQUESTS:
        parser.error(f"max-records must be 1..{MAX_RECORDS}; max-requests must be 1..{MAX_REQUESTS}")
    try:
        root = prepare(directory, config)
    except OSError as error:
        parser.error(str(error))
    print(f"Prepared optional injection at {root}")
    print(f"Startup/probe diagnostics: {'enabled' if args.diagnostic_log else 'disabled (use --diagnostic-log)'}")
    print("Add this directory to the service PYTHONPATH, then start vllm serve normally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
