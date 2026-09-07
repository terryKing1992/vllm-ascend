"""Prepare a persistent optional injection bundle; never launch or supervise vLLM."""

import argparse
import json
import math
import shutil
from pathlib import Path

from timing_probe import Config
from trace_transport import MAX_RECORDS, MAX_REQUESTS

BOOTSTRAP = """
# Generated optional observer bootstrap. Failure must not prevent Python startup.
def _install_optional_observer():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent
    if not (root / "enabled").is_file():
        return
    from timing_probe import install
    install(json.loads((root / "config.json").read_text(encoding="utf-8")))

try:
    _install_optional_observer()
except Exception:
    pass
"""

CHAIN_SITECUSTOMIZE = """
# Preserve the existing startup customization before installing observation.
import sys
from importlib.machinery import PathFinder
from pathlib import Path
_own_dir = Path(__file__).resolve().parent
_paths = [p for p in sys.path if Path(p).resolve() != _own_dir]
_spec = PathFinder.find_spec("sitecustomize", _paths)
if _spec is not None and _spec.loader is not None:
    _code = _spec.loader.get_code("sitecustomize")
    if _code is not None:
        exec(_code, dict(__name__="sitecustomize", __file__=_spec.origin))
"""


def prepare(directory, config):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("timing_probe.py", "trace_transport.py"):
        shutil.copyfile(Path(__file__).with_name(name), root / name)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # The bootstrap is installed last. Partial preparation never enables observation.
    (root / "sitecustomize.py").write_text(CHAIN_SITECUSTOMIZE + BOOTSTRAP, encoding="utf-8")
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
    args = parser.parse_args()
    config = vars(args).copy()
    directory = config.pop("output_dir")
    if not math.isfinite(args.sample_rate) or not 0 <= args.sample_rate <= 1:
        parser.error("sample-rate must be between 0 and 1")
    if args.every_n_steps < 1 or not 1024 <= args.collector_port <= 65535:
        parser.error("invalid step interval or collector port")
    if not 1 <= args.max_records <= MAX_RECORDS or not 1 <= args.max_requests <= MAX_REQUESTS:
        parser.error(f"max-records must be 1..{MAX_RECORDS}; max-requests must be 1..{MAX_REQUESTS}")
    try:
        root = prepare(directory, config)
    except OSError as error:
        parser.error(str(error))
    print(f"Prepared optional injection at {root}")
    print("Add this directory to the service PYTHONPATH, then start vllm serve normally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
