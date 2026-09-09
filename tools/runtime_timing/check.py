"""Inspect this Python process's automatic injection without importing the probe."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def module_path(name):
    # Importing sitecustomize or timing_probe here would hide startup failures.
    path = getattr(sys.modules.get(name), "__file__", None)
    return str(Path(path).resolve()) if path else None


def inspect_startup(directory):
    root = Path(directory).resolve()
    errors, warnings = [], []
    report = {
        "python": sys.executable,
        "no_site": sys.flags.no_site,
        "isolated": sys.flags.isolated,
        "ignore_environment": sys.flags.ignore_environment,
        "expected_inject_dir": str(root),
        "loaded_sitecustomize": module_path("sitecustomize"),
        "loaded_probe": module_path("timing_probe"),
        "enabled_marker": (root / "enabled").is_file(),
    }
    config = {}
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
    except Exception as error:
        errors.append(f"Cannot read bundle config: {type(error).__name__}")
        config = {}
    report["config"] = {key: config.get(key) for key in ("sample_rate", "diagnostic_log", "collector_port")}
    if not report["enabled_marker"]:
        errors.append("The bundle is disabled: enabled marker missing")
    if config.get("sample_rate") == 0:
        errors.append("sample_rate=0 deliberately disables installation")
    if not config.get("diagnostic_log"):
        warnings.append("Probe logs are disabled; prepare a new bundle with --diagnostic-log")
    if sys.flags.no_site or sys.flags.isolated or sys.flags.ignore_environment:
        errors.append("Python flags disable site or ignore PYTHONPATH; remove -S, -I or -E from the service command")
    for key, filename in (("loaded_sitecustomize", "sitecustomize.py"), ("loaded_probe", "timing_probe.py")):
        if report[key] != str(root / filename):
            errors.append(f"{filename} was not automatically loaded from the expected bundle")
    finder = next(
        (
            item
            for item in sys.meta_path
            if type(item).__module__ == "timing_probe" and type(item).__name__ == "HookFinder"
        ),
        None,
    )
    report["hook_installed"] = finder is not None
    report["observer_disabled"] = getattr(getattr(finder, "runtime", None), "disabled", None)
    if finder is None:
        errors.append("No timing import hook is installed; inspect timing-bootstrap stderr")
    elif report["observer_disabled"]:
        errors.append("The observer has disabled itself after a runtime failure")

    matches = {}
    for name in ("timing_probe.py", "trace_transport.py"):
        try:
            # Ignore checkout CRLF differences when comparing immutable copies.
            source = Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")
            injected = (root / name).read_bytes().replace(b"\r\n", b"\n")
            matches[name] = hashlib.sha256(source).digest() == hashlib.sha256(injected).digest()
        except OSError:
            matches[name] = False
        if not matches[name]:
            errors.append(f"{name} is missing or differs from this checkout; regenerate the injection bundle")
    report["bundle_matches_source"] = matches
    try:
        report["bootstrap_diagnostics_available"] = "[timing-bootstrap]" in (root / "sitecustomize.py").read_text(
            encoding="utf-8"
        )
    except OSError:
        report["bootstrap_diagnostics_available"] = False
    if not report["bootstrap_diagnostics_available"]:
        warnings.append("Old or missing bootstrap; regenerate the bundle to see startup failure diagnostics")
    report.update(status="FAIL" if errors else "PASS", errors=errors, warnings=warnings)
    report["scope"] = "Startup injection only; vLLM request hooks, UDP delivery and model execution are not exercised"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inject-dir", required=True, help="existing injection directory expected in service PYTHONPATH"
    )
    args = parser.parse_args()
    report = inspect_startup(args.inject_dir)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
