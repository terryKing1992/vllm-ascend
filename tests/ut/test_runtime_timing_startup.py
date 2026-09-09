"""Real Python startup regressions; keep failed observation optional."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "runtime_timing"
sys.path.insert(0, str(TOOL_DIR))

from run import prepare  # noqa: E402
from timing_probe import Config  # noqa: E402


class TestStartup(unittest.TestCase):
    def invoke(self, root, bundle, *args, extra_paths=()):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(map(str, (bundle, *extra_paths)))
        return subprocess.run([sys.executable, *args], env=env, cwd=root, capture_output=True, text=True, timeout=15)

    def check(self, root, bundle, *flags):
        return self.invoke(root, bundle, *flags, str(TOOL_DIR / "check.py"), "--inject-dir", str(bundle))

    def test_bootstrap_and_check_show_actual_automatic_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = prepare(root / "bundle", asdict(Config(diagnostic_log=True)))
            result = self.check(root, bundle)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["loaded_probe"], str(bundle / "timing_probe.py"))
            self.assertTrue(report["hook_installed"])
            self.assertTrue(report["bootstrap_diagnostics_available"])
            self.assertEqual(report["warnings"], [])
            self.assertIn("[timing-bootstrap] loading file=", result.stderr)
            self.assertIn("[timing-bootstrap] probe_loaded file=", result.stderr)
            self.assertIn("[timing-bootstrap] ready", result.stderr)
            self.assertIn("[timing-probe] installed", result.stderr)

    def test_default_silent_startup_is_reported_by_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = prepare(root / "bundle", asdict(Config()))
            result = self.check(root, bundle)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("[timing-", result.stderr)
            self.assertIn("--diagnostic-log", json.loads(result.stdout)["warnings"][0])

    def test_no_site_and_isolated_flags_are_not_hidden_by_manual_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = prepare(root / "bundle", asdict(Config(diagnostic_log=True)))
            for flag in ("-S", "-I", "-E"):
                with self.subTest(flag=flag):
                    result = self.check(root, bundle, flag)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertFalse(report["hook_installed"])
                    self.assertIsNone(report["loaded_probe"])

    def test_loading_failure_has_phase_but_service_still_runs(self):
        cases = (
            ("config.json", "broken", "config", "JSONDecodeError"),
            ("timing_probe.py", "raise ImportError('sensitive message')", "probe_import", "ImportError"),
            ("timing_probe.py", "syntax error!", "probe_import", "SyntaxError"),
            ("config.json", '{"unknown_option":true}', "install", "TypeError"),
        )
        for filename, contents, phase, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = prepare(root / "bundle", asdict(Config(diagnostic_log=True)))
                (bundle / filename).write_text(contents, encoding="utf-8")
                result = self.invoke(root, bundle, "-c", "print('service works')")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("service works", result.stdout)
                self.assertIn(f"failed phase={phase} error={error}", result.stderr)
                self.assertNotIn("sensitive message", result.stderr)

    def test_disabled_marker_and_zero_sampling_are_explained(self):
        for reason in ("enabled_missing", "sample_rate_zero"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = Config(diagnostic_log=True, sample_rate=0 if reason == "sample_rate_zero" else 1)
                bundle = prepare(root / "bundle", asdict(config))
                if reason == "enabled_missing":
                    (bundle / "enabled").unlink()
                result = self.check(root, bundle)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"skipped reason={reason}", result.stderr)
                self.assertFalse(json.loads(result.stdout)["hook_installed"])

    def test_stale_probe_and_wrong_sitecustomize_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = prepare(root / "bundle", asdict(Config(diagnostic_log=True)))
            with (bundle / "timing_probe.py").open("a", encoding="utf-8") as stream:
                stream.write("\n# old copy\n")
            result = self.check(root, bundle)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(result.stdout)["bundle_matches_source"]["timing_probe.py"])
            # A module found earlier on PYTHONPATH prevents the bundle from loading.
            shadow = root / "shadow"
            shadow.mkdir()
            (shadow / "sitecustomize.py").write_text("pass\n", encoding="utf-8")
            result = self.invoke(
                root, shadow, str(TOOL_DIR / "check.py"), "--inject-dir", str(bundle), extra_paths=(bundle,)
            )
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertEqual(report["loaded_sitecustomize"], str(shadow / "sitecustomize.py"))
            self.assertFalse(report["hook_installed"])

    def test_multiple_bundles_skip_stale_bootstraps_and_chain_real_customization_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = prepare(root / "new", asdict(Config(diagnostic_log=True)))
            previous = prepare(root / "previous", asdict(Config(diagnostic_log=True)))
            legacy, genuine = root / "legacy", root / "genuine"
            legacy.mkdir()
            genuine.mkdir()
            (legacy / "sitecustomize.py").write_text(
                "# Legacy generated bootstrap fingerprint\n"
                "_own_dir = None\n"
                "def _install_optional_observer(): pass\n"
                "raise AssertionError('old bootstrap must not run')\n",
                encoding="utf-8",
            )
            (genuine / "sitecustomize.py").write_text("print('existing customization ran')\n", encoding="utf-8")
            result = self.invoke(
                root,
                bundle,
                "-c",
                "import sys; from timing_probe import HookFinder; "
                "assert sum(isinstance(f, HookFinder) for f in sys.meta_path) == 1; "
                "assert 'langfuse' not in sys.modules; print('service works')",
                extra_paths=(previous, legacy, genuine),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("existing customization ran"), 1)
            self.assertEqual(result.stderr.count("skipped_previous_bundle"), 2)
            self.assertEqual(result.stderr.count("[timing-probe] installed"), 1)
            self.assertNotIn("RecursionError", result.stderr)

    def test_broken_stderr_cannot_break_optional_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = prepare(root / "bundle", asdict(Config(diagnostic_log=True)))
            genuine = root / "genuine"
            genuine.mkdir()
            (genuine / "sitecustomize.py").write_text(
                "import sys\nclass BrokenStream:\n"
                "    def write(self, value): raise OSError('closed')\n"
                "    def flush(self): pass\nsys.stderr = BrokenStream()\n",
                encoding="utf-8",
            )
            result = self.invoke(root, bundle, "-c", "print('service works')", extra_paths=(genuine,))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("service works", result.stdout)


if __name__ == "__main__":
    unittest.main()
