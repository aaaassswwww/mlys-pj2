"""Smoke tests for the submission entrypoint."""

from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class SubmissionEntrypointTest(unittest.TestCase):
    def test_selfcheck_submission_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "workspace/tools/selfcheck_submission.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[selfcheck] import=ok", result.stdout)
        self.assertIn("[selfcheck] prefill_decode_remove=ok", result.stdout)

    def test_run_sh_writes_results_log(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available in this environment")

        probe = subprocess.run(
            [bash, "-lc", "true"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            self.skipTest(f"bash is not usable in this environment: {probe.stderr or probe.stdout}")

        result = subprocess.run(
            [bash, "run.sh"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout, "")
        log_path = ROOT / "workspace" / "results.log"
        result_log_path = ROOT / "workspace" / "result.log"
        output_path = ROOT / "output3.txt"
        self.assertTrue(log_path.is_file())
        self.assertTrue(result_log_path.is_file())
        self.assertTrue(output_path.is_file())
        log_text = log_path.read_text(encoding="utf-8")
        result_log_text = result_log_path.read_text(encoding="utf-8")
        output_text = output_path.read_text(encoding="utf-8")
        self.assertIn("[run.sh] selfcheck=passed", log_text)
        self.assertIn("[selfcheck] prefill_decode_remove=ok", log_text)
        self.assertIn("[run.sh] selfcheck=passed", result_log_text)
        self.assertIn("---- result ----", output_text)
        self.assertIn("---- agent output ----", output_text)
        self.assertIn("workspace/engine.py", output_text)
        self.assertIn("Phase 0 through Phase 7 completed", output_text)
        self.assertIn("Decode Optimization Reasoning", output_text)
        self.assertIn("benchmark status in this run:", output_text)


if __name__ == "__main__":
    unittest.main()
