"""Smoke tests for local benchmark and profiling tools."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LocalToolsSmokeTest(unittest.TestCase):
    def test_benchmark_local_runs(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "workspace/tools/benchmark_local.py",
                "--batch-size",
                "2",
                "--prompt-len",
                "8",
                "--decode-steps",
                "4",
                "--repeat",
                "1",
                "--warmup",
                "0",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("prefill:", result.stdout)
        self.assertIn("decode:", result.stdout)
        self.assertIn("mixed:", result.stdout)

    def test_profile_decode_runs(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "workspace/tools/profile_decode.py",
                "--batch-size",
                "2",
                "--prompt-len",
                "8",
                "--decode-steps",
                "4",
                "--top",
                "5",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("function calls", result.stdout)


if __name__ == "__main__":
    unittest.main()
