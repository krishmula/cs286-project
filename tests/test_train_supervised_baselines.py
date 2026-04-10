from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SupervisedBaselineRunnerSmokeTests(unittest.TestCase):
    def test_runner_launches_phone_and_watch_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            command = [
                sys.executable,
                "-m",
                "src.train_supervised_baselines",
                "--project-root",
                str(PROJECT_ROOT),
                "--output-root",
                str(output_root),
                "--run-name-prefix",
                "smoke",
                "--experiment-prefix",
                "runner_smoke",
                "--no-resume",
                "--",
                "--label-fraction",
                "0.1",
                "--epochs",
                "1",
                "--patience",
                "1",
                "--batch-size",
                "32",
                "--device",
                "cpu",
                "--max-train-windows",
                "64",
                "--max-val-windows",
                "32",
                "--max-test-windows",
                "32",
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)

            phone_dir = output_root / "smoke-phone"
            watch_dir = output_root / "smoke-watch"
            self.assertIn('"channel_mode": "phone"', result.stdout)
            self.assertIn('"channel_mode": "watch"', result.stdout)
            self.assertTrue((phone_dir / "latest_checkpoint.pt").exists())
            self.assertTrue((watch_dir / "latest_checkpoint.pt").exists())
            self.assertEqual(len(list(phone_dir.glob("*_metrics.json"))), 1)
            self.assertEqual(len(list(watch_dir.glob("*_metrics.json"))), 1)


if __name__ == "__main__":
    unittest.main()
