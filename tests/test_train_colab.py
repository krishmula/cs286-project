from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ColabTrainingSmokeTests(unittest.TestCase):
    def test_colab_wrapper_runs_supervised_training_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            drive_root = Path(tmp_dir)
            command = [
                sys.executable,
                "-m",
                "src.train_colab",
                "--stage",
                "supervised",
                "--project-root",
                str(PROJECT_ROOT),
                "--drive-root",
                str(drive_root),
                "--run-name",
                "smoke",
                "--experiment-name",
                "colab_smoke",
                "--skip-drive-mount",
                "--allow-cpu",
                "--",
                "--channel-mode",
                "fusion",
                "--label-fraction",
                "0.1",
                "--epochs",
                "1",
                "--patience",
                "1",
                "--batch-size",
                "32",
                "--max-train-windows",
                "64",
                "--max-val-windows",
                "32",
                "--max-test-windows",
                "32",
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)

            output_dir = drive_root / "supervised" / "smoke"
            self.assertIn('"stage": "supervised"', result.stdout)
            self.assertTrue((output_dir / "latest_checkpoint.pt").exists())
            self.assertEqual(len(list(output_dir.glob("*_metrics.json"))), 1)


if __name__ == "__main__":
    unittest.main()
