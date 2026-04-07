from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.training import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SupervisedSmokeTests(unittest.TestCase):
    def test_supervised_runner_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            command = [
                sys.executable,
                "-m",
                "src.train_supervised",
                "--project-root",
                str(PROJECT_ROOT),
                "--output-dir",
                str(output_dir),
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
            self.assertIn("metrics_path", result.stdout)

            metrics_files = sorted(output_dir.glob("*_metrics.json"))
            checkpoint_files = sorted(output_dir.glob("*_checkpoint.pt"))
            confusion_files = sorted(output_dir.glob("*_confusion_matrix.png"))
            per_subject_files = sorted(output_dir.glob("*_per_subject_accuracy.csv"))

            self.assertEqual(len(metrics_files), 1)
            self.assertEqual(len(checkpoint_files), 2)
            self.assertEqual(len(confusion_files), 1)
            self.assertEqual(len(per_subject_files), 1)

            metrics = json.loads(metrics_files[0].read_text())
            latest_checkpoint = load_checkpoint(Path(metrics["artifacts"]["latest_checkpoint_path"]))
            self.assertEqual(metrics["channel_mode"], "fusion")
            self.assertEqual(metrics["label_fraction"], 0.1)
            self.assertEqual(metrics["config"]["train_windows_used"], 64)
            self.assertEqual(metrics["config"]["val_windows"], 32)
            self.assertEqual(metrics["config"]["test_windows"], 32)
            self.assertIn("accuracy", metrics["test"])
            self.assertIn("macro_f1", metrics["test"])
            self.assertEqual(latest_checkpoint["metadata"]["checkpoint_role"], "latest")


if __name__ == "__main__":
    unittest.main()
