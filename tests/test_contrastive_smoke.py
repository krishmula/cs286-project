from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.training import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ContrastiveSmokeTests(unittest.TestCase):
    def test_contrastive_runner_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            command = [
                sys.executable,
                "-m",
                "src.train_contrastive",
                "--project-root",
                str(PROJECT_ROOT),
                "--output-dir",
                str(output_dir),
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
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
            self.assertIn("metrics_path", result.stdout)

            metrics_files = sorted(output_dir.glob("*_metrics.json"))
            checkpoint_files = sorted(output_dir.glob("*_checkpoint.pt"))

            self.assertEqual(len(metrics_files), 1)
            self.assertEqual(len(checkpoint_files), 3)

            metrics = json.loads(metrics_files[0].read_text())
            self.assertIn("last_checkpoint_path", metrics["artifacts"])
            self.assertIn("latest_checkpoint_path", metrics["artifacts"])
            checkpoint = load_checkpoint(Path(metrics["artifacts"]["checkpoint_path"]))
            last_checkpoint = load_checkpoint(Path(metrics["artifacts"]["last_checkpoint_path"]))
            latest_checkpoint = load_checkpoint(Path(metrics["artifacts"]["latest_checkpoint_path"]))

            self.assertEqual(metrics["config"]["train_windows"], 64)
            self.assertEqual(metrics["config"]["val_windows"], 32)
            self.assertIn("loss", metrics["train"])
            self.assertIn("loss", metrics["val"])
            self.assertIn("loss", metrics["last_train"])
            self.assertIn("loss", metrics["last_val"])
            self.assertIn("phone_encoder_state_dict", checkpoint)
            self.assertIn("watch_encoder_state_dict", checkpoint)
            self.assertIn("phone_projector_state_dict", checkpoint)
            self.assertIn("watch_projector_state_dict", checkpoint)
            self.assertEqual(checkpoint["metadata"]["stage"], "contrastive")
            self.assertEqual(checkpoint["metadata"]["checkpoint_role"], "best")
            self.assertEqual(last_checkpoint["metadata"]["checkpoint_role"], "last")
            self.assertEqual(latest_checkpoint["metadata"]["checkpoint_role"], "latest")


if __name__ == "__main__":
    unittest.main()
