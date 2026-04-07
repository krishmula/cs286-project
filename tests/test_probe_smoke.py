from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from src.models import PhoneWatchContrastiveModel
from src.training import LinearProbeHead, freeze_module, load_checkpoint, train_probe_epoch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProbeSmokeTests(unittest.TestCase):
    def test_probe_runner_smoke_all_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            contrastive_dir = tmp_root / "contrastive"
            probes_dir = tmp_root / "probes"

            contrastive_command = [
                sys.executable,
                "-m",
                "src.train_contrastive",
                "--project-root",
                str(PROJECT_ROOT),
                "--output-dir",
                str(contrastive_dir),
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
            subprocess.run(contrastive_command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
            checkpoint_path = next(contrastive_dir.glob("*_checkpoint.pt"))

            for mode in ("pair", "phone", "watch"):
                mode_dir = probes_dir / mode
                probe_command = [
                    sys.executable,
                    "-m",
                    "src.train_probe",
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--encoder-ckpt-path",
                    str(checkpoint_path),
                    "--output-dir",
                    str(mode_dir),
                    "--evaluation-mode",
                    mode,
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
                result = subprocess.run(probe_command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
                self.assertIn("metrics_path", result.stdout)

                metrics_files = sorted(mode_dir.glob("*_metrics.json"))
                checkpoint_files = sorted(mode_dir.glob("*_checkpoint.pt"))
                confusion_files = sorted(mode_dir.glob("*_confusion_matrix.png"))
                per_subject_files = sorted(mode_dir.glob("*_per_subject_accuracy.csv"))

                self.assertEqual(len(metrics_files), 1)
                self.assertEqual(len(checkpoint_files), 2)
                self.assertEqual(len(confusion_files), 1)
                self.assertEqual(len(per_subject_files), 1)

                metrics = json.loads(metrics_files[0].read_text())
                latest_checkpoint = load_checkpoint(Path(metrics["artifacts"]["latest_checkpoint_path"]))
                self.assertEqual(metrics["probe_mode"], mode)
                self.assertEqual(metrics["label_fraction"], 0.1)
                self.assertEqual(metrics["config"]["train_windows_used"], 64)
                self.assertEqual(metrics["config"]["val_windows"], 32)
                self.assertEqual(metrics["config"]["test_windows"], 32)
                self.assertIn("accuracy", metrics["test"])
                self.assertIn("macro_f1", metrics["test"])
                self.assertEqual(latest_checkpoint["metadata"]["checkpoint_role"], "latest")

    def test_frozen_encoder_weights_do_not_change_during_probe_head_training(self) -> None:
        torch.manual_seed(42)
        model = PhoneWatchContrastiveModel()
        freeze_module(model.phone_encoder)
        freeze_module(model.watch_encoder)

        before_phone = {key: value.detach().clone() for key, value in model.phone_encoder.state_dict().items()}
        before_watch = {key: value.detach().clone() for key, value in model.watch_encoder.state_dict().items()}

        head = LinearProbeHead(in_dim=512, num_classes=3)
        head_before = {key: value.detach().clone() for key, value in head.state_dict().items()}

        features = torch.randn(12, 512)
        labels = torch.randint(low=0, high=3, size=(12,))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(features, labels),
            batch_size=4,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        train_probe_epoch(head=head, loader=loader, optimizer=optimizer, loss_fn=loss_fn, device=torch.device("cpu"))

        after_phone = model.phone_encoder.state_dict()
        after_watch = model.watch_encoder.state_dict()
        head_after = head.state_dict()

        for key, before in before_phone.items():
            self.assertTrue(torch.equal(before, after_phone[key]))
        for key, before in before_watch.items():
            self.assertTrue(torch.equal(before, after_watch[key]))

        head_changed = any(not torch.equal(before, head_after[key]) for key, before in head_before.items())
        self.assertTrue(head_changed)


if __name__ == "__main__":
    unittest.main()
