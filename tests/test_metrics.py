from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training import (
    EarlyStopper,
    checkpoint_metadata,
    compute_classification_metrics,
    load_checkpoint,
    save_checkpoint,
)


class MetricsTests(unittest.TestCase):
    def test_compute_classification_metrics(self) -> None:
        y_true = [0, 1, 2, 1, 0, 2]
        y_pred = [0, 2, 2, 1, 0, 1]
        subject_ids = [10, 10, 11, 11, 12, 12]
        index_to_label = {0: "A", 1: "B", 2: "C"}

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            subject_ids=subject_ids,
            index_to_label=index_to_label,
            loss=1.25,
        )

        self.assertAlmostEqual(metrics.accuracy, 4 / 6)
        self.assertEqual(metrics.confusion_matrix.shape, (3, 3))
        self.assertAlmostEqual(metrics.per_class_f1["A"], 1.0)
        self.assertAlmostEqual(metrics.per_class_f1["B"], 0.5)
        self.assertAlmostEqual(metrics.per_class_f1["C"], 0.5)
        self.assertAlmostEqual(metrics.macro_f1, (1.0 + 0.5 + 0.5) / 3)
        self.assertEqual(metrics.per_subject_accuracy, {10: 0.5, 11: 1.0, 12: 0.5})
        self.assertEqual(metrics.to_summary_dict()["loss"], 1.25)


class EarlyStopperTests(unittest.TestCase):
    def test_early_stopper_tracks_best_epoch_and_stop_signal(self) -> None:
        stopper = EarlyStopper(patience=2, mode="max")
        self.assertTrue(stopper.update(0.5, epoch=1))
        self.assertEqual(stopper.best_epoch, 1)
        self.assertFalse(stopper.should_stop)

        self.assertFalse(stopper.update(0.45, epoch=2))
        self.assertFalse(stopper.should_stop)

        self.assertFalse(stopper.update(0.44, epoch=3))
        self.assertTrue(stopper.should_stop)
        self.assertEqual(stopper.best_score, 0.5)
        self.assertEqual(len(stopper.history), 3)


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "round_trip.pt"
            payload = {
                "state_dict": {"layer.weight": np.arange(3, dtype=np.float32)},
                "metadata": checkpoint_metadata(
                    experiment_name="unit_test",
                    stage="smoke",
                    config={"batch_size": 8, "seed": 42},
                    manifest_path="/tmp/manifest.json",
                    extra={"best_epoch": 2},
                ),
            }

            save_checkpoint(checkpoint_path, payload)
            loaded = load_checkpoint(checkpoint_path)

            self.assertIn("state_dict", loaded)
            self.assertIn("metadata", loaded)
            np.testing.assert_allclose(loaded["state_dict"]["layer.weight"], np.arange(3, dtype=np.float32))
            self.assertEqual(loaded["metadata"]["experiment_name"], "unit_test")
            self.assertEqual(loaded["metadata"]["best_epoch"], 2)
            self.assertEqual(loaded["metadata"]["config"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
