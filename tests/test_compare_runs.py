from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.reporting.compare_runs import summarize_runs, write_comparison_outputs


class CompareRunsTests(unittest.TestCase):
    def test_write_comparison_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_dir = root / "baseline"
            probe_dir = root / "probes"
            output_dir = root / "reports"
            baseline_dir.mkdir()
            probe_dir.mkdir()

            (baseline_dir / "fusion_full_metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "fusion_full",
                        "channel_mode": "fusion",
                        "label_fraction": 1.0,
                        "test": {"accuracy": 0.5, "macro_f1": 0.4},
                    }
                ),
                encoding="utf-8",
            )
            (probe_dir / "contrastive_pair_probe_10pct_metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "contrastive_pair_probe_10pct",
                        "probe_mode": "pair",
                        "label_fraction": 0.1,
                        "test": {"accuracy": 0.6, "macro_f1": 0.55},
                    }
                ),
                encoding="utf-8",
            )

            outputs = write_comparison_outputs(
                baseline_dir=baseline_dir,
                probe_dir=probe_dir,
                output_dir=output_dir,
                expected_experiments=["fusion_full", "contrastive_pair_probe_10pct"],
            )

            self.assertTrue(outputs["json"].exists())
            self.assertTrue(outputs["markdown"].exists())
            payload = json.loads(outputs["json"].read_text())
            self.assertEqual([row["experiment_name"] for row in payload["results"]], ["fusion_full", "contrastive_pair_probe_10pct"])
            markdown = outputs["markdown"].read_text()
            self.assertIn("| fusion_full | supervised | fusion | 1.0 | 0.5000 | 0.4000 |", markdown)
            self.assertIn("| contrastive_pair_probe_10pct | probe | pair | 0.1 | 0.6000 | 0.5500 |", markdown)

    def test_missing_expected_experiment_raises(self) -> None:
        payloads = {
            "fusion_full": {
                "payload": {
                    "experiment_name": "fusion_full",
                    "channel_mode": "fusion",
                    "label_fraction": 1.0,
                    "test": {"accuracy": 0.5, "macro_f1": 0.4},
                },
                "metrics_path": "/tmp/fusion_full_metrics.json",
            }
        }
        with self.assertRaises(FileNotFoundError):
            summarize_runs(payloads, ["fusion_full", "contrastive_pair_probe_10pct"])

    def test_legacy_metrics_without_experiment_name_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_dir = root / "baseline"
            probe_dir = root / "probes"
            output_dir = root / "reports"
            baseline_dir.mkdir()
            probe_dir.mkdir()

            (baseline_dir / "legacy_metrics.json").write_text(
                json.dumps({"best_epoch": 2, "test": {"accuracy": 0.5, "macro_f1": 0.4}}),
                encoding="utf-8",
            )
            (baseline_dir / "fusion_full_metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "fusion_full",
                        "channel_mode": "fusion",
                        "label_fraction": 1.0,
                        "test": {"accuracy": 0.5, "macro_f1": 0.4},
                    }
                ),
                encoding="utf-8",
            )

            outputs = write_comparison_outputs(
                baseline_dir=baseline_dir,
                probe_dir=probe_dir,
                output_dir=output_dir,
                expected_experiments=["fusion_full"],
            )
            self.assertTrue(outputs["json"].exists())


if __name__ == "__main__":
    unittest.main()
