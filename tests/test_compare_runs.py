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

            (baseline_dir / "watch_full_metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "watch_full",
                        "channel_mode": "watch",
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
                expected_experiments=["watch_full", "contrastive_pair_probe_10pct"],
            )

            self.assertTrue(outputs["json"].exists())
            self.assertTrue(outputs["markdown"].exists())
            payload = json.loads(outputs["json"].read_text())
            self.assertEqual([row["experiment_name"] for row in payload["results"]], ["watch_full", "contrastive_pair_probe_10pct"])
            markdown = outputs["markdown"].read_text()
            self.assertIn("| watch_full | supervised | watch | 1.0 | 0.5000 | 0.4000 |", markdown)
            self.assertIn("| contrastive_pair_probe_10pct | probe | pair | 0.1 | 0.6000 | 0.5500 |", markdown)

    def test_missing_expected_experiment_raises(self) -> None:
        payloads = {
            "watch_full": {
                "payload": {
                    "experiment_name": "watch_full",
                    "channel_mode": "watch",
                    "label_fraction": 1.0,
                    "test": {"accuracy": 0.5, "macro_f1": 0.4},
                },
                "metrics_path": "/tmp/watch_full_metrics.json",
            }
        }
        with self.assertRaises(FileNotFoundError):
            summarize_runs(payloads, ["watch_full", "contrastive_pair_probe_10pct"])

    def test_experiment_aliases_allow_reusing_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_dir = root / "baseline"
            probe_dir = root / "probes"
            output_dir = root / "reports"
            baseline_dir.mkdir()
            probe_dir.mkdir()

            (baseline_dir / "standalone_supervised_watch_metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "standalone_supervised_watch",
                        "channel_mode": "watch",
                        "label_fraction": 1.0,
                        "test": {"accuracy": 0.62, "macro_f1": 0.63},
                    }
                ),
                encoding="utf-8",
            )

            outputs = write_comparison_outputs(
                baseline_dir=baseline_dir,
                probe_dir=probe_dir,
                output_dir=output_dir,
                expected_experiments=["watch_full"],
                experiment_aliases={"standalone_supervised_watch": "watch_full"},
            )

            payload = json.loads(outputs["json"].read_text())
            self.assertEqual(payload["results"][0]["experiment_name"], "watch_full")
            self.assertEqual(payload["results"][0]["mode"], "watch")

    def test_newest_duplicate_metrics_file_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_dir = root / "baseline"
            probe_dir = root / "probes"
            output_dir = root / "reports"
            baseline_dir.mkdir()
            probe_dir.mkdir()

            older = baseline_dir / "watch_full_old_metrics.json"
            newer = baseline_dir / "watch_full_new_metrics.json"
            older.write_text(
                json.dumps(
                    {
                        "experiment_name": "watch_full",
                        "channel_mode": "watch",
                        "label_fraction": 1.0,
                        "test": {"accuracy": 0.5, "macro_f1": 0.4},
                    }
                ),
                encoding="utf-8",
            )
            newer.write_text(
                json.dumps(
                    {
                        "experiment_name": "watch_full",
                        "channel_mode": "watch",
                        "label_fraction": 1.0,
                        "test": {"accuracy": 0.6, "macro_f1": 0.55},
                    }
                ),
                encoding="utf-8",
            )

            outputs = write_comparison_outputs(
                baseline_dir=baseline_dir,
                probe_dir=probe_dir,
                output_dir=output_dir,
                expected_experiments=["watch_full"],
            )

            payload = json.loads(outputs["json"].read_text())
            self.assertEqual(payload["results"][0]["test_accuracy"], 0.6)
            self.assertEqual(payload["results"][0]["test_macro_f1"], 0.55)

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
            (baseline_dir / "watch_full_metrics.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "watch_full",
                        "channel_mode": "watch",
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
                expected_experiments=["watch_full"],
            )
            self.assertTrue(outputs["json"].exists())


if __name__ == "__main__":
    unittest.main()
