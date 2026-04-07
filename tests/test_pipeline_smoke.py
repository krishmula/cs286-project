from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTests(unittest.TestCase):
    def test_end_to_end_phase2_pipeline_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_dir = root / "baseline"
            contrastive_dir = root / "contrastive"
            probe_dir = root / "probes"
            reports_dir = root / "reports"

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.train_supervised",
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--output-dir",
                    str(baseline_dir),
                    "--experiment-name",
                    "fusion_full",
                    "--channel-mode",
                    "fusion",
                    "--label-fraction",
                    "1.0",
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
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.train_supervised",
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--output-dir",
                    str(baseline_dir),
                    "--experiment-name",
                    "fusion_10pct",
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
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            subprocess.run(
                [
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
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            contrastive_checkpoint = next(contrastive_dir.glob("*_checkpoint.pt"))

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.train_probe",
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--encoder-ckpt-path",
                    str(contrastive_checkpoint),
                    "--output-dir",
                    str(probe_dir),
                    "--experiment-name",
                    "contrastive_pair_probe_10pct",
                    "--evaluation-mode",
                    "pair",
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
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.reporting.compare_runs",
                    "--baseline-dir",
                    str(baseline_dir),
                    "--probe-dir",
                    str(probe_dir),
                    "--output-dir",
                    str(reports_dir),
                    "--expected-experiments",
                    "fusion_full",
                    "fusion_10pct",
                    "contrastive_pair_probe_10pct",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            summary_json = reports_dir / "comparison_summary.json"
            summary_md = reports_dir / "comparison_summary.md"
            self.assertTrue(summary_json.exists())
            self.assertTrue(summary_md.exists())
            summary = json.loads(summary_json.read_text())
            self.assertEqual(
                [row["experiment_name"] for row in summary["results"]],
                ["fusion_full", "fusion_10pct", "contrastive_pair_probe_10pct"],
            )


if __name__ == "__main__":
    unittest.main()
