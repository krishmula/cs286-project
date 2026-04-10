from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping


CORE_EXPERIMENTS = [
    "watch_full",
    "watch_10pct",
    "contrastive_pair_probe_10pct",
    "contrastive_pair_probe_100pct",
    "contrastive_phone_probe_10pct",
    "contrastive_phone_probe_100pct",
    "contrastive_watch_probe_10pct",
    "contrastive_watch_probe_100pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Directory containing supervised *_metrics.json files.")
    parser.add_argument("--probe-dir", type=Path, required=True, help="Directory containing probe *_metrics.json files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write summary outputs.")
    parser.add_argument(
        "--experiment-alias",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Rename a metrics payload experiment name before summarizing. Useful for reusing older artifacts.",
    )
    parser.add_argument(
        "--expected-experiments",
        nargs="+",
        default=CORE_EXPERIMENTS,
        help="Experiment names that must be present in the output summary.",
    )
    return parser.parse_args()


def parse_experiment_aliases(alias_args: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for raw_alias in alias_args:
        source, separator, target = raw_alias.partition("=")
        source = source.strip()
        target = target.strip()
        if separator != "=" or not source or not target:
            raise ValueError(f"Invalid experiment alias {raw_alias!r}. Expected SOURCE=TARGET.")
        aliases[source] = target
    return aliases


def collect_metrics_payloads(*roots: Path, experiment_aliases: Mapping[str, str] | None = None) -> dict[str, dict]:
    payloads_by_experiment: dict[str, dict] = {}
    experiment_aliases = experiment_aliases or {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Metrics root does not exist: {root}")
        for metrics_path in sorted(root.rglob("*_metrics.json")):
            payload = json.loads(metrics_path.read_text())
            original_experiment_name = payload.get("experiment_name")
            experiment_name = experiment_aliases.get(original_experiment_name, original_experiment_name)
            if not experiment_name:
                continue
            if experiment_name in payloads_by_experiment:
                existing_entry = payloads_by_experiment[experiment_name]
                existing_path = Path(existing_entry["metrics_path"])
                existing_key = (existing_path.stat().st_mtime_ns, str(existing_path))
                candidate_key = (metrics_path.stat().st_mtime_ns, str(metrics_path))
                if candidate_key <= existing_key:
                    continue
            payloads_by_experiment[experiment_name] = {
                "payload": payload,
                "metrics_path": str(metrics_path),
                "original_experiment_name": original_experiment_name,
            }
    return payloads_by_experiment


def _row_from_payload(experiment_name: str, payload_entry: dict) -> dict:
    payload = payload_entry["payload"]
    metadata_config = payload.get("config", {})
    test_metrics = payload.get("test", {})
    probe_mode = payload.get("probe_mode")
    channel_mode = payload.get("channel_mode")
    if probe_mode:
        stage = "probe"
        mode = probe_mode
    else:
        stage = "supervised"
        mode = channel_mode
    return {
        "experiment_name": experiment_name,
        "stage": stage,
        "mode": mode,
        "label_fraction": payload.get("label_fraction", metadata_config.get("label_fraction")),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "metrics_path": payload_entry["metrics_path"],
    }


def summarize_runs(
    payloads_by_experiment: dict[str, dict],
    expected_experiments: Iterable[str],
) -> list[dict]:
    expected_list = list(expected_experiments)
    missing = [name for name in expected_list if name not in payloads_by_experiment]
    if missing:
        raise FileNotFoundError(
            "Missing expected experiment metrics: " + ", ".join(missing)
        )
    return [_row_from_payload(name, payloads_by_experiment[name]) for name in expected_list]


def _markdown_table(rows: list[dict]) -> str:
    lines = [
        "# Phase 2 Comparison Summary",
        "",
        "| Experiment | Stage | Mode | Label Fraction | Test Accuracy | Test Macro F1 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        label_fraction = row["label_fraction"]
        accuracy = row["test_accuracy"]
        macro_f1 = row["test_macro_f1"]
        lines.append(
            "| {experiment_name} | {stage} | {mode} | {label_fraction} | {accuracy:.4f} | {macro_f1:.4f} |".format(
                experiment_name=row["experiment_name"],
                stage=row["stage"],
                mode=row["mode"],
                label_fraction=label_fraction,
                accuracy=float(accuracy),
                macro_f1=float(macro_f1),
            )
        )
    return "\n".join(lines) + "\n"


def write_comparison_outputs(
    baseline_dir: Path,
    probe_dir: Path,
    output_dir: Path,
    expected_experiments: Iterable[str],
    experiment_aliases: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = collect_metrics_payloads(baseline_dir, probe_dir, experiment_aliases=experiment_aliases)
    rows = summarize_runs(payloads, expected_experiments)

    json_path = output_dir / "comparison_summary.json"
    markdown_path = output_dir / "comparison_summary.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle, indent=2)
    markdown_path.write_text(_markdown_table(rows), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def main() -> None:
    args = parse_args()
    outputs = write_comparison_outputs(
        baseline_dir=args.baseline_dir,
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        expected_experiments=args.expected_experiments,
        experiment_aliases=parse_experiment_aliases(args.experiment_alias),
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
