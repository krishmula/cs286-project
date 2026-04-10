from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .runtime import resolve_project_paths


RESERVED_RUNNER_ARGS = {
    "--project-root",
    "--output-dir",
    "--channel-mode",
    "--experiment-name",
    "--resume-from",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Project root containing artifacts/windows.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory where per-channel output directories should be created.",
    )
    parser.add_argument(
        "--channel-modes",
        nargs="+",
        choices=("phone", "watch"),
        default=("phone", "watch"),
        help="Supervised channel modes to run sequentially.",
    )
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="supervised-baseline",
        help="Each mode writes into output-root/<run-name-prefix>-<channel-mode>/.",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="supervised_baseline",
        help="Experiment names are formed as <experiment-prefix>_<channel-mode>.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from output-root/<run-name-prefix>-<channel-mode>/latest_checkpoint.pt.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to src.train_supervised. Prefix them with -- in the shell command.",
    )
    return parser.parse_args()


def strip_passthrough_delimiter(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def validate_runner_args(args: list[str]) -> None:
    reserved = sorted(argument for argument in args if argument in RESERVED_RUNNER_ARGS)
    if reserved:
        joined = ", ".join(reserved)
        raise ValueError(f"Do not forward reserved arguments via runner_args: {joined}")


def unique_channel_modes(channel_modes: list[str] | tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for channel_mode in channel_modes:
        if channel_mode in seen:
            continue
        seen.add(channel_mode)
        ordered.append(channel_mode)
    return ordered


def build_command(
    *,
    project_root: Path,
    output_root: Path,
    run_name_prefix: str,
    experiment_prefix: str,
    channel_mode: str,
    runner_args: list[str],
    no_resume: bool,
) -> tuple[list[str], Path, Path]:
    run_name = f"{run_name_prefix}-{channel_mode}"
    experiment_name = f"{experiment_prefix}_{channel_mode}"
    output_dir = (output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = output_dir / "latest_checkpoint.pt"

    command = [
        sys.executable,
        "-m",
        "src.train_supervised",
        "--project-root",
        str(project_root),
        "--output-dir",
        str(output_dir),
        "--channel-mode",
        channel_mode,
        "--experiment-name",
        experiment_name,
    ]
    if latest_checkpoint_path.exists() and not no_resume:
        command.extend(["--resume-from", str(latest_checkpoint_path)])
    command.extend(runner_args)
    return command, output_dir, latest_checkpoint_path


def main() -> None:
    args = parse_args()
    paths = resolve_project_paths(args.project_root)
    output_root = (args.output_root or (paths.baseline_root / "channel_baselines")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runner_args = strip_passthrough_delimiter(list(args.runner_args))
    validate_runner_args(runner_args)

    summaries = []
    for channel_mode in unique_channel_modes(args.channel_modes):
        command, output_dir, latest_checkpoint_path = build_command(
            project_root=paths.project_root,
            output_root=output_root,
            run_name_prefix=args.run_name_prefix,
            experiment_prefix=args.experiment_prefix,
            channel_mode=channel_mode,
            runner_args=runner_args,
            no_resume=args.no_resume,
        )

        print(
            json.dumps(
                {
                    "channel_mode": channel_mode,
                    "output_dir": str(output_dir),
                    "latest_checkpoint_path": str(latest_checkpoint_path),
                    "resuming": latest_checkpoint_path.exists() and not args.no_resume,
                    "command": command,
                },
                indent=2,
            )
        )
        subprocess.run(command, cwd=paths.project_root, check=True)
        summaries.append({"channel_mode": channel_mode, "output_dir": str(output_dir)})

    print(json.dumps({"runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
