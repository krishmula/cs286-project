from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


RUNNER_MODULES = {
    "supervised": "src.train_supervised",
    "contrastive": "src.train_contrastive",
    "probe": "src.train_probe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(RUNNER_MODULES), required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/cs286-project-runs"),
        help="Base directory on Google Drive where run artifacts should be written.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Subdirectory name under drive-root/<stage>/ used for checkpoints and metrics.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name forwarded to the underlying training module.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Overwrite latest_checkpoint.pt every N epochs in the Drive output directory.",
    )
    parser.add_argument(
        "--skip-drive-mount",
        action="store_true",
        help="Skip google.colab Drive mounting. Useful for local smoke tests.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running without CUDA. Intended for local validation rather than Colab training.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh run even if drive-root/<stage>/<run-name>/latest_checkpoint.pt exists.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying training module. Prefix them with -- in the shell command.",
    )
    return parser.parse_args()


def strip_passthrough_delimiter(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def mount_google_drive(skip_drive_mount: bool) -> None:
    if skip_drive_mount:
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "google.colab is unavailable. Run this inside Colab or pass --skip-drive-mount for local testing."
        ) from exc
    drive.mount("/content/drive", force_remount=False)


def resolve_device(*, allow_cpu: bool) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch must be installed before launching Colab training.") from exc

    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise RuntimeError("CUDA is not available. In Colab, switch the runtime to a GPU before training.")


def build_stage_command(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    device: str,
    latest_checkpoint_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        RUNNER_MODULES[args.stage],
        "--project-root",
        str(args.project_root.resolve()),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--save-every",
        str(args.save_every),
    ]
    if args.experiment_name is not None:
        command.extend(["--experiment-name", args.experiment_name])
    if latest_checkpoint_path.exists() and not args.no_resume:
        command.extend(["--resume-from", str(latest_checkpoint_path)])
    command.extend(strip_passthrough_delimiter(list(args.runner_args)))
    return command


def main() -> None:
    args = parse_args()
    mount_google_drive(args.skip_drive_mount)
    device = resolve_device(allow_cpu=args.allow_cpu)

    output_dir = (args.drive_root / args.stage / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = output_dir / "latest_checkpoint.pt"
    command = build_stage_command(
        args=args,
        output_dir=output_dir,
        device=device,
        latest_checkpoint_path=latest_checkpoint_path,
    )

    print(
        json.dumps(
            {
                "stage": args.stage,
                "device": device,
                "output_dir": str(output_dir),
                "latest_checkpoint_path": str(latest_checkpoint_path),
                "resuming": latest_checkpoint_path.exists() and not args.no_resume,
                "command": command,
            },
            indent=2,
        )
    )
    subprocess.run(command, cwd=args.project_root.resolve(), check=True)


if __name__ == "__main__":
    main()
