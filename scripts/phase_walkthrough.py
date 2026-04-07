#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_HZ = 20
WINDOW_SECONDS = 3.0
WINDOW_STRIDE_SECONDS = 1.0
WINDOW_LENGTH = int(TARGET_HZ * WINDOW_SECONDS)

REQUIRED_STREAMS: Tuple[Tuple[str, str], ...] = (
    ("phone", "accel"),
    ("phone", "gyro"),
    ("watch", "accel"),
    ("watch", "gyro"),
)


@dataclass(frozen=True)
class RawRow:
    subject_id: int
    activity_label: str
    timestamp_ns: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class StreamInfo:
    split: str
    device: str
    sensor: str
    subject_id: int
    path: Path


@dataclass
class StreamData:
    info: StreamInfo
    rows: List[RawRow]


@dataclass(frozen=True)
class SegmentRecord:
    split: str
    subject_id: int
    device: str
    sensor: str
    activity_label: str
    occurrence_index: int
    segment_index_global: int
    start_row: int
    end_row_exclusive: int
    n_rows: int
    start_ts_ns: int
    end_ts_ns: int
    duration_s: float
    native_hz: Optional[float]
    is_valid: bool
    drop_reason: str


@dataclass(frozen=True)
class PairedSegmentRecord:
    split: str
    subject_id: int
    activity_label: str
    occurrence_index: int
    common_duration_s: float
    is_valid_four_stream_pair: bool
    drop_reason: str
    segments_by_stream: Dict[Tuple[str, str], Optional[SegmentRecord]]


def find_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd
    if (cwd.parent / "data").exists():
        return cwd.parent
    raise FileNotFoundError("Could not locate the project root containing data/")


def parse_sensor_filename(path: Path) -> StreamInfo:
    parts = path.parts
    split = parts[-4]
    device = parts[-3]
    sensor = parts[-2]

    stem_parts = path.stem.split("_")
    if len(stem_parts) != 4 or stem_parts[0] != "data":
        raise ValueError(f"Unexpected filename format: {path.name}")

    subject_id = int(stem_parts[1])
    if stem_parts[2] != sensor or stem_parts[3] != device:
        raise ValueError(f"Filename stream mismatch for {path.name}")

    return StreamInfo(
        split=split,
        device=device,
        sensor=sensor,
        subject_id=subject_id,
        path=path,
    )


def iter_sensor_rows(path: Path) -> Iterator[RawRow]:
    info = parse_sensor_filename(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(";"):
                line = line[:-1]
            fields = line.split(",")
            if len(fields) != 6:
                raise ValueError(f"{path}:{line_number} expected 6 fields, found {len(fields)}")

            row = RawRow(
                subject_id=int(fields[0]),
                activity_label=fields[1],
                timestamp_ns=int(fields[2]),
                x=float(fields[3]),
                y=float(fields[4]),
                z=float(fields[5]),
            )
            if row.subject_id != info.subject_id:
                raise ValueError(
                    f"{path}:{line_number} subject id {row.subject_id} does not match filename {info.subject_id}"
                )
            yield row


def load_stream(path: Path) -> StreamData:
    info = parse_sensor_filename(path)
    rows = list(iter_sensor_rows(path))
    if not rows:
        raise ValueError(f"Empty stream file: {path}")
    return StreamData(info=info, rows=rows)


def load_subject_streams(data_root: Path, split: str, subject_id: int) -> Dict[Tuple[str, str], StreamData]:
    streams: Dict[Tuple[str, str], StreamData] = {}
    for device, sensor in REQUIRED_STREAMS:
        path = data_root / split / device / sensor / f"data_{subject_id}_{sensor}_{device}.txt"
        streams[(device, sensor)] = load_stream(path)
    return streams


def infer_native_hz(timestamps_ns: Sequence[int]) -> Optional[float]:
    positive_diffs = [
        next_ts - current_ts
        for current_ts, next_ts in zip(timestamps_ns, timestamps_ns[1:])
        if next_ts > current_ts
    ]
    if not positive_diffs:
        return None
    median_dt_ns = statistics.median(positive_diffs)
    if median_dt_ns <= 0:
        return None
    return 1_000_000_000.0 / median_dt_ns


def extract_segments(stream: StreamData) -> List[SegmentRecord]:
    labels = [row.activity_label for row in stream.rows]
    timestamps_ns = [row.timestamp_ns for row in stream.rows]

    segments: List[SegmentRecord] = []
    occurrence_counter: Dict[str, int] = defaultdict(int)

    start_row = 0
    current_label = labels[0]
    current_occurrence = occurrence_counter[current_label]
    segment_index_global = 0

    for row_index in range(1, len(labels) + 1):
        at_end = row_index == len(labels)
        label_changed = not at_end and labels[row_index] != current_label
        if not at_end and not label_changed:
            continue

        end_row_exclusive = row_index
        segment_timestamps = timestamps_ns[start_row:end_row_exclusive]
        start_ts_ns = segment_timestamps[0]
        end_ts_ns = segment_timestamps[-1]
        native_hz = infer_native_hz(segment_timestamps)
        duration_s = max(0.0, (end_ts_ns - start_ts_ns) / 1_000_000_000.0)

        is_valid = True
        drop_reason = ""
        if len(segment_timestamps) < 2:
            is_valid = False
            drop_reason = "fewer_than_two_rows"
        elif native_hz is None:
            is_valid = False
            drop_reason = "missing_positive_timestamp_deltas"
        elif any(next_ts <= current_ts for current_ts, next_ts in zip(segment_timestamps, segment_timestamps[1:])):
            is_valid = False
            drop_reason = "non_monotonic_timestamp_within_segment"

        segments.append(
            SegmentRecord(
                split=stream.info.split,
                subject_id=stream.info.subject_id,
                device=stream.info.device,
                sensor=stream.info.sensor,
                activity_label=current_label,
                occurrence_index=current_occurrence,
                segment_index_global=segment_index_global,
                start_row=start_row,
                end_row_exclusive=end_row_exclusive,
                n_rows=end_row_exclusive - start_row,
                start_ts_ns=start_ts_ns,
                end_ts_ns=end_ts_ns,
                duration_s=duration_s,
                native_hz=native_hz,
                is_valid=is_valid,
                drop_reason=drop_reason,
            )
        )

        occurrence_counter[current_label] += 1
        segment_index_global += 1

        if not at_end:
            start_row = row_index
            current_label = labels[row_index]
            current_occurrence = occurrence_counter[current_label]

    return segments


def pair_segments_for_subject(
    split: str,
    subject_id: int,
    segments_by_stream: Dict[Tuple[str, str], List[SegmentRecord]],
) -> List[PairedSegmentRecord]:
    keyed_segments: Dict[Tuple[str, str], Dict[Tuple[str, int], SegmentRecord]] = {}
    order_index: Dict[Tuple[str, int], int] = {}
    all_keys = set()

    for stream, segment_list in segments_by_stream.items():
        segment_map: Dict[Tuple[str, int], SegmentRecord] = {}
        keyed_segments[stream] = segment_map
        for segment in segment_list:
            key = (segment.activity_label, segment.occurrence_index)
            segment_map[key] = segment
            all_keys.add(key)
            existing_order = order_index.get(key)
            if existing_order is None or segment.segment_index_global < existing_order:
                order_index[key] = segment.segment_index_global

    paired_records: List[PairedSegmentRecord] = []
    ordered_keys = sorted(all_keys, key=lambda key: (order_index.get(key, 10**9), key[0], key[1]))

    for activity_label, occurrence_index in ordered_keys:
        segments = {
            stream: keyed_segments.get(stream, {}).get((activity_label, occurrence_index))
            for stream in REQUIRED_STREAMS
        }

        missing_streams = [stream for stream, segment in segments.items() if segment is None]
        if missing_streams:
            paired_records.append(
                PairedSegmentRecord(
                    split=split,
                    subject_id=subject_id,
                    activity_label=activity_label,
                    occurrence_index=occurrence_index,
                    common_duration_s=0.0,
                    is_valid_four_stream_pair=False,
                    drop_reason="missing_streams:" + "|".join(f"{d}_{s}" for d, s in missing_streams),
                    segments_by_stream=segments,
                )
            )
            continue

        duration_values = [segment.duration_s for segment in segments.values() if segment is not None]
        common_duration_s = min(duration_values)
        invalid_segments = [segment for segment in segments.values() if segment and not segment.is_valid]

        is_valid = True
        drop_reason = ""
        if invalid_segments:
            is_valid = False
            invalid_reasons = sorted({segment.drop_reason for segment in invalid_segments})
            drop_reason = "invalid_segment:" + "|".join(invalid_reasons)
        elif common_duration_s < WINDOW_SECONDS:
            is_valid = False
            drop_reason = "segment_shorter_than_window"

        paired_records.append(
            PairedSegmentRecord(
                split=split,
                subject_id=subject_id,
                activity_label=activity_label,
                occurrence_index=occurrence_index,
                common_duration_s=common_duration_s,
                is_valid_four_stream_pair=is_valid,
                drop_reason=drop_reason,
                segments_by_stream=segments,
            )
        )

    return paired_records


def extract_segment_times_and_values(
    stream: StreamData,
    segment: SegmentRecord,
) -> Tuple[List[float], List[Tuple[float, float, float]]]:
    rows = stream.rows[segment.start_row:segment.end_row_exclusive]
    start_ts_ns = rows[0].timestamp_ns
    times_s = [(row.timestamp_ns - start_ts_ns) / 1_000_000_000.0 for row in rows]
    values_xyz = [(row.x, row.y, row.z) for row in rows]
    return times_s, values_xyz


def generate_window_centers(common_duration_s: float) -> List[float]:
    half_window = WINDOW_SECONDS / 2.0
    centers: List[float] = []
    center = half_window
    max_center = common_duration_s - half_window
    epsilon = 1e-9
    while center <= max_center + epsilon:
        centers.append(center)
        center += WINDOW_STRIDE_SECONDS
    return centers


def build_query_times(window_start_s: float) -> List[float]:
    return [window_start_s + (index / TARGET_HZ) for index in range(WINDOW_LENGTH)]


def interpolate_stream(
    times_s: Sequence[float],
    values_xyz: Sequence[Tuple[float, float, float]],
    query_times_s: Sequence[float],
) -> List[List[float]]:
    axis_values = [
        [xyz[0] for xyz in values_xyz],
        [xyz[1] for xyz in values_xyz],
        [xyz[2] for xyz in values_xyz],
    ]
    output = [[0.0 for _ in query_times_s] for _ in range(3)]
    pointer = 0

    for query_index, query_time_s in enumerate(query_times_s):
        if query_time_s <= times_s[0]:
            left_index = 0
            right_index = 1
        elif query_time_s >= times_s[-1]:
            left_index = len(times_s) - 2
            right_index = len(times_s) - 1
        else:
            while pointer + 1 < len(times_s) and times_s[pointer + 1] < query_time_s:
                pointer += 1
            left_index = pointer
            right_index = pointer + 1

        left_time = times_s[left_index]
        right_time = times_s[right_index]
        if right_time <= left_time:
            alpha = 0.0
        else:
            alpha = (query_time_s - left_time) / (right_time - left_time)
            alpha = min(1.0, max(0.0, alpha))

        for axis in range(3):
            left_value = axis_values[axis][left_index]
            right_value = axis_values[axis][right_index]
            output[axis][query_index] = left_value + alpha * (right_value - left_value)

    return output


def print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def format_runs(labels: Sequence[str], max_runs: int = 16) -> str:
    runs: List[str] = []
    if not labels:
        return "(empty)"
    current = labels[0]
    length = 1
    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            runs.append(f"{current}x{length}")
            current = label
            length = 1
    runs.append(f"{current}x{length}")
    if len(runs) > max_runs:
        return " | ".join(runs[:max_runs]) + " | ..."
    return " | ".join(runs)


def print_stream_overview(stream: StreamData, segments: Sequence[SegmentRecord], raw_limit: int) -> None:
    labels = [row.activity_label for row in stream.rows]
    print(f"{stream.info.device}/{stream.info.sensor}")
    print(f"  file: {stream.info.path}")
    print(f"  rows: {len(stream.rows)}")
    print(f"  first {raw_limit} raw labels: {' '.join(labels[:raw_limit])}")
    print(f"  contiguous runs: {format_runs(labels[:raw_limit])}")
    print("  first segments:")
    for segment in segments[:8]:
        print(
            "   "
            + str(
                {
                    "segment": f"{segment.activity_label}{segment.occurrence_index}",
                    "rows": (segment.start_row, segment.end_row_exclusive),
                    "n_rows": segment.n_rows,
                    "duration_s": round(segment.duration_s, 3),
                    "native_hz": None if segment.native_hz is None else round(segment.native_hz, 2),
                    "valid": segment.is_valid,
                }
            )
        )


def plot_label_timeline(
    labels_by_name: Dict[str, Sequence[str]],
    output_path: Path,
    raw_limit: int,
) -> None:
    labels_present = sorted({label for labels in labels_by_name.values() for label in labels[:raw_limit]})
    label_to_y = {label: idx for idx, label in enumerate(labels_present)}

    fig, axes = plt.subplots(len(labels_by_name), 1, figsize=(12, 4 + 2 * len(labels_by_name)), sharex=True)
    if len(labels_by_name) == 1:
        axes = [axes]

    for ax, (name, labels) in zip(axes, labels_by_name.items()):
        rows = list(range(min(raw_limit, len(labels))))
        ys = [label_to_y[label] for label in labels[:raw_limit]]
        ax.step(rows, ys, where="post", linewidth=2)
        ax.scatter(rows, ys, s=18)
        ax.set_title(f"{name} raw label timeline")
        ax.set_ylabel("label")
        ax.set_yticks(list(label_to_y.values()))
        ax.set_yticklabels(list(label_to_y.keys()))
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("row index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_segment_and_window(
    pair: PairedSegmentRecord,
    subject_streams: Dict[Tuple[str, str], StreamData],
    query_times_s: Sequence[float],
    output_path: Path,
) -> None:
    phone_stream = subject_streams[("phone", "accel")]
    watch_stream = subject_streams[("watch", "accel")]
    phone_segment = pair.segments_by_stream[("phone", "accel")]
    watch_segment = pair.segments_by_stream[("watch", "accel")]
    assert phone_segment is not None
    assert watch_segment is not None

    phone_times_s, phone_values = extract_segment_times_and_values(phone_stream, phone_segment)
    watch_times_s, watch_values = extract_segment_times_and_values(watch_stream, watch_segment)

    phone_interp = interpolate_stream(phone_times_s, phone_values, query_times_s)
    watch_interp = interpolate_stream(watch_times_s, watch_values, query_times_s)

    phone_x = [xyz[0] for xyz in phone_values]
    watch_x = [xyz[0] for xyz in watch_values]
    window_start_s = query_times_s[0]
    window_end_s = query_times_s[-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(phone_times_s, phone_x, marker="o", markersize=3, linewidth=1.5, label="phone accel x")
    axes[0].plot(watch_times_s, watch_x, marker="o", markersize=3, linewidth=1.5, label="watch accel x")
    axes[0].axvspan(window_start_s, window_end_s, color="gold", alpha=0.2, label="selected window")
    axes[0].set_title(
        f"Selected paired segment {pair.activity_label}{pair.occurrence_index} for subject {pair.subject_id}"
    )
    axes[0].set_xlabel("relative segment time (s)")
    axes[0].set_ylabel("raw x value")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(query_times_s, phone_interp[0], linewidth=2, label="phone interpolated x")
    axes[1].plot(query_times_s, watch_interp[0], linewidth=2, label="watch interpolated x")
    axes[1].scatter(query_times_s, phone_interp[0], s=14)
    axes[1].scatter(query_times_s, watch_interp[0], s=14)
    axes[1].set_title("Window resampled onto the shared 20 Hz time grid")
    axes[1].set_xlabel("window-relative time (s)")
    axes[1].set_ylabel("interpolated x value")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def choose_pair(
    paired_segments: Sequence[PairedSegmentRecord],
    activity_label: Optional[str],
    occurrence_index: Optional[int],
) -> PairedSegmentRecord:
    candidates = [pair for pair in paired_segments if pair.is_valid_four_stream_pair]
    if activity_label is not None:
        candidates = [pair for pair in candidates if pair.activity_label == activity_label]
    if occurrence_index is not None:
        candidates = [pair for pair in candidates if pair.occurrence_index == occurrence_index]
    if not candidates:
        raise ValueError("No valid paired segment matched the requested filters.")
    return candidates[0]


def build_fused_window(
    pair: PairedSegmentRecord,
    subject_streams: Dict[Tuple[str, str], StreamData],
    query_times_s: Sequence[float],
) -> List[List[float]]:
    fused_channels: List[List[float]] = []
    for stream_key in REQUIRED_STREAMS:
        segment = pair.segments_by_stream[stream_key]
        assert segment is not None
        times_s, values_xyz = extract_segment_times_and_values(subject_streams[stream_key], segment)
        fused_channels.extend(interpolate_stream(times_s, values_xyz, query_times_s))
    return fused_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small, visual walkthrough of Phase 1 and Phase 2 on one subject and one activity episode."
    )
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Raw dataset split.")
    parser.add_argument("--subject", type=int, default=1600, help="Subject id to inspect.")
    parser.add_argument("--activity", default=None, help="Optional activity label filter, for example A.")
    parser.add_argument("--occurrence", type=int, default=None, help="Optional occurrence index filter.")
    parser.add_argument("--window-index", type=int, default=0, help="Which generated window to inspect.")
    parser.add_argument("--raw-limit", type=int, default=80, help="How many raw labels to print and plot.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots. Defaults to artifacts/demo_walkthrough.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    data_root = project_root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "artifacts" / "demo_walkthrough"
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_streams = load_subject_streams(data_root, args.split, args.subject)
    segments_by_stream = {stream_key: extract_segments(stream) for stream_key, stream in subject_streams.items()}
    paired_segments = pair_segments_for_subject(args.split, args.subject, segments_by_stream)
    pair = choose_pair(paired_segments, args.activity, args.occurrence)

    print_header("Phase 1: raw rows -> contiguous segments")
    print(
        f"Using split={args.split}, subject={args.subject}, "
        f"selected pair={pair.activity_label}{pair.occurrence_index}"
    )
    print(
        "Full pipeline requires 4 streams: "
        + ", ".join(f"{device}/{sensor}" for device, sensor in REQUIRED_STREAMS)
    )
    print()
    for stream_key in (("phone", "accel"), ("watch", "accel")):
        print_stream_overview(subject_streams[stream_key], segments_by_stream[stream_key], args.raw_limit)
        print()

    print_header("Phase 1: paired activity episodes across streams")
    print("First paired records:")
    for paired in paired_segments[:10]:
        print(
            {
                "pair": f"{paired.activity_label}{paired.occurrence_index}",
                "common_duration_s": round(paired.common_duration_s, 3),
                "valid": paired.is_valid_four_stream_pair,
                "drop_reason": paired.drop_reason,
            }
        )
    print()
    print("Selected pair details:")
    print(
        {
            "pair": f"{pair.activity_label}{pair.occurrence_index}",
            "common_duration_s": round(pair.common_duration_s, 3),
            "valid": pair.is_valid_four_stream_pair,
        }
    )
    for stream_key in REQUIRED_STREAMS:
        segment = pair.segments_by_stream[stream_key]
        assert segment is not None
        print(
            " "
            + str(
                {
                    "stream": f"{stream_key[0]}/{stream_key[1]}",
                    "rows": (segment.start_row, segment.end_row_exclusive),
                    "duration_s": round(segment.duration_s, 3),
                    "native_hz": None if segment.native_hz is None else round(segment.native_hz, 2),
                }
            )
        )

    print_header("Phase 2: paired segment -> sliding windows -> fused tensor")
    centers = generate_window_centers(pair.common_duration_s)
    if not centers:
        raise ValueError("Selected pair does not produce any full windows.")
    if args.window_index < 0 or args.window_index >= len(centers):
        raise ValueError(f"window-index must be between 0 and {len(centers) - 1}")

    center_s = centers[args.window_index]
    window_start_s = center_s - (WINDOW_SECONDS / 2.0)
    window_end_s = window_start_s + WINDOW_SECONDS
    query_times_s = build_query_times(window_start_s)
    fused_window = build_fused_window(pair, subject_streams, query_times_s)

    print(f"Window centers: {[round(center, 3) for center in centers[:12]]}")
    if len(centers) > 12:
        print(f"... plus {len(centers) - 12} more windows")
    print(
        {
            "selected_window_index": args.window_index,
            "start_s": round(window_start_s, 3),
            "end_s": round(window_end_s, 3),
            "query_times_head": [round(value, 3) for value in query_times_s[:10]],
        }
    )
    print("Fused window shape:", (len(fused_window), len(fused_window[0])))
    print("Channel order:")
    print(" phone/accel[x,y,z], phone/gyro[x,y,z], watch/accel[x,y,z], watch/gyro[x,y,z]")
    print("First 8 values of the first 4 channels:")
    for channel_index in range(4):
        print(f" channel_{channel_index}: {[round(value, 3) for value in fused_window[channel_index][:8]]}")

    label_plot_path = output_dir / f"subject_{args.subject}_{pair.activity_label}{pair.occurrence_index}_labels.png"
    segment_plot_path = output_dir / f"subject_{args.subject}_{pair.activity_label}{pair.occurrence_index}_window.png"

    plot_label_timeline(
        {
            "phone/accel": [row.activity_label for row in subject_streams[("phone", "accel")].rows],
            "watch/accel": [row.activity_label for row in subject_streams[("watch", "accel")].rows],
        },
        label_plot_path,
        args.raw_limit,
    )
    plot_segment_and_window(pair, subject_streams, query_times_s, segment_plot_path)

    print()
    print(f"Saved label timeline plot to: {label_plot_path}")
    print(f"Saved segment/window plot to: {segment_plot_path}")


if __name__ == "__main__":
    main()
