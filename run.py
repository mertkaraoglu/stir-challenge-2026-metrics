"""STIR 2026 metrics runner.

Scans a results directory for inference outputs, computes AJ / ATA / OA for
every model found, saves per-model ``metrics.json``, then generates comparison
plots and a summary table.

Usage::

    uv run run.py results/ --data_dir /path/to/dataset

Expected results layout (written by stir-challenge-2026-inference)::

    results/
      mono/<model_name>/preds.json
      stereo/<model_name>/preds.json

Outputs written by this script::

    results/
      mono/<model_name>/metrics.json
      stereo/<model_name>/metrics.json
      plots/
        mono_overall.png
        stereo_overall.png
        mono_<seq_id>.png        (one per sequence, mono)
        stereo_<seq_id>.png      (one per sequence, stereo)
        summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

from dataset import Annotation, CameraCalib
from utils import compute_metrics, unproject

MONO_THRESHOLDS: tuple[float, ...] = (2, 4, 8, 16, 32)
STEREO_THRESHOLDS: tuple[float, ...] = (0.002, 0.004, 0.008, 0.016, 0.032)
WARMUP_FRAMES = 10


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


def _load_annotations(seq_dir: Path) -> dict[int, Annotation]:
    ann_path = seq_dir / "annotations.json"
    if not ann_path.exists():
        return {}
    with open(ann_path) as fh:
        raw = json.load(fh)
    result: dict[int, Annotation] = {}
    for key, data in raw.items():
        frame_idx, instance_id = (int(v) for v in key.split("/"))
        if instance_id != 0:
            continue
        lx = np.array(data["left_x"], dtype=np.float32)
        ly = np.array(data["left_y"], dtype=np.float32)
        disp = np.array(data["disparity"], dtype=np.float32)
        result[frame_idx] = Annotation(
            left_coords=np.stack([lx, ly], axis=1),
            right_coords=np.stack([lx - disp, ly], axis=1),
            left_visibs=np.array(data["left_visibility"], dtype=bool),
            right_visibs=np.array(data["right_visibility"], dtype=bool),
        )
    return result


def _load_calib(seq_dir: Path) -> CameraCalib:
    with open(seq_dir / "calib.json") as fh:
        return CameraCalib.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------


def _process_mono_sequence(
    seq_id: str,
    seq_tracks: dict,
    data_dir: Path,
) -> tuple[
    dict[str, float],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    seq_dir = data_dir / seq_id
    annotations = _load_annotations(seq_dir)

    loc_gt_frames: list[np.ndarray] = []
    vis_gt_frames: list[np.ndarray] = []
    loc_pred_frames: list[np.ndarray] = []
    vis_pred_frames: list[np.ndarray] = []

    for frame_idx_str, frame_data in seq_tracks.items():
        frame_idx = int(frame_idx_str)
        # Skip the first frame (it's the reference frame)
        if frame_idx == 0:
            continue
        if frame_idx not in annotations:
            continue
        ann = annotations[frame_idx]
        coords = np.array(frame_data["coords"], dtype=np.float32)
        visibs = np.array(frame_data["visibilities"], dtype=bool)
        loc_gt_frames.append(ann.left_coords)
        vis_gt_frames.append(ann.left_visibs)
        loc_pred_frames.append(coords)
        vis_pred_frames.append(visibs)

    if not loc_gt_frames:
        print(f"  [{seq_id}] no annotated frames — skipping")
        return {}, [], [], [], []

    loc_gt = np.stack(loc_gt_frames).reshape(-1, 2)
    vis_gt = np.stack(vis_gt_frames).ravel()
    loc_pred = np.stack(loc_pred_frames).reshape(-1, 2)
    vis_pred = np.stack(vis_pred_frames).ravel()

    metrics = compute_metrics(loc_gt, vis_gt, loc_pred, vis_pred, MONO_THRESHOLDS)
    return metrics, [loc_gt], [vis_gt], [loc_pred], [vis_pred]


def _process_stereo_sequence(
    seq_id: str,
    seq_tracks: dict,
    data_dir: Path,
) -> tuple[
    dict[str, float],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    seq_dir = data_dir / seq_id
    annotations = _load_annotations(seq_dir)
    calib = _load_calib(seq_dir)

    loc_gt_frames: list[np.ndarray] = []
    vis_gt_frames: list[np.ndarray] = []
    loc_pred_frames: list[np.ndarray] = []
    vis_pred_frames: list[np.ndarray] = []

    for frame_idx_str, frame_data in seq_tracks.items():
        frame_idx = int(frame_idx_str)
        # Skip the first frame (it's the reference frame)
        if frame_idx == 0:
            continue
        if frame_idx not in annotations:
            continue
        ann = annotations[frame_idx]
        coords_3d = np.array(frame_data["coords"], dtype=np.float32)
        visibs = np.array(frame_data["visibilities"], dtype=bool)

        disp_gt = ann.left_coords[:, 0] - ann.right_coords[:, 0]
        loc_gt_3d = unproject(ann.left_coords, disp_gt, calib)

        loc_gt_frames.append(loc_gt_3d)
        vis_gt_frames.append(ann.left_visibs & ann.right_visibs)
        loc_pred_frames.append(coords_3d)
        vis_pred_frames.append(visibs)

    if not loc_gt_frames:
        print(f"  [{seq_id}] no annotated frames — skipping")
        return {}, [], [], [], []

    flat_gt = np.concatenate(loc_gt_frames)
    flat_vgt = np.concatenate(vis_gt_frames)
    flat_pred = np.concatenate(loc_pred_frames)
    flat_vpred = np.concatenate(vis_pred_frames)

    metrics = compute_metrics(
        flat_gt, flat_vgt, flat_pred, flat_vpred, STEREO_THRESHOLDS
    )
    return metrics, [flat_gt], [flat_vgt], [flat_pred], [flat_vpred]


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _compute_and_save(
    preds_path: Path,
    mode: str,
    data_dir: Path,
    thresholds: tuple[float, ...],
    process_seq_fn,
) -> dict:
    model_name = preds_path.parent.name
    print(f"\n[{mode}/{model_name}] computing metrics ...")

    with open(preds_path) as fh:
        all_tracks = json.load(fh)

    all_loc_gt: list[np.ndarray] = []
    all_vis_gt: list[np.ndarray] = []
    all_loc_pred: list[np.ndarray] = []
    all_vis_pred: list[np.ndarray] = []
    seq_metrics: dict[str, dict] = {}
    all_latencies: list[float] = []

    for seq_id, seq_tracks in all_tracks.items():
        metrics, lgt, vgt, lpred, vpred = process_seq_fn(seq_id, seq_tracks, data_dir)
        if metrics:
            seq_metrics[seq_id] = metrics
            all_loc_gt.extend(lgt)
            all_vis_gt.extend(vgt)
            all_loc_pred.extend(lpred)
            all_vis_pred.extend(vpred)

        frame_latencies = [
            v["latency_ms"]
            for k, v in sorted((int(k), v) for k, v in seq_tracks.items())
            if int(k) > WARMUP_FRAMES
        ]
        all_latencies.extend(frame_latencies)

    loc_gt = np.concatenate(all_loc_gt)
    vis_gt = np.concatenate(all_vis_gt)
    loc_pred = np.concatenate(all_loc_pred)
    vis_pred = np.concatenate(all_vis_pred)
    overall = compute_metrics(loc_gt, vis_gt, loc_pred, vis_pred, thresholds)
    if all_latencies:
        overall["p95_latency_ms"] = float(np.percentile(all_latencies, 95))

    output = {
        "model": model_name,
        "mode": mode,
        "thresholds": list(thresholds),
        "sequences": {sid: {"metrics": m} for sid, m in seq_metrics.items()},
        "overall": overall,
    }

    metrics_path = preds_path.parent / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"  saved {metrics_path}")
    print(
        f"  aj_avg={overall.get('aj_avg', 0):.4f}  ata_avg={overall.get('ata_avg', 0):.4f}  oa={overall.get('oa', 0):.4f}",
        end="",
    )
    if "p95_latency_ms" in overall:
        print(f"  p95={overall['p95_latency_ms']:.1f}ms", end="")
    print()

    return output


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_overall(
    mode: str,
    all_metrics: list[dict],
    plots_dir: Path,
) -> None:
    """One figure: AJ vs threshold + ATA vs threshold + OA bar."""
    if not all_metrics:
        return

    thresholds = all_metrics[0]["thresholds"]
    th_labels = [str(t) for t in thresholds]
    model_names = [m["model"] for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"STIR 2026 — {mode} overall")

    for m in all_metrics:
        overall = m["overall"]
        ajs = [overall.get(f"aj_{t}", 0) for t in thresholds]
        atas = [overall.get(f"ata_{t}", 0) for t in thresholds]
        axes[0].plot(th_labels, ajs, marker="o", label=m["model"])
        axes[1].plot(th_labels, atas, marker="o", label=m["model"])

    axes[0].set_title("AJ vs threshold")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("AJ")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].set_title("ATA vs threshold")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("ATA")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    oas = [m["overall"].get("oa", 0) for m in all_metrics]
    axes[2].bar(model_names, oas)
    axes[2].set_title("Occlusion Accuracy (OA)")
    axes[2].set_ylabel("OA")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out = plots_dir / f"{mode}_overall.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def _plot_per_sequence(
    mode: str,
    all_metrics: list[dict],
    plots_dir: Path,
) -> None:
    """One figure per sequence: AJ and ATA curves for all models."""
    if not all_metrics:
        return

    thresholds = all_metrics[0]["thresholds"]
    th_labels = [str(t) for t in thresholds]

    all_seq_ids: set[str] = set()
    for m in all_metrics:
        all_seq_ids.update(m["sequences"].keys())

    for seq_id in sorted(all_seq_ids):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"STIR 2026 — {mode} / sequence {seq_id}")

        for m in all_metrics:
            seq_data = m["sequences"].get(seq_id)
            if not seq_data:
                continue
            seq_m = seq_data["metrics"]
            ajs = [seq_m.get(f"aj_{t}", 0) for t in thresholds]
            atas = [seq_m.get(f"ata_{t}", 0) for t in thresholds]
            axes[0].plot(th_labels, ajs, marker="o", label=m["model"])
            axes[1].plot(th_labels, atas, marker="o", label=m["model"])

        axes[0].set_title("AJ vs threshold")
        axes[0].set_xlabel("threshold")
        axes[0].set_ylabel("AJ")
        axes[0].legend()
        axes[0].set_ylim(0, 1)

        axes[1].set_title("ATA vs threshold")
        axes[1].set_xlabel("threshold")
        axes[1].set_ylabel("ATA")
        axes[1].legend()
        axes[1].set_ylim(0, 1)

        fig.tight_layout()
        out = plots_dir / f"{mode}_{seq_id}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _build_table_rows(all_metrics: list[dict], mode: str) -> list[list]:
    rows = []
    for m in all_metrics:
        overall = m["overall"]
        rows.append(
            [
                mode,
                m["model"],
                f"{overall.get('aj_avg', 0):.4f}",
                f"{overall.get('ata_avg', 0):.4f}",
                f"{overall.get('oa', 0):.4f}",
                f"{overall['p95_latency_ms']:.1f}"
                if "p95_latency_ms" in overall
                else "—",
            ]
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="STIR 2026 metrics runner")
    parser.add_argument(
        "results_dir", help="Root results directory (contains mono/ and stereo/)"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Dataset root directory (for GT annotations)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: compute metrics
    mono_metrics: list[dict] = []
    for preds_path in sorted((results_dir / "mono").glob("*/preds.json")):
        result = _compute_and_save(
            preds_path, "mono", data_dir, MONO_THRESHOLDS, _process_mono_sequence
        )
        mono_metrics.append(result)

    stereo_metrics: list[dict] = []
    for preds_path in sorted((results_dir / "stereo").glob("*/preds.json")):
        result = _compute_and_save(
            preds_path, "stereo", data_dir, STEREO_THRESHOLDS, _process_stereo_sequence
        )
        stereo_metrics.append(result)

    if not mono_metrics and not stereo_metrics:
        print("No preds.json files found. Run inference first.")
        return

    # Phase 2: plots
    print("\nGenerating plots ...")
    _plot_overall("mono", mono_metrics, plots_dir)
    _plot_overall("stereo", stereo_metrics, plots_dir)
    _plot_per_sequence("mono", mono_metrics, plots_dir)
    _plot_per_sequence("stereo", stereo_metrics, plots_dir)

    # Phase 3: summary table
    headers = ["mode", "model", "AJ_avg", "ATA_avg", "OA", "p95_ms"]
    rows = _build_table_rows(mono_metrics, "mono") + _build_table_rows(
        stereo_metrics, "stereo"
    )

    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))

    table_md = tabulate(rows, headers=headers, tablefmt="github")
    summary_path = plots_dir / "summary.md"
    with open(summary_path, "w") as fh:
        fh.write("# STIR 2026 Results\n\n")
        fh.write(table_md)
        fh.write("\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
