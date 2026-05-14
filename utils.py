"""Metric helpers and geometry for STIR 2026 metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np

from dataset import CameraCalib


def counts_per_threshold(
    loc_gt: np.ndarray,
    vis_gt: np.ndarray,
    loc_pred: np.ndarray,
    vis_pred: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[float, tuple[int, int, int]]:
    """Compute AJ ``(tp, fp, fn)`` counts at each distance threshold."""
    assert loc_gt.shape == loc_pred.shape, f"loc shape mismatch: {loc_gt.shape} vs {loc_pred.shape}"
    assert vis_gt.shape == vis_pred.shape, f"vis shape mismatch: {vis_gt.shape} vs {vis_pred.shape}"

    dist = np.linalg.norm(loc_gt - loc_pred, axis=-1)
    counts: dict[float, tuple[int, int, int]] = {}
    for th in thresholds:
        within = dist < th
        tp = int((within & vis_gt & vis_pred).sum())
        fp = int((vis_pred & (~vis_gt | ~within)).sum())
        fn = int((vis_gt & (~vis_pred | ~within)).sum())
        counts[th] = (tp, fp, fn)
    return counts


def tracking_accuracy_per_threshold(
    loc_gt: np.ndarray,
    vis_gt: np.ndarray,
    loc_pred: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[float, tuple[int, int]]:
    """Compute distance accuracy counts over ground-truth visible points."""
    assert loc_gt.shape == loc_pred.shape, f"loc shape mismatch: {loc_gt.shape} vs {loc_pred.shape}"
    assert vis_gt.shape == loc_gt.shape[:-1], f"vis shape mismatch: {vis_gt.shape} vs {loc_gt.shape[:-1]}"

    dist = np.linalg.norm(loc_gt - loc_pred, axis=-1)
    totals: dict[float, tuple[int, int]] = {}
    visible_total = int(vis_gt.sum())
    for th in thresholds:
        within = dist < th
        correct = int((within & vis_gt).sum())
        totals[th] = (correct, visible_total)
    return totals


def compute_metrics(
    loc_gt: np.ndarray,
    vis_gt: np.ndarray,
    loc_pred: np.ndarray,
    vis_pred: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    """Compute AJ, ATA, and OA metrics.

    Returns dict keyed by ``aj_<threshold>``, ``ata_<threshold>``,
    ``aj_avg``, ``ata_avg``, and ``oa``.
    """
    counts = counts_per_threshold(loc_gt, vis_gt, loc_pred, vis_pred, thresholds)
    ata_counts = tracking_accuracy_per_threshold(loc_gt, vis_gt, loc_pred, thresholds)

    results: dict[str, float] = {}
    ajs: list[float] = []
    atas: list[float] = []
    for th in thresholds:
        tp, fp, fn = counts[th]
        aj_denom = tp + fp + fn
        aj = tp / aj_denom if aj_denom > 0 else 0.0
        results[f"aj_{th}"] = aj
        ajs.append(aj)

        correct, total = ata_counts[th]
        ata = correct / total if total > 0 else 0.0
        results[f"ata_{th}"] = ata
        atas.append(ata)

    results["aj_avg"] = float(np.mean(ajs))
    results["ata_avg"] = float(np.mean(atas))

    total = vis_gt.size
    results["oa"] = float((vis_gt == vis_pred).sum() / total) if total > 0 else 0.0
    return results


def _intrinsics(
    calib: CameraCalib, cam: Literal["left", "right"]
) -> tuple[float, float, float, float, float]:
    """Return ``(fx, fy, cx, cy, baseline)`` for the chosen camera."""
    K = calib.K_left if cam == "left" else calib.K_right
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2], float(-calib.t_left_to_right[0])


def unproject(
    uv: np.ndarray,
    disparity: np.ndarray,
    calib: CameraCalib,
    cam: Literal["left", "right"] = "left",
) -> np.ndarray:
    """2D pixel + disparity → 3D in left camera frame.

    Parameters
    ----------
    uv : (N, 2) float — ``[x, y]`` pixel coordinates in the chosen camera.
    disparity : (N,) float — pixel disparity ``left_x - right_x``.
    calib : CameraCalib
    cam : which camera's intrinsics to use (almost always ``"left"``).

    Returns
    -------
    (N, 3) float32 — ``[X, Y, Z]`` in left camera frame (metres).
    """
    fx, fy, cx, cy, baseline = _intrinsics(calib, cam)
    Z = fx * baseline / np.clip(disparity, 1e-6, None)
    X = (uv[:, 0] - cx) * Z / fx
    Y = (uv[:, 1] - cy) * Z / fy
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


def project(
    xyz: np.ndarray,
    calib: CameraCalib,
    cam: Literal["left", "right"],
) -> np.ndarray:
    """3D in left camera frame → 2D pixel coordinates.

    Parameters
    ----------
    xyz : (N, 3) float — ``[X, Y, Z]`` in left camera frame.
    calib : CameraCalib
    cam : target image; ``"right"`` first transforms *xyz* via extrinsics.

    Returns
    -------
    (N, 2) float32 — ``[u, v]`` pixel coordinates.
    """
    if cam == "right":
        xyz = (calib.R_left_to_right @ xyz.T).T + calib.t_left_to_right
    fx, fy, cx, cy, _ = _intrinsics(calib, cam)
    Z = np.clip(xyz[:, 2], 1e-6, None)
    u = fx * xyz[:, 0] / Z + cx
    v = fy * xyz[:, 1] / Z + cy
    return np.stack([u, v], axis=1).astype(np.float32)
