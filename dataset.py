"""Minimal dataset types for STIR 2026 metrics.

Only the data structures needed by the metrics scripts — no video reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Annotation:
    left_coords: np.ndarray   # (N, 2) float32 — pixel coordinates in left image
    right_coords: np.ndarray  # (N, 2) float32 — pixel coordinates in right image
    left_visibs: np.ndarray   # (N,) bool
    right_visibs: np.ndarray  # (N,) bool


@dataclass
class CameraCalib:
    K_left: np.ndarray           # (3, 3) intrinsic matrix, left camera
    K_right: np.ndarray          # (3, 3) intrinsic matrix, right camera
    D_left: np.ndarray           # (5,) distortion coefficients, left camera
    D_right: np.ndarray          # (5,) distortion coefficients, right camera
    R_left_to_right: np.ndarray  # (3, 3) rotation from left to right camera frame
    t_left_to_right: np.ndarray  # (3,) translation from left to right camera frame (m)

    @classmethod
    def from_dict(cls, d: dict) -> CameraCalib:
        return cls(
            K_left=np.array(d["K_left"]),
            K_right=np.array(d["K_right"]),
            D_left=np.array(d["D_left"]),
            D_right=np.array(d["D_right"]),
            R_left_to_right=np.array(d["R_left_to_right"]),
            t_left_to_right=np.array(d["t_left_to_right"]),
        )
