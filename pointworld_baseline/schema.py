from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

POINTWORLD_CONTEXT_HORIZON = 1
POINTWORLD_PRED_HORIZON = 10
POINTWORLD_HORIZON = POINTWORLD_CONTEXT_HORIZON + POINTWORLD_PRED_HORIZON


@dataclass(frozen=True)
class PointWorldOutput:
    """PointWorld prediction output for one sample.

    Attributes:
        object_points_pred: Absolute predicted scene point trajectory, shape (T, N, 3).
        object_displacement_pred: Predicted displacement relative to input point positions, shape (T, N, 3).
        object_displacement_pred_norm: Normalized displacement, shape (T, N, 3), if returned.
        log_var: Aleatoric uncertainty log variance, shape (T, N, 1), if returned.
        confidence: Confidence derived from uncertainty, shape (T, N), if returned.
        raw_outputs: Raw PointWorld output dict, kept for debugging/experiments.
    """

    object_points_pred: np.ndarray
    object_displacement_pred: Optional[np.ndarray]
    object_displacement_pred_norm: Optional[np.ndarray]
    log_var: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    raw_outputs: dict[str, np.ndarray]

    @property
    def scene_points_pred(self) -> np.ndarray:
        return self.object_points_pred

    @property
    def scene_displacement_pred(self) -> Optional[np.ndarray]:
        return self.object_displacement_pred

    @property
    def scene_displacement_pred_norm(self) -> Optional[np.ndarray]:
        return self.object_displacement_pred_norm



def as_numpy(array, *, dtype=None, name: str = "array") -> np.ndarray:
    if array is None:
        raise ValueError(f"{name} is required")
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    out = np.asarray(array)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def require_shape(array: np.ndarray, shape: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have ndim {len(shape)}, got shape {array.shape}")
    for idx, (actual, expected) in enumerate(zip(array.shape, shape)):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} dim {idx} must be {expected}, got shape {array.shape}")


def validate_horizon(name: str, value: np.ndarray) -> None:
    if value.shape[0] != POINTWORLD_HORIZON:
        raise ValueError(f"{name} first dimension must be T={POINTWORLD_HORIZON}, got shape {value.shape}")
