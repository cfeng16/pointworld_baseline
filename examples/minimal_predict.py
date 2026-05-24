"""Minimal PointWorld baseline API example.

This file intentionally uses placeholder arrays. Replace them with data from
any dataset that can provide clean scene points, RGB-D, camera calibration, qpos,
and gripper state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pointworld_baseline import PointWorldPredictor
from pointworld_baseline.schema import POINTWORLD_HORIZON


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointworld_repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3_weights", required=True, type=Path)
    args = parser.parse_args()

    n_points = 128
    scene_points = np.zeros((n_points, 3), dtype=np.float32)
    ofi_mask = np.zeros((n_points,), dtype=bool)
    ofi_mask[:16] = True

    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    depth = np.ones((224, 224), dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)
    qpos = np.zeros((POINTWORLD_HORIZON, 7), dtype=np.float32)
    gripper_pos = np.zeros((POINTWORLD_HORIZON,), dtype=np.float32)

    predictor = PointWorldPredictor(
        pointworld_repo=args.pointworld_repo,
        checkpoint=args.checkpoint,
        dinov3_weights=args.dinov3_weights,
        device="cuda",
    )
    out = predictor.predict_scene_motion(
        scene_points=scene_points,
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        qpos=qpos,
        gripper_pos=gripper_pos,
        ofi_mask=ofi_mask,
    )
    print("scene prediction", out.scene_points_pred.shape)
    print("ofi prediction", None if out.ofi_points_pred is None else out.ofi_points_pred.shape)


if __name__ == "__main__":
    main()
