from __future__ import annotations

import numpy as np

from .schema import POINTWORLD_HORIZON


def finite_difference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    velocity = np.zeros_like(values)
    acceleration = np.zeros_like(values)
    if values.shape[0] >= 2:
        velocity[0] = values[1] - values[0]
        velocity[-1] = values[-1] - values[-2]
    if values.shape[0] >= 3:
        velocity[1:-1] = (values[2:] - values[:-2]) / 2.0
        acceleration[0] = velocity[1] - velocity[0]
        acceleration[-1] = velocity[-1] - velocity[-2]
    if values.shape[0] >= 4:
        acceleration[1:-1] = (velocity[2:] - velocity[:-2]) / 2.0
    return velocity, acceleration


def pairwise_min_dist(points: np.ndarray, robot_points: np.ndarray) -> np.ndarray:
    dists = []
    for t in range(robot_points.shape[0]):
        diff = points[:, None, :] - robot_points[t][None, :, :]
        dists.append(np.linalg.norm(diff, axis=-1).min(axis=1))
    return np.stack(dists, axis=0).astype(np.float32)


def project_world_to_pixel(points_world: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=-1)
    cam = points_h @ extrinsics.T
    z = cam[:, 2]
    valid = z > 1e-6
    pix_h = cam[:, :3] @ intrinsics.T
    xy = pix_h[:, :2] / np.maximum(pix_h[:, 2:3], 1e-6)
    return xy, valid


def sample_rgb_at_points(
    rgb: np.ndarray,
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb
    xy, z_valid = project_world_to_pixel(points_world, intrinsics, extrinsics)
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    mask = (x >= 0) & (x < w) & (y >= 0) & (y < h) & z_valid
    if valid_mask is not None:
        mask &= valid_mask.astype(bool)
    colors = np.zeros((points_world.shape[0], 3), dtype=np.float32)
    colors[mask] = rgb_u8[y[mask], x[mask]].astype(np.float32) / 255.0
    return colors


def make_scene_features(
    object_points: np.ndarray,
    robot_points: np.ndarray,
    gripper_pos: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    object_normals: np.ndarray | None = None,
    object_colors: np.ndarray | None = None,
    object_exists: np.ndarray | None = None,
) -> np.ndarray:
    """Build the released PointWorld 31-D scene feature tensor.

    Returns shape (T, N, 31). Only t=0 is consumed by the released model, but
    PointWorld's batch schema is time-major.
    """
    object_points = np.asarray(object_points, dtype=np.float32)
    robot_points = np.asarray(robot_points, dtype=np.float32)
    gripper_pos = np.asarray(gripper_pos, dtype=np.float32).reshape(POINTWORLD_HORIZON, -1)
    if gripper_pos.shape[1] != 1:
        raise ValueError(f"gripper_pos must be scalar per step, got shape {gripper_pos.shape}")
    n_points = object_points.shape[0]
    if object_normals is None:
        object_normals = np.zeros((n_points, 3), dtype=np.float32)
    if object_colors is None:
        object_colors = sample_rgb_at_points(rgb, object_points, intrinsics, extrinsics, object_exists)
    gripper_context = np.repeat(gripper_pos.reshape(1, 1, POINTWORLD_HORIZON), n_points, axis=1)
    dist2robot = pairwise_min_dist(object_points, robot_points).T.reshape(1, n_points, POINTWORLD_HORIZON)
    feat0 = np.concatenate(
        [
            object_points.reshape(1, n_points, 3),
            np.asarray(object_colors, dtype=np.float32).reshape(1, n_points, 3),
            np.asarray(object_normals, dtype=np.float32).reshape(1, n_points, 3),
            gripper_context,
            dist2robot,
        ],
        axis=-1,
    ).astype(np.float32)
    if feat0.shape[-1] != 31:
        raise AssertionError(f"Expected 31 scene feature dims, got {feat0.shape[-1]}")
    return np.repeat(feat0, POINTWORLD_HORIZON, axis=0)


def make_robot_features(
    robot_points: np.ndarray,
    gripper_pos: np.ndarray,
    robot_normals: np.ndarray | None = None,
    robot_colors: np.ndarray | None = None,
) -> np.ndarray:
    """Build the released PointWorld 16-D robot feature tensor, shape (T, Nr, 16)."""
    robot_points = np.asarray(robot_points, dtype=np.float32)
    t_steps, n_robot, _ = robot_points.shape
    if t_steps != POINTWORLD_HORIZON:
        raise ValueError(f"robot_points must have T={POINTWORLD_HORIZON}, got shape {robot_points.shape}")
    if robot_normals is None:
        robot_normals = np.zeros_like(robot_points, dtype=np.float32)
    if robot_colors is None:
        robot_colors = np.zeros_like(robot_points, dtype=np.float32)
        robot_colors[..., 0] = 1.0
        robot_colors[..., 2] = 1.0
    robot_colors = np.asarray(robot_colors, dtype=np.float32)
    if robot_colors.max(initial=0.0) > 1.0:
        robot_colors = robot_colors / 255.0
    gripper = np.asarray(gripper_pos, dtype=np.float32).reshape(t_steps, 1, 1)
    gripper_feat = np.repeat(gripper, n_robot, axis=1)
    velocity, acceleration = finite_difference(robot_points)
    feats = np.concatenate(
        [robot_points, robot_colors, np.asarray(robot_normals, dtype=np.float32), gripper_feat, velocity, acceleration],
        axis=-1,
    ).astype(np.float32)
    if feats.shape[-1] != 16:
        raise AssertionError(f"Expected 16 robot feature dims, got {feats.shape[-1]}")
    return feats
