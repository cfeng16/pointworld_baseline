# Examples

Examples are dataset adapters. They should convert a dataset into the generic
PointWorld baseline API inputs:

```text
scene_points: (N, 3)      # all clean scene points
rgb: (H, W, 3)
depth: (H, W)
intrinsics: (3, 3)
extrinsics: (4, 4)       # world/base-to-camera
qpos: (11, 7)
gripper_pos: (11,)
```

The predictor returns one trajectory for every input `scene_points` row. If an
adapter has a separate evaluation mask, apply it outside the predictor:

```python
selected_pred = out.scene_points_pred[:, eval_mask]
```

For Step3/DROID-style adapters, `scene_points` should stay in robot-base
coordinates and `extrinsics` should map robot-base/world coordinates into the
external RGB-D camera. Use `obs/qpos` and `obs/gripper_pos` for robot
conditioning; do not use `act` as the current robot pose.
