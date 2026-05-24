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
ofi_mask: optional (N,)   # output-side object-of-interest mask
```

`ofi_mask` is not a model input mask. It is used only after prediction for
metrics, visualization, or optimization.

For Step3/DROID-style adapters, `scene_points` should stay in robot-base
coordinates and `extrinsics` should map robot-base/world coordinates into the
external RGB-D camera. Use `obs/qpos` and `obs/gripper_pos` for robot
conditioning; do not use `act` as the current robot pose.
