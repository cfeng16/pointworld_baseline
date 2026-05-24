# Examples

Examples are dataset adapters. They should convert a dataset into the generic
PointWorld baseline API inputs:

```text
scene_points: (N, 3)
scene_exists: optional (N,)
rgb: (H, W, 3)
depth: (H, W)
intrinsics: (3, 3)
extrinsics: (4, 4)
qpos: (11, 7)
gripper_pos: (11,)
```

Keep object/moving-point masks adapter-side for metrics and visualization. The
core package is intentionally dataset-agnostic.
