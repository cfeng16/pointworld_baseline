# Examples

Examples are dataset adapters. They should convert a dataset into the generic
PointWorld baseline API inputs:

```text
object_points: (N, 3)
rgb: (H, W, 3)
depth: (H, W)
intrinsics: (3, 3)
extrinsics: (4, 4)
qpos: (11, 7)
gripper_pos: (11,)
object_exists: optional (N,)
```

The core package is intentionally dataset-agnostic.
