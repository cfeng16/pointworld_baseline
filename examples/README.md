# Examples

Examples are dataset adapters. They should convert a dataset into the generic
PointWorld baseline API inputs:

```text
scene_points: (N, 3)      # all clean scene points
rgb: (H, W, 3)
depth: (H, W)
intrinsics: (3, 3)
extrinsics: (4, 4)
qpos: (11, 7)
gripper_pos: (11,)
ofi_mask: optional (N,)   # output-side object-of-interest mask
```

`ofi_mask` is not a model input mask. It is used only after prediction for
metrics, visualization, or optimization.
