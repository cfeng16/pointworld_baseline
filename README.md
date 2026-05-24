# PointWorld Baseline API

This repository is a thin, dataset-agnostic inference wrapper around NVIDIA PointWorld.

The intended high-level use is simple:

```text
Input:  all clean scene points + RGB-D/camera + planned robot trajectory
Output: predicted future 3D trajectory for every input scene point
```

If you care about one object, pass an `ofi_mask` over those same input points. The `ofi_mask` is applied only to the output for metrics/visualization; it is not used as a model input mask.

## What PointWorld Predicts

The released PointWorld model uses a fixed horizon:

```text
context horizon:    1
prediction horizon: 10
total T:            11
```

For `N` input scene points:

```python
out.scene_points_pred.shape == (11, N, 3)
```

Meaning:

```text
out.scene_points_pred[t, i] = predicted xyz of input scene point i at timestep t
```

The model predicts every input point ID. Object-of-interest selection happens afterward with `ofi_mask`.

## Recommended API

```python
from pointworld_baseline import PointWorldPredictor

predictor = PointWorldPredictor(
    pointworld_repo="/path/to/PointWorld",
    checkpoint="/path/to/model-best.pt",
    dinov3_weights="/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    device="cuda",
)

out = predictor.predict_scene_motion(
    scene_points=scene_points,      # (N, 3), all clean scene points
    rgb=rgb,                        # (H, W, 3), uint8 RGB, current image
    depth=depth,                    # (H, W), float32 depth in meters
    intrinsics=intrinsics,          # (3, 3), camera intrinsics
    extrinsics=extrinsics,          # (4, 4), camera extrinsics
    qpos=qpos,                      # (11, 7), robot arm joint trajectory
    gripper_pos=gripper_pos,        # (11,) or (11, 1), gripper open/close trajectory
    ofi_mask=ofi_mask,              # optional (N,), output-side object-of-interest mask
)

scene_pred = out.scene_points_pred  # (11, N, 3), all scene points
ofi_pred = out.ofi_points_pred      # (11, M, 3), only if ofi_mask was provided
```

Important: `ofi_mask` is not conditioning. It does not hide points from PointWorld. It only slices the prediction output.

## Input Meaning

### `scene_points`, shape `(N, 3)`

All clean scene points you want PointWorld to predict. Filter invalid/padded/depth-bad points before calling the API.

### `ofi_mask`, optional shape `(N,)`

Object-of-interest mask over `scene_points`. This is output-side only:

```python
out.ofi_points_pred == out.scene_points_pred[:, ofi_mask]
```

Use this for object metrics, visualization, or downstream optimization.

### `rgb`, shape `(H, W, 3)`

Current RGB image at the beginning of the prediction window. Expected format: uint8 RGB, values 0..255.

### `depth`, shape `(H, W)`

Current depth image at the beginning of the prediction window. Expected format: float32 depth in meters.

### `intrinsics`, shape `(3, 3)`

Camera intrinsic matrix:

```text
fx  0 cx
 0 fy cy
 0  0  1
```

### `extrinsics`, shape `(4, 4)`

Camera extrinsic matrix. Your scene points and robot points must use the coordinate convention expected by this transform.

### `qpos`, shape `(11, 7)`

Future Franka/Panda arm joint trajectory:

```text
qpos[t] = panda_joint1 ... panda_joint7 at timestep t
```

### `gripper_pos`, shape `(11,)` or `(11, 1)`

Future gripper open/close state. This is not the 6D gripper pose. The spatial gripper pose is computed from `qpos` with FK.

## Robot Conditioning Internally

The high-level API does this internally:

```text
qpos + gripper_pos
  -> PointWorld RobotSampler / FK
  -> robot point trajectory, shape (11, Nr, 3)
  -> robot features, shape (11, Nr, 16)
  -> PointWorld model
```

FK means forward kinematics: joint angles -> robot/gripper pose and mesh points.

IK means inverse kinematics: desired end-effector pose -> joint angles.

This wrapper does not need IK when `qpos` is available.

## Low-Level API

If you already have robot point trajectories/features, use:

```python
out = predictor.predict(
    scene_points=scene_points,
    scene_features=scene_features,
    robot_points=robot_points,
    robot_features=robot_features,
    rgb=rgb,
    depth=depth,
    intrinsics=intrinsics,
    extrinsics=extrinsics,
    ofi_mask=ofi_mask,
)
```


## Outputs

| Output field | Shape | Meaning |
| --- | --- | --- |
| `scene_points_pred` | `(11, N, 3)` | Absolute predicted xyz trajectory for every input scene point. |
| `ofi_points_pred` | `(11, M, 3)` or `None` | `scene_points_pred[:, ofi_mask]`; output-side object selection. |
| `scene_displacement_pred` | `(11, N, 3)` | Predicted displacement relative to input point positions. |
| `ofi_displacement_pred` | `(11, M, 3)` or `None` | `scene_displacement_pred[:, ofi_mask]`. |
| `log_var` | `(11, N, 1)` | Predicted aleatoric uncertainty. |
| `confidence` | `(11, N)` | Confidence derived from uncertainty. |
| `raw_outputs` | dict | Raw PointWorld tensor outputs converted to numpy. |

The `object_*` output names are kept only as backward-compatible aliases.

## Checkpoints

Do not commit checkpoints to git. Suggested local layout:

```text
checkpoints/
  model-best.pt                         # symlink/copy to PointWorld large-droid checkpoint
  dinov3_vitl16_pretrain_*.pth          # symlink/copy to DINOv3 ViT-L/16 checkpoint
```

Example:

```bash
mkdir -p checkpoints
ln -s /share/fang/path/to/model-best.pt checkpoints/model-best.pt
ln -s /share/fang/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## Step3 Data

Step3 support should live under `examples/`. A Step3 adapter should pass:

```text
scene_points = all clean/valid tracked scene points at t=0
ofi_mask = object-of-interest mask over those scene_points
```

The core package should not depend on Step3.
