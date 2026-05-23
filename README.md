# PointWorld Baseline API

This repository is a thin, dataset-agnostic inference wrapper around NVIDIA PointWorld.

The baseline task is:

```text
Given current object points, current RGB-D observation, camera calibration,
and a planned robot state trajectory, predict a future 3D trajectory for every input object point.
```

PointWorld itself calls these `scene` points. This wrapper exposes them as `object_points` because the common robotics use case is object motion prediction.

## What PointWorld Predicts

The released PointWorld model uses a fixed horizon:

```text
context horizon:    1
prediction horizon: 10
total T:            11
```

For `N` input object points, the main output is:

```python
out.object_points_pred.shape == (11, N, 3)
```

Meaning:

```text
out.object_points_pred[t, i] = predicted xyz of input object point i at timestep t
```

The model predicts a trajectory for every input point ID. Visibility/missing GT should be handled by your dataset masks and evaluation code.

## Recommended High-Level API

```python
from pointworld_baseline import PointWorldPredictor

predictor = PointWorldPredictor(
    pointworld_repo="/path/to/PointWorld",
    checkpoint="/path/to/model-best.pt",
    dinov3_weights="/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    device="cuda",
)

out = predictor.predict_object_motion(
    object_points=object_points,    # (N, 3), xyz points to forecast
    rgb=rgb,                        # (H, W, 3), uint8 RGB, current image
    depth=depth,                    # (H, W), float32 depth in meters
    intrinsics=intrinsics,          # (3, 3), camera intrinsics
    extrinsics=extrinsics,          # (4, 4), camera extrinsics
    qpos=qpos,                      # (11, 7), robot arm joint trajectory
    gripper_pos=gripper_pos,        # (11,) or (11, 1), gripper open/close trajectory
    object_exists=object_exists,    # optional (N,), bool mask
)

pred = out.object_points_pred       # (11, N, 3)
conf = out.confidence               # (11, N), if returned
```

## Input Meaning

### `object_points`, shape `(N, 3)`

The 3D points whose future motion you want to predict.

Use object points for an object-motion baseline. You can pass full-scene points, but PointWorld will then forecast every static/background point too.

### `rgb`, shape `(H, W, 3)`

Current RGB image at the beginning of the prediction window.

Expected format:

```text
uint8 RGB, values 0..255
```

### `depth`, shape `(H, W)`

Current depth image at the beginning of the prediction window.

Expected format:

```text
float32 depth in meters
```

### `intrinsics`, shape `(3, 3)`

Camera intrinsic matrix:

```text
fx  0 cx
 0 fy cy
 0  0  1
```

### `extrinsics`, shape `(4, 4)`

Camera extrinsic matrix. Your object points and robot points must use the coordinate convention expected by this transform.

### `qpos`, shape `(11, 7)`

Future Franka/Panda arm joint trajectory.

```text
qpos[t] = panda_joint1 ... panda_joint7 at timestep t
```

### `gripper_pos`, shape `(11,)` or `(11, 1)`

Future gripper open/close state.

This is not the 6D gripper pose. The spatial gripper pose is computed from `qpos` with FK. `gripper_pos` controls finger opening.

### `object_exists`, optional shape `(N,)`

Boolean valid mask for object points. If omitted, all input object points are treated as valid.

## Robot Conditioning Internally

The high-level API does this internally:

```text
qpos + gripper_pos
  -> PointWorld RobotSampler / FK
  -> robot point trajectory, shape (11, Nr, 3)
  -> robot features, shape (11, Nr, 16)
  -> PointWorld model
```

FK means forward kinematics:

```text
joint angles -> robot/gripper pose and mesh points
```

IK means inverse kinematics:

```text
desired end-effector pose -> joint angles
```

This wrapper does not need IK when `qpos` is available.

## Low-Level API

If you already have robot point trajectories, use:

```python
out = predictor.predict(
    object_points=object_points,
    object_features=object_features,
    object_exists=object_exists,
    robot_points=robot_points,
    robot_features=robot_features,
    robot_exists=robot_exists,
    rgb=rgb,
    depth=depth,
    intrinsics=intrinsics,
    extrinsics=extrinsics,
)
```

Expected low-level shapes:

| Input | Shape | Description |
| --- | --- | --- |
| `object_points` | `(N, 3)` | Current object points to forecast. |
| `object_features` | `(11, N, 31)` | PointWorld scene feature tensor. |
| `object_exists` | `(N,)` | Optional valid mask. |
| `robot_points` | `(11, Nr, 3)` | Robot/gripper point trajectory. |
| `robot_features` | `(11, Nr, 16)` | Robot point features. |
| `robot_exists` | `(11, Nr)` | Optional robot valid mask. |
| `rgb` | `(H, W, 3)` | Current RGB image. |
| `depth` | `(H, W)` | Current depth in meters. |
| `intrinsics` | `(3, 3)` | Camera intrinsics. |
| `extrinsics` | `(4, 4)` | Camera extrinsics. |

## Outputs

| Output field | Shape | Meaning |
| --- | --- | --- |
| `object_points_pred` | `(11, N, 3)` | Absolute predicted xyz trajectory for each input object point. |
| `object_displacement_pred` | `(11, N, 3)` | Predicted displacement relative to input object point positions. |
| `object_displacement_pred_norm` | `(11, N, 3)` | Normalized displacement from PointWorld. |
| `log_var` | `(11, N, 1)` | Predicted aleatoric uncertainty. |
| `confidence` | `(11, N)` | Confidence derived from uncertainty. |
| `raw_outputs` | dict | Raw PointWorld tensor outputs converted to numpy. |

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

Step3 support should live under `examples/`. It is an adapter that produces the generic API inputs:

```text
object_points, rgb, depth, intrinsics, extrinsics, qpos, gripper_pos, object_exists
```

The core package should not depend on Step3.
