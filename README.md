# PointWorld Baseline API

This repository is a thin, dataset-agnostic inference wrapper around NVIDIA PointWorld.

The baseline task is:

```text
Given current scene points, current RGB-D observation, camera calibration,
and a planned robot state trajectory, predict a future 3D trajectory for every input scene point.
```

PointWorld predicts scene flow. For a proper scene-flow baseline, pass all useful non-robot scene points as `scene_points` and use a separate object/target mask only for evaluation or visualization. For an object-only ablation, you can pass only object points.

## What PointWorld Predicts

The released PointWorld model uses a fixed horizon:

```text
context horizon:    1
prediction horizon: 10
total T:            11
```

For `N` input scene points, the main output is:

```python
out.scene_points_pred.shape == (11, N, 3)
```

Meaning:

```text
out.scene_points_pred[t, i] = predicted xyz of input scene point i at timestep t
```

The model predicts a trajectory for every input point ID. Visibility, moving-object masks, and object-of-interest masks should be handled by your dataset adapter and evaluation code.

## Recommended High-Level API

```python
from pointworld_baseline import PointWorldPredictor

predictor = PointWorldPredictor(
    pointworld_repo="/path/to/PointWorld",
    checkpoint="/path/to/model-best.pt",
    dinov3_weights="/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    device="cuda",
)

out = predictor.predict_scene_motion(
    scene_points=scene_points,      # (N, 3), xyz scene points to forecast
    scene_exists=scene_exists,      # optional (N,), bool valid mask for model input
    rgb=rgb,                        # (H, W, 3), uint8 RGB, current image
    depth=depth,                    # (H, W), float32 depth in meters
    intrinsics=intrinsics,          # (3, 3), camera intrinsics
    extrinsics=extrinsics,          # (4, 4), camera extrinsics
    qpos=qpos,                      # (11, 7), robot arm joint trajectory
    gripper_pos=gripper_pos,        # (11,) or (11, 1), gripper open/close trajectory
)

pred = out.scene_points_pred       # (11, N, 3)
conf = out.confidence               # (11, N), if returned
```

`predict_object_motion(...)` is kept as a compatibility alias for object-only experiments. New code should call `predict_scene_motion(...)`.

## Scene Points And Target Masks

Use two different masks in your adapter:

```text
scene_exists: valid scene points that PointWorld should see and predict
target_mask: object-of-interest points for metrics, visualization, or downstream optimization
```

`target_mask` is not passed to the predictor. PointWorld predicts all input points; your evaluation code selects the object points afterward.

Recommended default for our Step3-style evaluation:

```text
scene_points = all valid tracked non-robot scene points at t=0
scene_exists = scene_valid[0]
target_mask = flow_moving_mask & scene_valid
```

## Input Meaning

### `scene_points`, shape `(N, 3)`

The 3D scene points whose future motion you want to predict. Use all useful scene points for the main baseline. Use only object points for an object-only ablation.

### `scene_exists`, optional shape `(N,)`

Boolean valid mask for `scene_points`. If omitted, all input scene points are treated as valid.

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

Camera extrinsic matrix. Your scene points and robot points must use the coordinate convention expected by this transform.

### `qpos`, shape `(11, 7)`

Future Franka/Panda arm joint trajectory.

```text
qpos[t] = panda_joint1 ... panda_joint7 at timestep t
```

### `gripper_pos`, shape `(11,)` or `(11, 1)`

Future gripper open/close state.

This is not the 6D gripper pose. The spatial gripper pose is computed from `qpos` with FK. `gripper_pos` controls finger opening.

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
    scene_points=scene_points,
    scene_features=scene_features,
    scene_exists=scene_exists,
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
| `scene_points` | `(N, 3)` | Current scene points to forecast. |
| `scene_features` | `(11, N, 31)` | PointWorld scene feature tensor. |
| `scene_exists` | `(N,)` | Optional valid mask for model input. |
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
| `scene_points_pred` | `(11, N, 3)` | Alias for `object_points_pred`; absolute predicted xyz trajectory for each input point. |
| `object_points_pred` | `(11, N, 3)` | Backward-compatible name for the same prediction. |
| `scene_displacement_pred` | `(11, N, 3)` | Alias for `object_displacement_pred`. |
| `object_displacement_pred` | `(11, N, 3)` | Predicted displacement relative to input point positions. |
| `scene_displacement_pred_norm` | `(11, N, 3)` | Alias for `object_displacement_pred_norm`. |
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
scene_points, scene_exists, rgb, depth, intrinsics, extrinsics, qpos, gripper_pos
```

Object/moving-point masks should remain adapter-side evaluation or visualization masks, not core model inputs. The core package should not depend on Step3.
