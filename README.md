# PointWorld Baseline API

This repository is a thin, dataset-agnostic inference wrapper around NVIDIA PointWorld.

The intended high-level use is simple:

```text
Input:  all clean scene points + RGB-D/camera + planned robot trajectory
Output: predicted future 3D trajectory for every input scene point
```

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

The model predicts every input point ID, preserving the input `scene_points` ordering.

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
    extrinsics=extrinsics,          # (4, 4), world/base-to-camera extrinsics
    qpos=qpos,                      # (11, 7), robot arm joint trajectory
    gripper_pos=gripper_pos,        # (11,) or (11, 1), gripper open/close trajectory
)

scene_pred = out.scene_points_pred  # (11, N, 3), all scene points
```

## Input Meaning

### `scene_points`, shape `(N, 3)`

All clean scene points you want PointWorld to predict. Filter invalid/padded/depth-bad points before calling the API.

Use the same world/robot-base frame as the robot trajectory. For Step3/DROID-style data this is the robot base frame, not the camera frame.

### `rgb`, shape `(H, W, 3)`

Current RGB image at the beginning of the prediction window. Expected format: uint8 RGB, values 0..255.

### `depth`, shape `(H, W)`

Current depth image at the beginning of the prediction window. Expected format: float32 depth in meters. If `scene_normals` is not provided, the wrapper estimates scene normals from this depth image and transforms them into the same frame as `scene_points`.

### `intrinsics`, shape `(3, 3)`

Camera intrinsic matrix:

```text
fx  0 cx
 0 fy cy
 0  0  1
```

### `extrinsics`, shape `(4, 4)`

Camera extrinsic matrix mapping scene/robot points from world or robot-base coordinates into the RGB-D camera frame:

```text
point_cam = extrinsics @ [point_world_or_base, 1]
pixel     = intrinsics @ point_cam
```

Do not pass already-camera-frame points with a non-identity extrinsic. If your points are already in camera coordinates, use identity extrinsics and keep the robot points in the same camera frame.

### `qpos`, shape `(11, 7)`

Future Franka/Panda arm joint trajectory:

```text
qpos[t] = panda_joint1 ... panda_joint7 at timestep t
```

For Step3/DROID-style data, use observed `obs/qpos` over the prediction window. Do not substitute the action vector for `qpos`; the action can be a command/target and is not the robot joint state.

### `gripper_pos`, shape `(11,)` or `(11, 1)`

Future gripper open/close state. This is not the 6D gripper pose. The spatial gripper pose is computed from `qpos` with FK.

For Step3/DROID-style data, `obs/gripper_pos` is the scalar gripper state. The observed end-effector pose is redundant for this high-level path when `qpos` is available; PointWorld reconstructs the gripper pose with FK.

## Coordinate Checks

For a new dataset, verify these before trusting predictions:

1. Project `scene_points` with `extrinsics` and `intrinsics`; they should land on the RGB-D image and projected camera `z` should match depth.
2. Inverting `extrinsics` should not be the better projection if your points are in world/base coordinates.
3. FK from `qpos` should produce an end-effector link near the dataset's observed end-effector pose, if that pose is available.

On Step3, `episode_context/origin_is_robot_base=True`; `achieved_flows` are robot-base points, and `episode_context/extrinsics` equals per-step `obs/extrinsics`. The correct transform direction is base/world to camera.

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
)
```

## Outputs

| Output field | Shape | Meaning |
| --- | --- | --- |
| `scene_points_pred` | `(11, N, 3)` | Absolute predicted xyz trajectory for every input scene point. |
| `scene_displacement_pred` | `(11, N, 3)` | Predicted displacement relative to input point positions. |
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

