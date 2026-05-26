from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .features import make_robot_features, make_scene_features
from .schema import (
    POINTWORLD_HORIZON,
    PointWorldOutput,
    as_numpy,
    require_shape,
    validate_horizon,
)


@contextmanager
def _working_directory(path: Path):
    """Run PointWorld code from its repo root for relative asset paths."""
    import os

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PointWorldPredictor:
    """Small inference API around the released NVIDIA PointWorld code.

    The recommended high-level method is `predict_scene_motion`, where callers
    provide scene points plus qpos/gripper_pos. The wrapper uses PointWorld's
    RobotSampler/FK internally to construct robot point conditioning.
    """

    def __init__(
        self,
        pointworld_repo: str | Path,
        checkpoint: str | Path,
        dinov3_weights: str | Path | None = None,
        device: str = "cuda",
        robot_device: str = "cpu",
        domain: str = "droid",
        num_robot_points: int = 500,
        robot_seed: int = 0,
    ) -> None:
        self.pointworld_repo = Path(pointworld_repo).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dinov3_weights = Path(dinov3_weights).expanduser().resolve() if dinov3_weights else None
        self.device = torch.device(device)
        self.robot_device = robot_device
        self.domain = domain
        self.num_robot_points = int(num_robot_points)
        self.robot_seed = int(robot_seed)
        self._model = None

        if not self.pointworld_repo.exists():
            raise FileNotFoundError(f"PointWorld repo not found: {self.pointworld_repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"PointWorld checkpoint not found: {self.checkpoint}")
        if str(self.pointworld_repo) not in sys.path:
            sys.path.insert(0, str(self.pointworld_repo))
        if self.dinov3_weights is not None:
            self._prepare_dinov3_weights(self.dinov3_weights)

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _prepare_dinov3_weights(self, weights: Path) -> None:
        if not weights.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
        target_dir = self.pointworld_repo / "third_party" / "dinov3" / "checkpoints"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / weights.name
        if not target.exists():
            try:
                target.symlink_to(weights)
            except OSError:
                import shutil

                shutil.copy2(weights, target)

    def _load_model(self):
        arguments = importlib.import_module("arguments")
        base = importlib.import_module("pointworld.base")
        contract = importlib.import_module("pointworld.checkpoint_contract")

        checkpoint = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        model_contract, _ = contract.read_checkpoint_contract(checkpoint, context=f"checkpoint {self.checkpoint}")
        args = arguments.parse_args(skip_command_line=True)
        args.model_path = str(self.checkpoint)
        args.device = str(self.device)
        args.distributed = False
        args.disable_compile = True
        args.domains = [self.domain]
        args.data_dirs = []
        contract.apply_model_contract_to_args(args, model_contract, context=f"checkpoint {self.checkpoint}")
        norm_stats_path = Path(args.norm_stats_path)
        if not norm_stats_path.is_absolute():
            args.norm_stats_path = str(self.pointworld_repo / norm_stats_path)

        data_info_dict = {
            "robot_features_dim": 16,
            "scene_features_dim": 31,
        }
        with _working_directory(self.pointworld_repo):
            model = base.BaseModel(args, data_info_dict, rank=0, cpu_pg=None).to(self.device)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Missing PointWorld checkpoint keys: {missing[:10]}")
        if unexpected:
            raise RuntimeError(f"Unexpected PointWorld checkpoint keys: {unexpected[:10]}")
        model.eval()
        return model

    def sample_robot_points_from_qpos(
        self,
        qpos: np.ndarray,
        gripper_pos: np.ndarray,
        *,
        num_robot_points: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qpos = as_numpy(qpos, dtype=np.float32, name="qpos")
        gripper_pos = as_numpy(gripper_pos, dtype=np.float32, name="gripper_pos").reshape(POINTWORLD_HORIZON, 1)
        require_shape(qpos, (POINTWORLD_HORIZON, 7), "qpos")

        dataset_robot = importlib.import_module("dataset_components.robot")
        robot_sampler_mod = importlib.import_module("robot_sampler")
        utils = importlib.import_module("utils")
        urdf_path = Path(utils.resolve_robot_urdf(self.domain))
        if not urdf_path.is_absolute():
            urdf_path = self.pointworld_repo / urdf_path

        with _working_directory(self.pointworld_repo):
            sampler = robot_sampler_mod.RobotSampler(
                urdf_path=str(urdf_path),
                gripper_only=True,
                device=self.robot_device,
            )
            sampler.presample(num_robot_points or self.num_robot_points, seed=self.robot_seed if seed is None else seed)

            qpos_t = torch.as_tensor(qpos, dtype=torch.float32, device=sampler.device)
            gripper_t = torch.as_tensor(gripper_pos.reshape(-1), dtype=torch.float32, device=sampler.device)
            joint_dict = {f"panda_joint{i + 1}": qpos_t[:, i] for i in range(7)}
            joint_dict.update(dataset_robot.build_robotiq_joint_dict(gripper_t, sampler.joint_names))
            with torch.no_grad():
                points, colors, normals = sampler.compute_points(joint_dict)
        return (
            points.detach().cpu().numpy().astype(np.float32),
            colors.detach().cpu().numpy(),
            normals.detach().cpu().numpy().astype(np.float32),
        )

    def predict_scene_motion(
        self,
        *,
        scene_points: np.ndarray,
        qpos: np.ndarray,
        gripper_pos: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        scene_normals: np.ndarray | None = None,
        scene_colors: np.ndarray | None = None,
    ) -> PointWorldOutput:
        """Predict the full PointWorld horizon for input scene points.

        Args:
            scene_points: Current scene points to forecast, shape (N, 3).
            qpos: Franka/Panda arm joint trajectory, shape (T=11, 7).
            gripper_pos: Gripper open/close state trajectory, shape (T=11,) or (T=11, 1).
            rgb: Current RGB image, shape (H, W, 3), uint8 0..255.
            depth: Current depth image in meters, shape (H, W).
            intrinsics: Camera intrinsics, shape (3, 3).
            extrinsics: Camera extrinsics, shape (4, 4).
        """
        scene_points = as_numpy(scene_points, dtype=np.float32, name="scene_points")
        require_shape(scene_points, (None, 3), "scene_points")
        qpos = as_numpy(qpos, dtype=np.float32, name="qpos")
        validate_horizon("qpos", qpos)
        gripper_pos = as_numpy(gripper_pos, dtype=np.float32, name="gripper_pos").reshape(POINTWORLD_HORIZON, 1)
        robot_points, robot_colors, robot_normals = self.sample_robot_points_from_qpos(qpos, gripper_pos)
        robot_features = make_robot_features(robot_points, gripper_pos, robot_normals, robot_colors)
        scene_features = make_scene_features(
            scene_points,
            robot_points,
            gripper_pos,
            as_numpy(rgb, name="rgb"),
            as_numpy(depth, dtype=np.float32, name="depth"),
            as_numpy(intrinsics, dtype=np.float32, name="intrinsics"),
            as_numpy(extrinsics, dtype=np.float32, name="extrinsics"),
            scene_normals=scene_normals,
            scene_colors=scene_colors,
        )
        return self.predict(
            scene_points=scene_points,
            scene_features=scene_features,
            robot_points=robot_points,
            robot_features=robot_features,
            robot_exists=None,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    def predict_object_motion(
        self,
        *,
        object_points: np.ndarray,
        qpos: np.ndarray,
        gripper_pos: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        object_exists: np.ndarray | None = None,
        object_normals: np.ndarray | None = None,
        object_colors: np.ndarray | None = None,
    ) -> PointWorldOutput:
        """Compatibility alias for object-only scene-point prediction."""
        return self.predict_scene_motion(
            scene_points=object_points,
            qpos=qpos,
            gripper_pos=gripper_pos,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            scene_normals=object_normals,
            scene_colors=object_colors,
        )

    def predict(
        self,
        *,
        scene_points: np.ndarray | None = None,
        scene_features: np.ndarray | None = None,
        scene_exists: np.ndarray | None = None,
        robot_points: np.ndarray,
        robot_features: np.ndarray,
        robot_exists: np.ndarray | None = None,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        object_points: np.ndarray | None = None,
        object_features: np.ndarray | None = None,
        object_exists: np.ndarray | None = None,
    ) -> PointWorldOutput:
        """Low-level API when robot point trajectory/features are already built."""
        if scene_points is None:
            scene_points = object_points
        if scene_features is None:
            scene_features = object_features
        if scene_exists is None:
            scene_exists = object_exists
        if scene_points is None:
            raise ValueError("scene_points is required")
        if scene_features is None:
            raise ValueError("scene_features is required")

        scene_points = as_numpy(scene_points, dtype=np.float32, name="scene_points")
        require_shape(scene_points, (None, 3), "scene_points")
        n_points = scene_points.shape[0]
        scene_features = as_numpy(scene_features, dtype=np.float32, name="scene_features")
        require_shape(scene_features, (POINTWORLD_HORIZON, n_points, 31), "scene_features")
        robot_points = as_numpy(robot_points, dtype=np.float32, name="robot_points")
        require_shape(robot_points, (POINTWORLD_HORIZON, None, 3), "robot_points")
        robot_features = as_numpy(robot_features, dtype=np.float32, name="robot_features")
        require_shape(robot_features, (POINTWORLD_HORIZON, robot_points.shape[1], 16), "robot_features")
        if scene_exists is None:
            scene_exists_arr = np.ones((POINTWORLD_HORIZON, n_points), dtype=bool)
        else:
            scene_exists_1d = as_numpy(scene_exists, dtype=bool, name="scene_exists").reshape(n_points)
            scene_exists_arr = np.repeat(scene_exists_1d.reshape(1, n_points), POINTWORLD_HORIZON, axis=0)
        if robot_exists is None:
            robot_exists_arr = np.ones((POINTWORLD_HORIZON, robot_points.shape[1]), dtype=bool)
        else:
            robot_exists_arr = as_numpy(robot_exists, dtype=bool, name="robot_exists")
            require_shape(robot_exists_arr, (POINTWORLD_HORIZON, robot_points.shape[1]), "robot_exists")

        rgb_arr = as_numpy(rgb, name="rgb")
        depth_arr = as_numpy(depth, dtype=np.float32, name="depth")
        intr_arr = as_numpy(intrinsics, dtype=np.float32, name="intrinsics")
        ext_arr = as_numpy(extrinsics, dtype=np.float32, name="extrinsics")
        require_shape(rgb_arr, (None, None, 3), "rgb")
        require_shape(depth_arr, (rgb_arr.shape[0], rgb_arr.shape[1]), "depth")
        require_shape(intr_arr, (3, 3), "intrinsics")
        require_shape(ext_arr, (4, 4), "extrinsics")

        scene_flows = np.repeat(scene_points.reshape(1, n_points, 3), POINTWORLD_HORIZON, axis=0)

        def tensor(array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
            out = torch.from_numpy(np.ascontiguousarray(array).copy())
            if dtype is not None:
                out = out.to(dtype=dtype)
            return out.unsqueeze(0).to(self.device)

        batch: dict[str, Any] = {
            "scene_flows": tensor(scene_flows, torch.float32),
            "scene_features": tensor(scene_features, torch.float32),
            "scene_exists": tensor(scene_exists_arr, torch.bool),
            "robot_flows": tensor(robot_points, torch.float32),
            "robot_features": tensor(robot_features, torch.float32),
            "robot_exists": tensor(robot_exists_arr, torch.bool),
            "cam0_initial_rgb": tensor(rgb_arr, None),
            "cam0_initial_depth": tensor(depth_arr, torch.float32),
            "cam0_intrinsic": tensor(intr_arr, torch.float32),
            "cam0_extrinsic": tensor(ext_arr, torch.float32),
            "cam0_exists": torch.ones((1,), dtype=torch.bool, device=self.device),
            "__domain__": [self.domain],
            "__key__": ["pointworld_baseline_sample"],
        }
        with torch.inference_mode():
            outputs_t = self.model(batch, training=False)
        outputs = {
            key: value.detach().cpu().numpy()
            for key, value in outputs_t.items()
            if isinstance(value, torch.Tensor)
        }
        return PointWorldOutput(
            object_points_pred=outputs["scene_flows"][0],
            object_displacement_pred=outputs.get("scene_relative", None)[0] if "scene_relative" in outputs else None,
            object_displacement_pred_norm=outputs.get("scene_relative_norm", None)[0]
            if "scene_relative_norm" in outputs
            else None,
            log_var=outputs.get("log_var", None)[0] if "log_var" in outputs else None,
            confidence=outputs.get("confidence", None)[0] if "confidence" in outputs else None,
            raw_outputs=outputs,
        )
