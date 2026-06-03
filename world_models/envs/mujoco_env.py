from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image


RewardFn = Callable[[Any, Any, np.ndarray, dict[str, Any]], float]
TerminalFn = Callable[[Any, Any, dict[str, Any]], bool]


def _load_mujoco():
    if "mujoco" in sys.modules:
        return sys.modules["mujoco"]
    if importlib.util.find_spec("mujoco") is None:
        raise ImportError(
            "The native MuJoCo bindings are required for MuJoCoImageEnv. "
            "Install them with `pip install mujoco` or `pip install torchwm[mujoco]`."
        )
    return importlib.import_module("mujoco")


def _validate_model_source(
    *,
    xml_path: str | Path | None,
    xml_string: str | None,
    binary_path: str | Path | None,
) -> None:
    sources = [xml_path is not None, xml_string is not None, binary_path is not None]
    if sum(sources) != 1:
        raise ValueError(
            "Provide exactly one of xml_path, xml_string, or binary_path for MuJoCoImageEnv."
        )


def _infer_model_source(model: str | Path) -> dict[str, str | Path]:
    model_text = str(model)
    if model_text.lstrip().startswith("<"):
        return {"xml_string": model_text}
    if model_text.endswith(".mjb"):
        return {"binary_path": model}
    return {"xml_path": model}


def _model_source_from_config(args) -> dict[str, str | Path]:
    if getattr(args, "mujoco_xml_string", None) is not None:
        return {"xml_string": args.mujoco_xml_string}
    if getattr(args, "mujoco_binary_path", None) is not None:
        return {"binary_path": args.mujoco_binary_path}
    return {"xml_path": getattr(args, "mujoco_xml_path", None) or args.env}


def make_mujoco_env_from_config(args, size: tuple[int, int]) -> "MuJoCoImageEnv":
    """Build a MuJoCoImageEnv from a DreamerConfig-like object."""
    return MuJoCoImageEnv(
        **_model_source_from_config(args),
        seed=args.seed,
        size=size,
        camera=getattr(args, "mujoco_camera", None),
        frame_skip=int(getattr(args, "mujoco_frame_skip", 1)),
        reset_noise_scale=float(getattr(args, "mujoco_reset_noise_scale", 0.0)),
    )


class MuJoCoImageEnv:
    """Native MuJoCo environment adapter for pixel-based world-model training.

    The adapter uses the low-level ``mujoco`` Python package directly: models are
    compiled from MJCF XML strings/files or MJB binaries via ``mujoco.MjModel``;
    simulation state lives in ``mujoco.MjData``; actions are written to
    ``data.ctrl``; and images are produced with ``mujoco.Renderer``. Observations
    follow TorchWM's Dreamer-style contract: ``{"image": uint8[C, H, W]}``.

    Native MuJoCo models do not define task rewards or episode termination by
    themselves, so callers can supply ``reward_fn`` and ``terminal_fn`` callbacks.
    By default, rewards are ``0.0`` and episodes terminate only through external
    wrappers such as ``TimeLimit``.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        xml_string: str | None = None,
        binary_path: str | Path | None = None,
        assets: dict[str, bytes] | None = None,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        camera: str | int | None = None,
        reward_fn: RewardFn | None = None,
        terminal_fn: TerminalFn | None = None,
        frame_skip: int = 1,
        reset_noise_scale: float = 0.0,
        default_control_range: tuple[float, float] = (-1.0, 1.0),
    ):
        _validate_model_source(
            xml_path=xml_path,
            xml_string=xml_string,
            binary_path=binary_path,
        )

        self._mujoco = _load_mujoco()
        self._size = (int(size[0]), int(size[1]))
        self._camera = camera
        self._reward_fn = reward_fn
        self._terminal_fn = terminal_fn
        self._frame_skip = max(1, int(frame_skip))
        self._reset_noise_scale = float(reset_noise_scale)
        self._rng = np.random.default_rng(seed)
        self._closed = False

        if xml_string is not None:
            self.model = self._mujoco.MjModel.from_xml_string(xml_string, assets)
        elif binary_path is not None:
            self.model = self._mujoco.MjModel.from_binary_path(str(binary_path))
        else:
            self.model = self._mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = self._mujoco.MjData(self.model)

        height, width = self._size
        self._renderer = self._mujoco.Renderer(self.model, height=height, width=width)

        self._observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, height, width),
                    dtype=np.uint8,
                )
            }
        )
        self._action_space = self._build_action_space(default_control_range)
        self.max_episode_steps = 1000

    def _build_action_space(
        self, default_control_range: tuple[float, float]
    ) -> gym.spaces.Box:
        action_dim = int(getattr(self.model, "nu", 0))
        if action_dim <= 0:
            return gym.spaces.Box(
                low=np.zeros((0,), dtype=np.float32),
                high=np.zeros((0,), dtype=np.float32),
                dtype=np.float32,
            )

        ctrlrange = np.asarray(getattr(self.model, "actuator_ctrlrange", []), dtype=np.float32)
        limited = np.asarray(getattr(self.model, "actuator_ctrllimited", []), dtype=bool)
        if ctrlrange.shape == (action_dim, 2) and limited.shape[0] == action_dim:
            default_low, default_high = default_control_range
            low = np.where(limited, ctrlrange[:, 0], float(default_low)).astype(np.float32)
            high = np.where(limited, ctrlrange[:, 1], float(default_high)).astype(np.float32)
        else:
            low = np.full((action_dim,), float(default_control_range[0]), dtype=np.float32)
            high = np.full((action_dim,), float(default_control_range[1]), dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _render_chw(self) -> np.ndarray:
        if self._camera is None:
            self._renderer.update_scene(self.data)
        else:
            self._renderer.update_scene(self.data, camera=self._camera)
        image = np.asarray(self._renderer.render())
        if image.shape[:2] != self._size:
            image = np.asarray(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.BILINEAR
                )
            )
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image.astype(np.uint8, copy=False).transpose(2, 0, 1).copy()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mujoco.mj_resetData(self.model, self.data)
        if self._reset_noise_scale > 0.0:
            if getattr(self.model, "nq", 0):
                self.data.qpos[:] += self._rng.normal(
                    0.0, self._reset_noise_scale, size=self.data.qpos.shape
                )
            if getattr(self.model, "nv", 0):
                self.data.qvel[:] += self._rng.normal(
                    0.0, self._reset_noise_scale, size=self.data.qvel.shape
                )
        self._mujoco.mj_forward(self.model, self.data)
        return {"image": self._render_chw()}

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        clipped = np.clip(action_arr, self.action_space.low, self.action_space.high)
        if clipped.size:
            self.data.ctrl[:] = clipped
        self._mujoco.mj_step(self.model, self.data, nstep=self._frame_skip)

        info = {
            "action": clipped.astype(np.float32, copy=True),
            "time": float(getattr(self.data, "time", 0.0)),
            "qpos": np.asarray(getattr(self.data, "qpos", []), dtype=np.float64).copy(),
            "qvel": np.asarray(getattr(self.data, "qvel", []), dtype=np.float64).copy(),
        }
        if self._reward_fn is None:
            reward = 0.0
        else:
            reward = float(self._reward_fn(self.model, self.data, clipped, info))
        done = (
            bool(self._terminal_fn(self.model, self.data, info))
            if self._terminal_fn
            else False
        )
        return {"image": self._render_chw()}, reward, done, info

    def render(self):
        return self._render_chw().transpose(1, 2, 0).copy()

    def close(self):
        if self._closed:
            return
        self._closed = True
        close = getattr(self._renderer, "close", None)
        if callable(close):
            close()


def make_mujoco_env(
    model: str | Path | None = None,
    **kwargs,
) -> MuJoCoImageEnv:
    """Create a native MuJoCo image environment from an XML/MJB path or XML string.

    ``model`` is treated as an MJCF XML string when it starts with ``<``;
    otherwise it is interpreted as a filesystem path. Use explicit
    ``xml_path``, ``xml_string``, or ``binary_path`` keyword arguments when the
    source type should not be inferred.
    """
    if model is not None:
        kwargs.update({k: v for k, v in _infer_model_source(model).items() if k not in kwargs})
    return MuJoCoImageEnv(**kwargs)


def make_humanoid_env(
    version: str = "v4",
    xml_file: str = "humanoid.xml",
    forward_reward_weight: float = 1.25,
    ctrl_cost_weight: float = 0.1,
    contact_cost_weight: float = 5e-7,
    healthy_reward: float = 5.0,
    terminate_when_unhealthy: bool = True,
    healthy_z_range: tuple[float, float] = (1.0, 2.0),
    reset_noise_scale: float = 1e-2,
    exclude_current_positions_from_observation: bool = True,
    include_cinert_in_observation: bool = True,
    include_cvel_in_observation: bool = True,
    include_qfrc_actuator_in_observation: bool = True,
    include_cfrc_ext_in_observation: bool = True,
) -> gym.Env:
    """Create Gymnasium's task-level Humanoid MuJoCo environment."""
    env_id = f"Humanoid-{version}"
    return gym.make(
        env_id,
        xml_file=xml_file,
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        contact_cost_weight=contact_cost_weight,
        healthy_reward=healthy_reward,
        terminate_when_unhealthy=terminate_when_unhealthy,
        healthy_z_range=healthy_z_range,
        reset_noise_scale=reset_noise_scale,
        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        include_cinert_in_observation=include_cinert_in_observation,
        include_cvel_in_observation=include_cvel_in_observation,
        include_qfrc_actuator_in_observation=include_qfrc_actuator_in_observation,
        include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
    )


def make_half_cheetah_env(
    version: str = "v4",
    forward_reward_weight: float = 0.1,
    reset_noise_scale: float = 0.1,
    exclude_current_positions_from_observation: bool = True,
    render_mode: str = "rgb_array",
) -> gym.Env:
    """Create Gymnasium's task-level HalfCheetah MuJoCo environment."""
    env_id = f"HalfCheetah-{version}"
    return gym.make(
        env_id,
        forward_reward_weight=forward_reward_weight,
        reset_noise_scale=reset_noise_scale,
        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        render_mode=render_mode,
    )
