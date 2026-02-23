from __future__ import annotations

import base64
import importlib
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Support both module paths:
# - world_models.ui.server
# - torchwm.world_models.ui.server
if "world_models" not in sys.modules:
    package_name = __package__ or ""
    if package_name.endswith("world_models.ui"):
        maybe_world_models_pkg = package_name.rsplit(".ui", 1)[0]
        if maybe_world_models_pkg != "world_models":
            try:
                sys.modules["world_models"] = importlib.import_module(
                    maybe_world_models_pkg
                )
            except Exception:
                pass

from ..configs.dreamer_config import DreamerConfig
from ..envs import list_available_atari_envs
from ..models.dreamer import DreamerAgent
from ..models.planet import Planet
from ..training.train_planet import train as planet_train
from ..utils.utils import flatten_dict
from ..catalog import get_catalog, refresh_catalog


CATALOG = get_catalog()

SUPPORTED_MODELS = CATALOG["models"]
ENVIRONMENTS_BY_MODEL = {
    k: v.get("supported_environments", []) for k, v in SUPPORTED_MODELS.items()
}
DEFAULT_MODEL_CONFIGS = {
    k: v.get("default_config", {}) for k, v in SUPPORTED_MODELS.items()
}
DEFAULT_TRAINING_CONFIGS = CATALOG["default_training_configs"]


def _build_env_catalog() -> dict[str, list[str]]:
    atari_envs: list[str] = []
    try:
        atari_envs = list_available_atari_envs()
    except Exception:
        atari_envs = []

    all_envs = {}
    for model_key in SUPPORTED_MODELS:
        base_envs = ENVIRONMENTS_BY_MODEL.get(model_key, [])
        if model_key == "planet":
            base_envs = base_envs + atari_envs[:80]
        all_envs[model_key] = base_envs

    return all_envs


ENVIRONMENTS_BY_MODEL = _build_env_catalog()


MODEL_TRAINERS: dict[str, Any] = {}


def register_model_trainer(model_key: str, trainer_class: type) -> None:
    """Register a trainer class for a model.

    Args:
        model_key: Unique identifier for the model
        trainer_class: Class with train() method
    """
    MODEL_TRAINERS[model_key] = trainer_class


def get_model_trainer(model_key: str) -> type | None:
    """Get the trainer class for a model.

    Args:
        model_key: Unique identifier for the model

    Returns:
        Trainer class or None if not found
    """
    if model_key in MODEL_TRAINERS:
        return MODEL_TRAINERS[model_key]

    trainer_map = {
        "dreamer": DreamerAgent,
        "planet": Planet,
    }

    return trainer_map.get(model_key)


class LoadModelRequest(BaseModel):
    model: str
    config: dict[str, Any] = Field(default_factory=dict)


class LoadEnvironmentRequest(BaseModel):
    environment: str
    config: dict[str, Any] = Field(default_factory=dict)


class StartTrainingRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_like_template(template: Any, value: Any) -> Any:
    if isinstance(template, bool):
        return _coerce_bool(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _encode_frame_as_data_url(frame: Any) -> str | None:
    if frame is None:
        return None

    arr = np.asarray(frame)
    if arr.ndim != 3:
        return None

    # CHW -> HWC
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        return None

    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mn >= -0.6 and mx <= 0.6:
        arr = (arr + 0.5) * 255.0
    elif mn >= 0.0 and mx <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    encoded_ok, png = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not encoded_ok:
        return None
    raw = base64.b64encode(png.tobytes()).decode("ascii")
    return f"data:image/png;base64,{raw}"


def _reward_stats(values: Any, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {}
    return {
        f"{prefix}/avg_reward": float(arr.mean()),
        f"{prefix}/max_reward": float(arr.max()),
        f"{prefix}/min_reward": float(arr.min()),
        f"{prefix}/std_reward": float(arr.std()),
    }


class TrainingController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._model_name: str | None = None
        self._model_config: dict[str, Any] = {}
        self._model_module: str | None = None
        self._model_class: str | None = None
        self._environment_name: str | None = None
        self._environment_config: dict[str, Any] = {}
        self._training_config: dict[str, Any] = {}

        self._status = "idle"
        self._message = "Ready."
        self._traceback: str | None = None
        self._results_dir: str | None = None

        self._progress_current = 0
        self._progress_total = 1
        self._progress_unit = "steps"

        self._metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
        self._latest_frame_data_url: str | None = None

        self._started_at: float | None = None
        self._finished_at: float | None = None

    def available_environments(self, model_name: str | None) -> list[str]:
        if model_name is None:
            merged = sorted(
                set(ENVIRONMENTS_BY_MODEL["dreamer"])
                | set(ENVIRONMENTS_BY_MODEL["planet"])
            )
            return merged
        key = model_name.strip().lower()
        if key not in ENVIRONMENTS_BY_MODEL:
            raise ValueError(f"Unknown model '{model_name}'.")
        return ENVIRONMENTS_BY_MODEL[key]

    def load_model(self, model_name: str, config: dict[str, Any] | None = None) -> None:
        key = model_name.strip().lower()
        if key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. Available: {list(SUPPORTED_MODELS.keys())}"
            )

        with self._lock:
            if self._is_running_locked():
                raise RuntimeError("Cannot change model while training is running.")
            self._model_name = key
            self._model_config = dict(config or {})

            model_info = SUPPORTED_MODELS.get(key, {})
            self._model_module = model_info.get("module")
            self._model_class = model_info.get("class_name")

            valid_envs = set(ENVIRONMENTS_BY_MODEL.get(key, []))
            if self._environment_name not in valid_envs:
                self._environment_name = None
                self._environment_config = {}
            self._status = "idle"
            self._message = f"Model '{key}' loaded."
            self._traceback = None

    def load_environment(
        self, environment_name: str, config: dict[str, Any] | None = None
    ) -> None:
        env = environment_name.strip()
        with self._lock:
            if self._is_running_locked():
                raise RuntimeError(
                    "Cannot change environment while training is running."
                )
            if self._model_name is not None:
                valid = set(ENVIRONMENTS_BY_MODEL[self._model_name])
                if env not in valid:
                    raise ValueError(
                        f"Environment '{env}' is not valid for model '{self._model_name}'."
                    )
            self._environment_name = env
            self._environment_config = dict(config or {})
            self._status = "idle"
            self._message = f"Environment '{env}' loaded."
            self._traceback = None

    def start_training(self, config: dict[str, Any] | None = None) -> None:
        with self._lock:
            if self._is_running_locked():
                raise RuntimeError("Training is already running.")
            if self._model_name is None:
                raise ValueError("Load a model before starting training.")
            if self._environment_name is None:
                raise ValueError("Load an environment before starting training.")

            defaults = DEFAULT_TRAINING_CONFIGS.get(self._model_name, {})
            merged = dict(defaults)
            merged.update(config or {})
            self._training_config = merged

            self._metrics = defaultdict(list)
            self._latest_frame_data_url = None
            self._progress_current = 0
            self._progress_total = 1
            self._progress_unit = "steps"
            self._results_dir = None
            self._traceback = None

            self._status = "running"
            self._message = "Training started."
            self._started_at = time.time()
            self._finished_at = None
            self._stop_event = threading.Event()

            self._thread = threading.Thread(
                target=self._run_training, name="torchwm-ui-trainer", daemon=True
            )
            self._thread.start()

    def stop_training(self) -> bool:
        with self._lock:
            if not self._is_running_locked():
                return False
            self._stop_event.set()
            self._message = "Stop requested. Waiting for current step to finish."
            return True

    def snapshot_state(self) -> dict[str, Any]:
        with self._lock:
            ratio = (
                float(self._progress_current) / float(self._progress_total)
                if self._progress_total > 0
                else 0.0
            )
            return {
                "model": self._model_name,
                "environment": self._environment_name,
                "status": self._status,
                "message": self._message,
                "traceback": self._traceback,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "results_dir": self._results_dir,
                "progress": {
                    "current": self._progress_current,
                    "total": self._progress_total,
                    "unit": self._progress_unit,
                    "ratio": max(0.0, min(1.0, ratio)),
                },
            }

    def snapshot_metrics(self, limit: int) -> dict[str, Any]:
        limit = max(1, min(limit, 5000))
        with self._lock:
            payload: dict[str, list[dict[str, float]]] = {}
            for key, values in self._metrics.items():
                payload[key] = values[-limit:]
            return {"series": payload}

    def snapshot_frame(self) -> dict[str, str | None]:
        with self._lock:
            return {"image": self._latest_frame_data_url}

    def _is_running_locked(self) -> bool:
        return (
            self._status == "running"
            and self._thread is not None
            and self._thread.is_alive()
        )

    def _set_progress(self, current: int, total: int, unit: str) -> None:
        with self._lock:
            self._progress_current = int(max(0, current))
            self._progress_total = int(max(1, total))
            self._progress_unit = unit

    def _set_message(self, message: str) -> None:
        with self._lock:
            self._message = message

    def _append_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            for key, value in metrics.items():
                numeric = _to_float(value)
                if numeric is None:
                    continue
                self._metrics[key].append(
                    {"step": float(step), "value": numeric, "timestamp": now}
                )
                if len(self._metrics[key]) > 3000:
                    self._metrics[key] = self._metrics[key][-3000:]

    def _set_frame(self, frame: Any) -> None:
        encoded = _encode_frame_as_data_url(frame)
        if encoded is None:
            return
        with self._lock:
            self._latest_frame_data_url = encoded

    def _run_training(self) -> None:
        try:
            model_name = self._model_name
            if model_name == "dreamer":
                self._run_dreamer()
            elif model_name == "planet":
                self._run_planet()
            else:
                raise ValueError(
                    f"No training implementation for model '{model_name}'."
                )

            with self._lock:
                self._finished_at = time.time()
                if self._stop_event.is_set():
                    self._status = "stopped"
                    self._message = "Training stopped by user."
                else:
                    self._status = "completed"
                    self._message = "Training completed."
        except Exception as exc:
            with self._lock:
                self._status = "failed"
                self._message = f"{type(exc).__name__}: {exc}"
                self._traceback = traceback.format_exc(limit=12)
                self._finished_at = time.time()

    def _run_dreamer(self) -> None:
        with self._lock:
            env_name = self._environment_name
            model_cfg = dict(DEFAULT_MODEL_CONFIGS["dreamer"])
            model_cfg.update(self._model_config)
            train_cfg = dict(self._training_config)

        if env_name is None:
            raise ValueError("Dreamer requires an environment.")

        cfg = DreamerConfig()
        cfg.env = env_name
        for key, value in {**model_cfg, **train_cfg}.items():
            if hasattr(cfg, key):
                current = getattr(cfg, key)
                setattr(cfg, key, _coerce_like_template(current, value))

        cfg.log_video_freq = -1
        cfg.restore = False

        agent = DreamerAgent(cfg)
        total_steps = int(cfg.total_steps)
        self._set_progress(0, total_steps, "steps")

        seed_steps = max(1, int(cfg.seed_steps // max(1, cfg.action_repeat)))
        seed_rewards = agent.dreamer.collect_random_episodes(
            agent.train_env, seed_steps
        )
        self._append_metrics(0, _reward_stats(seed_rewards, "seed"))

        global_step = int(agent.dreamer.data_buffer.steps * cfg.action_repeat)
        self._set_progress(global_step, total_steps, "steps")

        while global_step <= total_steps:
            if self._stop_event.is_set():
                return

            model_loss = 0.0
            actor_loss = 0.0
            value_loss = 0.0
            for _ in range(int(cfg.update_steps)):
                if self._stop_event.is_set():
                    return
                model_loss, actor_loss, value_loss = agent.dreamer.train_one_batch()

            collect_steps = max(1, int(cfg.collect_steps // max(1, cfg.action_repeat)))
            train_rewards = agent.dreamer.act_and_collect_data(
                agent.train_env, collect_steps
            )

            logs: dict[str, Any] = {
                "train/model_loss": model_loss,
                "train/actor_loss": actor_loss,
                "train/value_loss": value_loss,
            }
            logs.update(_reward_stats(train_rewards, "train"))

            should_eval = global_step % max(1, int(cfg.test_interval)) == 0
            if should_eval:
                eval_episodes = max(1, int(train_cfg.get("preview_eval_episodes", 1)))
                eval_rewards, eval_videos = agent.dreamer.evaluate(
                    agent.test_env, eval_episodes, render=True
                )
                logs.update(_reward_stats(eval_rewards, "eval"))
                frame = self._extract_dreamer_preview_frame(eval_videos)
                if frame is not None:
                    self._set_frame(frame)

            self._append_metrics(global_step, logs)
            global_step = int(agent.dreamer.data_buffer.steps * cfg.action_repeat)
            self._set_progress(min(global_step, total_steps), total_steps, "steps")

    def _run_planet(self) -> None:
        with self._lock:
            env_name = self._environment_name
            model_cfg = dict(DEFAULT_MODEL_CONFIGS["planet"])
            model_cfg.update(self._model_config)
            train_cfg = dict(self._training_config)

        if env_name is None:
            raise ValueError("Planet requires an environment.")

        results_dir = str(
            train_cfg.get(
                "results_dir", Path("results") / f"ui_planet_{int(time.time())}"
            )
        )
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._results_dir = results_dir

        planet = Planet(
            env=env_name,
            bit_depth=int(model_cfg.get("bit_depth", 5)),
            memory_size=int(model_cfg.get("memory_size", 100)),
            action_repeats=int(model_cfg.get("action_repeats", 1)),
            headless=True,
            results_dir=results_dir,
        )

        warmup_episodes = int(train_cfg.get("warmup_episodes", 1))
        if warmup_episodes > 0:
            planet.warmup(n_episodes=warmup_episodes, random_policy=True)

        epochs = max(1, int(train_cfg.get("epochs", 20)))
        steps_per_epoch = max(1, int(train_cfg.get("steps_per_epoch", 50)))
        batch_size = max(1, int(train_cfg.get("batch_size", 32)))
        horizon = max(2, int(train_cfg.get("horizon", 50)))
        beta = float(train_cfg.get("beta", 1.0))
        save_every = max(1, int(train_cfg.get("save_every", 10)))
        record_grads = _coerce_bool(train_cfg.get("record_grads", False))

        self._set_progress(0, epochs, "epochs")

        for epoch in range(1, epochs + 1):
            if self._stop_event.is_set():
                return

            epoch_metrics: dict[str, list[float]] = defaultdict(list)
            for _ in range(steps_per_epoch):
                if self._stop_event.is_set():
                    return
                train_metrics = planet_train(
                    planet.memory,
                    planet.rssm.train(),
                    planet.optimizer,
                    planet.device,
                    N=batch_size,
                    H=horizon,
                    beta=beta,
                    grads=record_grads,
                )
                flat = flatten_dict(train_metrics)
                for key, value in flat.items():
                    numeric = _to_float(value)
                    if numeric is not None:
                        epoch_metrics[f"train/{key.replace('.', '/')}"].append(numeric)

            epoch_summary: dict[str, float] = {
                key: float(np.mean(values))
                for key, values in epoch_metrics.items()
                if values
            }

            planet.memory.append(planet.rollout_gen.rollout_once(explore=True))
            eval_episode, eval_frames, eval_metrics = planet.rollout_gen.rollout_eval()
            planet.memory.append(eval_episode)

            flat_eval = flatten_dict(eval_metrics)
            for key, value in flat_eval.items():
                metric_name = key.replace(".", "/")
                if isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value, dtype=np.float32).reshape(-1)
                    if arr.size > 0:
                        epoch_summary[f"{metric_name}_mean"] = float(arr.mean())
                else:
                    numeric = _to_float(value)
                    if numeric is not None:
                        epoch_summary[metric_name] = numeric

            if eval_frames is not None and len(eval_frames) > 0:
                self._set_frame(eval_frames[-1])

            self._append_metrics(epoch, epoch_summary)
            self._set_progress(epoch, epochs, "epochs")
            self._set_message(f"Planet epoch {epoch}/{epochs}")

            if epoch % save_every == 0:
                ckpt = Path(results_dir) / f"ckpt_{epoch}.pth"
                torch.save(planet.rssm.state_dict(), ckpt)

    @staticmethod
    def _extract_dreamer_preview_frame(videos: Any) -> np.ndarray | None:
        if videos is None:
            return None

        if isinstance(videos, np.ndarray) and videos.dtype != object:
            if videos.ndim >= 5:
                return np.asarray(videos[0, -1])
            if videos.ndim == 4:
                return np.asarray(videos[-1])
            return None

        try:
            first_video = videos[0]
            if len(first_video) == 0:
                return None
            return np.asarray(first_video[-1])
        except Exception:
            return None


controller = TrainingController()
app = FastAPI(title="TorchWM UI Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/catalog")
def catalog(refresh: bool = Query(default=False)) -> dict[str, Any]:
    global CATALOG, SUPPORTED_MODELS, ENVIRONMENTS_BY_MODEL, DEFAULT_MODEL_CONFIGS, DEFAULT_TRAINING_CONFIGS

    if refresh:
        CATALOG = refresh_catalog()
        SUPPORTED_MODELS = CATALOG["models"]
        ENVIRONMENTS_BY_MODEL = {
            k: v.get("supported_environments", []) for k, v in SUPPORTED_MODELS.items()
        }
        DEFAULT_MODEL_CONFIGS = {
            k: v.get("default_config", {}) for k, v in SUPPORTED_MODELS.items()
        }
        DEFAULT_TRAINING_CONFIGS = CATALOG["default_training_configs"]

    return {
        "models": SUPPORTED_MODELS,
        "environments_by_model": ENVIRONMENTS_BY_MODEL,
        "default_model_configs": DEFAULT_MODEL_CONFIGS,
        "default_training_configs": DEFAULT_TRAINING_CONFIGS,
        "all_environments": CATALOG["environments"],
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"items": list(SUPPORTED_MODELS.keys())}


class RegisterModelRequest(BaseModel):
    model_key: str
    label: str
    module: str
    class_name: str
    description: str = ""
    supported_environments: list[str] = []
    default_config: dict[str, Any] = {}


@app.post("/api/models/register")
def register_model(payload: RegisterModelRequest) -> dict[str, Any]:
    global SUPPORTED_MODELS, ENVIRONMENTS_BY_MODEL, DEFAULT_MODEL_CONFIGS

    SUPPORTED_MODELS[payload.model_key] = {
        "label": payload.label,
        "module": payload.module,
        "class_name": payload.class_name,
        "description": payload.description,
        "supported_environments": payload.supported_environments,
        "default_config": payload.default_config,
    }
    ENVIRONMENTS_BY_MODEL[payload.model_key] = payload.supported_environments
    DEFAULT_MODEL_CONFIGS[payload.model_key] = payload.default_config

    return {"status": "registered", "model": payload.model_key}


@app.post("/api/models/register-trainer/{model_key}")
def register_trainer(
    model_key: str, trainer_module: str, trainer_class: str
) -> dict[str, Any]:
    """Register a trainer for a model dynamically.

    Args:
        model_key: Model identifier
        trainer_module: Python module path (e.g., 'my_package.trainer')
        trainer_class: Class name in the module
    """
    try:
        module = importlib.import_module(trainer_module)
        trainer_cls = getattr(module, trainer_class)
        register_model_trainer(model_key, trainer_cls)
        return {
            "status": "registered",
            "model": model_key,
            "trainer": f"{trainer_module}.{trainer_class}",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/environments")
def environments(model: str | None = Query(default=None)) -> dict[str, Any]:
    if model is None:
        all_envs = set()
        for env_list in ENVIRONMENTS_BY_MODEL.values():
            all_envs.update(env_list)
        return {"model": None, "items": sorted(all_envs)}

    try:
        items = controller.available_environments(model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"model": model, "items": items}


@app.post("/api/load-model")
def load_model(payload: LoadModelRequest) -> dict[str, Any]:
    try:
        controller.load_model(payload.model, payload.config)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return controller.snapshot_state()


@app.post("/api/load-environment")
def load_environment(payload: LoadEnvironmentRequest) -> dict[str, Any]:
    try:
        controller.load_environment(payload.environment, payload.config)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return controller.snapshot_state()


@app.post("/api/train/start")
def start_training(payload: StartTrainingRequest) -> dict[str, Any]:
    try:
        controller.start_training(payload.config)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return controller.snapshot_state()


@app.post("/api/train/stop")
def stop_training() -> dict[str, Any]:
    stopped = controller.stop_training()
    return {"stop_requested": stopped, **controller.snapshot_state()}


@app.get("/api/state")
def state() -> dict[str, Any]:
    return controller.snapshot_state()


@app.get("/api/metrics")
def metrics(limit: int = Query(default=400, ge=1, le=5000)) -> dict[str, Any]:
    return controller.snapshot_metrics(limit=limit)


@app.get("/api/frame")
def frame() -> dict[str, str | None]:
    return controller.snapshot_frame()
