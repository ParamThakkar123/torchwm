# Unity ML-Agents

The Unity ML-Agents backend connects TorchWM to an external Unity executable. It supports continuous-action ML-Agents behaviors, extracts visual observations when available, and converts vector observations into image-like inputs when necessary.

## Install

```bash
pip install torchwm[ml-agents]
```

You also need a Unity environment executable built for the platform running training.

## Main API

```python
from torchwm import UnityMLAgentsEnv

env = UnityMLAgentsEnv(
    file_name="/path/to/UnityEnvironment.x86_64",
    behavior_name=None,
    seed=0,
    size=(64, 64),
    worker_id=0,
    base_port=5005,
    no_graphics=True,
    time_scale=20.0,
    quality_level=1,
    max_episode_steps=1000,
)
```

If `behavior_name` is omitted, TorchWM uses the first behavior advertised by the Unity executable. The wrapper currently supports continuous action spaces only.

## Dreamer configuration

```python
from torchwm import DreamerConfig

cfg = DreamerConfig()
cfg.env_backend = "unity_mlagents"
cfg.unity_file_name = "/path/to/UnityEnvironment.x86_64"
cfg.unity_behavior_name = None
cfg.unity_no_graphics = True
cfg.unity_time_scale = 20.0
cfg.unity_worker_id = 0
cfg.unity_base_port = 5005
cfg.unity_quality_level = 1
```

`env_backend` can be `"unity"`, `"unity_mlagents"`, or `"mlagents"`. Dreamer creation requires `unity_file_name`; without it, `make_env()` raises a clear configuration error.

## Observation contract

`UnityMLAgentsEnv` exposes:

```python
{"image": uint8 array with shape (3, H, W)}
```

For visual Unity observations, the wrapper normalizes and resizes the image, handles grayscale/RGBA/channel-order variants, and returns channel-first RGB. For vector-only observations, it synthesizes a simple RGB image using value bands.

## Action contract

The action space is a continuous Gymnasium `Box` with shape `(continuous_size,)` and bounds `[-1.0, 1.0]`. Discrete ML-Agents action spaces are not supported by the current wrapper.

## Engine settings

TorchWM configures Unity through `EngineConfigurationChannel`:

- `width` and `height` come from the requested TorchWM image size.
- `quality_level` controls graphics quality.
- `time_scale` speeds up or slows down simulation.
- `no_graphics=True` can improve throughput for headless training when the executable supports it.

Use unique `worker_id` values when launching multiple Unity environments on the same machine so ML-Agents ports do not collide.

## Troubleshooting

- **No behaviors found**: verify the executable starts and has an ML-Agents behavior configured.
- **Behavior name not found**: inspect available behavior names or leave `behavior_name=None` to select the first one.
- **Discrete action error**: configure the Unity behavior for continuous actions or extend the wrapper before using discrete branches.
- **Port conflicts**: change `worker_id` or `base_port` for parallel runs.
- **Slow simulation**: increase `time_scale`, lower `quality_level`, and use `no_graphics=True` where supported.
