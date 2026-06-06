"""Local browser for HDF5 replay buffers."""

from __future__ import annotations

import importlib
import json
import logging
import math
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import click

logger = logging.getLogger("torchwm.replay_browser")

OBSERVATION_KEYS = ("observations", "obs", "frames", "images")
ACTION_KEYS = ("actions", "action")
REWARD_KEYS = ("rewards", "reward")
DONE_KEYS = ("dones", "terminals", "terminated", "done")


def _ensure(module: str) -> Any | None:
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def _find_first_dataset(group: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in group and hasattr(group[name], "shape"):
            return group[name]
    return None


def _iter_episode_groups(h5file: Any) -> list[tuple[str, Any]]:
    groups = []
    for key in h5file.keys():
        value = h5file[key]
        if hasattr(value, "keys") and _find_first_dataset(value, OBSERVATION_KEYS) is not None:
            groups.append((key, value))
    return sorted(groups, key=lambda item: item[0])


def _split_flat_episodes(done_values: Any, length: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    episodes: list[tuple[int, int]] = []
    start = 0
    if done_values is not None:
        for index in range(min(length, len(done_values))):
            if bool(done_values[index]):
                episodes.append((start, index + 1))
                start = index + 1
    if start < length:
        episodes.append((start, length))
    return episodes or [(0, length)]


def summarize_replay_buffer(path: Path) -> dict[str, Any]:
    """Return episode and tensor-shape metadata for a replay-buffer HDF5 file."""
    h5py = _ensure("h5py")
    if h5py is None:
        raise click.ClickException("Please install h5py to browse replay buffers")

    with h5py.File(path, "r") as h5file:
        root_obs = _find_first_dataset(h5file, OBSERVATION_KEYS)
        root_actions = _find_first_dataset(h5file, ACTION_KEYS)
        root_rewards = _find_first_dataset(h5file, REWARD_KEYS)
        root_dones = _find_first_dataset(h5file, DONE_KEYS)
        episode_groups = _iter_episode_groups(h5file)

        episodes: list[dict[str, Any]] = []
        if root_obs is not None:
            ranges = _split_flat_episodes(root_dones, int(root_obs.shape[0]))
            for episode_index, (start, stop) in enumerate(ranges):
                reward_sum = float(root_rewards[start:stop].sum()) if root_rewards is not None else None
                episodes.append(
                    {
                        "name": f"episode_{episode_index}",
                        "start": start,
                        "stop": stop,
                        "length": stop - start,
                        "reward_sum": reward_sum,
                    }
                )
            obs_shape = tuple(root_obs.shape[1:])
            action_shape = tuple(root_actions.shape[1:]) if root_actions is not None and len(root_actions.shape) > 1 else ()
        elif episode_groups:
            obs0 = _find_first_dataset(episode_groups[0][1], OBSERVATION_KEYS)
            obs_shape = tuple(obs0.shape[1:]) if obs0 is not None else ()
            action_shape = ()
            for name, group in episode_groups:
                obs = _find_first_dataset(group, OBSERVATION_KEYS)
                actions = _find_first_dataset(group, ACTION_KEYS)
                rewards = _find_first_dataset(group, REWARD_KEYS)
                if obs is None:
                    continue
                reward_sum = float(rewards[:].sum()) if rewards is not None else None
                episodes.append(
                    {
                        "name": name,
                        "start": 0,
                        "stop": int(obs.shape[0]),
                        "length": int(obs.shape[0]),
                        "reward_sum": reward_sum,
                    }
                )
                if actions is not None and len(actions.shape) > 1:
                    action_shape = tuple(actions.shape[1:])
        else:
            raise click.ClickException(
                "No observation dataset found. Expected root observations/obs/frames/images or per-episode groups."
            )

    return {
        "path": str(path),
        "episodes": episodes,
        "episode_count": len(episodes),
        "transition_count": int(sum(ep["length"] for ep in episodes)),
        "observation_shape": obs_shape,
        "action_shape": action_shape,
    }


def _array_to_image(array: Any) -> Any:
    numpy = _ensure("numpy")
    image_mod = _ensure("PIL.Image")
    if numpy is None or image_mod is None:
        raise click.ClickException("Please install numpy and Pillow to render replay frames")

    arr = numpy.asarray(array)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = numpy.moveaxis(arr, 0, -1)
    if arr.dtype.kind == "f":
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = numpy.clip(arr, 0, 255).astype("uint8")
    elif arr.dtype != numpy.uint8:
        arr = numpy.clip(arr, 0, 255).astype("uint8")
    if arr.ndim == 2:
        return image_mod.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return image_mod.fromarray(arr[..., 0], mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return image_mod.fromarray(arr).convert("RGB")
    raise click.ClickException(f"Unsupported observation shape for rendering: {arr.shape}")


def _format_value(value: Any) -> str:
    numpy = _ensure("numpy")
    if numpy is not None:
        arr = numpy.asarray(value)
        if arr.ndim == 0:
            value = arr.item()
        elif arr.size <= 6:
            return numpy.array2string(arr, precision=3, separator=", ")
        else:
            return f"shape={tuple(arr.shape)}"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_frame_png(path: Path, episode_index: int, step_index: int) -> bytes:
    """Render one replay frame as PNG with step, reward, action, and done overlay."""
    h5py = _ensure("h5py")
    draw_mod = _ensure("PIL.ImageDraw")
    font_mod = _ensure("PIL.ImageFont")
    if h5py is None or draw_mod is None or font_mod is None:
        raise click.ClickException("Please install h5py and Pillow to browse replay buffers")

    with h5py.File(path, "r") as h5file:
        root_obs = _find_first_dataset(h5file, OBSERVATION_KEYS)
        if root_obs is not None:
            rewards = _find_first_dataset(h5file, REWARD_KEYS)
            actions = _find_first_dataset(h5file, ACTION_KEYS)
            dones = _find_first_dataset(h5file, DONE_KEYS)
            ranges = _split_flat_episodes(dones, int(root_obs.shape[0]))
            start, stop = ranges[episode_index]
            absolute_index = start + step_index
            if absolute_index >= stop:
                raise IndexError("step out of range")
            obs = root_obs[absolute_index]
            reward = rewards[absolute_index] if rewards is not None else "n/a"
            action = actions[absolute_index] if actions is not None else "n/a"
            done = dones[absolute_index] if dones is not None else "n/a"
        else:
            _name, group = _iter_episode_groups(h5file)[episode_index]
            obs_ds = _find_first_dataset(group, OBSERVATION_KEYS)
            if obs_ds is None or step_index >= int(obs_ds.shape[0]):
                raise IndexError("step out of range")
            rewards = _find_first_dataset(group, REWARD_KEYS)
            actions = _find_first_dataset(group, ACTION_KEYS)
            dones = _find_first_dataset(group, DONE_KEYS)
            obs = obs_ds[step_index]
            reward = rewards[step_index] if rewards is not None else "n/a"
            action = actions[step_index] if actions is not None else "n/a"
            done = dones[step_index] if dones is not None else "n/a"

    image = _array_to_image(obs)
    if image.width < 320:
        scale = max(1, math.ceil(320 / image.width))
        image = image.resize((image.width * scale, image.height * scale), resample=0)
    draw = draw_mod.Draw(image)
    font = font_mod.load_default()
    lines = [
        f"episode {episode_index}  step {step_index}",
        f"reward: {_format_value(reward)}",
        f"action: {_format_value(action)}",
        f"done: {_format_value(done)}",
    ]
    padding = 5
    line_height = 13
    width = max(220, min(image.width, 520))
    height = padding * 2 + line_height * len(lines)
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    for line_no, line in enumerate(lines):
        draw.text((padding, padding + line_no * line_height), line, fill=(255, 255, 255), font=font)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def replay_browser_html(metadata: dict[str, Any]) -> str:
    metadata_json = json.dumps(metadata)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>TorchWM replay browser</title>
  <style>
    body {{ margin: 0; background: #101418; color: #e6edf3; font-family: system-ui, sans-serif; }}
    header {{ padding: 0.75rem 1rem; background: #17202a; position: sticky; top: 0; }}
    main {{ display: grid; grid-template-columns: 19rem 1fr; gap: 1rem; padding: 1rem; }}
    img {{ max-width: 100%; image-rendering: pixelated; border: 1px solid #36414f; background: #000; }}
    button, select, input {{ margin: 0.25rem 0; width: 100%; }}
    .stats {{ color: #9fb0c0; font-size: 0.9rem; }}
  </style>
</head>
<body>
<header><strong>TorchWM replay browser</strong> <span id=\"title\" class=\"stats\"></span></header>
<main>
  <aside>
    <label>Episode<select id=\"episode\"></select></label>
    <label>Step<input id=\"step\" type=\"range\" min=\"0\" value=\"0\" /></label>
    <div id=\"stepLabel\" class=\"stats\"></div>
    <button id=\"prev\">◀ previous</button>
    <button id=\"next\">next ▶</button>
    <p class=\"stats\">Use ←/→ to step, ↑/↓ to switch episodes.</p>
  </aside>
  <section><img id=\"frame\" alt=\"Replay frame with action/reward overlay\" /></section>
</main>
<script>
const metadata = {metadata_json};
let episode = 0;
let step = 0;
const episodeEl = document.getElementById('episode');
const stepEl = document.getElementById('step');
const frameEl = document.getElementById('frame');
const titleEl = document.getElementById('title');
const stepLabel = document.getElementById('stepLabel');
for (const [i, ep] of metadata.episodes.entries()) {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `${{i}}: ${{ep.name}} (${{ep.length}} steps, return ${{ep.reward_sum ?? 'n/a'}})`;
  episodeEl.appendChild(opt);
}}
function loadFrame() {{
  const ep = metadata.episodes[episode];
  step = Math.max(0, Math.min(step, ep.length - 1));
  stepEl.max = Math.max(0, ep.length - 1);
  stepEl.value = step;
  titleEl.textContent = `${{metadata.path}} · ${{metadata.episode_count}} episodes · ${{metadata.transition_count}} steps`;
  stepLabel.textContent = `episode ${{episode}}, step ${{step + 1}} / ${{ep.length}}, return ${{ep.reward_sum ?? 'n/a'}}`;
  frameEl.src = `/frame.png?episode=${{episode}}&step=${{step}}&t=${{Date.now()}}`;
}}
episodeEl.addEventListener('change', () => {{ episode = Number(episodeEl.value); step = 0; loadFrame(); }});
stepEl.addEventListener('input', () => {{ step = Number(stepEl.value); loadFrame(); }});
document.getElementById('prev').onclick = () => {{ step -= 1; loadFrame(); }};
document.getElementById('next').onclick = () => {{ step += 1; loadFrame(); }};
document.addEventListener('keydown', (ev) => {{
  if (ev.key === 'ArrowLeft') step -= 1;
  else if (ev.key === 'ArrowRight') step += 1;
  else if (ev.key === 'ArrowUp') {{ episode = Math.max(0, episode - 1); episodeEl.value = episode; step = 0; }}
  else if (ev.key === 'ArrowDown') {{ episode = Math.min(metadata.episodes.length - 1, episode + 1); episodeEl.value = episode; step = 0; }}
  else return;
  ev.preventDefault(); loadFrame();
}});
loadFrame();
</script>
</body>
</html>"""


def serve_replay_browser(path: Path, metadata: dict[str, Any], host: str, port: int, open_browser: bool) -> str:
    """Serve the replay browser until interrupted and return the served URL."""
    html = replay_browser_html(metadata).encode("utf-8")

    class ReplayBrowserHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug("replay browser: " + format, *args)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", html)
                    return
                if parsed.path == "/metadata.json":
                    self._send(200, "application/json", json.dumps(metadata).encode("utf-8"))
                    return
                if parsed.path == "/frame.png":
                    query = parse_qs(parsed.query)
                    episode = int(query.get("episode", ["0"])[0])
                    step = int(query.get("step", ["0"])[0])
                    self._send(200, "image/png", render_frame_png(path, episode, step))
                    return
            except Exception as exc:
                self._send(500, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
                return
            self.send_response(404)
            self.end_headers()

        def _send(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((host, port), ReplayBrowserHandler)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
    click.echo(f"Serving replay browser at {url}")
    click.echo("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopping replay browser.")
    finally:
        server.server_close()
    return url
