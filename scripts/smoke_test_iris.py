"""Smoke test: run a short IRIS evaluation and print the /api/latents payload.

This script performs an in-process end-to-end check without starting the HTTP
server: it runs IRIS evaluation (render=True) for a single short episode,
installs the latents into the UI controller and calls the same `get_latents`
handler used by the `/api/latents` endpoint, then prints the returned payload.

Run from the repository root:
  python scripts/smoke_test_iris.py

Notes:
- The test tries to shorten the environment episode length to keep runtime low.
- Requires Atari/ALE envs installed and available in `world_models.ui.server.ATARI_ENVS`.
"""

import sys
import time
import base64
import asyncio
import threading
import os
import requests

from world_models.training.train_iris import IRISTrainer
from world_models.ui import server as ui_server


def main():
    # Select an Atari env if available
    env = None
    if getattr(ui_server, "ATARI_ENVS", None):
        env = ui_server.ATARI_ENVS[0]
    if env is None:
        env = "ALE/Pong-v5"

    print(f"Smoke test: IRIS eval on env={env} (device=cpu, 1 episode)")

    trainer = IRISTrainer(game=env, device="cpu")

    # Try to shorten episodes for a quick smoke run (best-effort)
    if hasattr(trainer.env, "spec") and trainer.env.spec is not None:
        setattr(trainer.env, "_max_episode_steps", 200)

    # Run a single rendered evaluation to get videos + latents
    episode_returns, videos, latents = trainer.evaluate(num_episodes=1, render=True)
    print("Evaluation finished. episode_returns=", episode_returns)

    # Place latents into the controller (mimic training flow)
    ui_server.controller.last_latents_ref[0] = latents

    # Call the same handler used by /api/latents and print its payload
    payload = ui_server.get_latents()
    latents_b64 = payload.get("latents")
    shape = payload.get("shape")
    print("/api/latents payload:")
    print("  shape:", shape)
    if isinstance(latents_b64, str):
        print("  base64 length:", len(latents_b64))
        raw = base64.b64decode(latents_b64)
        print("  raw bytes:", len(raw))

    # Also report if a preview gif was set in the controller
    frame_snapshot = ui_server.controller.snapshot_frame()
    print("Preview snapshot: gif present=", frame_snapshot.get("gif") is not None)

    # Now simulate the training path: set preview frame/gif into controller
    if videos and len(videos) > 0:
        eval_videos = videos[0]
        frame = ui_server.controller._extract_dreamer_preview_frame(eval_videos)
        frames_for_gif = ui_server.controller._extract_dreamer_preview_frames(
            eval_videos
        )
        if frame is not None:
            ui_server.controller._set_frame(frame)
        if frames_for_gif:
            ui_server.controller._set_gif(frames_for_gif)

    # Simulate an SSE subscriber and verify broadcast_update delivers an update.
    loop = None
    q = None
    thread = None

    try:
        loop = asyncio.get_running_loop()

        async def _make_q():
            return asyncio.Queue()

        q = asyncio.run_coroutine_threadsafe(_make_q(), loop).result(timeout=2)
        ui_server.SUBSCRIBERS.append((q, loop))
        ui_server.broadcast_update(
            ["state", "metrics", "frame", "latents"], extra={"test": True}
        )
        msg = asyncio.run_coroutine_threadsafe(q.get(), loop).result(timeout=5)
        print("Received SSE message:", msg)
        ui_server.SUBSCRIBERS.remove((q, loop))
    except RuntimeError:
        loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()

        async def _make_q_bg():
            return asyncio.Queue()

        q = asyncio.run_coroutine_threadsafe(_make_q_bg(), loop).result(timeout=2)
        ui_server.SUBSCRIBERS.append((q, loop))
        ui_server.broadcast_update(
            ["state", "metrics", "frame", "latents"], extra={"test": True}
        )
        msg = asyncio.run_coroutine_threadsafe(q.get(), loop).result(timeout=5)
        print("Received SSE message:", msg)
        ui_server.SUBSCRIBERS.remove((q, loop))
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)

    # POST to the running FastAPI server's /api/visualize and save HTML to a file.
    base_url = os.environ.get("TORCHWM_UI_BASE", "http://127.0.0.1:8000")
    if latents is not None and getattr(latents, "tobytes", None):
        latents_b64 = base64.b64encode(latents.tobytes()).decode("ascii")
        shape_param = ",".join(str(int(x)) for x in latents.shape)
        # Send as JSON body to avoid URL length limits which can corrupt base64
        body = {"latents": latents_b64, "shape": shape_param, "method": "tsne"}
        resp = requests.post(f"{base_url}/api/visualize", json=body, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            html = data.get("html")
            if html:
                out_path = "smoke_iris_visualize.html"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print("Wrote visualization HTML to", out_path)
        else:
            print("/api/visualize returned status", resp.status_code, resp.text)


if __name__ == "__main__":
    main()
