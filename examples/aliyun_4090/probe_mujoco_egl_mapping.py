#!/usr/bin/env python3
"""Hold a minimal MuJoCo EGL renderer active for external GPU mapping evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold-seconds", type=float, default=12.0)
    args = parser.parse_args()
    if args.hold_seconds <= 0:
        raise SystemExit("--hold-seconds must be positive")

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <light pos="0 0 2"/>
            <geom type="sphere" size="0.1" rgba="0.2 0.7 0.3 1"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        pixels = renderer.render()
        print(
            json.dumps(
                {
                    "event": "egl_ready",
                    "pid": os.getpid(),
                    "python": platform.python_version(),
                    "mujoco": mujoco.__version__,
                    "mujoco_gl": os.environ.get("MUJOCO_GL"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                    "frame_shape": list(pixels.shape),
                    "frame_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        deadline = time.monotonic() + args.hold_seconds
        frames = 1
        while time.monotonic() < deadline:
            mujoco.mj_step(model, data)
            renderer.update_scene(data)
            renderer.render()
            frames += 1
        print(json.dumps({"event": "egl_closed", "frames": frames}), flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
