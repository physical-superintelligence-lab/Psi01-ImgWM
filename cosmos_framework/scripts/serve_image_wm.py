"""Serve an Image-WM checkpoint over ``POST /wm`` (``cosmos_framework/inference/psix_wm/server.py``).

The deployment configuration::

    python -m cosmos_framework.scripts.serve_image_wm --checkpoint-path <run> --host 0.0.0.0 --quantize fp8 --compile

``<run>`` is a run's or a released checkpoint's folder (its latest checkpoint) or one ``checkpoints/iter_*``.

``--compile`` captures CUDA graphs: unset ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments``.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from cosmos_framework.inference.psix_wm.sampler import PRESETS
from cosmos_framework.psix_wm.caption import DEPLOYMENT_SECONDS, Shot

# (H, W) each robot camera records at; the model's canvas follows from it, as in training.
_CAMERA_SHAPE = {"zed_mini": (384, 672), "realsense": (480, 640)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", help="default: config.yaml of the checkpoint's folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8016)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/serve_image_wm"))
    parser.add_argument("--robot", default="g1", choices=("g1", "h1"))
    parser.add_argument("--camera", default="zed_mini_neck", choices=("zed_mini", "zed_mini_neck", "realsense"))
    parser.add_argument("--preset", default="fast", choices=sorted(PRESETS))
    parser.add_argument(
        "--seconds", type=float, default=DEPLOYMENT_SECONDS, help="horizon of a future request that names none"
    )
    parser.add_argument("--quantize", default="none", choices=("none", "fp8"))
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    import uvicorn

    from cosmos_framework.inference.psix_wm.generator import Generator
    from cosmos_framework.inference.psix_wm.server import WmServer

    generator = Generator(
        args.checkpoint_path, args.config_file, args.output_dir, quantize=args.quantize, compile=args.compile
    )
    canvas = generator.canvas(_CAMERA_SHAPE[args.camera.removesuffix("_neck")])
    server = WmServer(
        generator,
        Shot(args.robot, args.camera, "teleop", canvas),
        sampler=PRESETS[args.preset],
        default_seconds=args.seconds,
    )
    # The port answers /health and /ready (false) while the model warms up.
    threading.Thread(target=server.startup, daemon=True).start()
    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
