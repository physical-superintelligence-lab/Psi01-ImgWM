"""The Image-WM ``/wm`` service: a camera frame and an instruction in, the predicted frame out.

``POST /wm``::

    {"transport": "jpeg", "ego_jpeg": <base64 JPEG>, "task": str, "subtask": str,
     "mode": "future" | "subgoal", "seconds": float, "seed": int,
     "robot_episode_session_id": str, "req_id": int, "prompt_gen": int}

* ``future`` (default): the frame ``seconds`` ahead (0.8-4.0, clamped) as the robot continues ``task``;
* ``subgoal``: the frame when ``subtask`` is complete.

The answer carries the frame as ``subgoal_jpeg``. A ``req_id`` already answered in the same session
replays its response (``req_id`` -1 never does; the last four answers are kept), and with other inputs it is
refused (409); one request runs at a time (429 otherwise).
``GET /health``, ``/ready`` and ``/state``; ``POST /reset`` forgets the replays.
"""

from __future__ import annotations

import base64
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image

from cosmos_framework.inference.psix_wm.gate import WmRequestGate, request_digest
from cosmos_framework.inference.psix_wm.generator import Generator
from cosmos_framework.inference.psix_wm.sampler import Sampler
from cosmos_framework.psix_wm.caption import HORIZON_BINS, MAX_HORIZON, MIN_HORIZON, Shot, future_prompt, goal_prompt
from cosmos_framework.utils import log

MAX_TEXT_LENGTH = 300
MAX_IMAGE_SIDE = 4096
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
JPEG_QUALITY = 90
_REQUIRED = ("transport", "ego_jpeg", "task", "robot_episode_session_id", "req_id", "prompt_gen")


def prompt_for(shot: Shot, payload: dict[str, Any], default_seconds: float) -> tuple[str, float, float | None]:
    """(caption, mRoPE rate, seconds ahead or None for a goal) of a valid request."""

    task = payload["task"].strip()
    if payload.get("mode", "future") == "subgoal":
        return *goal_prompt(shot, task, payload["subtask"].strip()), None
    seconds = round(min(max(float(payload.get("seconds", default_seconds)), MIN_HORIZON), MAX_HORIZON), 1)
    return *future_prompt(shot, task, seconds), seconds


class WmServer:
    def __init__(self, generator: Generator, shot: Shot, *, sampler: Sampler, default_seconds: float) -> None:
        if not MIN_HORIZON <= default_seconds <= MAX_HORIZON:
            raise ValueError(f"default seconds must be in [{MIN_HORIZON}, {MAX_HORIZON}]")
        self.generator, self.shot, self.sampler, self.default_seconds = generator, shot, sampler, default_seconds
        self._gate = WmRequestGate()
        self._ready = False
        self._started = time.time()
        self.gen_count = 0
        self.last_error: str | None = None
        # Every model call runs on this one thread: compiled CUDA graphs are thread-local.
        self._model_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wm-model")
        self.app = FastAPI()
        self._routes()

    def _on_model_thread(self, fn, *args, **kwargs):
        return self._model_thread.submit(fn, *args, **kwargs).result()

    def startup(self) -> None:
        """Warm both modes at the longest accepted text, so no later request recompiles, then report ready."""

        text = ("pick up the towel and fold it " * MAX_TEXT_LENGTH)[:MAX_TEXT_LENGTH]
        frame = Image.new("RGB", self.shot.canvas, (32, 32, 32))
        try:
            for mode in ("future", "subgoal"):
                caption, mrope_fps, _ = prompt_for(
                    self.shot, {"task": text, "subtask": text, "mode": mode}, self.default_seconds
                )
                self._on_model_thread(
                    self.generator.generate,
                    [(frame, caption, mrope_fps, 0)],
                    canvas=self.shot.canvas,
                    sampler=self.sampler,
                )
        except Exception as error:
            self.last_error = f"warmup: {type(error).__name__}: {error}"
            log.exception(self.last_error)
            raise
        self._ready = True

    @staticmethod
    def _error(status: int, message: str, payload: dict[str, Any] | None = None) -> JSONResponse:
        body = {
            "error": message,
            **{key: payload[key] for key in ("req_id", "prompt_gen") if payload and key in payload},
        }
        return JSONResponse(status_code=status, content=body)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        missing = [field for field in _REQUIRED if field not in payload]
        if missing:
            raise ValueError(f"missing fields {missing}")
        if payload["transport"] != "jpeg":
            raise ValueError("transport must be 'jpeg'")
        for field in ("task", "robot_episode_session_id"):
            if (
                not isinstance(payload[field], str)
                or not payload[field].strip()
                or len(payload[field]) > MAX_TEXT_LENGTH
            ):
                raise ValueError(f"{field} must be a non-empty string of at most {MAX_TEXT_LENGTH} characters")
        mode, subtask = payload.get("mode", "future"), payload.get("subtask", "")
        if mode not in ("future", "subgoal"):
            raise ValueError("mode must be 'future' or 'subgoal'")
        if (
            not isinstance(subtask, str)
            or len(subtask) > MAX_TEXT_LENGTH
            or (mode == "subgoal" and not subtask.strip())
        ):
            raise ValueError(
                f"subtask must be a string of at most {MAX_TEXT_LENGTH} characters, non-empty in subgoal mode"
            )
        seconds = payload.get("seconds", MIN_HORIZON)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError("seconds must be a number")
        for field in ("req_id", "prompt_gen", "seed"):
            value = payload.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < (-1 if field == "req_id" else 0):
                raise ValueError(f"{field} must be a non-negative int" + (" or -1" if field == "req_id" else ""))
        if payload.get("seed", 0) >= 2**32:
            raise ValueError("seed must be below 2**32")

    @staticmethod
    def _decode(value: Any) -> Image.Image:
        image = Image.open(io.BytesIO(base64.b64decode(value, validate=True)), formats=["JPEG"])
        if min(image.size) < 8 or max(image.size) > MAX_IMAGE_SIDE:
            raise ValueError(f"image size {image.size} out of bounds")
        return image.convert("RGB")

    def predict(self, payload: dict[str, Any]) -> JSONResponse:
        if not self._ready:
            return self._error(503, "not_ready", payload)
        try:
            self._validate(payload)
            seed = payload.get("seed", 0)
            caption, mrope_fps, seconds = prompt_for(self.shot, payload, self.default_seconds)
            key = self._gate.key_of(payload)
            digest = request_digest(
                {
                    "caption": caption,
                    "mrope_fps": mrope_fps,
                    "seed": seed,
                    "ego_jpeg": payload["ego_jpeg"],
                    "prompt_gen": payload["prompt_gen"],
                }
            )
        except Exception as error:
            return self._error(400, str(error), payload)
        admission = self._gate.try_begin(key, digest)
        if admission.kind != "run":
            if admission.kind == "cached":
                return JSONResponse(content=admission.cached)
            return self._error(429 if admission.kind == "busy" else 409, admission.kind, payload)

        start = time.perf_counter()
        try:
            frame = self._decode(payload["ego_jpeg"])
        except Exception as error:
            self._gate.abort()
            return self._error(400, f"ego_jpeg: {error}", payload)
        try:
            frames = self._on_model_thread(
                self.generator.generate,
                [(frame, caption, mrope_fps, seed)],
                canvas=self.shot.canvas,
                sampler=self.sampler,
            )
            inference_ms = (time.perf_counter() - start) * 1000
            buffer = io.BytesIO()
            frames[0][-1].save(buffer, format="JPEG", quality=JPEG_QUALITY)
            self.gen_count += 1  # one request runs at a time
            response = {
                "subgoal_jpeg": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "task": payload["task"],
                "subtask": payload.get("subtask"),
                "mode": payload.get("mode", "future"),
                "horizon_seconds": seconds,
                "mrope_fps": mrope_fps,
                "instruction_used": caption,
                "robot_episode_session_id": payload["robot_episode_session_id"],
                "req_id": payload["req_id"],
                "prompt_gen": payload["prompt_gen"],
                "gen_id": self.gen_count,
                "seed": seed,
                "inference_time_ms": round(inference_ms, 2),
                "total_time_ms": round((time.perf_counter() - start) * 1000, 2),
            }
            answer = JSONResponse(content=response)  # serialized before it can be replayed
            self._gate.commit(key, digest, response)
            return answer
        except Exception as error:
            self._gate.abort()
            self.last_error = f"{type(error).__name__}: {error}"
            log.exception(f"/wm failed: {self.last_error}")
            return self._error(500, f"generation failed: {self.last_error}", payload)

    def state(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "gen_count": self.gen_count,
            "busy": self._gate.busy,
            "last_error": self.last_error,
            "shot": {
                "robot_type": self.shot.robot_type,
                "camera": self.shot.camera,
                "control": self.shot.control,
                "canvas": f"{self.shot.canvas[0]}x{self.shot.canvas[1]}",
            },
            "sampler": vars(self.sampler),
            "default_seconds": self.default_seconds,
            "horizon_bins": {str(k): list(v) for k, v in HORIZON_BINS.items()},
            "uptime_s": round(time.time() - self._started, 1),
        }

    def _routes(self) -> None:
        @self.app.post("/wm")
        async def wm(request: Request):
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_PAYLOAD_BYTES:
                    return self._error(413, f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if not isinstance(payload, dict):
                return self._error(400, "body is not a JSON object")
            return await run_in_threadpool(self.predict, payload)

        @self.app.get("/health")
        async def health():
            return {"status": "ok", "ready": self._ready, "last_error": self.last_error}

        @self.app.get("/ready")
        async def ready():
            return {"ready": self._ready}

        @self.app.get("/state")
        async def state():
            return self.state()

        @self.app.post("/reset")
        async def reset():
            self._gate.reset()
            return {"status": "reset"}
