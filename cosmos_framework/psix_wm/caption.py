"""The JSON caption of a five-frame Image-WM clip (schema ``psix_wm_caption``).

A clip is a GOAL query (it ends when the current subtask is complete) or a FUTURE query (it spans
a horizon in seconds). Its caption is a JSON object serialized by ``canonical``; those bytes are
what the model is trained and served on, so every sentence here is fixed.

Text forms: a goal caption carries the subtask, with the task in ``context`` (``both``) or
without it (``subtask``). A future caption names the task (``task``), the subtask in the task's
place (``subtask``), or the task with the subtask in ``context`` (``both``).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA = "psix_wm_caption"
NUM_FRAMES = 5
GOAL_MROPE_FPS = 0.25
# A future's horizon falls in one of three bins, each named by its nominal horizon. The caption states
# the actual seconds; the mRoPE rate is the bin's (``future_mrope_fps`` of the nominal horizon).
HORIZON_BINS = {0.8: (0.8, 1.2), 1.6: (1.2, 2.4), 3.2: (2.4, 4.0)}
HORIZON_STEP = 0.1  # a caption states its horizon with one decimal, so a drawn horizon is a multiple of it
FUTURE_HORIZONS = tuple(HORIZON_BINS)
DEPLOYMENT_SECONDS = 1.6  # the future horizon the robot asks for; serving and goal tracks default to it
MIN_HORIZON, MAX_HORIZON = HORIZON_BINS[FUTURE_HORIZONS[0]][0], HORIZON_BINS[FUTURE_HORIZONS[-1]][1]
ROBOT_TYPES = ("g1", "h1", "human")
CAMERAS = ("zed_mini", "zed_mini_neck", "realsense", "egocentric_cam", "aria", "vision_pro")
CONTROLS = ("teleop", "natural")
_SUBTASK_CONTEXT = "Current subtask: "

_ACTORS = {
    "g1": ("A G1 humanoid robot performing manipulation, first-person egocentric view.", "robot"),
    "h1": ("An H1 humanoid robot performing manipulation, first-person egocentric view.", "robot"),
    "human": ("A human performing manipulation, first-person egocentric view.", "person"),
}
_FIXED_HEAD = {
    "camera_motion": "Egocentric head-mounted camera, minimal motion.",
    "camera_angle": "First-person, angled downward at the workspace.",
    "framing": "Close-range first-person view.",
}
# A camera on an actuated neck pans and tilts on its own, independently of the hands.
_MOVING_NECK = {
    "camera_motion": (
        "Egocentric camera on an actuated neck: the head pans and tilts to follow the "
        "workspace, so the view can swing widely and independently of the hands."
    ),
    "camera_angle": "First-person, angled downward at the workspace; the angle shifts as the neck turns.",
    "framing": "Close-range first-person view, reframed by head motion.",
}


@dataclass(frozen=True)
class Shot:
    """Everything in a caption besides its text: who is filmed, by which camera, on what canvas."""

    robot_type: str
    camera: str
    control: str
    canvas: tuple[int, int]
    """(W, H) the model receives."""

    def __post_init__(self) -> None:
        if self.robot_type not in ROBOT_TYPES:
            raise ValueError(f"robot_type {self.robot_type!r} not in {ROBOT_TYPES}")
        if self.camera not in CAMERAS:
            raise ValueError(f"camera {self.camera!r} not in {CAMERAS}")
        if self.control not in CONTROLS:
            raise ValueError(f"control {self.control!r} not in {CONTROLS}")
        width, height = self.canvas
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError(f"canvas must be positive, got {self.canvas!r}")

    def _common(self) -> dict[str, Any]:
        return {
            "cinematography": dict(_MOVING_NECK if self.camera.endswith("_neck") else _FIXED_HEAD),
            "resolution": {"W": int(self.canvas[0]), "H": int(self.canvas[1])},
            "meta": {"embodiment": self.robot_type, "camera": self.camera, "control": self.control},
        }


def canonical(caption: Mapping[str, Any]) -> str:
    """The caption's bytes. Default separators on purpose: compact ones change the tokens."""

    return json.dumps(dict(caption), sort_keys=True)


def frame_indexes(current: int, target: int) -> tuple[int, int, int, int, int]:
    """Five frames from ``current`` to ``target``, evenly spaced."""

    if any(isinstance(v, bool) or not isinstance(v, int) for v in (current, target)):
        raise TypeError("current and target frame indexes must be integers")
    if current < 0 or target - current < NUM_FRAMES - 1:
        raise ValueError(f"target frame {target} must be at least four frames after current frame {current}")
    indexes = [int(round(current + step * (target - current) / (NUM_FRAMES - 1))) for step in range(NUM_FRAMES)]
    for step in range(1, NUM_FRAMES):
        indexes[step] = max(indexes[step], indexes[step - 1] + 1)
    return tuple(indexes)  # type: ignore[return-value]


def _low(text: str) -> str:
    """Lower the first character and drop trailing full stops."""

    text = str(text)
    return (text[0].lower() + text[1:]).rstrip(".") if text else text


def future_mrope_fps(horizon: float) -> float:
    """The mRoPE rate of a future clip at a nominal ``horizon``: its five frames span the horizon."""

    return (NUM_FRAMES - 1) / horizon


def goal_caption(shot: Shot, task: str, subtask: str) -> dict[str, Any]:
    """The clip ends when ``subtask`` is complete, with the task as context (form ``both``)."""

    description, actor = _ACTORS[shot.robot_type]
    caption = {
        "subjects": [
            {
                "description": description,
                "action": f"The {actor} completes the current subtask: {_low(subtask)}.",
                "state_changes": "The final frame shows the subtask completed.",
            }
        ],
        "actions": [
            {"time": "completion", "description": f"{str(subtask).rstrip('.')}, carried through until complete."}
        ],
        "temporal_caption": (
            f"The clip ends at the completion of the subtask '{_low(subtask)}'; the final frame shows it completed."
        ),
        "prediction": {"type": "subgoal"},
        "context": f"Task: {_low(task)}.",
        **shot._common(),
    }
    return caption


def future_caption(shot: Shot, horizon: float, text: str, *, subtask: str | None = None) -> dict[str, Any]:
    """The clip spans ``horizon`` seconds of ``text`` (the task or the subtask); ``subtask`` adds it as context."""

    horizon = float(horizon)
    if not MIN_HORIZON <= horizon <= MAX_HORIZON:
        raise ValueError(f"future horizon {horizon!r} outside [{MIN_HORIZON}, {MAX_HORIZON}] s")
    description, actor = _ACTORS[shot.robot_type]
    caption = {
        "subjects": [
            {
                "description": description,
                "action": f"The {actor} continues performing the task: {_low(text)}.",
                "state_changes": f"The scene evolves according to the {actor}'s ongoing motion.",
            }
        ],
        "actions": [{"time": f"0:00-0:0{horizon:.1f}", "description": f"Continuing the task: {_low(text)}."}],
        "temporal_caption": (
            f"In {horizon:.1f} seconds after the current frame, show how the scene has evolved "
            f"as the {actor} continues the task: {_low(text)}."
        ),
        "prediction": {"type": "future", "horizon_seconds": horizon},
        "duration_seconds": horizon,
        **shot._common(),
    }
    if subtask:
        caption["context"] = f"{_SUBTASK_CONTEXT}{_low(subtask)}."
    return caption


# ------------------------------------------------------------------ rewrites of a built caption
# They change only the text sentences, so the caption's horizon is kept. The forms of each kind of clip,
# in the order a draw picks them.
TEXT_FORMS = {"goal": ("both", "subtask"), "future": ("both", "subtask", "task")}
_FUTURE_ACTION = re.compile(r"^The (?P<actor>.+?) continues performing the task: ")


def narrow(caption: Mapping[str, Any], form: str) -> dict[str, Any]:
    """A caption built with its context (form ``both``) in form ``subtask``, or for a future, ``task``.

    A goal in form ``subtask`` and a future in form ``task`` lose the context; a future in form
    ``subtask`` names the subtask where it named the task.
    """

    decoded = copy.deepcopy(dict(caption))
    context = decoded.pop("context")
    if decoded["prediction"]["type"] == "subgoal" or form == "task":
        return decoded
    if not context.startswith(_SUBTASK_CONTEXT):
        raise ValueError(f"not a future caption with a subtask context: {context!r}")
    subtask = context[len(_SUBTASK_CONTEXT) :].rstrip(".")
    actor = _FUTURE_ACTION.match(decoded["subjects"][0]["action"]).group("actor")
    decoded["subjects"][0]["action"] = f"The {actor} continues performing the task: {subtask}."
    decoded["actions"][0]["description"] = f"Continuing the task: {subtask}."
    sentence = decoded["temporal_caption"]
    tail = f" as the {actor} continues"  # the sentence's fixed start never contains it
    decoded["temporal_caption"] = sentence[: sentence.index(tail)] + f"{tail} the task: {subtask}."
    return decoded


def horizon_bin(seconds: float) -> float:
    """The nominal horizon whose bin holds ``seconds`` (clamped to the bins' range)."""

    return next((horizon for horizon, (_, high) in HORIZON_BINS.items() if seconds <= high + 1e-9), FUTURE_HORIZONS[-1])


# ------------------------------------------------------------------ prompts at inference
def future_prompt(shot: Shot, task: str, seconds: float) -> tuple[str, float]:
    """The caption and mRoPE rate asking for the frame ``seconds`` ahead, as training builds them: the
    caption states ``seconds``, the rate is that of the bin holding ``seconds``."""

    return canonical(future_caption(shot, seconds, task)), future_mrope_fps(horizon_bin(seconds))


def goal_prompt(shot: Shot, task: str, subtask: str) -> tuple[str, float]:
    """The caption and mRoPE rate asking for the frame where ``subtask`` is complete."""

    return canonical(goal_caption(shot, task, subtask)), GOAL_MROPE_FPS
