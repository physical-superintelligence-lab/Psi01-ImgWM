"""How the Image-WM generates: its negative prompt and the two measured sampler presets (the captions it
is prompted with are built in ``cosmos_framework/psix_wm/caption.py``).

Validation, offline evaluation and serving all use these, so their numbers describe one sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed objects, warped robot hand, artifacts, noise, oversaturated"


@dataclass(frozen=True)
class Sampler:
    num_steps: int
    guidance: float
    shift: float


# UniPC. 20 steps is the quality reference; the fast preset keeps quality at 8 steps with a lower shift and guidance.
PRESETS = {
    "quality": Sampler(num_steps=20, guidance=6.0, shift=10.0),
    "fast": Sampler(num_steps=8, guidance=3.0, shift=3.0),
}
