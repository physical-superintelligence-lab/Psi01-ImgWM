"""Pixel scores of an Image-WM prediction, shared by in-loop validation and offline evaluation.

Every frame is resized (LANCZOS) to ``SCORE_SIZE`` and rounded to uint8 before scoring. MSE is the
RGB mean squared error on [0, 1]; PSNR is on [0, 255], 99 for an exact match.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image

SCORE_SIZE = (512, 384)
LAST_FRAME_METRICS = ("mse_last", "psnr_last")  # of the last generated frame; what in-loop validation logs
METRICS = (*LAST_FRAME_METRICS, "psnr_mean")  # what frame_metrics returns


def score_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.resize(SCORE_SIZE, Image.LANCZOS))


def mse01(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) / 255.0 - b.astype(np.float64) / 255.0) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def frame_metrics(predictions: Sequence[np.ndarray], ground_truth: Sequence[np.ndarray]) -> dict:
    """Scores of predicted frames 1..4 against the real ones (both already at ``SCORE_SIZE``)."""

    if not predictions or len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must be equally long and non-empty")
    mses = [mse01(prediction, truth) for prediction, truth in zip(predictions, ground_truth)]
    psnrs = [psnr(prediction, truth) for prediction, truth in zip(predictions, ground_truth)]
    return dict(zip(METRICS, (round(mses[-1], 5), round(psnrs[-1], 2), round(float(np.mean(psnrs)), 2))))
