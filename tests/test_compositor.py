"""Test del compositing su un caso controllato."""

import cv2
import numpy as np

from greenframe.compositor import Compositor
from greenframe.config import Config

GREEN = (70, 175, 45)
SKIN = (125, 160, 210)


def test_replace_green_keep_hand():
    frame = np.full((300, 400, 3), 90, np.uint8)
    frame[50:250, 100:300] = GREEN                       # pannello
    cv2.circle(frame, (120, 150), 30, SKIN, -1)          # "mano" sul bordo sinistro
    quad = np.array([[99.5, 49.5], [299.5, 49.5], [299.5, 249.5], [99.5, 249.5]])
    red = np.zeros((500, 500, 3), np.uint8)
    red[:] = (0, 0, 255)
    cfg = Config()
    cfg.render.lighting = 0.0
    cfg.render.grain = 0.0
    cfg.render.mode = "stretch"
    out = Compositor(red, 1.0, cfg).render(frame, quad)
    # Pannello -> rosso.
    assert np.abs(out[150, 250].astype(int) - [0, 0, 255]).max() <= 2
    # La mano resta com'era.
    assert np.abs(out[150, 115].astype(int) - SKIN).max() <= 2
    # Lo sfondo lontano non viene toccato.
    assert (out[10, 10] == frame[10, 10]).all()
    # Nessun pixel verde residuo dentro il quad.
    roi = out[50:250, 100:300].astype(int)
    greenness = roi[..., 1] - np.maximum(roi[..., 0], roi[..., 2])
    assert greenness.max() < 20


def test_nan_quad_returns_original():
    frame = np.zeros((50, 50, 3), np.uint8)
    out = Compositor(np.zeros((10, 10, 3), np.uint8), 1.0, Config()).render(frame, np.full((4, 2), np.nan))
    assert out is frame
