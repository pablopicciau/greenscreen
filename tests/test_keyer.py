"""Test del riconoscimento del verde e del modello di colore del pannello."""

import numpy as np

from greenframe.color import from_linear_np, to_linear_np
from greenframe.keyer import PanelColor, generic_green_mask

GREEN = (70, 175, 45)      # BGR
SKIN = (125, 160, 210)
WHITE = (235, 235, 235)
PLANT = (30, 150, 80)      # verde saturo ma di tinta più gialla


def _shade(color_bgr, k):
    """Lo stesso colore con k volte la luce (in luce lineare, come un'ombra vera)."""
    lin = to_linear_np(np.array(color_bgr, np.float32) / 255.0) * k
    return np.clip(from_linear_np(lin) * 255 + 0.5, 0, 255).astype(np.uint8)


def test_generic_mask_classes():
    img = np.zeros((10, 30, 3), np.uint8)
    img[:, :10] = GREEN
    img[:, 10:20] = SKIN
    img[:, 20:] = WHITE
    m = generic_green_mask(img, 14, 0.16)
    assert m[:, :10].all() and not m[:, 10:].any()


def test_generic_mask_works_in_shadow():
    """Il verde in ombra resta verde (la purezza non dipende dalla luminosità)."""
    img = np.zeros((4, 4, 3), np.uint8)
    img[:] = _shade(GREEN, 0.35)
    assert generic_green_mask(img, 14, 0.16).all()


def test_panel_color_separates_other_greens():
    """Il modello del pannello rifiuta un verde saturo diverso (pianta) ma non le ombre."""
    rng = np.random.default_rng(0)
    panel = np.clip(np.array(GREEN) + rng.normal(0, 2.5, (2000, 3)), 0, 255).astype(np.uint8)
    model = PanelColor.fit(panel)
    assert model is not None
    probe = np.array([[GREEN, _shade(GREEN, 0.4), _shade(GREEN, 1.3), PLANT, SKIN, WHITE]], np.uint8)
    m = model.mask(probe)[0]
    assert list(m) == [1, 1, 1, 0, 0, 0]
    # La pianta sta ben oltre la soglia (in deviazioni standard del pannello).
    assert model.distance(np.array([PLANT], np.uint8))[0] > 1.5 * model.tau
    assert generic_green_mask(np.array([[PLANT]], np.uint8), 14, 0.16).all()   # il vecchio criterio no


def test_panel_color_roundtrip():
    rng = np.random.default_rng(1)
    panel = np.clip(np.array(GREEN) + rng.normal(0, 2, (500, 3)), 0, 255).astype(np.uint8)
    m1 = PanelColor.fit(panel)
    m2 = PanelColor.from_dict(m1.to_dict())
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert (m1.mask(img) == m2.mask(img)).all()
