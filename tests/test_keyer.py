"""Test della chiave cromatica."""

import numpy as np

from greenframe.config import KeyParams
from greenframe.keyer import detection_mask, despill, screen_plate, soft_alpha, green_difference

GREEN = (70, 175, 45)      # BGR
SKIN = (125, 160, 210)
WHITE = (235, 235, 235)


def test_detection_mask_classes():
    img = np.zeros((10, 30, 3), np.uint8)
    img[:, :10] = GREEN
    img[:, 10:20] = SKIN
    img[:, 20:] = WHITE
    m = detection_mask(img, KeyParams())
    assert m[:, :10].all() and not m[:, 10:].any()


def test_detection_mask_works_in_shadow():
    """Il verde in ombra (tutti i canali dimezzati) resta verde."""
    img = np.zeros((4, 4, 3), np.uint8)
    img[:] = (35, 88, 22)
    assert detection_mask(img, KeyParams()).all()


def test_soft_alpha_mixed_pixel():
    """Un pixel per metà dito e metà pannello ha alpha intermedio."""
    kp = KeyParams()
    mixed = (np.array(GREEN, float) + np.array(SKIN, float)) / 2
    img = np.array([[GREEN, SKIN, mixed]], np.float32)
    d, _ = green_difference(img)
    plate = np.array([[GREEN] * 3], np.float32)
    dp, _ = green_difference(plate)
    a = soft_alpha(d, dp, kp)
    assert a[0, 0] == 1.0 and a[0, 1] == 0.0 and 0.1 < a[0, 2] < 0.9


def test_screen_plate_fills_holes():
    """Sotto una "mano" il pannello pulito viene ricostruito dai vicini."""
    img = np.zeros((120, 160, 3), np.float32)
    img[:] = GREEN
    img[40:80, 60:100] = SKIN
    clean = np.ones((120, 160), np.uint8)
    clean[40:80, 60:100] = 0
    plate = screen_plate(img, clean)
    assert np.abs(plate[60, 80] - np.array(GREEN)).max() < 3


def test_despill_removes_green_cast():
    img = np.array([[[120, 200, 180]]], np.float32)   # pelle con riflesso verde
    out = despill(img, 1.0)
    assert out[0, 0, 1] == 180   # G limitato a max(R, B)
