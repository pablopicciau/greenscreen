"""Test della preparazione dell'immagine (proporzioni, EXIF)."""

import numpy as np
import pytest

from greenframe.image_fit import fit_to_aspect, load_image


def test_fill_crops_to_aspect():
    img = np.zeros((1000, 1000, 3), np.uint8)
    out = fit_to_aspect(img, 0.8, "fill")
    assert out.shape[:2] == (1000, 800)


def test_fit_adds_bands_without_distortion():
    img = np.full((1000, 1000, 3), 200, np.uint8)
    out = fit_to_aspect(img, 0.8, "fit", "white")
    assert out.shape[:2] == (1250, 1000)
    # L'immagine originale è intatta al centro, le bande sono bianche.
    assert (out[125:1125] == 200).all() and (out[:100] == 255).all()


def test_exif_orientation_is_applied(tmp_path):
    """Le foto del telefono hanno la rotazione nei metadati EXIF: va rispettata."""
    Image = pytest.importorskip("PIL.Image")
    arr = np.zeros((100, 200, 3), np.uint8)
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = 6                       # "ruota di 90° in visualizzazione"
    path = tmp_path / "foto.jpg"
    img.save(path, exif=exif.tobytes())
    assert load_image(str(path)).shape[:2] == (200, 100)
