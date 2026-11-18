"""Test delle funzioni geometriche (proiezioni ideali, senza rumore)."""

import numpy as np
import pytest

from greenframe import geometry as geo
from make_synthetic_video import Scene

W, H = 1080, 1920
FOCAL = 0.6 * float(np.hypot(W, H))


def projected_quads(n=40, seed=3):
    """Quad veri di un pannello 40x50 cm che si muove in 3D."""
    scene = Scene(W, H, 6.0, 0.4, 0.5, FOCAL, set(), seed)
    out = []
    for t in np.linspace(0, 6, n):
        R, tr = scene.pose(t)
        q, _ = scene.project(scene.corners3d(), R, tr)
        out.append(q)
    return out


def test_lines_roundtrip():
    """Quad -> rette dei lati -> intersezioni restituisce gli stessi angoli."""
    q = np.array([[10.0, 20.0], [300.0, 35.0], [310.0, 400.0], [5.0, 380.0]])
    back = geo.quad_from_lines(geo.quad_lines(q))
    assert np.allclose(back, q, atol=1e-9)


def test_order_and_align():
    """L'ordinamento TL,TR,BR,BL è indipendente dall'ordine di ingresso."""
    q = np.array([[10.0, 20.0], [300.0, 35.0], [310.0, 400.0], [5.0, 380.0]])
    shuffled = q[[2, 0, 3, 1]]
    assert np.allclose(geo.order_quad(shuffled), q)
    # Rotazione ciclica: align_to_reference la annulla.
    assert np.allclose(geo.align_to_reference(np.roll(q, 1, axis=0), q), q)


def test_offset_quad_expands():
    q = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]])
    big = geo.offset_quad(q, 2.0)
    assert np.allclose(big, [[-2, -2], [102, -2], [102, 52], [-2, 52]])


def test_aspect_ratio_estimation_exact():
    """Il metodo di Zhang-He recupera il rapporto 0.8 da qualunque prospettiva."""
    for q in projected_quads():
        ratio, focal = geo.estimate_aspect_ratio(q, (W, H))
        assert ratio == pytest.approx(0.8, rel=1e-6)
        if focal is not None:
            assert focal == pytest.approx(FOCAL, rel=1e-6)


def test_complete_by_ratio_static():
    """Senza movimento, il lato mancante ricostruito coincide con quello vero."""
    for q in projected_quads(10):
        for missing in range(4):
            rec = geo.complete_by_ratio(geo.quad_lines(q), missing, q)
            assert np.abs(rec - q).max() < 1e-6


def test_complete_rectangle_exact():
    """Il metodo dei punti di fuga è esatto su dati ideali."""
    for q in projected_quads(10):
        for missing in range(4):
            rec = geo.complete_rectangle(geo.quad_lines(q), missing, FOCAL, 0.8, (W, H), q.mean(axis=0))
            assert rec is not None and np.abs(rec - q).max() < 1e-4


@pytest.mark.parametrize("text,expected", [("40x50", 0.8), ("4:5", 0.8), ("0.8", 0.8), ("16/9", 16 / 9), ("", None)])
def test_parse_aspect(text, expected):
    assert geo.parse_aspect(text) == (pytest.approx(expected) if expected else None)
