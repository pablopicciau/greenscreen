"""Test del rilevatore su fotogrammi sintetici (senza compressione video)."""

import numpy as np
import pytest

from greenframe import geometry as geo
from greenframe.config import DetectParams, KeyParams
from greenframe.detector import _fit_side_line, detect_quad
from make_synthetic_video import Scene, make_background, make_panel_texture, render_frame

W, H = 540, 960


def make_scene(scenarios):
    focal = 0.6 * float(np.hypot(W, H))
    scene = Scene(W, H, 6.0, 0.4, 0.5, focal, set(scenarios), seed=5)
    rng = np.random.default_rng(5)
    return scene, make_background(W, H, rng), make_panel_texture(400, 500, rng)


def frame_and_truth(scene, bg, tex, t):
    frame = np.clip(render_frame(scene, bg, tex, t, np.zeros(2)) + 0.5, 0, 255).astype(np.uint8)
    R, tr = scene.pose(t)
    truth, _ = scene.project(scene.corners3d(), R, tr)
    return frame, truth


@pytest.mark.parametrize("t", [0.3, 1.7, 4.1, 5.5])
def test_detect_with_hands(t):
    """Mani che reggono i lati: errore sugli angoli sotto 1.5 px."""
    scene, bg, tex = make_scene({"hold"})
    frame, truth = frame_and_truth(scene, bg, tex, t)
    det = detect_quad(frame, KeyParams(), DetectParams())
    assert det is not None
    err = np.linalg.norm(geo.align_to_reference(det.quad, truth) - truth, axis=1)
    assert err.max() < 1.5


def test_detect_with_occluded_corner():
    """Un braccio copre completamente l'angolo in alto a sinistra."""
    scene, bg, tex = make_scene({"hold", "corner"})
    t = 0.41 * scene.T        # a metà dello scenario "corner"
    frame, truth = frame_and_truth(scene, bg, tex, t)
    det = detect_quad(frame, KeyParams(), DetectParams())
    assert det is not None
    err = np.linalg.norm(geo.align_to_reference(det.quad, truth) - truth, axis=1)
    assert err.max() < 2.0


def test_no_panel_gives_reason():
    frame = np.full((200, 200, 3), 128, np.uint8)
    diag = {}
    assert detect_quad(frame, KeyParams(), DetectParams(), diag=diag) is None
    assert diag["reason"]


def test_side_line_prefers_contiguous_edge():
    """Un tratto di bordo continuo (30% del lato) vince su punti sparsi allineati.

    I punti sparsi simulano le creste di un bordo irregolare o i profili delle
    dita: sono tanti e allineati, ma non formano un tratto continuo.
    """
    t_edge = np.linspace(0.6, 0.9, 120)
    edge = np.stack([t_edge * 300, np.full_like(t_edge, 100.0)], axis=1)        # retta y = 100
    t_crest = np.repeat(np.arange(0.02, 0.98, 0.08), 6) + np.tile(np.linspace(0, 0.01, 6), 12)
    crests = np.stack([t_crest * 300, 80.0 + 0.05 * t_crest * 300], axis=1)       # retta inclinata
    pts = np.vstack([edge, crests])
    t = np.concatenate([t_edge, t_crest])
    line, coverage = _fit_side_line(pts, t, threshold=1.0)
    assert line is not None
    # La retta trovata è quella del bordo vero (y = 100) e la copertura è ~30%.
    assert np.abs(edge[:, 0] * line[0] + edge[:, 1] * line[1] + line[2]).max() < 0.5
    assert coverage == pytest.approx(0.3, abs=0.05)
