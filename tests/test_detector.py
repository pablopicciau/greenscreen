"""Test del rilevatore su fotogrammi sintetici (senza compressione video)."""

import numpy as np
import pytest

from greenframe import geometry as geo
from greenframe.config import DetectParams, KeyParams
from greenframe.detector import _fit_side_line, detect_quad
from make_synthetic_video import Scene, make_background, make_panel_texture, render_frame, to_display_u8

W, H = 540, 960


def make_scene(scenarios):
    focal = 0.6 * float(np.hypot(W, H))
    scene = Scene(W, H, 6.0, 0.4, 0.5, focal, set(scenarios), seed=5)
    rng = np.random.default_rng(5)
    bg = make_background(W, H, rng, green_object="greenbg" in scenarios)
    return scene, bg, make_panel_texture(400, 500, rng)


def frame_and_truth(scene, bg, tex, t):
    frame = to_display_u8(render_frame(scene, bg, tex, t, np.zeros(2)))
    R, tr = scene.pose(t)
    truth, _ = scene.project(scene.corners3d(), R, tr)
    return frame, truth


@pytest.mark.parametrize("t", [0.3, 1.7, 4.1, 5.5])
def test_detect_with_hands(t):
    """Mani che reggono i lati: errore sugli angoli sotto 0.5 px (immagine senza compressione)."""
    scene, bg, tex = make_scene({"hold"})
    frame, truth = frame_and_truth(scene, bg, tex, t)
    det = detect_quad(frame, KeyParams(), DetectParams())
    assert det is not None
    err = np.linalg.norm(geo.align_to_reference(det.quad, truth) - truth, axis=1)
    assert err.max() < 0.5


@pytest.mark.parametrize("t", [0.5, 2.2, 3.9])
def test_detect_next_to_saturated_green(t):
    """Una pianta di verde saturo tocca il pannello: prima il pannello si fondeva con
    lei (errore di decine di pixel); ora i due verdi vengono separati per colore."""
    scene, bg, tex = make_scene({"hold", "greenbg"})
    frame, truth = frame_and_truth(scene, bg, tex, t)
    det = detect_quad(frame, KeyParams(), DetectParams())
    assert det is not None
    err = np.linalg.norm(geo.align_to_reference(det.quad, truth) - truth, axis=1)
    assert err.max() < 0.8


def test_straight_occluder_is_recognized():
    """Un oggetto con bordo dritto copre il lato inferiore: il suo bordo prosegue
    oltre gli angoli (giunzione a T) e il lato viene ricostruito, non misurato."""
    scene = Scene(W, H, 6.0, 0.4, 0.5, 0.6 * float(np.hypot(W, H)), {"hold", "straightedge"}, seed=5)
    rng = np.random.default_rng(5)
    bg, tex = make_background(W, H, rng), make_panel_texture(400, 500, rng)
    t_prev, t = 0.745 * scene.T, 0.75 * scene.T
    frame, truth = frame_and_truth(scene, bg, tex, t)
    prior = scene.corners_at(t_prev)
    # Riferimento per le proporzioni: l'ultimo fotogramma "pulito" (qui il precedente;
    # nel video le occlusioni lunghe vengono poi corrette offline per interpolazione).
    det = detect_quad(frame, KeyParams(), DetectParams(), prior=prior, reference=prior)
    assert det is not None
    assert det.impostor[2] >= DetectParams().impostor_threshold
    assert 2 in det.completed
    err = np.linalg.norm(det.quad - truth, axis=1)
    # Stima del singolo fotogramma (la traccia la rifinisce offline interpolando
    # tra prima e dopo l'occlusione); senza il test l'errore sarebbe ~10%
    # dell'altezza del pannello (>40 px).
    assert err.max() < 8.0


def test_detect_with_occluded_corner():
    """Un braccio copre completamente l'angolo in alto a sinistra."""
    scene, bg, tex = make_scene({"hold", "corner"})
    t = 0.41 * scene.T        # a metà dello scenario "corner"
    frame, truth = frame_and_truth(scene, bg, tex, t)
    det = detect_quad(frame, KeyParams(), DetectParams())
    assert det is not None
    err = np.linalg.norm(geo.align_to_reference(det.quad, truth) - truth, axis=1)
    assert err.max() < 1.0


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
