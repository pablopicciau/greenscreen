"""Test dello scenario "reveal": persona di spalle che si gira e mostra il quadro.

All'inizio il pannello non è visibile: il programma non deve agganciare altro
(verdi dello sfondo) e, quando il quadro compare di taglio, deve coprirlo
subito (niente "lampo verde"). Vedi tools/make_reveal_video.py.
"""

import json

import numpy as np
import pytest

from greenframe.config import Config
from greenframe.tracker import (STATUS_LOST, STATUS_OK, Track, discard_weak_segments,
                                merge_backward, track_segments)


def _track(quads, strong, status=None):
    T = len(quads)
    return Track(width=540, height=960, fps=30.0, raw=np.array(quads, float),
                 confidence=np.where(strong, 0.9, 0.35), status=status or [STATUS_OK] * T,
                 panel_luma=np.full(T, 0.5), hidden_side=np.full(T, -1), support=np.ones((T, 4)),
                 edge_width=np.full((T, 4), np.nan), impostor=np.full((T, 4), np.nan),
                 strong=np.array(strong, bool), reasons=[""] * T)


SMALL = np.array([[0, 470], [70, 470], [55, 540], [0, 545]], float)     # zona verdastra dello sfondo
PANEL = np.array([[100, 200], [440, 210], [445, 760], [95, 770]], float)  # pannello vero


def test_weak_false_target_is_discarded_and_panel_kept():
    quads = [SMALL] * 20 + [PANEL + k for k in range(30)]
    strong = [False] * 20 + [True] * 30
    tr = _track(quads, strong)
    assert track_segments(tr) == [(0, 19), (20, 49)]
    dropped = discard_weak_segments(tr, min_strong=3)
    assert dropped == 20
    assert all(s == STATUS_LOST for s in tr.status[:20])
    assert all(s == STATUS_OK for s in tr.status[20:])


def test_backward_merge_stops_at_discontinuity():
    """Il tracking all'indietro si innesta solo finché resta continuo."""
    quads = [np.full((4, 2), np.nan)] * 10 + [PANEL] * 5
    status = [STATUS_LOST] * 10 + [STATUS_OK] * 5
    tr = _track(quads, [False] * 10 + [True] * 5, status)
    # Risultati all'indietro per i fotogrammi 9, 8, 7, 6: i primi due seguono il
    # pannello, poi un "salto" su un altro bersaglio (da non innestare).
    back = _track([PANEL - 2, PANEL - 4, SMALL, SMALL], [True, True, False, False])
    n = merge_backward(tr, back, [9, 8, 7, 6])
    assert n == 2
    assert tr.status[9] == STATUS_OK and tr.status[8] == STATUS_OK
    assert tr.status[7] == STATUS_LOST and np.isnan(tr.raw[7]).all()


@pytest.fixture(scope="module")
def reveal_clip(tmp_path_factory):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg non disponibile")
    import make_reveal_video

    d = tmp_path_factory.mktemp("reveal")
    video = d / "reveal.mp4"
    make_reveal_video.main(["--out", str(video), "--width", "360", "--height", "640",
                            "--seconds", "4", "--subframes", "1"])
    return video, json.loads(video.with_suffix(".gt.json").read_text())


def test_reveal_tracking(reveal_clip):
    """Reveal completo: nessun aggancio sul retro, quadro seguito fin da quando compare."""
    from greenframe import pipeline

    video, meta = reveal_clip
    track = pipeline.analyze(str(video), Config(), None, log=lambda m: None, show_progress=False)
    gt = meta["ground_truth"]
    phase = np.array([g["phase"] for g in gt])
    truth = np.array([g["corners"] for g in gt])
    valid = ~np.isnan(track.smooth).any(axis=(1, 2))
    first = int(np.argmax(phase != "retro"))
    # Sul retro (lontano dalla comparsa) nessun quadro: niente falsi inserimenti.
    assert not valid[:max(0, first - 2)].any()
    # Dove il quadro si vede, è sempre agganciato.
    assert valid[phase != "retro"].all()
    err = np.linalg.norm(track.smooth - truth, axis=2)
    assert np.nanmean(err[phase == "frontale"]) < 1.0
    assert np.nanmax(err[(phase == "taglio") | (phase == "rotazione")]) < 15.0


def test_order_quad_grazing_view_rising_top_edge():
    """Quadro di taglio con il lato superiore che sale verso destra: TL deve
    restare l'estremo sinistro del lato superiore (bug: slittamento di 1)."""
    from greenframe import geometry as geo

    tl, tr, br, bl = [348, 321], [391, 202], [392, 669], [350, 805]
    assert np.allclose(geo.order_quad(np.array([br, tl, bl, tr], float)), [tl, tr, br, bl])
