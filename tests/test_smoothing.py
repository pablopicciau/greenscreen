"""Test dello smoothing temporale."""

import numpy as np

from greenframe.smoothing import smooth_track
from greenframe.tracker import STATUS_LOST, STATUS_OK, Track


def make_track(T=60, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None, None]
    base = np.array([[100.0, 100.0], [300.0, 110.0], [310.0, 400.0], [90.0, 390.0]])[None]
    truth = base + np.concatenate([2.0 * t, 0.05 * t ** 2], axis=2) * np.ones((1, 4, 1))
    raw = truth + rng.normal(0, noise, truth.shape)
    tr = Track(width=1080, height=1920, fps=30.0, raw=raw, confidence=np.ones(T),
               status=[STATUS_OK] * T, panel_luma=np.full(T, 150.0), hidden_side=np.full(T, -1))
    return tr, truth


def test_smoothing_reduces_jitter_without_lag():
    tr, truth = make_track()
    sm = smooth_track(tr, half_window=2)
    raw_err = np.abs(tr.raw - truth).mean()
    sm_err = np.abs(sm - truth).mean()
    assert sm_err < 0.8 * raw_err
    # Nessun ritardo sistematico nonostante l'accelerazione.
    assert abs((sm - truth).mean()) < 0.1


def test_outlier_rejected_and_gap_filled():
    tr, truth = make_track(noise=0.1)
    tr.raw[20] += 25.0                         # misura sbagliata
    tr.raw[40:43] = np.nan                     # tre fotogrammi persi
    for i in range(40, 43):
        tr.status[i] = STATUS_LOST
        tr.confidence[i] = 0.0
    sm = smooth_track(tr, half_window=2)
    assert np.abs(sm[20] - truth[20]).max() < 1.5
    assert not np.isnan(sm[40:43]).any()
    assert np.abs(sm[41] - truth[41]).max() < 2.0


def test_consistency_check_fixes_impostor_side():
    """Un oggetto dritto davanti al lato inferiore "accorcia" il quad misurato.

    Il controllo di coerenza (rapporto d'aspetto implicato dal quad) deve
    riconoscere i fotogrammi sbagliati e, con l'interpolazione offline,
    riportare gli angoli vicino al vero.
    """
    from greenframe import geometry as geo
    from greenframe.smoothing import flag_inconsistent_sides
    from make_synthetic_video import Scene

    W, H = 1080, 1920
    focal = 0.6 * float(np.hypot(W, H))
    scene = Scene(W, H, 3.0, 0.4, 0.5, focal, set(), seed=4)
    truth = []
    for t in np.linspace(0, 3.0, 90):
        R, tr_ = scene.pose(t)
        truth.append(scene.project(scene.corners3d(), R, tr_)[0])
    truth = np.array(truth)
    raw = truth.copy()
    support = np.full((90, 4), 0.95)
    for i in range(40, 52):
        # Il bordo dell'oggetto sta all'85% dell'altezza: BR e BL "salgono" lungo i lati.
        raw[i, 2] = truth[i, 1] + 0.85 * (truth[i, 2] - truth[i, 1])
        raw[i, 3] = truth[i, 0] + 0.85 * (truth[i, 3] - truth[i, 0])
        support[i, 2] = 0.5
    track = Track(width=W, height=H, fps=30.0, raw=raw, confidence=np.full(90, 0.9),
                  status=[STATUS_OK] * 90, panel_luma=np.full(90, 150.0),
                  hidden_side=np.full(90, -1), support=support,
                  panel_aspect=0.8, focal=focal, focal_reliable=True)
    fixed = flag_inconsistent_sides(track)
    assert set(fixed) == set(range(40, 52))          # tutti e soli i fotogrammi sbagliati
    assert all(track.hidden_side[i] == 2 for i in fixed)
    sm = smooth_track(track, half_window=2)
    err = np.linalg.norm(sm - truth, axis=2)
    # Prima della correzione ~100 px; con interpolazione lineare ~25 px;
    # con l'interpolazione di Hermite ~1.6 px.
    assert err[40:52].max() < 3.0
    # I fotogrammi lontani dall'occlusione restano esatti (lo smoothing
    # sparge solo un poco l'errore sui vicini immediati).
    assert err[:36].max() < 0.5 and err[56:].max() < 0.5
    assert err.max() < 3.0
