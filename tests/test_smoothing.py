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


def test_long_occlusion_follows_tilt():
    """Lato inferiore coperto per 1.5 s mentre il pannello si inclina.

    Stesso movimento del benchmark Full HD "standard". La sola interpolazione
    nel tempo sbaglia di ~40 px a metà occlusione (non sa che il pannello si
    inclina e poi torna); con la geometria del rettangolo, ricostruita in ogni
    fotogramma dai tre lati visibili e pesata con la sua incertezza, l'errore
    resta sotto i 2 px. Un fotogramma in cui il rilevamento indica il lato
    sbagliato viene ricostruito dai vicini.
    """
    from greenframe import geometry as geo
    from greenframe.tracker import STATUS_MODEL
    from make_synthetic_video import Scene

    W, H, fps = 1080, 1920, 30.0
    focal = 0.6 * float(np.hypot(W, H))
    scene = Scene(W, H, 15.0, 0.4, 0.5, focal, set(), seed=7)
    T = 450
    truth = np.array([scene.corners_at(i / fps) for i in range(T)])
    rng = np.random.default_rng(0)
    raw = truth + rng.normal(0, 0.1, truth.shape)          # misura sub-pixel
    hidden = np.full(T, -1)
    status = [STATUS_OK] * T
    support = np.full((T, 4), 0.95)
    occl = range(316, 360)
    for i in occl:
        # Come il tracking: lati visibili misurati, angoli in basso posti con
        # le proporzioni dell'ultimo fotogramma pulito (che non segue l'inclinazione).
        raw[i] = geo.complete_by_ratio(geo.quad_lines(raw[i]), 2, raw[315])
        hidden[i], status[i], support[i, 2] = 2, STATUS_MODEL, 0.0
    # Fotogramma con il lato sbagliato: il bordo dell'oggetto preso per il lato
    # inferiore e il lato destro "ricostruito".
    raw[330, 2] = truth[330, 1] + 0.9 * (truth[330, 2] - truth[330, 1])
    raw[330, 3] = truth[330, 0] + 0.9 * (truth[330, 3] - truth[330, 0])
    raw[330, 2, 0] += 25.0
    hidden[330] = 1

    def run(with_geometry: bool) -> np.ndarray:
        track = Track(width=W, height=H, fps=fps, raw=raw.copy(), confidence=np.full(T, 0.9),
                      status=list(status), panel_luma=np.full(T, 150.0), hidden_side=hidden.copy(),
                      support=support.copy(), panel_aspect=0.8,
                      focal=focal if with_geometry else None, focal_reliable=with_geometry)
        sm = smooth_track(track)
        return np.linalg.norm(sm - truth, axis=2)

    err_time_only = run(False)
    err = run(True)
    assert err_time_only[list(occl)].max() > 20.0
    assert err[list(occl)].max() < 2.0
    assert err[list(occl)].mean() < 0.6
    assert err[:300].max() < 0.5 and err[375:].max() < 0.5
