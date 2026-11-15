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
