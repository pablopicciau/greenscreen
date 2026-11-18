"""Test della stima di nitidezza e tempo di esposizione dai bordi."""

import numpy as np
import pytest

from greenframe.blur_estimation import edge_iqr, estimate_blur, normal_speeds, render_blur_sigma


def test_edge_iqr_limits():
    # Solo gaussiana: larghezza 25-75% = 2 · 0.6745 · sigma.
    assert edge_iqr(1.0, np.array([0.0]))[0] == pytest.approx(1.349, abs=0.02)
    # Rettangolo dominante: metà della sua larghezza.
    assert edge_iqr(0.2, np.array([20.0]))[0] == pytest.approx(10.0, abs=0.1)
    w = edge_iqr(0.8, np.linspace(0, 30, 31))
    assert np.all(np.diff(w) > 0)


def _moving_track(T=120, speed=6.0):
    """Quad che trasla e ruota: velocità dei lati variabili e note."""
    t = np.arange(T)[:, None, None]
    base = np.array([[300.0, 300.0], [700.0, 310.0], [690.0, 800.0], [290.0, 790.0]])[None]
    shift = np.concatenate([speed * np.sin(t / 15.0) * 15.0, 0.5 * speed * np.cos(t / 11.0) * 11.0], axis=2)
    return base + shift


def test_estimate_blur_recovers_sigma_and_shutter():
    quads = _moving_track()
    rng = np.random.default_rng(0)
    sigma, shutter = 0.9, 0.45
    widths = np.full((len(quads), 4), np.nan)
    for t in range(1, len(quads) - 1):
        sp = normal_speeds(quads, t)
        widths[t] = edge_iqr(sigma, shutter * sp) + rng.normal(0, 0.05, 4)
    est = estimate_blur(widths, quads)
    assert est is not None
    s_hat, e_hat, info = est
    assert s_hat == pytest.approx(sigma, abs=0.05)
    assert e_hat == pytest.approx(shutter, abs=0.04)
    assert info["shutter_observable"]


def test_static_panel_shutter_defaults_to_180():
    quads = np.repeat(_moving_track(T=2)[:1], 60, axis=0)
    widths = np.full((60, 4), 1.4)
    est = estimate_blur(widths, quads)
    assert est is not None and est[1] == 0.5 and not est[2]["shutter_observable"]


def test_render_blur_sigma():
    assert render_blur_sigma(None) == 0.0
    assert render_blur_sigma(0.5) == 0.0                       # già più nitido della misura
    assert render_blur_sigma(1.1) == pytest.approx(0.70, abs=0.01)
