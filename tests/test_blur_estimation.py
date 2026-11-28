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


# ---------------------------------------------------------------------------
# Spessore della tela: lati accanto ai quali si vede la faccia laterale
# ---------------------------------------------------------------------------

def _project_panel(yaw_deg: float, center=(0.0, 0.0, 3.0), focal=1000.0, size=(1080, 1920),
                   width=1.0, height=1.2) -> np.ndarray:
    """Angoli (TL, TR, BR, BL) di un rettangolo ruotato attorno all'asse verticale."""
    a = np.radians(yaw_deg)
    u = np.array([np.cos(a), 0.0, -np.sin(a)])       # asse orizzontale del pannello
    v = np.array([0.0, 1.0, 0.0])                    # asse verticale (y verso il basso)
    c = np.array(center)
    pts3 = [c - width / 2 * u - height / 2 * v, c + width / 2 * u - height / 2 * v,
            c + width / 2 * u + height / 2 * v, c - width / 2 * u + height / 2 * v]
    cx, cy = (size[0] - 1) / 2, (size[1] - 1) / 2
    return np.array([[focal * p[0] / p[2] + cx, focal * p[1] / p[2] + cy] for p in pts3])


def test_side_face_visibility_frontal_and_turned():
    from greenframe.geometry import side_face_visibility

    # Frontale e centrato: nessuna faccia laterale visibile (si guarda il fronte).
    vis = side_face_visibility(_project_panel(0.0), 1000.0, (1080, 1920))
    assert np.all(vis < -0.1)
    # Ruotato di 40° con il lato destro verso la camera: si vede la faccia
    # laterale del lato destro (lato 1), non quella del sinistro (lato 3).
    vis = side_face_visibility(_project_panel(40.0), 1000.0, (1080, 1920))
    assert vis[1] > 0.3 and vis[3] < 0
    assert vis[0] < 0 and vis[2] < 0                 # alto e basso: nascosti
    # Rotazione opposta: si inverte.
    vis = side_face_visibility(_project_panel(-40.0), 1000.0, (1080, 1920))
    assert vis[3] > 0.3 and vis[1] < 0
    # Una focale sbagliata del 30% non cambia la decisione.
    vis = side_face_visibility(_project_panel(40.0), 1300.0, (1080, 1920))
    assert vis[1] > 0.2 and vis[3] < 0


def test_estimate_blur_ignores_sides_with_visible_thickness():
    """Il lato con lo spessore visibile dà larghezze falsate: va escluso."""
    size, focal = (1080, 1920), 1000.0
    yaws = 60.0 * np.cos(np.linspace(0, np.pi, 90)) ** 2          # rotazione verso il frontale
    quads = np.array([_project_panel(y, center=(0.0, 0.0, 3.0)) for y in yaws])
    rng = np.random.default_rng(1)
    sigma, shutter = 0.9, 0.5
    widths = np.full((len(quads), 4), np.nan)
    for t in range(1, len(quads) - 1):
        sp = normal_speeds(quads, t)
        widths[t] = edge_iqr(sigma, shutter * sp) + rng.normal(0, 0.03, 4)
        # Lato destro (vicino, faccia laterale visibile): misura falsata del 40%.
        widths[t, 1] *= 0.6
    biased = estimate_blur(widths, quads)
    fixed = estimate_blur(widths, quads, focal=focal, image_size=size)
    assert biased is not None and fixed is not None
    assert biased[1] < 0.4                         # senza geometria l'esposizione risulta corta
    assert fixed[1] == pytest.approx(shutter, abs=0.06)
    assert fixed[2]["excluded_side_faces"] > 0


def test_blur_shaped_rejects_strip_next_to_edge():
    """Profilo pulito (gaussiana ⊗ rettangolo) accettato; con un secondo gradino no."""
    from scipy.special import ndtr

    from greenframe.detector import _blur_shaped, _first_crossing

    x = np.arange(-18, 18.01, 0.5)                                 # posizioni in px
    clean = 1.0 - ndtr(x / 1.0)                                     # bordo sfocato, 1 dentro -> 0 fuori
    # Bordo con la striscia: lo stesso bordo seguito da una striscia (5 px) di
    # un colore che si proietta a -0.35, poi lo sfondo.
    strip = clean - 0.35 * (ndtr(x / 1.0) - ndtr((x - 5.0) / 1.0))
    p = np.stack([clean, strip])
    x25, _ = _first_crossing(p, 0.25)
    x50, _ = _first_crossing(p, 0.5)
    x75, _ = _first_crossing(p, 0.75)
    ok = _blur_shaped(p, x25, x50, x75)
    assert ok[0] and not ok[1]
