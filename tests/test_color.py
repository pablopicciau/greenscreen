"""Test delle conversioni di colore: luce lineare, YUV 4:2:0, posizione della crominanza."""

import numpy as np
import pytest
import torch
from scipy.special import erf

from greenframe.color import (LINEAR_LUT, YuvFormat, chroma_siting, from_linear, from_linear_np,
                              rgb_to_yuv420, to_linear, to_linear_np, yuv420_to_rgb)


def test_transfer_roundtrip():
    v = np.linspace(-0.2, 1.2, 1001, dtype=np.float32)
    assert np.allclose(from_linear_np(to_linear_np(v)), v, atol=1e-5)
    t = torch.from_numpy(v)
    assert torch.allclose(from_linear(to_linear(t)), t, atol=1e-5)
    assert LINEAR_LUT[0] == 0 and abs(LINEAR_LUT[255] - 1) < 1e-6
    assert np.all(np.diff(LINEAR_LUT) > 0)


def test_chroma_siting_rotation():
    """FFmpeg traspone i piani quando ruota il video: la crominanza cambia posizione."""
    assert chroma_siting("left") == (0.0, 0.5)
    assert chroma_siting(None) == (0.0, 0.5)              # default H.264/HEVC
    assert chroma_siting("left", 90) == (0.5, 0.0)        # "top"
    assert chroma_siting("left", 270) == (0.5, 1.0)       # "bottom"
    assert chroma_siting("left", 180) == (1.0, 0.5)
    assert chroma_siting("center", 90) == (0.5, 0.5)


def test_yuv_roundtrip_smooth_image():
    yy, xx = np.mgrid[0:64, 0:96].astype(np.float32)
    rgb = np.stack([0.2 + 0.6 * xx / 96, 0.5 + 0.3 * np.sin(yy / 10), 0.3 + 0.4 * yy / 64])
    t = torch.from_numpy(rgb)
    y, u, v = rgb_to_yuv420(t, YuvFormat())
    back = yuv420_to_rgb(y, u, v, YuvFormat()).numpy()
    assert np.abs(back - rgb)[:, 2:-2, 2:-2].max() < 3 / 255


def _edge_error(bg, x_edge, detail=True):
    green = to_linear_np(np.array([45, 175, 70]) / 255.0)
    back = to_linear_np(np.array(bg) / 255.0)
    xs = np.arange(96)[None, :].repeat(48, 0)
    cov = 0.5 * (1 + erf((xs - x_edge) / (np.sqrt(2) * np.sqrt(0.7 ** 2 + 1 / 12))))
    lin = back[:, None, None] * (1 - cov[None]) + green[:, None, None] * cov[None]
    y, u, v = rgb_to_yuv420(torch.from_numpy(from_linear_np(lin)), YuvFormat())
    rgb = yuv420_to_rgb(y, u, v, YuvFormat(), detail=detail).numpy()
    prof = to_linear_np(rgb[:, 24, :]).T                         # (96, 3)
    c_in, c_out = prof[-6:].mean(0), prof[:6].mean(0)
    d = c_in - c_out
    p = (prof - c_out) @ d / (d @ d)
    k = np.nonzero(p >= 0.5)[0][0]
    x50 = k - 1 + (0.5 - p[k - 1]) / (p[k] - p[k - 1])
    return x50 - x_edge


@pytest.mark.parametrize("detail", [True, False])
def test_edge_position_is_unbiased(detail):
    """Un bordo verde/sfondo (miscelato in luce lineare) torna nella posizione giusta.

    È il test che garantisce la precisione delle linee: crominanza 4:2:0 nella
    posizione corretta + misura in luce lineare. La vecchia catena (conversione
    rapida di FFmpeg + soglia sui valori gamma) sbagliava di 0.3-0.6 px.
    """
    errs = [_edge_error(bg, x, detail) for bg in [(200, 170, 150), (40, 30, 35), (150, 60, 60)]
            for x in np.arange(40.0, 42.01, 0.25)]
    errs = np.array(errs)
    assert np.abs(errs).max() < 0.25
    assert abs(errs.mean()) < 0.06
