"""Test del rendering (GPU se disponibile, sempre anche su CPU) su casi controllati."""

import cv2
import numpy as np
import pytest
import torch

from greenframe.color import LINEAR_LUT, YuvFormat, from_linear_np, rgb_to_yuv420, to_linear_np
from greenframe.config import Config
from greenframe.keyer import PanelColor
from greenframe.render import Renderer
from greenframe.tracker import STATUS_OK, Track
from greenframe.video_io import YuvFrame

GREEN = (70, 175, 45)      # BGR
SKIN = (125, 160, 210)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def to_yuv(bgr: np.ndarray) -> YuvFrame:
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1].transpose(2, 0, 1))).float() / 255
    y, u, v = rgb_to_yuv420(rgb, YuvFormat())
    return YuvFrame(y.numpy(), u.numpy(), v.numpy())


def make_track(quads, green_px):
    T = len(quads)
    lin = LINEAR_LUT[np.array(GREEN)]
    luma = 0.0722 * lin[0] + 0.7152 * lin[1] + 0.2126 * lin[2]
    return Track(width=400, height=300, fps=30.0, raw=np.array(quads), confidence=np.ones(T),
                 status=[STATUS_OK] * T, panel_luma=np.full(T, luma), smooth=np.array(quads),
                 panel_aspect=1.0, shutter=0.5, lens_sigma=None,
                 panel_color=PanelColor.fit(green_px).to_dict())


def base_frame():
    frame = np.full((300, 400, 3), 90, np.uint8)
    frame[50:250, 100:300] = GREEN                       # pannello
    cv2.circle(frame, (120, 150), 30, SKIN, -1)          # "mano" sul bordo sinistro
    quad = np.array([[99.5, 49.5], [299.5, 49.5], [299.5, 249.5], [99.5, 249.5]])
    return frame, quad


def config():
    cfg = Config()
    cfg.render.lighting = 0.0
    cfg.render.grain = 0.0
    cfg.render.mode = "stretch"
    cfg.render.lens_blur = 0.0
    return cfg


@pytest.mark.parametrize("device", DEVICES)
def test_replace_green_keep_hand(device):
    frame, quad = base_frame()
    track = make_track([quad] * 3, frame[60:240, 160:290].reshape(-1, 3))
    red = np.zeros((500, 500, 3), np.uint8)
    red[:] = (0, 0, 255)
    rend = Renderer(red, track, config(), torch.device(device), YuvFormat())
    yuv = to_yuv(frame)
    res = rend.render(yuv, 1, want_bgr=True)
    out = res.bgr
    assert np.abs(out[150, 250].astype(int) - [0, 0, 255]).max() <= 4       # pannello -> rosso
    assert np.abs(out[150, 115].astype(int) - SKIN).max() <= 4              # la mano resta
    roi = out[52:248, 102:298].astype(int)
    greenness = roi[..., 1] - np.maximum(roi[..., 0], roi[..., 2])
    assert greenness.max() < 20                                             # niente verde residuo
    # Lontano dal quadro i piani YUV sono identici all'originale (nessuna perdita).
    assert (res.y[:20] == yuv.y[:20]).all() and (res.u[:10] == yuv.u[:10]).all()
    assert (res.y[:, :60] == yuv.y[:, :60]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_pixels_unmixed_exactly(device):
    """Bordo di un dito semi-trasparente (50% pelle, 50% pannello, in luce lineare).

    Il risultato deve essere il 50% pelle + 50% quadro: con la sola
    differenza di verde (pelle con G < R) alpha verrebbe sottostimato e il
    bordo resterebbe quasi tutto color pelle.
    """
    frame, quad = base_frame()
    frame[:, :] = 90
    frame[50:250, 100:300] = GREEN
    frame[100:200, 180:200] = SKIN
    mix = from_linear_np(0.5 * to_linear_np(np.array(SKIN) / 255) + 0.5 * to_linear_np(np.array(GREEN) / 255))
    frame[100:200, 200:208] = np.round(mix * 255).astype(np.uint8)
    track = make_track([quad] * 3, frame[60:95, 110:290].reshape(-1, 3))
    red = np.zeros((500, 500, 3), np.uint8)
    red[:] = (0, 0, 255)
    rend = Renderer(red, track, config(), torch.device(device), YuvFormat())
    out = rend.render(to_yuv(frame), 1, want_bgr=True).bgr
    got = to_linear_np(out[120:180, 202:206].reshape(-1, 3).mean(0) / 255)
    want = 0.5 * to_linear_np(np.array(SKIN) / 255) + 0.5 * to_linear_np(np.array([0, 0, 255]) / 255)
    assert np.abs(from_linear_np(got) - from_linear_np(want)).max() * 255 < 14


@pytest.mark.parametrize("device", DEVICES)
def test_motion_blur_samples(device):
    frame, quad = base_frame()
    quads = [quad + [8.0 * k, 0.0] for k in range(5)]
    track = make_track(quads, frame[60:240, 160:290].reshape(-1, 3))
    red = np.zeros((50, 50, 3), np.uint8)
    rend = Renderer(red, track, config(), torch.device(device), YuvFormat())
    times = rend._time_samples(2)
    assert len(times) > 3 and min(times) == pytest.approx(-max(times))
    assert max(times) < 0.25 + 1e-9                                  # otturatore 0.5
    cfg = config()
    cfg.render.motion_blur = False
    assert Renderer(red, track, cfg, torch.device(device), YuvFormat())._time_samples(2) == [0.0]


@pytest.mark.parametrize("device", DEVICES)
def test_green_plant_next_to_panel_untouched(device):
    """Una pianta verde (tinta diversa dal pannello) attaccata al bordo resta com'è.

    Per la differenza di verde la pianta sembra pannello: senza il modello di
    colore il quadro "sbordava" di 2 px sulla pianta e il despill la
    scoloriva per una decina di pixel.
    """
    frame, quad = base_frame()
    plant = (40, 150, 95)                                 # BGR: verde più giallo e scuro
    frame[50:250, 300:360] = plant
    track = make_track([quad] * 3, frame[60:240, 160:290].reshape(-1, 3))
    red = np.zeros((500, 500, 3), np.uint8)
    red[:] = (0, 0, 255)
    rend = Renderer(red, track, config(), torch.device(device), YuvFormat())
    out = rend.render(to_yuv(frame), 1, want_bgr=True).bgr.astype(int)
    assert np.abs(out[150, 296] - [0, 0, 255]).max() <= 6                  # dentro: quadro
    assert np.abs(out[150, 301] - plant).max() <= 20                       # crominanza 4:2:0
    for x in (302, 304, 308, 312):
        assert np.abs(out[150, x] - plant).max() <= 6


@pytest.mark.parametrize("device", DEVICES)
def test_sample_chunks_same_result(device):
    """Gli istanti del motion blur elaborati a gruppi (meno memoria) danno lo stesso quadro."""
    frame, quad = base_frame()
    quads = [quad + [6.0 * k, 2.0 * k] for k in range(5)]
    track = make_track(quads, frame[60:240, 160:290].reshape(-1, 3))
    rng = np.random.default_rng(1)
    painting = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
    rend = Renderer(painting, track, config(), torch.device(device), YuvFormat())
    assert len(rend._time_samples(2)) > rend.sample_chunk
    yuv = to_yuv(frame)
    a = rend.render(yuv, 2)
    rend.sample_chunk = 64                                      # tutti insieme
    b = rend.render(yuv, 2)
    for pa, pb in ((a.y, b.y), (a.u, b.u), (a.v, b.v)):
        assert np.abs(pa.astype(int) - pb.astype(int)).max() <= 1


@pytest.mark.parametrize("device", DEVICES)
def test_nan_quad_passthrough(device):
    frame, quad = base_frame()
    track = make_track([quad, np.full((4, 2), np.nan), quad], frame[60:240, 160:290].reshape(-1, 3))
    rend = Renderer(np.zeros((10, 10, 3), np.uint8), track, config(), torch.device(device), YuvFormat())
    yuv = to_yuv(frame)
    res = rend.render(yuv, 1)
    assert res.y is yuv.y and res.u is yuv.u


@pytest.mark.parametrize("device", DEVICES)
def test_markers_removed_but_hand_kept(device):
    """Marcatori di tracking a croce sul pannello: sostituiti come il pannello.

    Caso reale (schermi verdi con crocette nere): senza la rimozione le
    crocette restano visibili sopra il quadro. La mano, che entra dal bordo,
    non deve essere toccata; con l'opzione disattivata la crocetta resta.
    """
    frame, quad = base_frame()
    cv2.line(frame, (190, 140), (210, 140), (20, 20, 20), 3)       # crocetta nera
    cv2.line(frame, (200, 130), (200, 150), (20, 20, 20), 3)
    track = make_track([quad] * 3, frame[60:120, 230:290].reshape(-1, 3))
    red = np.zeros((500, 500, 3), np.uint8)
    red[:] = (0, 0, 255)
    cfg = config()
    out = Renderer(red, track, cfg, torch.device(device), YuvFormat()).render(to_yuv(frame), 1, want_bgr=True).bgr
    patch = out[132:149, 192:209].astype(int)
    assert np.abs(patch - [0, 0, 255]).max() <= 25                  # niente crocetta né alone scuro
    assert np.abs(out[150, 115].astype(int) - SKIN).max() <= 4      # la mano resta
    cfg.render.remove_markers = False
    out2 = Renderer(red, track, cfg, torch.device(device), YuvFormat()).render(to_yuv(frame), 1, want_bgr=True).bgr
    assert out2[140, 200].max() < 90                                # crocetta ancora visibile
