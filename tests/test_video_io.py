"""Test della lettura video: rotazione degli smartphone e video HDR (richiedono ffmpeg)."""

import shutil
import subprocess

import numpy as np
import pytest
import torch
from scipy.special import erf

from greenframe.color import YuvFormat, from_linear_np, rgb_to_yuv420, to_linear_np
from greenframe.video_io import FrameConverter, FrameReader, has_filter, probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg non disponibile")

GREEN = np.array([45, 175, 70]) / 255.0       # RGB
BACK = np.array([200, 170, 150]) / 255.0


def _encode(frames_yuv: list[bytes], w: int, h: int, path, extra=()):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
           "-r", "30", "-i", "-", "-c:v", "libx264", "-qp", "0", "-colorspace", "bt709",
           "-color_range", "tv", "-chroma_sample_location", "left", *extra, str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames_yuv:
        p.stdin.write(f)
    p.stdin.close()
    assert p.wait() == 0


@pytest.mark.parametrize("rotation", [90, -90])
def test_rotated_video_keeps_edge_position(tmp_path, rotation):
    """Video verticale del telefono (salvato orizzontale + metadato di rotazione).

    FFmpeg ruota i piani e con loro la crominanza: se la sua nuova posizione
    non venisse ricalcolata, il bordo del pannello si sposterebbe.
    """
    w, h, x_edge = 320, 240, 160.3
    xs = np.arange(w)[None, :].repeat(h, 0)
    cov = 0.5 * (1 + erf((xs - x_edge) / (np.sqrt(2) * 0.8)))
    lin = (to_linear_np(BACK)[:, None, None] * (1 - cov[None])
           + to_linear_np(GREEN)[:, None, None] * cov[None])
    y, u, v = rgb_to_yuv420(torch.from_numpy(from_linear_np(lin)), YuvFormat())
    raw = y.numpy().tobytes() + u.numpy().tobytes() + v.numpy().tobytes()
    plain = tmp_path / "plain.mp4"
    _encode([raw] * 3, w, h, plain)
    rotated = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-display_rotation:v:0", str(rotation), "-i", str(plain),
                    "-c", "copy", str(rotated)], check=True)
    info = probe(str(rotated))
    assert (info.width, info.height) == (h, w)
    with FrameReader(info) as reader:
        conv = FrameConverter(reader.fmt, torch.device("cpu"))
        frame = next(iter(reader))
        rgb = conv.rgb(frame).numpy()
    # Dopo la rotazione il bordo è orizzontale: dove dovrebbe trovarsi?
    if info.rotation == 270:          # antiorario: (x, y) -> (y, W-1-x)
        y_edge, green_above = w - 1 - x_edge, True
    else:                             # orario: (x, y) -> (H-1-y, x)
        y_edge, green_above = x_edge, False
    prof = to_linear_np(rgb[:, :, h // 2]).T                    # colonna centrale (W', 3)
    if green_above:
        prof = prof[::-1]
        y_edge = w - 1 - y_edge
    c_out, c_in = prof[:8].mean(0), prof[-8:].mean(0)
    d = c_in - c_out
    p = (prof - c_out) @ d / (d @ d)
    k = np.nonzero(p >= 0.5)[0][0]
    pos = k - 1 + (0.5 - p[k - 1]) / (p[k] - p[k - 1])
    assert abs(pos - y_edge) < 0.15


@pytest.mark.skipif(not (has_filter("zscale") or has_filter("libplacebo")), reason="serve zscale o libplacebo")
def test_hdr_video_is_tonemapped(tmp_path):
    """Video HDR (HLG, come l'iPhone): letto e convertito in SDR BT.709."""
    hlg = tmp_path / "hlg.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
                    "-vf", "zscale=transferin=bt709:primariesin=bt709:matrixin=bt709:rangein=full:"
                           "transfer=arib-std-b67:primaries=bt2020:matrix=bt2020nc:range=tv:npl=100,"
                           "format=yuv420p10le",
                    "-c:v", "libx265", "-x265-params", "log-level=error", "-color_primaries", "bt2020",
                    "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", str(hlg)], check=True)
    info = probe(str(hlg))
    assert info.hdr
    with FrameReader(info) as reader:
        frames = list(reader)
        conv = FrameConverter(reader.fmt, torch.device("cpu"))
        img = conv.bgr_u8(frames[0])
    assert len(frames) == 30
    assert frames[0].u.shape == frames[0].y.shape          # 4:4:4 dopo il tone mapping
    assert 30 < img.mean() < 220 and img.std() > 20
