"""Test end-to-end: video sintetico -> analisi -> rendering (richiede ffmpeg)."""

import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from greenframe import pipeline
from greenframe.config import Config
from greenframe.tracker import Track
import make_synthetic_video
import make_test_painting

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg non disponibile")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    video = d / "clip.mp4"
    make_synthetic_video.main(["--out", str(video), "--width", "360", "--height", "640",
                               "--seconds", "2", "--subframes", "1", "--scenarios", "hold"])
    image = d / "quadro.png"
    make_test_painting.main(["--out", str(image), "--style", "pattern", "--width", "400", "--height", "500"])
    gt = json.loads(video.with_suffix(".gt.json").read_text())
    return d, video, image, gt


def test_analyze_accuracy(synthetic):
    d, video, image, gt = synthetic
    track = pipeline.analyze(str(video), Config(), str(d / "t.json"), log=lambda m: None,
                             show_progress=False)
    truth = np.array([g["corners"] for g in gt["ground_truth"]])
    assert track.n_frames == gt["frames"]
    err = np.linalg.norm(track.smooth - truth, axis=2)
    assert not np.isnan(err).any()
    assert err.mean() < 0.4
    assert err.max() < 1.5
    assert track.panel_aspect == pytest.approx(0.8, rel=0.01)
    # Nitidezza ed esposizione stimate dai bordi, modello di colore del pannello.
    assert track.lens_sigma is not None and 0.5 < track.lens_sigma < 2.0
    assert track.shutter is not None and 0.0 <= track.shutter <= 1.0
    assert track.panel_color is not None
    # La traccia salvata si ricarica identica.
    loaded = Track.load(d / "t.json")
    assert np.allclose(loaded.smooth, track.smooth, atol=1e-3)
    assert loaded.shutter == track.shutter and loaded.panel_color == track.panel_color
    assert np.allclose(loaded.edge_width, track.edge_width, atol=1e-3, equal_nan=True)


def test_render_output(synthetic):
    d, video, image, gt = synthetic
    track = Track.load(d / "t.json")
    out = d / "out.mp4"
    pipeline.render(str(video), track, str(image), str(out), Config(), log=lambda m: None,
                    show_progress=False)
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    assert ok and frame.shape[:2] == (640, 360)
    assert n == gt["frames"]
    # L'audio originale è stato copiato; colore e crominanza dichiarati.
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_type,chroma_location,color_space",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert "audio" in probe
    assert "left" in probe and "bt709" in probe
    # Dentro il quadro non resta verde.
    q = np.round(track.smooth[0]).astype(np.int32)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, q, 1)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    px = frame[mask > 0].astype(int)
    green_frac = np.mean(px[:, 1] - np.maximum(px[:, 0], px[:, 2]) > 40)
    assert green_frac < 0.01
