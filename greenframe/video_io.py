"""Lettura e scrittura video tramite FFmpeg (con ripiego su OpenCV in lettura).

Scelte e motivazioni
--------------------
* **Lettura con FFmpeg in pipe** (``rawvideo bgr24``): gestisce qualunque
  formato degli smartphone (HEVC, rotazione, ecc.), permette di forzare il
  **frame rate costante** (i telefoni registrano a frame rate variabile: senza
  questa conversione l'audio, copiato dall'originale, andrebbe fuori sincrono)
  e di convertire i video **HDR** (iPhone: HLG/Dolby Vision) in SDR.
* **Scrittura con FFmpeg in pipe**: H.264 (libx264 di qualità, oppure NVENC
  sulla GPU NVIDIA), pixel format yuv420p e colori **BT.709** dichiarati
  esplicitamente (altrimenti la conversione RGB->YUV userebbe BT.601 e i
  colori del quadro risulterebbero leggermente alterati), audio originale
  ricodificato in AAC, ``+faststart`` per la riproduzione via web/Instagram.
* FFmpeg viene cercato nel PATH; in alternativa si usa quello fornito dal
  pacchetto ``imageio-ffmpeg`` (installabile con pip, comodo su Windows).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import EncodeParams

# Frame rate standard a cui "agganciare" valori medi leggermente irregolari.
_STANDARD_FPS = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                 Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60)]


def find_ffmpeg() -> str | None:
    """Percorso dell'eseguibile ffmpeg (PATH o imageio-ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def find_ffprobe() -> str | None:
    """Percorso di ffprobe, se disponibile (accanto a ffmpeg o nel PATH)."""
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    ff = find_ffmpeg()
    if ff:
        cand = Path(ff).with_name("ffprobe" + Path(ff).suffix)
        if cand.exists():
            return str(cand)
    return None


def require_ffmpeg() -> str:
    """Come :func:`find_ffmpeg` ma con un errore chiaro se manca."""
    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError(
            "FFmpeg non trovato. Su Windows: 'winget install Gyan.FFmpeg' "
            "(poi riapri il terminale) oppure 'pip install imageio-ffmpeg'."
        )
    return exe


def has_filter(name: str) -> bool:
    """True se l'ffmpeg disponibile include il filtro indicato (es. 'zscale')."""
    ff = find_ffmpeg()
    if ff is None:
        return False
    out = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if len(line.split()) > 1)


@dataclass
class VideoInfo:
    """Metadati essenziali di un video."""

    path: str
    width: int              # larghezza *visualizzata* (rotazione già applicata)
    height: int
    fps: float
    fps_str: str            # frazione esatta, es. "30000/1001"
    n_frames: int           # stima (il conteggio esatto si ha leggendo)
    duration: float
    has_audio: bool
    rotation: int = 0
    color_transfer: str = ""
    hdr: bool = False


def _snap_fps(rate: Fraction) -> Fraction:
    """Aggancia un frame rate medio irregolare (es. 29.98) al valore standard più vicino."""
    best = min(_STANDARD_FPS, key=lambda std: abs(float(rate) / float(std) - 1.0))
    if abs(float(rate) / float(best) - 1.0) < 0.005:
        return best
    return rate.limit_denominator(1001)


def probe(path: str) -> VideoInfo:
    """Legge i metadati con ffprobe (o con OpenCV se ffprobe manca)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Video non trovato: {path}")
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return _probe_opencv(path)
    res = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe non riesce a leggere {path}: {res.stderr.strip()}")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"Nessuna traccia video in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    # Rotazione: nei video degli smartphone è spesso un metadato, non nei pixel.
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = int(video["tags"]["rotate"])
    w, h = int(video["width"]), int(video["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w

    # Frame rate: quello nominale (r_frame_rate) se coincide con la media
    # (video a frame rate costante), altrimenti la media agganciata al valore
    # standard più vicino (video a frame rate variabile degli smartphone).
    def _frac(txt: str) -> Fraction | None:
        try:
            f = Fraction(txt)
            return f if f > 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    r_rate = _frac(video.get("r_frame_rate", ""))
    avg_rate = _frac(video.get("avg_frame_rate", ""))
    if r_rate and avg_rate and abs(float(r_rate) / float(avg_rate) - 1.0) < 0.002:
        rate = r_rate
    else:
        rate = _snap_fps(avg_rate or r_rate or Fraction(30))
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    n_frames = int(video.get("nb_frames") or round(duration * float(rate)))
    transfer = video.get("color_transfer", "") or ""
    return VideoInfo(
        path=path, width=w, height=h, fps=float(rate),
        fps_str=f"{rate.numerator}/{rate.denominator}", n_frames=n_frames, duration=duration,
        has_audio=has_audio, rotation=rotation, color_transfer=transfer,
        hdr=transfer in ("smpte2084", "arib-std-b67"),
    )


def _probe_opencv(path: str) -> VideoInfo:
    """Ripiego senza ffprobe: metadati via OpenCV (audio sconosciuto -> assunto presente)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {path}")
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        raise RuntimeError(f"Impossibile leggere fotogrammi da: {path}")
    rate = _snap_fps(Fraction(fps).limit_denominator(1001))
    return VideoInfo(path=path, width=frame.shape[1], height=frame.shape[0], fps=float(rate),
                     fps_str=f"{rate.numerator}/{rate.denominator}", n_frames=n,
                     duration=n / float(rate), has_audio=True)


class FrameReader:
    """Iteratore sui fotogrammi BGR uint8 di un video.

    Uso::

        with FrameReader(info) as reader:
            for frame in reader:
                ...
    """

    def __init__(self, info: VideoInfo, tonemap_hdr: bool = True):
        self.info = info
        self.proc: subprocess.Popen | None = None
        self.cap = None
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            # Ripiego: OpenCV (niente conversione HDR né frame rate costante).
            import cv2

            self.cap = cv2.VideoCapture(info.path)
            return
        filters = []
        if info.hdr and tonemap_hdr and has_filter("zscale"):
            # HDR (HLG/PQ) -> SDR BT.709 con tone mapping "hable".
            filters.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                           "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
        filters.append(f"fps={info.fps_str}")   # frame rate costante (sincronia audio)
        cmd = [ffmpeg, "-v", "error", "-i", info.path, "-map", "0:v:0",
               "-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.frame_bytes = info.width * info.height * 3
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.frame_bytes * 2)

    def __iter__(self):
        if self.cap is not None:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    return
                yield frame
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            buf = self.proc.stdout.read(self.frame_bytes)
            if len(buf) < self.frame_bytes:
                return
            yield np.frombuffer(buf, np.uint8).reshape(self.info.height, self.info.width, 3)

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.proc is not None:
            if self.proc.stdout:
                self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FrameWriter:
    """Codifica fotogrammi BGR uint8 in un MP4 H.264, con l'audio del video originale."""

    def __init__(self, out_path: str, info: VideoInfo, enc: EncodeParams,
                 audio_source: str | None = None):
        ffmpeg = require_ffmpeg()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
               "-r", info.fps_str, "-i", "-"]
        use_audio = audio_source is not None and enc.keep_audio and info.has_audio
        if use_audio:
            cmd += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a:0?"]
        # Conversione RGB -> YUV esplicitamente in BT.709 (standard HD) + metadati.
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
        if enc.codec == "h264_nvenc":
            preset = enc.preset if enc.preset.startswith("p") else "p6"
            cmd += ["-c:v", "h264_nvenc", "-preset", preset, "-rc", "vbr",
                    "-cq", str(enc.crf + 2), "-b:v", "0", "-profile:v", "high"]
        else:
            cmd += ["-c:v", enc.codec, "-preset", enc.preset, "-crf", str(enc.crf),
                    "-profile:v", "high"]
        if use_audio:
            cmd += ["-c:a", "aac", "-b:a", enc.audio_bitrate, "-shortest"]
        cmd += ["-movflags", "+faststart", out_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.out_path = out_path

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        if self.proc.wait() != 0:
            raise RuntimeError(f"FFmpeg ha restituito un errore durante la codifica:\n{err}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            self.close()
        else:
            # In caso di eccezione chiude senza mascherarla.
            try:
                self.proc.kill()
            finally:
                self.proc.wait()
