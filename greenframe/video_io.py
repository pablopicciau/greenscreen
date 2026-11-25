"""Lettura e scrittura video tramite FFmpeg, nel formato nativo YUV.

Scelte e motivazioni
--------------------
* **Lettura dei piani YUV originali** (``rawvideo yuv420p``): FFmpeg decodifica
  ma *non* converte in RGB. La conversione la fa :class:`FrameConverter`
  (GPU o CPU) rispettando la posizione dei campioni di crominanza
  (``chroma_location``) e ricostruendo il colore a piena risoluzione con
  l'aiuto della luminanza. La conversione rapida di FFmpeg verso ``bgr24``
  replica ogni campione di crominanza su 2×2 pixel: nei test spostava i
  bordi del pannello verde di ~0.4 px in orizzontale.
* Il **frame rate costante** viene forzato (i telefoni registrano a frame rate
  variabile: senza conversione l'audio, copiato dall'originale, andrebbe
  fuori sincrono). La **rotazione** degli smartphone viene applicata da FFmpeg
  e la posizione della crominanza ricalcolata di conseguenza.
* I video **HDR** (iPhone: HLG/Dolby Vision) vengono convertiti in SDR BT.709
  con tone mapping (``libplacebo`` sulla GPU se disponibile, altrimenti
  ``zscale``+``tonemap``), in YUV 4:4:4 per non perdere colore.
* **Scrittura**: i piani YUV 4:2:0 preparati dal programma (crominanza
  "left", BT.709) vengono passati all'encoder H.264/HEVC (libx264 di qualità
  oppure NVENC sulla GPU NVIDIA) con i metadati di colore espliciti; audio
  originale copiato (o ricodificato in AAC se necessario), ``+faststart``
  per la riproduzione via web/Instagram.
* FFmpeg viene cercato nel PATH; in alternativa si usa quello fornito dal
  pacchetto ``imageio-ffmpeg`` (installabile con pip, comodo su Windows).
"""

from __future__ import annotations

import functools
import json
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .color import YuvFormat, chroma_siting, rgb_to_yuv420, yuv420_to_rgb
from .config import EncodeParams

# Frame rate standard a cui "agganciare" valori medi leggermente irregolari.
_STANDARD_FPS = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                 Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60)]

# Formati sorgente 4:2:0 (letti come yuv420p senza ricampionare la crominanza).
_FMT_420 = {"yuv420p", "yuvj420p", "nv12", "nv21", "yuv420p10le", "yuv420p10be", "p010le",
            "yuv420p12le", "yuv420p16le", "p016le"}


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


@functools.lru_cache(maxsize=None)
def _ffmpeg_list(kind: str) -> str:
    ff = find_ffmpeg()
    if ff is None:
        return ""
    return subprocess.run([ff, "-hide_banner", f"-{kind}"], capture_output=True, text=True).stdout


def has_filter(name: str) -> bool:
    """True se l'ffmpeg disponibile include il filtro indicato (es. 'zscale')."""
    return any(line.split()[1:2] == [name] for line in _ffmpeg_list("filters").splitlines()
               if len(line.split()) > 1)


def has_encoder(name: str) -> bool:
    """True se l'ffmpeg disponibile include l'encoder indicato (es. 'h264_nvenc')."""
    return any(line.split()[1:2] == [name] for line in _ffmpeg_list("encoders").splitlines()
               if len(line.split()) > 1)


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
    rotation: int = 0       # rotazione oraria applicata in visualizzazione (0/90/180/270)
    color_transfer: str = ""
    hdr: bool = False
    pix_fmt: str = "yuv420p"
    chroma_location: str = ""
    color_space: str = ""
    color_range: str = ""
    audio_codec: str = ""


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
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # Rotazione: nei video degli smartphone è spesso un metadato, non nei pixel.
    # ffprobe riporta l'angolo antiorario della matrice di visualizzazione:
    # FFmpeg ruota il video in senso orario di (-rotation) gradi.
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = -int(video["tags"]["rotate"])
    rotation_cw = (-rotation) % 360
    w, h = int(video["width"]), int(video["height"])
    if rotation_cw in (90, 270):
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
        has_audio=audio is not None, rotation=rotation_cw, color_transfer=transfer,
        hdr=transfer in ("smpte2084", "arib-std-b67"),
        pix_fmt=video.get("pix_fmt", "") or "", chroma_location=video.get("chroma_location", "") or "",
        color_space=video.get("color_space", "") or "", color_range=video.get("color_range", "") or "",
        audio_codec=(audio or {}).get("codec_name", "") or "",
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


# ---------------------------------------------------------------------------
# Fotogrammi YUV
# ---------------------------------------------------------------------------

@dataclass
class YuvFrame:
    """Piani di un fotogramma (uint8). ``u``/``v`` a metà risoluzione (4:2:0) o piena (4:4:4)."""

    y: np.ndarray
    u: np.ndarray
    v: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.y.shape

    def tobytes(self) -> bytes:
        return self.y.tobytes() + self.u.tobytes() + self.v.tobytes()


@functools.lru_cache(maxsize=None)
def _libplacebo_works() -> bool:
    """True se libplacebo è presente E un dispositivo Vulkan si inizializza davvero.

    Avere il filtro compilato in FFmpeg non basta: senza driver Vulkan
    funzionanti (driver assente o incompatibile, desktop remoto, alcuni
    portatili) FFmpeg termina subito con errore e il video HDR non verrebbe
    letto affatto. Si prova quindi una conversione minima, una volta sola.
    """
    if not has_filter("libplacebo"):
        return False
    ff = find_ffmpeg()
    if ff is None:
        return False
    try:
        res = subprocess.run(
            [ff, "-v", "error", "-init_hw_device", "vulkan", "-f", "lavfi",
             "-i", "color=c=gray:s=16x16:d=0.04", "-vf", "libplacebo", "-f", "null", "-"],
            capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.returncode == 0


def _hdr_filter() -> str | None:
    """Catena di filtri FFmpeg HDR -> SDR BT.709 (YUV 4:4:4), o None se non disponibile."""
    if _libplacebo_works():
        # Tone mapping BT.2390 sulla GPU (Vulkan); applica anche i metadati Dolby Vision.
        return ("libplacebo=tonemapping=bt.2390:colorspace=bt709:color_primaries=bt709:"
                "color_trc=bt709:range=tv:format=yuv444p")
    if has_filter("zscale"):
        return ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv444p")
    return None


class FrameReader:
    """Iteratore sui fotogrammi :class:`YuvFrame` di un video (lettura anticipata in un thread).

    Uso::

        with FrameReader(info) as reader:
            for frame in reader:          # YuvFrame
                ...

    ``reader.fmt`` descrive matrice, gamma dei valori e posizione della
    crominanza dei piani prodotti.
    """

    def __init__(self, info: VideoInfo, tonemap_hdr: bool = True, prefetch: int = 6):
        self.info = info
        ffmpeg = require_ffmpeg()
        filters = []
        use_hdr = info.hdr and tonemap_hdr and _hdr_filter() is not None
        if use_hdr:
            filters.append(_hdr_filter())
            self.pix = "yuv444p"
            self.fmt = YuvFormat("bt709", False, (0.0, 0.0))
        else:
            self.pix = "yuv420p" if (info.pix_fmt in _FMT_420 or not info.pix_fmt) else "yuv444p"
            # I valori "pieni" (yuvj, color_range=pc) vengono riportati al range "tv".
            filters.append("scale=out_range=tv")
            siting = chroma_siting(info.chroma_location, info.rotation)
            self.fmt = YuvFormat(info.color_space or "bt709", False, siting)
        filters.append(f"fps={info.fps_str}")   # frame rate costante (sincronia audio)
        cmd = [ffmpeg, "-v", "error"]
        if use_hdr and "libplacebo" in filters[0]:
            cmd += ["-init_hw_device", "vulkan"]
        cmd += ["-i", info.path, "-map", "0:v:0", "-vf", ",".join(filters),
                "-f", "rawvideo", "-pix_fmt", self.pix, "-"]
        w, h = info.width, info.height
        cw, ch = ((w + 1) // 2, (h + 1) // 2) if self.pix == "yuv420p" else (w, h)
        self._sizes = (w * h, cw * ch)
        self._shapes = ((h, w), (ch, cw))
        self.frame_bytes = w * h + 2 * cw * ch
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.frame_bytes * 2)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        ys, cs = self._sizes
        try:
            while not self._stop.is_set():
                buf = self.proc.stdout.read(self.frame_bytes)
                if len(buf) < self.frame_bytes:
                    break
                arr = np.frombuffer(buf, np.uint8)
                frame = YuvFrame(arr[:ys].reshape(self._shapes[0]),
                                 arr[ys:ys + cs].reshape(self._shapes[1]),
                                 arr[ys + cs:].reshape(self._shapes[1]))
                while not self._stop.is_set():
                    try:
                        self._queue.put(frame, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        finally:
            self._queue.put(None)

    def __iter__(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            yield frame

    def close(self):
        self._stop.set()
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()
        self._thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FrameConverter:
    """Conversioni YUV <-> RGB su GPU (o CPU) con PyTorch.

    * :meth:`rgb`: tensore (3, H, W) float32 in valori video 0-1 (gamma BT.709);
    * :meth:`bgr_u8`: immagine NumPy BGR uint8 (per OpenCV);
    * :meth:`yuv`: piani YUV 4:2:0 uint8 con crominanza "left" per l'encoder.
    """

    OUT_FORMAT = YuvFormat("bt709", False, (0.0, 0.5))

    def __init__(self, fmt: YuvFormat, device=None, detail: bool = True):
        import torch

        from .device import get_device

        self.fmt = fmt
        self.device = device if device is not None else get_device()
        self.detail = detail
        self.torch = torch

    def planes(self, frame: YuvFrame):
        """Piani YUV sul dispositivo (un'unica copia: 1.5 byte per pixel)."""
        t = self.torch
        # Nota: niente pin_memory per fotogramma (allocare memoria bloccata
        # costa più della copia stessa: ~25 ms contro ~1 ms in Full HD).
        buf = t.from_numpy(np.concatenate([frame.y.ravel(), frame.u.ravel(), frame.v.ravel()]))
        buf = buf.to(self.device)
        ny, nc = frame.y.size, frame.u.size
        return (buf[:ny].view(frame.y.shape), buf[ny:ny + nc].view(frame.u.shape),
                buf[ny + nc:].view(frame.v.shape))

    def rgb(self, frame: YuvFrame):
        y, u, v = self.planes(frame)
        return yuv420_to_rgb(y, u, v, self.fmt, detail=self.detail)

    def bgr_u8(self, frame: YuvFrame) -> np.ndarray:
        rgb = self.rgb(frame)
        out = (rgb.flip(0).permute(1, 2, 0) * 255.0 + 0.5).clamp(0, 255).to(self.torch.uint8)
        return out.cpu().numpy()

    def yuv(self, rgb) -> YuvFrame:
        y, u, v = rgb_to_yuv420(rgb, self.OUT_FORMAT)
        return YuvFrame(y.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy())


def iter_bgr(info: VideoInfo, device=None, detail: bool = True):
    """Generatore di fotogrammi BGR uint8 (conversione accurata del colore)."""
    with FrameReader(info) as reader:
        conv = FrameConverter(reader.fmt, device, detail)
        for frame in reader:
            yield conv.bgr_u8(frame)


class FrameWriter:
    """Codifica fotogrammi in un MP4 (H.264/HEVC), con l'audio del video originale.

    ``input_format``: ``"yuv420p"`` (piani preparati con crominanza "left",
    metodo :meth:`write_yuv`) oppure ``"bgr24"`` (immagini BGR, metodo
    :meth:`write`, convertite da FFmpeg: usato per i video di debug).
    """

    def __init__(self, out_path: str, info: VideoInfo, enc: EncodeParams,
                 audio_source: str | None = None, input_format: str = "yuv420p"):
        ffmpeg = require_ffmpeg()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self.input_format = input_format
        cmd = [ffmpeg, "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", input_format, "-s", f"{info.width}x{info.height}",
               "-r", info.fps_str, "-i", "-"]
        use_audio = audio_source is not None and enc.keep_audio and info.has_audio
        if use_audio:
            cmd += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a:0?"]
        if input_format != "yuv420p":
            # Conversione RGB -> YUV esplicitamente in BT.709.
            cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"]
        cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-color_range", "tv", "-chroma_sample_location", "left"]
        cmd += encoder_args(enc)
        if use_audio:
            if info.audio_codec == "aac" and enc.copy_audio:
                # Copia senza ricodifica. Niente -shortest: con la copia il taglio
                # avviene a pacchetti interi e farebbe perdere l'ultimo fotogramma.
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", enc.audio_bitrate, "-shortest"]
        cmd += ["-movflags", "+faststart", out_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.out_path = out_path
        # Lo stderr viene letto in un thread: se FFmpeg scrive molti messaggi
        # il buffer non si riempie (altrimenti la scrittura si bloccherebbe).
        self._err = bytearray()
        self._err_thread = threading.Thread(target=self._drain, daemon=True)
        self._err_thread.start()

    def _drain(self):
        if self.proc.stderr:
            for chunk in iter(lambda: self.proc.stderr.read(4096), b""):
                self._err.extend(chunk)

    def write_yuv(self, frame: YuvFrame) -> None:
        assert self.proc.stdin is not None and self.input_format == "yuv420p"
        self.proc.stdin.write(frame.tobytes())

    def write(self, frame_bgr: np.ndarray) -> None:
        assert self.proc.stdin is not None and self.input_format == "bgr24"
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        code = self.proc.wait()
        self._err_thread.join(timeout=5)
        if code != 0:
            raise RuntimeError("FFmpeg ha restituito un errore durante la codifica:\n"
                               + self._err.decode(errors="replace"))

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


def encoder_args(enc: EncodeParams) -> list[str]:
    """Argomenti FFmpeg dell'encoder video scelto."""
    codec = enc.codec
    if codec in ("h264_nvenc", "hevc_nvenc"):
        preset = enc.preset if enc.preset.startswith("p") else "p7"
        args = ["-c:v", codec, "-preset", preset, "-tune", "hq", "-rc", "vbr",
                "-cq", str(enc.crf + 2), "-b:v", "0", "-spatial-aq", "1", "-pix_fmt", "yuv420p"]
        if codec == "h264_nvenc":
            args += ["-profile:v", "high"]
        else:
            args += ["-profile:v", "main", "-tag:v", "hvc1"]   # hvc1: compatibile con Apple/Instagram
        return args
    if codec == "libx265":
        return ["-c:v", "libx265", "-preset", enc.preset, "-crf", str(enc.crf + 2),
                "-pix_fmt", "yuv420p", "-tag:v", "hvc1"]
    return ["-c:v", codec, "-preset", enc.preset, "-crf", str(enc.crf), "-profile:v", "high",
            "-pix_fmt", "yuv420p"]
