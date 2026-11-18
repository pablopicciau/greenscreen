"""Pipeline completa: analisi (tracking) e rendering (sostituzione).

Due fasi separate, collegate da un file di traccia JSON:

* ``analyze``: legge il video una volta, trova il quadro in ogni fotogramma,
  liscia le traiettorie e salva la traccia (pochi secondi per un Reel);
* ``render``: rilegge il video e inserisce l'immagine usando la traccia.

Così, per pubblicare lo stesso video con quadri diversi, l'analisi si fa una
volta sola.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import geometry as geo
from .blur_estimation import estimate_blur
from .config import Config
from .device import describe, get_device
from .image_fit import load_image
from .render import Renderer
from .smoothing import flag_inconsistent_sides, smooth_track
from .tracker import STATUS_LOST, STATUS_MODEL, STATUS_OK, STATUS_PREDICTED, Track, track_frames
from .video_io import FrameConverter, FrameReader, FrameWriter, VideoInfo, YuvFrame, iter_bgr, probe

Logger = Callable[[str], None]


class Progress:
    """Barra di avanzamento minimale su stderr (nessuna dipendenza esterna)."""

    def __init__(self, label: str, total: int, enabled: bool = True):
        self.label, self.total, self.enabled = label, max(total, 1), enabled
        self.t0 = time.time()
        self.last = -1.0

    def __call__(self, i: int) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last < 0.25 and i + 1 < self.total:
            return
        self.last = now
        fps = (i + 1) / max(now - self.t0, 1e-6)
        pct = min(100.0, 100.0 * (i + 1) / self.total)
        sys.stderr.write(f"\r[{self.label}] {pct:5.1f}%  {i + 1}/{self.total}  {fps:5.1f} fot/s")
        sys.stderr.flush()

    def done(self) -> None:
        if self.enabled:
            sys.stderr.write(f"\r[{self.label}] completato in {time.time() - self.t0:.1f} s" + " " * 20 + "\n")


def _log_default(msg: str) -> None:
    print(msg)


def check_video(info: VideoInfo, log: Logger) -> None:
    """Avvisi utili sui video degli smartphone."""
    if info.hdr:
        log("ATTENZIONE: video HDR (" + info.color_transfer + "). Verrà convertito in SDR; "
            "per colori più fedeli disattiva l'HDR nelle impostazioni della fotocamera.")
    if info.width * info.height > 1920 * 1080 * 1.1:
        log(f"Nota: video {info.width}x{info.height} oltre il Full HD: l'elaborazione sarà più lenta.")


def analyze(video_path: str, cfg: Config, track_path: str | None = None,
            debug_video: str | None = None, log: Logger = _log_default,
            show_progress: bool = True) -> Track:
    """Fase 1+2: tracking del quadro e smoothing. Ritorna (e salva) la traccia."""
    info = probe(video_path)
    log(f"Video: {info.width}x{info.height}, {info.fps:.3f} fps, ~{info.n_frames} fotogrammi, "
        f"audio: {'sì' if info.has_audio else 'no'}")
    check_video(info, log)

    prog = Progress("analisi", info.n_frames, show_progress)
    device = get_device(cfg.render.device)
    with FrameReader(info) as reader:
        conv = FrameConverter(reader.fmt, device, detail=cfg.detect.chroma_detail)
        # La conversione del colore (GPU) del fotogramma successivo procede in
        # parallelo al rilevamento (CPU) di quello attuale.
        frames = prefetch((conv.bgr_u8(f) for f in reader), 3)
        track = track_frames(frames, (info.width, info.height), info.fps, cfg, prog)
    prog.done()
    fixed = flag_inconsistent_sides(track, cfg.track.aspect_tolerance, cfg.track.trusted_support)
    if fixed:
        log(f"Controllo di coerenza: {len(fixed)} fotogrammi con un lato coperto da un oggetto "
            f"sono stati ricostruiti.")
    track.smooth = smooth_track(track, cfg.track.smooth_half_window, cfg.track.outlier_px)
    if track.edge_width is not None:
        est = estimate_blur(track.edge_width, track.smooth)
        if est is not None:
            track.lens_sigma, track.shutter, info_blur = est
            if not info_blur["shutter_observable"]:
                log("Nota: pannello quasi fermo, tempo di esposizione non misurabile (uso 180°).")
    log(track.summary())
    if track.n_frames and track.status.count(STATUS_LOST) == track.n_frames:
        log("ERRORE: quadro verde mai trovato. Controlla che il pannello sia ben illuminato "
            "e di un verde saturo (vedi docs/GUIDA_RIPRESE.md).")
    if track_path:
        track.save(track_path)
        log(f"Traccia salvata in {track_path}")
    if debug_video:
        write_debug_video(info, track, debug_video, cfg, show_progress)
        log(f"Video di debug salvato in {debug_video}")
    return track


def prefetch(iterable, depth: int = 3):
    """Esegue un generatore in un thread, tenendo pronti fino a ``depth`` elementi.

    PyTorch, OpenCV e le pipe di FFmpeg rilasciano il GIL: lettura,
    conversione sulla GPU ed elaborazione sulla CPU si sovrappongono.
    """
    q: queue.Queue = queue.Queue(maxsize=max(1, depth))
    done = object()
    err: list[BaseException] = []

    def worker():
        try:
            for item in iterable:
                q.put(item)
        except BaseException as exc:  # noqa: BLE001 - riportata al chiamante
            err.append(exc)
        finally:
            q.put(done)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is done:
            break
        yield item
    th.join()
    if err:
        raise err[0]


def render_frames(video_path: str, track: Track, image_path: str, cfg: Config,
                  log: Logger = _log_default, want_bgr: bool = True):
    """Generatore: ``(indice, YuvFrame originale, RenderResult)`` in ordine.

    Con ``want_bgr=True`` ogni risultato contiene anche l'immagine BGR finale
    (anteprime, benchmark); per scrivere il video bastano i piani YUV.
    """
    if track.smooth is None:
        raise ValueError("La traccia non contiene gli angoli finali (smooth)")
    info = probe(video_path)
    device = get_device(cfg.render.device)
    image = load_image(image_path)
    with FrameReader(info) as reader:
        rend = Renderer(image, track, cfg, device, reader.fmt)
        log(f"Immagine {image.shape[1]}x{image.shape[0]} -> pannello "
            f"{cfg.render.panel_aspect or track.panel_aspect or 1.0:.3f} (modalità '{cfg.render.mode}', "
            f"preparata {rend.prepared.shape[1]}x{rend.prepared.shape[0]}); calcolo su {describe(device)}; "
            f"motion blur: {'sì, otturatore %.0f°' % (360 * rend.shutter) if rend.shutter > 0 else 'no'}; "
            f"nitidezza adattata: sfocatura {rend.lens_sigma:.2f} px")
        for i, frame in enumerate(reader):
            yield i, frame, rend.render(frame, i, want_bgr=want_bgr)


def render(video_path: str, track: Track, image_path: str, out_path: str, cfg: Config,
           log: Logger = _log_default, show_progress: bool = True) -> None:
    """Fase 3: genera il video finale con l'immagine inserita."""
    info = probe(video_path)
    prog = Progress("rendering", info.n_frames, show_progress)
    with FrameWriter(out_path, info, cfg.encode, video_path, input_format="yuv420p") as writer:
        frames = prefetch(render_frames(video_path, track, image_path, cfg, log, want_bgr=False), 4)
        for i, _, res in frames:
            writer.write_yuv(YuvFrame(res.y, res.u, res.v))
            prog(i)
    prog.done()
    log(f"Video salvato in {out_path}")


def preview(video_path: str, track: Track, image_path: str, frame_index: int, out_png: str,
            cfg: Config) -> None:
    """Salva un PNG di un singolo fotogramma: originale | risultato."""
    info = probe(video_path)
    device = get_device(cfg.render.device)
    with FrameReader(info) as reader:
        conv = FrameConverter(reader.fmt, device)
        rend = Renderer(load_image(image_path), track, cfg, device, reader.fmt)
        for i, frame in enumerate(reader):
            if i == frame_index:
                src = conv.bgr_u8(frame)
                out = rend.render(frame, i, want_bgr=True).bgr
                Path(out_png).parent.mkdir(parents=True, exist_ok=True)
                ok, buf = cv2.imencode(".png", np.hstack([src, out]))
                buf.tofile(out_png)  # tofile: funziona anche con percorsi non ASCII su Windows
                return
    raise IndexError(f"Il video ha meno di {frame_index + 1} fotogrammi")


# ---------------------------------------------------------------------------
# Video di debug
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    STATUS_OK: (60, 220, 60),          # verde: misurato
    STATUS_MODEL: (0, 200, 255),       # arancio: un lato ricostruito
    STATUS_PREDICTED: (0, 140, 255),   # arancio scuro: solo predizione
    STATUS_LOST: (40, 40, 255),        # rosso: perso
}


def draw_debug(frame: np.ndarray, track: Track, i: int) -> np.ndarray:
    """Disegna sul fotogramma i quad (grezzo e finale), gli angoli e lo stato."""
    out = frame.copy()
    status = track.status[i] if i < track.n_frames else STATUS_LOST
    color = _STATUS_COLORS.get(status, (255, 255, 255))
    scale = max(out.shape[:2]) / 1920.0
    th = max(1, int(round(2 * scale)))
    raw = track.raw[i] if i < track.n_frames else None
    fin = track.smooth[i] if track.smooth is not None and i < track.n_frames else None
    if raw is not None and not np.isnan(raw).any():
        cv2.polylines(out, [np.round(raw * 4).astype(np.int32)], True, (0, 0, 255), th, cv2.LINE_AA, shift=2)
    if fin is not None and not np.isnan(fin).any():
        cv2.polylines(out, [np.round(fin * 4).astype(np.int32)], True, color, th, cv2.LINE_AA, shift=2)
        for k, name in enumerate(geo.CORNER_NAMES):
            p = tuple(np.round(fin[k]).astype(int))
            cv2.circle(out, p, int(6 * scale) + 2, color, th, cv2.LINE_AA)
            cv2.putText(out, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale + 0.2,
                        color, th, cv2.LINE_AA)
    text = f"#{i} {status} conf {track.confidence[i]:.2f}" if i < track.n_frames else f"#{i} fuori traccia"
    if i < track.n_frames and track.reasons and track.reasons[i]:
        text += f" ({track.reasons[i]})"
    # Scritta su un riquadro scuro semitrasparente (leggibile su qualunque
    # sfondo; un contorno disegnato con spessore maggiore non combacia col
    # testo in tutte le versioni di OpenCV).
    font_scale = 1.0 * scale + 0.3
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, th)
    org = (int(20 * scale) + 5, int(50 * scale) + 10)
    x0, y0 = org[0] - 6, org[1] - tht - 8
    x1, y1 = min(out.shape[1], org[0] + tw + 6), org[1] + base + 6
    box = out[max(0, y0):y1, max(0, x0):x1]
    box[:] = (box * 0.35).astype(np.uint8)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, th, cv2.LINE_AA)
    return out


def write_debug_video(info: VideoInfo, track: Track, out_path: str, cfg: Config,
                      show_progress: bool = True) -> None:
    """Video con l'overlay del tracking (rosso = misura grezza, colore = finale)."""
    prog = Progress("debug", info.n_frames, show_progress)
    with FrameWriter(out_path, info, cfg.encode, None, input_format="bgr24") as writer:
        for i, frame in enumerate(iter_bgr(info, get_device(cfg.render.device))):
            writer.write(draw_debug(frame, track, i))
            prog(i)
    prog.done()
