"""Interfaccia a riga di comando.

Esempi (PowerShell o prompt dei comandi, dalla cartella del progetto)::

    # Tutto in un colpo: analisi + video finale
    python -m greenframe run --video ripresa.mp4 --image quadro.jpg --out reel.mp4

    # Analisi una volta sola, poi un video per ogni quadro
    python -m greenframe analyze --video ripresa.mp4 --debug-video controllo.mp4
    python -m greenframe render --video ripresa.mp4 --image quadro1.jpg quadro2.jpg --out output/

    # Anteprima di un singolo fotogramma (PNG originale | risultato)
    python -m greenframe preview --video ripresa.mp4 --image quadro.jpg --frame 120 --out prova.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .geometry import parse_aspect
from .tracker import TRACK_VERSION, Track


def _default_track_path(video: str) -> str:
    """Traccia accanto al video: ripresa.mp4 -> ripresa.track.json."""
    p = Path(video)
    return str(p.with_name(p.stem + ".track.json"))


def _add_common(ap: argparse.ArgumentParser) -> None:
    """Opzioni condivise tra i comandi."""
    ap.add_argument("--video", required=True, help="video originale con il pannello verde")
    ap.add_argument("--track", help="file della traccia (default: <video>.track.json)")
    ap.add_argument("--aspect", help="proporzioni del pannello LxA, es. 40x50, 4:5 o 0.8 "
                                     "(default: stimate dal video)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="calcolo su GPU NVIDIA (cuda), CPU o automatico (default)")
    ap.add_argument("--quiet", action="store_true", help="niente barra di avanzamento")


def _add_render_opts(ap: argparse.ArgumentParser) -> None:
    """Opzioni di inserimento e codifica."""
    ap.add_argument("--mode", choices=["fill", "fit", "stretch"], default="fill",
                    help="fill = riempi (ritaglia), fit = adatta (bande), stretch = deforma")
    ap.add_argument("--fit-bg", default="blur", help="sfondo bande in modalità fit: blur, white, black, #rrggbb")
    ap.add_argument("--lighting", type=float, default=None,
                    help="trasferimento di luce e ombre 0-1 (default 1 = come il pannello)")
    ap.add_argument("--grain", type=float, default=None, help="grana della camera sull'immagine 0-1 (default 1)")
    ap.add_argument("--shutter", default="auto",
                    help="otturatore in gradi per il motion blur (es. 180) o 'auto' = stimato dal video")
    ap.add_argument("--no-motion-blur", action="store_true", help="niente sfocatura di movimento sul quadro")
    ap.add_argument("--lens-blur", default="auto",
                    help="sfocatura (px) del quadro per eguagliare la nitidezza della camera; "
                         "'auto' = stimata dal video, 0 = nessuna")
    ap.add_argument("--codec", choices=["libx264", "h264_nvenc", "hevc_nvenc", "libx265"], default="libx264",
                    help="libx264 = qualità massima (CPU); h264_nvenc/hevc_nvenc = GPU NVIDIA, più veloce")
    ap.add_argument("--crf", type=int, default=None, help="qualità (più basso = migliore, default 17)")
    ap.add_argument("--no-audio", action="store_true", help="non copiare l'audio originale")
    ap.add_argument("--keep-markers", action="store_true",
                    help="non sostituire le piccole zone non verdi interne al pannello "
                         "(marcatori di tracking, macchie, riflessi)")


def build_config(args: argparse.Namespace) -> Config:
    """Traduce gli argomenti in una configurazione."""
    cfg = Config()
    if getattr(args, "aspect", None):
        cfg.render.panel_aspect = parse_aspect(args.aspect)
    cfg.render.device = getattr(args, "device", "auto")
    if hasattr(args, "mode"):
        cfg.render.mode = args.mode
        cfg.render.fit_background = args.fit_bg
        if args.lighting is not None:
            cfg.render.lighting = args.lighting
        if args.grain is not None:
            cfg.render.grain = args.grain
        if str(args.shutter).lower() != "auto":
            cfg.render.shutter = min(max(float(args.shutter) / 360.0, 0.0), 1.0)
        cfg.render.motion_blur = not args.no_motion_blur
        if str(args.lens_blur).lower() != "auto":
            cfg.render.lens_blur = max(float(args.lens_blur), 0.0)
        cfg.encode.codec = args.codec
        if args.codec.endswith("_nvenc"):
            cfg.encode.preset = "p7"
        if args.crf is not None:
            cfg.encode.crf = args.crf
        cfg.encode.keep_audio = not args.no_audio
        cfg.render.remove_markers = not args.keep_markers
    return cfg


def _outputs_for(images: list[str], out: str) -> list[str]:
    """Un file di uscita per ogni immagine (cartella se le immagini sono più di una)."""
    out_p = Path(out)
    if len(images) == 1 and out_p.suffix.lower() in (".mp4", ".mov", ".m4v"):
        return [str(out_p)]
    out_p.mkdir(parents=True, exist_ok=True)
    return [str(out_p / (Path(img).stem + ".mp4")) for img in images]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="greenframe",
        description="Green Frame Replacer: sostituisce il pannello verde tenuto in mano con un'immagine.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", aliases=["analizza"], help="analizza il video e salva la traccia")
    _add_common(p_an)
    p_an.add_argument("--debug-video", help="salva un video con l'overlay del tracking")

    p_re = sub.add_parser("render", aliases=["renderizza"], help="crea il video finale da una traccia")
    _add_common(p_re)
    _add_render_opts(p_re)
    p_re.add_argument("--image", nargs="+", required=True, help="immagine/i da inserire")
    p_re.add_argument("--out", required=True, help="file .mp4 (una immagine) o cartella (più immagini)")

    p_run = sub.add_parser("run", aliases=["esegui"], help="analisi + video finale in un solo comando")
    _add_common(p_run)
    _add_render_opts(p_run)
    p_run.add_argument("--image", nargs="+", required=True, help="immagine/i da inserire")
    p_run.add_argument("--out", required=True, help="file .mp4 (una immagine) o cartella (più immagini)")
    p_run.add_argument("--debug-video", help="salva anche un video con l'overlay del tracking")

    p_pv = sub.add_parser("preview", aliases=["anteprima"], help="anteprima PNG di un fotogramma")
    _add_common(p_pv)
    _add_render_opts(p_pv)
    p_pv.add_argument("--image", required=True)
    p_pv.add_argument("--frame", type=int, default=0, help="indice del fotogramma (da 0)")
    p_pv.add_argument("--out", required=True, help="file .png")

    args = ap.parse_args(argv)
    # Import qui per mostrare l'help anche se mancano dipendenze opzionali.
    from . import pipeline

    cfg = build_config(args)
    track_path = args.track or _default_track_path(args.video)
    show = not args.quiet

    try:
        if args.cmd in ("analyze", "analizza"):
            pipeline.analyze(args.video, cfg, track_path, args.debug_video, show_progress=show)
            return 0

        if args.cmd in ("run", "esegui"):
            track = pipeline.analyze(args.video, cfg, track_path, args.debug_video, show_progress=show)
        else:
            track = Track.load(track_path) if Path(track_path).exists() else None
            if track is None:
                print(f"Traccia non trovata ({track_path}): eseguo prima l'analisi.")
            elif track.version < TRACK_VERSION:
                print("Traccia creata da una versione precedente del programma: rifaccio l'analisi.")
                track = None
            if track is None:
                track = pipeline.analyze(args.video, cfg, track_path, show_progress=show)

        if args.cmd in ("preview", "anteprima"):
            pipeline.preview(args.video, track, args.image, args.frame, args.out, cfg)
            print(f"Anteprima salvata in {args.out}")
            return 0

        for img, out in zip(args.image, _outputs_for(args.image, args.out)):
            pipeline.render(args.video, track, img, out, cfg, show_progress=show)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, IndexError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 1
