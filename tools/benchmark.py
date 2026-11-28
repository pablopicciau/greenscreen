"""Benchmark di precisione e velocità su video sintetici con verità di riferimento.

Per ogni scenario genera (una volta sola, poi li riusa) un video sintetico in
``samples/generated/bench/``, esegue l'analisi e misura:

* **errore sugli angoli** rispetto alla posizione vera, separato per
  situazione (libero, angolo coperto, lato coperto);
* **tremolio**: variazione seconda dell'errore (quanto "balla" il quadro
  inserito tra un fotogramma e l'altro, anche se la posizione media è giusta);
* rapporto d'aspetto, focale e otturatore stimati;
* tempi di analisi e di rendering;
* con ``--render``: **qualità del risultato finale**, confrontando ogni
  fotogramma prodotto con il "composito perfetto" (lo stesso fotogramma
  sintetico con il quadro stampato sul pannello, stessa luce, stesse mani,
  stessa sfocatura di movimento e stessa grana). PSNR in dB, più alto è
  meglio; "bordi" considera solo la fascia di ±3 px attorno ai lati veri,
  dove si vedono gli errori di linea, di chiave e di motion blur.

Scenari "rivelazione" (``tools/make_reveal_video.py``): persona di spalle che
si gira e mostra un quadro grande. Gli errori sono separati per fase (quadro
di taglio, in rotazione, frontale) e con ``--render`` si misurano anche:

* **lampo verde**: pixel di verde rimasti nel risultato sulla sagoma della
  tela (fronte e lati), fotogramma per fotogramma;
* **falsi inserimenti**: fotogrammi in cui il fronte non è visibile (si vede
  il retro) ma il programma ha modificato l'immagine (più di 50 pixel cambiati
  di oltre 8 livelli). Un quad presente su un fotogramma di retro non basta:
  il rendering sostituisce solo dove c'è il verde del pannello.

Uso:
    python tools/benchmark.py                       # tutti gli scenari, Full HD, 15 s
    python tools/benchmark.py --quick               # 540x960, 4 s
    python tools/benchmark.py --scenarios standard,veloce --render --tag dopo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import make_reveal_video as mrv  # noqa: E402
import make_synthetic_video as msv  # noqa: E402
import make_test_painting as mtp  # noqa: E402

# GREENFRAME_ROOT permette di misurare un'altra versione del programma (es.
# quella originale estratta da git) sugli stessi video.
import os  # noqa: E402

if os.environ.get("GREENFRAME_ROOT"):
    for mod in [m for m in sys.modules if m == "greenframe" or m.startswith("greenframe.")]:
        del sys.modules[mod]
    sys.path.insert(0, os.environ["GREENFRAME_ROOT"])

from greenframe import geometry as geo  # noqa: E402
from greenframe import pipeline  # noqa: E402
from greenframe.config import Config  # noqa: E402

print(f"[bench] greenframe da: {Path(pipeline.__file__).parent}")


def _render_frames_compat(video, track, painting, cfg):
    """``pipeline.render_frames`` oppure, per la versione originale, il ciclo equivalente."""
    if hasattr(pipeline, "render_frames"):
        for i, src, res in pipeline.render_frames(str(video), track, str(painting), cfg, log=lambda m: None):
            yield i, src, (res.bgr if hasattr(res, "bgr") else res)
        return
    from greenframe.compositor import Compositor
    from greenframe.image_fit import load_image
    from greenframe.video_io import FrameReader, probe

    info = probe(str(video))
    comp = Compositor(load_image(str(painting)), cfg.render.panel_aspect or track.panel_aspect,
                      cfg, pipeline.reference_luma(track))
    with FrameReader(info) as reader:
        for i, frame in enumerate(reader):
            quad = track.smooth[i] if i < track.n_frames else None
            yield i, frame, comp.render(frame, quad)

# Argomenti del generatore per ogni scenario.
PRESETS = {
    "standard": ["--scenarios", "hold,corner,bottom"],
    "bordo_dritto": ["--scenarios", "hold,straightedge"],
    "verde_sfondo": ["--scenarios", "hold,greenbg"],
    "veloce": ["--scenarios", "hold", "--motion", "2.5", "--subframes", "9", "--shutter", "0.5"],
}

# Scenari "rivelazione" (persona di spalle che si gira e mostra un quadro grande).
REVEAL_PRESETS = {
    "rivelazione": ["--panel-cm", "100x120", "--fill", "0.65"],
    "rivelazione_sinistra": ["--panel-cm", "100x120", "--fill", "0.65", "--direction", "-1"],
    "rivelazione_grande": ["--panel-cm", "120x150", "--fill", "0.9"],
    "rivelazione_orizzontale": ["--panel-cm", "150x100", "--fill", "1.0"],
    "rivelazione_lati_verdi": ["--panel-cm", "100x120", "--fill", "0.65", "--green-sides"],
    "rivelazione_veloce": ["--panel-cm", "100x120", "--fill", "0.65", "--turn", "0.30,0.50",
                           "--subframes", "9"],
}
REVEAL_PHASES = ("taglio", "rotazione", "frontale")


def _generator(meta: dict):
    """Modulo che ha generato il video (per ricostruire la scena)."""
    return mrv if meta.get("generator") == "reveal" else msv


def ensure_video(name: str, outdir: Path, width: int, height: int, seconds: float) -> Path:
    """Genera il video dello scenario se non esiste già con gli stessi parametri."""
    video = outdir / f"{name}_{width}x{height}_{seconds:g}s.mp4"
    meta_path = video.with_suffix(".gt.json")
    if video.exists() and meta_path.exists():
        return video
    print(f"[bench] genero {video.name} ...", flush=True)
    t0 = time.time()
    gen, presets = (mrv, REVEAL_PRESETS) if name in REVEAL_PRESETS else (msv, PRESETS)
    gen.main(["--out", str(video), "--width", str(width), "--height", str(height),
              "--seconds", str(seconds)] + presets[name])
    print(f"[bench]   generato in {time.time() - t0:.0f} s", flush=True)
    return video


def ensure_painting(outdir: Path) -> Path:
    path = outdir / "quadro_bench.png"
    if not path.exists():
        mtp.main(["--out", str(path), "--style", "art", "--width", "1600", "--height", "2000"])
    return path


def track_metrics(track, meta: dict) -> dict:
    """Errori degli angoli finali rispetto alla verità."""
    gt = meta["ground_truth"]
    truth = np.array([g["corners"] for g in gt])
    est = track.smooth
    n = min(len(truth), len(est))
    truth, est = truth[:n], est[:n]
    err = np.linalg.norm(est - truth, axis=2)                       # (T, 4)
    valid = ~np.isnan(err).any(axis=1)
    out = {"frames": int(n), "lost": int((~valid).sum()), "groups": {}}
    if meta.get("generator") == "reveal":
        phase = np.array([g["phase"] for g in gt[:n]])
        visible = phase != "retro"
        groups = {p: phase == p for p in REVEAL_PHASES}
        groups["tutti (visibile)"] = visible
        # Persi: fronte visibile ma nessun quad. Falsi: si vede il retro ma c'è un quad.
        out["lost"] = int((visible & ~valid).sum())
        out["lost_by_phase"] = {p: int(((phase == p) & ~valid).sum()) for p in REVEAL_PHASES}
        out["false_quads"] = int((~visible & valid).sum())
        out["first_visible"] = int(np.argmax(visible)) if visible.any() else -1
        out["first_tracked"] = int(np.argmax(visible & valid)) if (visible & valid).any() else -1
    else:
        corner = np.array([g["corner_occluded"] for g in gt[:n]])
        bottom = np.array([g["bottom_occluded"] for g in gt[:n]])
        groups = {"libero": ~(corner | bottom), "angolo coperto": corner,
                  "lato coperto": bottom, "tutti": np.ones(n, bool)}
    for gname, sel in groups.items():
        s = sel & valid
        if not s.any():
            continue
        worst = err[s].max(axis=1)
        out["groups"][gname] = {
            "n": int(s.sum()),
            "mean": float(err[s].mean()),
            "p50": float(np.median(worst)),
            "p95": float(np.percentile(worst, 95)),
            "max": float(worst.max()),
        }
    e = est - truth
    d2 = e[2:] - 2 * e[1:-1] + e[:-2]
    ok = valid[2:] & valid[1:-1] & valid[:-2]
    out["jitter"] = float(np.linalg.norm(d2[ok], axis=2).mean()) if ok.any() else float("nan")
    out["aspect"] = track.panel_aspect
    out["aspect_true"] = meta["panel_aspect"]
    out["focal"] = track.focal
    out["focal_true"] = meta["focal"]
    out["shutter"] = getattr(track, "shutter", None)
    out["shutter_true"] = meta.get("shutter")
    return out


def _frame_changed(i: int, track) -> bool:
    """True se il programma modifica il fotogramma ``i`` (c'è un quad valido)."""
    sm = track.smooth
    return sm is not None and i < len(sm) and not np.isnan(sm[i]).any()


def _poly_mask(quad: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(quad * 16).astype(np.int32), 1, cv2.LINE_8, shift=4)
    return m.astype(bool)


def render_metrics(video: Path, track, painting: Path, meta: dict, cfg: Config,
                   step: int = 3) -> dict:
    """PSNR del risultato rispetto al composito perfetto (un fotogramma ogni ``step``)."""
    gen = _generator(meta)
    reveal = gen is mrv
    scene, bg, _tex = gen.build_scene(meta)
    img = cv2.imread(str(painting))
    # Il quadro va portato alle proporzioni del pannello come fa il programma
    # (fill/fit): altrimenti, con un pannello di proporzioni diverse dal quadro
    # (es. 100x120 contro 4:5), il riferimento sarebbe deformato e il PSNR
    # misurerebbe la differenza di ritaglio invece della qualità.
    from greenframe.image_fit import fit_to_aspect
    img = fit_to_aspect(img, meta["panel_aspect"], cfg.render.mode, cfg.render.fit_background)
    # Quadro alla sua risoluzione piena: il riferimento è reso con anti-aliasing 3×3.
    ptex = msv.painting_texture(img, img.shape[1], img.shape[0])
    truth = np.array([g["corners"] for g in meta["ground_truth"]])
    se_reg = n_reg = se_edge = n_edge = se_hand = n_hand = 0.0
    psnr_frames = []
    phases = [g.get("phase", "") for g in meta["ground_truth"]]
    leak = np.zeros(len(truth), np.int64)          # pixel verdi rimasti (lampo verde)
    changed_retro = []                             # fotogrammi "retro" modificati
    for i, src, out in _render_frames_compat(video, track, painting, cfg):
        if i >= len(truth):
            continue
        if reveal:
            # Verde residuo sulla sagoma della tela (fronte e lati), in ogni fotogramma.
            sil = cv2.convexHull(scene.silhouette_at(i / meta["fps"]).astype(np.float32)).reshape(-1, 2)
            zone = _poly_mask(geo.offset_quad(sil, 2.0) if len(sil) == 4 else sil, out.shape[:2])
            zone = cv2.dilate(zone.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            o16 = out.astype(np.int16)
            green = (o16[..., 1] - np.maximum(o16[..., 0], o16[..., 2]) > 35) & (o16[..., 1] > 50)
            leak[i] = int((green & zone).sum())
            if phases[i] == "retro":
                # Si vede il retro: il programma non deve inserire nulla
                # (conta il risultato, non la sola presenza di un quad).
                if _frame_changed(i, track):
                    diff = np.abs(out.astype(np.int16) - src.astype(np.int16)).max(axis=2)
                    if int((diff > 8).sum()) > 50:
                        changed_retro.append(i)
                continue
        if i % step:
            continue
        ref = gen.synthesize_frame(scene, bg, ptex, i, meta["fps"], meta["subframes"],
                                   meta["shutter"], meta["noise"], meta["seed"], ss=3)
        ref = np.clip(ref + 0.5, 0, 255).astype(np.uint8).astype(np.float64)
        q = truth[i]
        shape = out.shape[:2]
        region = _poly_mask(geo.offset_quad(q, 6.0), shape)
        edge = region & ~_poly_mask(geo.offset_quad(q, -3.0), shape) & _poly_mask(geo.offset_quad(q, 3.0), shape)
        o = out.astype(np.float64)
        # Guadagno globale (la luminosità di riferimento è una scelta, non un errore di forma).
        g = float((o[region] * ref[region]).sum() / max((ref[region] ** 2).sum(), 1e-9))
        d2 = (o - g * ref) ** 2
        se_reg += d2[region].sum(); n_reg += region.sum() * 3
        se_edge += d2[edge].sum(); n_edge += edge.sum() * 3
        if not reveal:
            # Bordi delle mani sopra il quadro: fascia di ±2 px attorno alla pelle, dentro il quadro.
            R, tr = scene.pose(i / meta["fps"])
            skin, dark = msv.render_occluders(scene, i / meta["fps"], R, tr)
            occ = np.maximum(skin, dark)
            hand_edge = (occ > 0.02) & (occ < 0.98)
            hand_edge = cv2.dilate(hand_edge.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            hand_edge &= _poly_mask(geo.offset_quad(q, -2.0), shape)
            se_hand += d2[hand_edge].sum(); n_hand += hand_edge.sum() * 3
        if region.any():
            psnr_frames.append((i, 10 * np.log10(255 ** 2 / max(d2[region].mean(), 1e-9))))
    def psnr(se, n):
        return float(10 * np.log10(255 ** 2 / max(se / max(n, 1), 1e-9)))
    res = {"psnr": psnr(se_reg, n_reg), "psnr_edge": psnr(se_edge, n_edge),
           "psnr_hands": psnr(se_hand, n_hand) if n_hand else float("nan"),
           "psnr_worst": float(min(p for _, p in psnr_frames)) if psnr_frames else float("nan"),
           "worst_frame": int(min(psnr_frames, key=lambda x: x[1])[0]) if psnr_frames else -1,
           "frames": len(psnr_frames)}
    if reveal:
        ph = np.array(phases[:len(leak)])
        res["green_leak"] = {p: {"max_px": int(leak[ph == p].max()) if (ph == p).any() else 0,
                                 "frames_over_50px": int((leak[ph == p] > 50).sum())}
                             for p in ("retro",) + REVEAL_PHASES}
        res["green_leak_worst_frame"] = int(np.argmax(leak))
        res["false_insertions"] = len(changed_retro)
    return res


def run(args) -> dict:
    outdir = ROOT / "samples" / "generated" / "bench"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        width, height, seconds = 540, 960, 4.0
    else:
        width, height, seconds = args.width, args.height, args.seconds
    painting = ensure_painting(outdir)
    results = {"tag": args.tag, "size": [width, height], "seconds": seconds, "scenarios": {}}
    names = []
    for name in args.scenarios.split(","):
        name = name.strip()
        names += list(REVEAL_PRESETS) if name == "rivelazioni" else [name]
    for name in names:
        video = ensure_video(name, outdir, width, height, seconds)
        meta = json.loads(video.with_suffix(".gt.json").read_text())
        cfg = Config()
        cfg.render.lighting = 1.0
        if args.aspetto_vero:
            # Come se l'utente indicasse le misure del quadro (--aspect LxA).
            cfg.render.panel_aspect = meta["panel_aspect"]
        t0 = time.time()
        track = pipeline.analyze(str(video), cfg, str(video.with_suffix(f".{args.tag}.track.json")),
                                 log=lambda m: None, show_progress=False)
        t_an = time.time() - t0
        res = track_metrics(track, meta)
        res["t_analysis"] = t_an
        if args.time_render:
            t0 = time.time()
            pipeline.render(str(video), track, str(painting), str(outdir / f"{name}.{args.tag}.mp4"),
                            cfg, log=lambda m: None, show_progress=False)
            res["t_render"] = time.time() - t0
        if args.render:
            res["render"] = render_metrics(video, track, painting, meta, cfg, step=args.step)
        results["scenarios"][name] = res
        print_scenario(name, res, flush=True)
    path = outdir / f"results_{args.tag}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"[bench] risultati salvati in {path}")
    return results


def print_scenario(name: str, r: dict, flush: bool = False) -> None:
    print(f"\n== {name} ==  fotogrammi {r['frames']}, persi {r['lost']}, analisi {r['t_analysis']:.1f} s"
          + (f", rendering {r['t_render']:.1f} s" if "t_render" in r else ""))
    for g, v in r["groups"].items():
        print(f"  {g:15s} n={v['n']:4d}  media {v['mean']:.3f} px  peggiore: mediana {v['p50']:.2f} "
              f"p95 {v['p95']:.2f} max {v['max']:.2f}")
    print(f"  tremolio {r['jitter']:.3f} px   aspetto {r['aspect']:.4f} (vero {r['aspect_true']:.4f})   "
          f"focale {r['focal']:.0f} (vera {r['focal_true']:.0f})"
          + (f"   otturatore {r['shutter']:.2f} (vero {r['shutter_true']:.2f})" if r.get("shutter") else ""))
    if "lost_by_phase" in r:
        print(f"  persi per fase {r['lost_by_phase']}   quad su fotogrammi di retro: {r['false_quads']}   "
              f"fronte visibile dal fotogramma {r['first_visible']}, agganciato dal {r['first_tracked']}")
    if "render" in r:
        rr = r["render"]
        if "green_leak" in rr:
            print(f"  lampo verde (px max / fotogrammi oltre 50 px): "
                  + ", ".join(f"{k} {v['max_px']}/{v['frames_over_50px']}" for k, v in rr["green_leak"].items())
                  + f"   peggiore: fotogramma {rr['green_leak_worst_frame']}   "
                  f"falsi inserimenti: {rr['false_insertions']}")
        print(f"  composito: PSNR {rr['psnr']:.2f} dB, bordi {rr['psnr_edge']:.2f} dB, "
              f"bordi delle mani {rr.get('psnr_hands', float('nan')):.2f} dB, "
              f"fotogramma peggiore {rr['psnr_worst']:.2f} dB ({rr['frames']} fotogrammi)")
    if flush:
        sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=",".join(PRESETS),
                    help="scenari separati da virgola; 'rivelazioni' = tutti gli scenari di rivelazione "
                         f"({', '.join(REVEAL_PRESETS)})")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--quick", action="store_true", help="540x960, 4 s")
    ap.add_argument("--render", action="store_true", help="misura anche la qualità del composito")
    ap.add_argument("--time-render", action="store_true", help="misura il tempo del rendering completo")
    ap.add_argument("--step", type=int, default=3, help="con --render: un fotogramma ogni N")
    ap.add_argument("--tag", default="run", help="etichetta dei risultati (es. prima, dopo)")
    ap.add_argument("--aspetto-vero", action="store_true",
                    help="usa le proporzioni vere del pannello (come --aspect indicato dall'utente)")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
