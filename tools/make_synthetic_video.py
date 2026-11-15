"""Generatore di video di prova con verità di riferimento (ground truth).

Simula una persona che tiene in mano un pannello verde davanti a una camera:

* pannello rettangolare 3D (default 40x50 cm) che ruota (yaw/pitch/roll),
  trasla e si avvicina/allontana, con illuminazione che cambia;
* mani che lo reggono ai lati (pollici sopra il bordo, con ombra);
* scenari di occlusione difficili: un braccio che copre completamente un
  angolo, un oggetto in primo piano che nasconde l'intero lato inferiore;
* sfocatura di movimento (media di sotto-fotogrammi), rumore del sensore,
  compressione H.264 reale e traccia audio.

Gli angoli veri del pannello (proiezione esatta del rettangolo 3D) vengono
salvati in un JSON accanto al video, così i test misurano l'errore in pixel.

Uso:
    python tools/make_synthetic_video.py --out samples/generated/test.mp4
    python tools/make_synthetic_video.py --out prova.mp4 --width 540 --height 960 --seconds 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# Colori in BGR.
GREEN_BGR = np.array([70.0, 175.0, 45.0])
SKIN_BGR = np.array([125.0, 160.0, 210.0])
SLEEVE_BGR = np.array([70.0, 50.0, 45.0])


def rot_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Matrice di rotazione da angoli in radianti (Rz · Ry · Rx)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ ry @ rx


class Scene:
    """Descrizione parametrica della scena in funzione del tempo."""

    def __init__(self, width: int, height: int, seconds: float, panel_w: float, panel_h: float,
                 focal: float, scenarios: set[str], seed: int):
        self.w, self.h = width, height
        self.T = seconds
        self.pw, self.ph = panel_w, panel_h
        self.f = focal
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0
        self.scenarios = scenarios
        rng = np.random.default_rng(seed)
        # Fasi casuali per rendere ogni video un po' diverso.
        self.ph_ = rng.uniform(0, 2 * np.pi, size=8)
        # Distanza tale che il pannello occupi ~55% del fotogramma nella
        # dimensione più "stretta" (vale sia per video verticali che orizzontali).
        self.z0 = self.f * max(panel_w / (0.55 * width), panel_h / (0.55 * height))

    def pose(self, t: float):
        """Rotazione e traslazione del pannello al tempo t (secondi)."""
        p = self.ph_
        u = t / self.T
        yaw = np.radians(32) * np.sin(2 * np.pi * 0.23 * t + p[0])
        pitch = np.radians(18) * np.sin(2 * np.pi * 0.31 * t + p[1])
        roll = np.radians(14) * np.sin(2 * np.pi * 0.17 * t + p[2])
        # Un "colpo" veloce a metà video (movimento rapido della mano).
        roll += np.radians(10) * np.exp(-((u - 0.62) / 0.04) ** 2)
        x = 0.10 * np.sin(2 * np.pi * 0.19 * t + p[3])
        y = 0.06 * np.sin(2 * np.pi * 0.27 * t + p[4])
        z = self.z0 * (1.0 + 0.18 * np.sin(2 * np.pi * 0.13 * t + p[5]))
        return rot_matrix(yaw, pitch, roll), np.array([x, y, z])

    def project(self, pts3: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Proietta punti 3D del sistema pannello (metri) in pixel."""
        cam = (R @ pts3.T).T + t
        return np.stack([self.f * cam[:, 0] / cam[:, 2] + self.cx,
                         self.f * cam[:, 1] / cam[:, 2] + self.cy], axis=1), cam[:, 2]

    def corners3d(self) -> np.ndarray:
        """Angoli 3D TL, TR, BR, BL (y verso il basso, come nell'immagine)."""
        a, b = self.pw / 2, self.ph / 2
        return np.array([[-a, -b, 0], [a, -b, 0], [a, b, 0], [-a, b, 0]], dtype=np.float64)

    def brightness(self, R: np.ndarray) -> float:
        """Illuminazione lambertiana: cambia quando il pannello ruota."""
        normal = R @ np.array([0, 0, -1.0])
        to_light = np.array([-0.35, -0.55, -1.0])
        to_light /= np.linalg.norm(to_light)
        return 0.55 + 0.45 * max(0.0, float(normal @ to_light))


def random_room_color(rng: np.random.Generator, hard: bool) -> np.ndarray:
    """Colore BGR casuale di una stanza: toni caldi/neutri, niente verdi.

    Con ``hard=True`` sono ammessi anche verdi spenti (stress test).
    """
    if hard:
        return rng.uniform(30, 220, size=3)
    # Tonalità OpenCV (0-180): il verde sta circa tra 35 e 95, la escludiamo.
    hue = rng.choice([rng.uniform(0, 30), rng.uniform(100, 180)])
    hsv = np.uint8([[[hue, rng.uniform(15, 150), rng.uniform(70, 230)]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float64)


def make_background(w: int, h: int, rng: np.random.Generator, hard: bool = False) -> np.ndarray:
    """Sfondo "stanza": macchie di colore a bassa frequenza, mobili, texture."""
    low = np.array([[random_room_color(rng, hard) for _ in range(4)] for _ in range(6)], np.float32)
    bg = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(7):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = x0 + rng.integers(w // 8, w // 2), y0 + rng.integers(h // 10, h // 3)
        color = random_room_color(rng, hard)
        cv2.rectangle(bg, (int(x0), int(y0)), (int(x1), int(y1)), color.tolist(), -1, cv2.LINE_AA)
    # Tocco di verde nello sfondo (una pianta): il rilevatore non deve confondersi.
    cv2.ellipse(bg, (int(0.12 * w), int(0.9 * h)), (int(0.05 * w), int(0.04 * h)), 0, 0, 360,
                (40, 120, 50), -1, cv2.LINE_AA)
    # Nota: GaussianBlur su un array (h, w, 1) restituisce (h, w): si riaggiunge l'asse.
    texture = cv2.GaussianBlur(rng.normal(0, 12, size=(h, w)).astype(np.float32), (0, 0), 1.2)
    bg = cv2.GaussianBlur(bg, (0, 0), 3) + texture[..., None]
    return np.clip(bg, 0, 255).astype(np.float32)


def make_panel_texture(tw: int, th: int, rng: np.random.Generator) -> np.ndarray:
    """Texture del pannello verde: gradiente di luce e leggera trama del tessuto."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    grad = 1.0 + 0.12 * (xx / tw - 0.5) - 0.10 * (yy / th - 0.5)
    weave = cv2.GaussianBlur(rng.normal(0, 4, size=(th, tw)).astype(np.float32), (0, 0), 1.0)
    tex = GREEN_BGR[None, None, :] * grad[..., None] + weave[..., None] * np.array([0.3, 1.0, 0.3])
    return tex.astype(np.float32)


def draw_capsule(mask: np.ndarray, p: np.ndarray, q: np.ndarray, radius: float, ss: int = 4):
    """Disegna una capsula (segmento spesso) anti-aliased con coordinate sub-pixel."""
    r = max(1, int(round(radius)))
    cv2.line(mask, tuple(np.round(p * ss).astype(int)), tuple(np.round(q * ss).astype(int)),
             1.0, thickness=2 * r, lineType=cv2.LINE_AA, shift=2)


def render_occluders(scene: Scene, t: float, R, tr) -> tuple[np.ndarray, np.ndarray]:
    """Maschere (H, W) float delle mani (pelle) e della manica/oggetto scuro."""
    h, w = scene.h, scene.w
    skin = np.zeros((h, w), np.float32)
    dark = np.zeros((h, w), np.float32)
    pw, ph = scene.pw, scene.ph
    u = t / scene.T

    if "hold" in scene.scenarios:
        # Due mani che reggono i lati sinistro e destro: palmo fuori dal bordo,
        # pollice sopra la superficie verde.
        for side in (-1, 1):
            palm = np.array([[side * (pw / 2 + 0.035), 0.04, 0.0]])
            thumb_base = np.array([[side * (pw / 2 + 0.01), 0.02, -0.005]])
            thumb_tip = np.array([[side * (pw / 2 - 0.045), -0.015, -0.01]])
            pts, z = scene.project(np.vstack([palm, thumb_base, thumb_tip]), R, tr)
            r_palm = scene.f * 0.05 / z[0]
            cv2.ellipse(skin, tuple(np.round(pts[0] * 4).astype(int)),
                        (int(r_palm * 4 * 0.8), int(r_palm * 4 * 1.25)), 0, 0, 360, 1.0, -1,
                        cv2.LINE_AA, shift=2)
            draw_capsule(skin, pts[1], pts[2], scene.f * 0.012 / z[1])

    if "corner" in scene.scenarios and 0.30 < u < 0.52:
        # Un braccio entra dall'alto a sinistra, copre del tutto l'angolo TL e riesce.
        k = (u - 0.30) / 0.22
        c, _ = scene.project(scene.corners3d()[:1], R, tr)
        tip = c[0] + np.array([1.0, 0.8]) * (-0.12 + 0.30 * np.sin(np.pi * k)) * min(w, h)
        shoulder = np.array([-0.3 * w, -0.2 * h])
        draw_capsule(skin, shoulder, tip, 0.055 * min(w, h))

    if ("bottom" in scene.scenarios or "straightedge" in scene.scenarios) and 0.70 < u < 0.80:
        # Oggetto scuro in primo piano che nasconde l'intero lato inferiore.
        c, _ = scene.project(scene.corners3d(), R, tr)
        y_bottom = max(c[2, 1], c[3, 1])
        panel_h_px = c[2, 1] - c[1, 1]
        y_top = y_bottom - 0.10 * panel_h_px
        if "straightedge" in scene.scenarios:
            # Bordo dritto: caso ambiguo (limite noto del metodo).
            cv2.rectangle(dark, (0, int(y_top)), (w - 1, int(y_bottom + 0.08 * h)), 1.0, -1)
        else:
            # Bordo irregolare (es. coperta, gruppo di dita, fogliame): il lato
            # risulta davvero nascosto e va ricostruito.
            xs = np.arange(0, w + 8, 8, dtype=np.float64)
            panel_w_px = abs(c[1, 0] - c[0, 0])
            period = max(panel_w_px / 5, 8)
            # Somma di onde con periodi incommensurabili: bordo irregolare, le
            # creste non risultano allineate su una retta (come accade nella realtà).
            ys = y_top + 0.04 * panel_h_px * (np.sin(2 * np.pi * xs / period)
                                             + 0.7 * np.sin(2 * np.pi * xs / (period * 0.37) + 1.3)
                                             + 0.5 * np.sin(2 * np.pi * xs / (period * 2.71) + 0.4))
            poly = np.vstack([np.stack([xs, ys], 1), [[w, h + 10], [0, h + 10]]])
            cv2.fillPoly(dark, [np.round(poly * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)

    skin = cv2.GaussianBlur(skin, (0, 0), 0.8)
    dark = cv2.GaussianBlur(dark, (0, 0), 0.8)
    return skin, dark


def render_frame(scene: Scene, bg: np.ndarray, tex: np.ndarray, t: float,
                 shake: np.ndarray) -> np.ndarray:
    """Rende un singolo (sotto)fotogramma in float32."""
    h, w = scene.h, scene.w
    R, tr = scene.pose(t)
    corners, _ = scene.project(scene.corners3d(), R, tr)
    th, tw = tex.shape[:2]
    src = np.array([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]], np.float32)
    H = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    panel = cv2.warpPerspective(tex * scene.brightness(R), H, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    pmask = cv2.warpPerspective(np.ones((th, tw), np.float32), H, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)

    # Sfondo con leggero tremolio della camera a mano.
    M = np.float32([[1, 0, shake[0]], [0, 1, shake[1]]])
    frame = cv2.warpAffine(bg, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    skin, dark = render_occluders(scene, t, R, tr)
    # Ombra delle mani sul pannello: maschera spostata e sfocata.
    shadow = cv2.GaussianBlur(np.roll(np.roll(skin, 10, axis=0), 6, axis=1), (0, 0), 6)
    panel = panel * (1.0 - 0.35 * np.clip(shadow, 0, 1))[..., None]

    frame = frame * (1 - pmask[..., None]) + panel  # panel è già pre-moltiplicato dalla maschera
    shade = 0.85 + 0.15 * np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    frame = frame * (1 - skin[..., None]) + (SKIN_BGR * shade) * skin[..., None]
    frame = frame * (1 - dark[..., None]) + SLEEVE_BGR * dark[..., None]
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="file .mp4 di uscita (il JSON viene scritto accanto)")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--panel-cm", default="40x50", help="dimensioni del pannello LxA in cm")
    ap.add_argument("--scenarios", default="hold,corner,bottom",
                    help="occlusioni: hold (mani ai lati), corner (angolo coperto), "
                         "bottom (lato coperto da oggetto irregolare), "
                         "straightedge (lato coperto da oggetto dritto: limite noto)")
    ap.add_argument("--subframes", type=int, default=3, help="campioni per la sfocatura di movimento")
    ap.add_argument("--noise", type=float, default=2.5, help="deviazione standard del rumore (0-255)")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hard-background", action="store_true",
                    help="sfondo con verdi spenti accanto al pannello (stress test)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pw_cm, ph_cm = (float(v) for v in args.panel_cm.lower().split("x"))
    w, h = args.width, args.height
    focal = 0.6 * float(np.hypot(w, h))  # camera tipica da smartphone
    scenarios = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    scene = Scene(w, h, args.seconds, pw_cm / 100, ph_cm / 100, focal, scenarios, args.seed)
    rng = np.random.default_rng(args.seed)
    bg = make_background(w, h, rng, hard=args.hard_background)
    tex = make_panel_texture(800, int(800 * ph_cm / pw_cm), rng)

    n = int(round(args.seconds * args.fps))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg non trovato nel PATH")
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
           "-f", "lavfi", "-i", f"sine=frequency=440:duration={args.seconds}",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
           "-c:a", "aac", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    gt = []
    shutter = 0.5  # frazione dell'intervallo tra fotogrammi in cui l'otturatore è aperto
    for i in range(n):
        t = i / args.fps
        # Tremolio della camera (pochi pixel, lento).
        shake = np.array([3.0 * np.sin(2 * np.pi * 0.9 * t), 2.0 * np.sin(2 * np.pi * 1.3 * t + 1)])
        if args.subframes > 1:
            offs = (np.arange(args.subframes) / (args.subframes - 1) - 0.5) * shutter / args.fps
        else:
            offs = np.array([0.0])
        acc = np.zeros((h, w, 3), np.float32)
        for dt in offs:
            acc += render_frame(scene, bg, tex, t + dt, shake)
        frame = acc / len(offs)
        # Ottica leggermente morbida + rumore del sensore.
        frame = cv2.GaussianBlur(frame, (0, 0), 0.7)
        frame += rng.normal(0, args.noise, size=frame.shape).astype(np.float32)
        proc.stdin.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())

        R, tr = scene.pose(t)
        corners, _ = scene.project(scene.corners3d(), R, tr)
        u = t / args.seconds
        gt.append({
            "frame": i,
            "corners": corners.round(3).tolist(),
            # Etichette utili ai test per valutare separatamente i casi difficili.
            "corner_occluded": bool("corner" in scenarios and 0.30 < u < 0.52),
            "bottom_occluded": bool(("bottom" in scenarios or "straightedge" in scenarios)
                                    and 0.70 < u < 0.80),
        })
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg ha restituito un errore")

    meta = {
        "width": w, "height": h, "fps": args.fps, "frames": n,
        "panel_aspect": pw_cm / ph_cm, "focal": focal, "scenarios": sorted(scenarios),
        "ground_truth": gt,
    }
    out.with_suffix(".gt.json").write_text(json.dumps(meta))
    print(f"Creato {out} ({n} fotogrammi) e {out.with_suffix('.gt.json')}")


if __name__ == "__main__":
    main()
