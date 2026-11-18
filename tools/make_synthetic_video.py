"""Generatore di video di prova con verità di riferimento (ground truth).

Simula una persona che tiene in mano un pannello verde davanti a una camera:

* pannello rettangolare 3D (default 40x50 cm) che ruota (yaw/pitch/roll),
  trasla e si avvicina/allontana, con illuminazione che cambia;
* mani che lo reggono ai lati (pollici sopra il bordo, con ombra);
* scenari di occlusione difficili: un braccio che copre completamente un
  angolo, un oggetto in primo piano che nasconde l'intero lato inferiore
  (con bordo irregolare o dritto);
* un oggetto di verde saturo nello sfondo che tocca il pannello;
* sfocatura di movimento (media di sotto-fotogrammi, otturatore regolabile),
  rumore del sensore, compressione H.264 reale e traccia audio.

Formazione dell'immagine fisicamente corretta, come in una fotocamera vera:
la scena viene composta, sfocata (movimento e obiettivo) e mescolata nei
pixel di bordo in **luce lineare**; solo alla fine si applica la curva gamma
BT.709. La conversione in YUV 4:2:0 è fatta qui, con la crominanza campionata
nella posizione standard "left" (come gli encoder degli smartphone) e
dichiarata nel file. Così i bordi del pannello nel video hanno esattamente le
caratteristiche di quelli reali e il benchmark non premia correzioni sbagliate.

Gli angoli veri del pannello (proiezione esatta del rettangolo 3D) vengono
salvati in un JSON accanto al video, così i test misurano l'errore in pixel.

Lo stesso fotogramma può essere ri-generato con un **quadro al posto del
verde** (:func:`synthesize_frame` con ``texture`` = quadro illuminato): è il
"composito perfetto" con cui ``tools/benchmark.py`` confronta il risultato
del programma. Il rumore di ogni fotogramma dipende solo da (seed, indice),
quindi le due versioni hanno esattamente la stessa grana.

Uso:
    python tools/make_synthetic_video.py --out samples/generated/test.mp4
    python tools/make_synthetic_video.py --out prova.mp4 --width 540 --height 960 --seconds 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenframe.color import YuvFormat, from_linear_np, rgb_to_yuv420, to_linear_np  # noqa: E402

# Colori in BGR (valori video 0-255, come li si vedrebbe sullo schermo).
GREEN_BGR = np.array([70.0, 175.0, 45.0])
SKIN_BGR = np.array([125.0, 160.0, 210.0])
SLEEVE_BGR = np.array([70.0, 50.0, 45.0])
# Verde "pianta": saturo quasi quanto il pannello ma di tinta più gialla.
PLANT_BGR = np.array([30.0, 150.0, 80.0])


def lin(color_bgr: np.ndarray) -> np.ndarray:
    """Colore (valori video 0-255) -> riflettanza in luce lineare (0-1)."""
    return to_linear_np(np.asarray(color_bgr, np.float32) / 255.0)


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
                 focal: float, scenarios: set[str], seed: int, motion: float = 1.0):
        self.w, self.h = width, height
        self.T = seconds
        self.pw, self.ph = panel_w, panel_h
        self.f = focal
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0
        self.scenarios = scenarios
        self.motion = motion
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
        m = self.motion  # moltiplica tutte le velocità (scenario "movimento veloce")
        yaw = np.radians(32) * np.sin(2 * np.pi * 0.23 * m * t + p[0])
        pitch = np.radians(18) * np.sin(2 * np.pi * 0.31 * m * t + p[1])
        roll = np.radians(14) * np.sin(2 * np.pi * 0.17 * m * t + p[2])
        # Un "colpo" veloce a metà video (movimento rapido della mano).
        roll += np.radians(10) * np.exp(-((u - 0.62) / 0.04) ** 2)
        x = 0.10 * np.sin(2 * np.pi * 0.19 * m * t + p[3])
        y = 0.06 * np.sin(2 * np.pi * 0.27 * m * t + p[4])
        z = self.z0 * (1.0 + 0.18 * np.sin(2 * np.pi * 0.13 * m * t + p[5]))
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

    def corners_at(self, t: float) -> np.ndarray:
        """Angoli veri (4, 2) in pixel al tempo t."""
        R, tr = self.pose(t)
        return self.project(self.corners3d(), R, tr)[0]

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


def make_background(w: int, h: int, rng: np.random.Generator, hard: bool = False,
                    green_object: bool = False) -> np.ndarray:
    """Sfondo "stanza" in luce lineare (0-1): macchie di colore, mobili, texture.

    ``green_object=True`` aggiunge una grande pianta di verde saturo proprio
    dietro il bordo destro del pannello (il pannello la tocca e la copre in
    parte durante il movimento): è il caso "verde saturo attaccato al pannello".
    """
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
    if green_object:
        # Pianta grande e satura: foglie (ellissi) di verde leggermente variabile.
        prng = np.random.default_rng(99)
        for _ in range(40):
            cx = 0.86 * w + prng.normal(0, 0.06 * w)
            cy = 0.45 * h + prng.normal(0, 0.12 * h)
            ax, ay = prng.uniform(0.03, 0.08) * w, prng.uniform(0.02, 0.05) * h
            col = PLANT_BGR * prng.uniform(0.8, 1.15)
            cv2.ellipse(bg, (int(cx), int(cy)), (int(ax), int(ay)), float(prng.uniform(0, 180)),
                        0, 360, col.tolist(), -1, cv2.LINE_AA)
    # Nota: GaussianBlur su un array (h, w, 1) restituisce (h, w): si riaggiunge l'asse.
    texture = cv2.GaussianBlur(rng.normal(0, 12, size=(h, w)).astype(np.float32), (0, 0), 1.2)
    bg = cv2.GaussianBlur(bg, (0, 0), 3) + texture[..., None]
    return to_linear_np(np.clip(bg, 0, 255) / 255.0)


def panel_illumination(tw: int, th: int) -> np.ndarray:
    """Gradiente di luce (th, tw) sulla superficie del pannello (fattore lineare)."""
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    return 1.0 + 0.12 * (xx / tw - 0.5) - 0.10 * (yy / th - 0.5)


def make_panel_texture(tw: int, th: int, rng: np.random.Generator) -> np.ndarray:
    """Radianza del pannello verde (luce lineare): gradiente di luce e trama del tessuto."""
    grad = panel_illumination(tw, th)
    weave = cv2.GaussianBlur(rng.normal(0, 4, size=(th, tw)).astype(np.float32), (0, 0), 1.0)
    albedo = GREEN_BGR[None, None, :] + weave[..., None] * np.array([0.3, 1.0, 0.3])
    return (to_linear_np(np.clip(albedo, 0, 255) / 255.0) * grad[..., None]).astype(np.float32)


def painting_texture(painting_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Il quadro come apparirebbe stampato sul pannello (stessa luce del verde)."""
    img = cv2.resize(painting_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    return to_linear_np(img.astype(np.float32) / 255.0) * panel_illumination(tw, th)[..., None]


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


def _panel_layer(tex: np.ndarray, corners: np.ndarray, w: int, h: int, ss: int):
    """Strato del pannello (radianza pre-moltiplicata e copertura), anti-aliasing ``ss``×``ss``.

    Con ``ss`` > 1 il pannello viene disegnato a risoluzione ``ss`` volte
    maggiore nel solo rettangolo che lo contiene e poi ridotto "ad area":
    ogni pixel riceve la media esatta della texture che copre (serve per il
    composito di riferimento con il quadro, ricco di dettagli fini).
    """
    th, tw = tex.shape[:2]
    src = np.array([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]], np.float32)
    if ss <= 1:
        H = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
        panel = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        pmask = cv2.warpPerspective(np.ones((th, tw), np.float32), H, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        return panel, pmask
    x0 = int(max(0, np.floor(corners[:, 0].min()) - 2))
    y0 = int(max(0, np.floor(corners[:, 1].min()) - 2))
    x1 = int(min(w, np.ceil(corners[:, 0].max()) + 3))
    y1 = int(min(h, np.ceil(corners[:, 1].max()) + 3))
    panel = np.zeros((h, w, 3), np.float32)
    pmask = np.zeros((h, w), np.float32)
    if x1 <= x0 or y1 <= y0:
        return panel, pmask
    # Coordinate del canvas sovracampionato: il centro del pixel (i, j) dell'immagine
    # corrisponde al centro del blocco ss×ss, cioè (ss*i + (ss-1)/2).
    dst = (corners - [x0, y0]) * ss + (ss - 1) / 2.0
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    size = ((x1 - x0) * ss, (y1 - y0) * ss)
    big = cv2.warpPerspective(tex, H, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bigm = cv2.warpPerspective(np.ones((th, tw), np.float32), H, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
    panel[y0:y1, x0:x1] = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    pmask[y0:y1, x0:x1] = cv2.resize(bigm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return panel, pmask


def render_frame(scene: Scene, bg: np.ndarray, tex: np.ndarray, t: float,
                 shake: np.ndarray, ss: int = 1) -> np.ndarray:
    """Rende un singolo (sotto)fotogramma in luce lineare (float32, 0-1)."""
    h, w = scene.h, scene.w
    R, tr = scene.pose(t)
    corners, _ = scene.project(scene.corners3d(), R, tr)
    panel, pmask = _panel_layer(tex * scene.brightness(R), corners, w, h, ss)

    # Sfondo con leggero tremolio della camera a mano.
    M = np.float32([[1, 0, shake[0]], [0, 1, shake[1]]])
    frame = cv2.warpAffine(bg, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    skin, dark = render_occluders(scene, t, R, tr)
    # Ombra delle mani sul pannello: maschera spostata e sfocata.
    shadow = cv2.GaussianBlur(np.roll(np.roll(skin, 10, axis=0), 6, axis=1), (0, 0), 6)
    panel = panel * (1.0 - 0.45 * np.clip(shadow, 0, 1))[..., None]

    frame = frame * (1 - pmask[..., None]) + panel  # panel è già pre-moltiplicato dalla maschera
    shade = 0.75 + 0.25 * np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    frame = frame * (1 - skin[..., None]) + (lin(SKIN_BGR) * shade) * skin[..., None]
    frame = frame * (1 - dark[..., None]) + lin(SLEEVE_BGR) * dark[..., None]
    return frame


def to_display_u8(frame_lin: np.ndarray) -> np.ndarray:
    """Luce lineare -> immagine BGR uint8 (gamma BT.709), senza rumore."""
    return np.clip(from_linear_np(frame_lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def camera_shake(t: float) -> np.ndarray:
    """Tremolio della camera (pochi pixel, lento)."""
    return np.array([3.0 * np.sin(2 * np.pi * 0.9 * t), 2.0 * np.sin(2 * np.pi * 1.3 * t + 1)])


def synthesize_frame(scene: Scene, bg: np.ndarray, texture: np.ndarray, index: int, fps: float,
                     subframes: int = 3, shutter: float = 0.5, noise: float = 2.5,
                     seed: int = 7, ss: int = 1) -> np.ndarray:
    """Fotogramma finale BGR in valori video 0-255 (float32, prima della quantizzazione).

    ``shutter`` è la frazione dell'intervallo tra fotogrammi in cui l'otturatore
    resta aperto (0.5 = 180°): la sfocatura di movimento è la media, in luce
    lineare, dei ``subframes`` istanti distribuiti in quell'intervallo. Poi
    sfocatura dell'obiettivo (in luce lineare), curva gamma e rumore, che
    dipende solo da ``(seed, index)``.
    """
    t = index / fps
    shake = camera_shake(t)
    if subframes > 1:
        offs = (np.arange(subframes) / (subframes - 1) - 0.5) * shutter / fps
    else:
        offs = np.array([0.0])
    acc = np.zeros((scene.h, scene.w, 3), np.float32)
    for dt in offs:
        acc += render_frame(scene, bg, texture, t + dt, shake, ss)
    frame = cv2.GaussianBlur(acc / len(offs), (0, 0), 0.7)      # obiettivo
    frame = from_linear_np(np.clip(frame, 0, None)) * 255.0
    rng = np.random.default_rng([seed, index])
    frame += rng.normal(0, noise, size=frame.shape).astype(np.float32)
    return frame


def build_scene(meta: dict) -> tuple[Scene, np.ndarray, np.ndarray]:
    """Ricostruisce scena, sfondo e texture verde dai metadati salvati nel JSON."""
    w, h = meta["width"], meta["height"]
    pw_cm, ph_cm = meta["panel_cm"]
    scene = Scene(w, h, meta["seconds"], pw_cm / 100, ph_cm / 100, meta["focal"],
                  set(meta["scenarios"]), meta["seed"], meta.get("motion", 1.0))
    rng = np.random.default_rng(meta["seed"])
    bg = make_background(w, h, rng, hard=meta.get("hard_background", False),
                         green_object="greenbg" in meta["scenarios"])
    tex = make_panel_texture(800, int(800 * ph_cm / pw_cm), rng)
    return scene, bg, tex


def frame_to_yuv420(frame_bgr: np.ndarray) -> bytes:
    """BGR (valori video 0-255, float) -> byte YUV 4:2:0 BT.709 "tv", crominanza "left"."""
    import torch

    rgb = torch.from_numpy(np.ascontiguousarray(frame_bgr[..., ::-1].transpose(2, 0, 1))) / 255.0
    y, u, v = rgb_to_yuv420(rgb, YuvFormat())
    return y.numpy().tobytes() + u.numpy().tobytes() + v.numpy().tobytes()


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
                         "straightedge (lato coperto da oggetto dritto), "
                         "greenbg (pianta verde satura che tocca il pannello)")
    ap.add_argument("--subframes", type=int, default=3, help="campioni per la sfocatura di movimento")
    ap.add_argument("--shutter", type=float, default=0.5,
                    help="frazione del tempo tra fotogrammi con otturatore aperto (0.5 = 180°)")
    ap.add_argument("--motion", type=float, default=1.0, help="moltiplicatore della velocità dei movimenti")
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
    if w % 2 or h % 2:
        raise SystemExit("larghezza e altezza devono essere pari (YUV 4:2:0)")
    focal = 0.6 * float(np.hypot(w, h))  # camera tipica da smartphone
    scenarios = sorted({s.strip() for s in args.scenarios.split(",") if s.strip()})
    meta = {
        "width": w, "height": h, "fps": args.fps, "seconds": args.seconds,
        "panel_cm": [pw_cm, ph_cm], "panel_aspect": pw_cm / ph_cm, "focal": focal,
        "scenarios": scenarios, "seed": args.seed, "motion": args.motion,
        "subframes": args.subframes, "shutter": args.shutter, "noise": args.noise,
        "hard_background": bool(args.hard_background),
        "linear_light": True, "chroma_location": "left", "lens_sigma": 0.7,
    }
    scene, bg, tex = build_scene(meta)

    n = int(round(args.seconds * args.fps))
    meta["frames"] = n
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg non trovato nel PATH")
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
           "-f", "lavfi", "-i", f"sine=frequency=440:duration={args.seconds}",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-chroma_sample_location", "left",
           "-c:a", "aac", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    gt = []
    for i in range(n):
        frame = synthesize_frame(scene, bg, tex, i, args.fps, args.subframes, args.shutter,
                                 args.noise, args.seed)
        proc.stdin.write(frame_to_yuv420(frame))

        t = i / args.fps
        corners = scene.corners_at(t)
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

    meta["ground_truth"] = gt
    out.with_suffix(".gt.json").write_text(json.dumps(meta))
    print(f"Creato {out} ({n} fotogrammi) e {out.with_suffix('.gt.json')}")


if __name__ == "__main__":
    main()
