"""Scena di prova "reveal": persona di spalle che si gira e mostra un quadro grande.

È il Reel tipico degli artisti su Instagram: all'inizio la persona è di spalle
e regge il quadro davanti a sé (la camera vede la schiena e il **retro** della
tela con il telaio in legno), poi si gira lentamente e il quadro compare,
prima **di taglio** (una striscia sottile, vista quasi di lato), poi sempre più
frontale, infine viene mostrato in camera.

Cosa rende difficile questa scena (e quindi cosa misura):

* il verde compare gradualmente, partendo da una striscia sottilissima con una
  prospettiva estrema: se il programma lo aggancia tardi, il verde resta
  visibile proprio nel momento della rivelazione ("lampo verde");
* la tela ha uno **spessore**: durante la rotazione si vedono i lati (tela
  grezza, oppure verdi se l'artista li ha dipinti: variante ``--green-sides``);
* il **corpo** copre il retro del quadro all'inizio, il quadro copre il corpo
  alla fine; braccia e mani reggono i lati, con le dita sul fronte;
* un quadro grande riempie l'inquadratura e può **uscire dai bordi**
  (``--fill`` vicino a 1);
* la luce sul quadro cambia molto durante la rotazione.

Geometria: camera fissa (cavalletto) nell'origine, asse Z in avanti, Y verso
il basso. La persona ruota attorno al proprio asse verticale: l'angolo va da
180° (di spalle) a 0° (di fronte) con un'accelerazione e una frenata morbide.
Il quadro è una scatola 3D (fronte, retro, quattro lati); corpo, testa,
braccia e mani sono primitive semplici. La visibilità è risolta per pixel con
uno z-buffer (ordine di profondità + test di profondità).

Formazione dell'immagine come in ``make_synthetic_video.py``: composizione,
sfocatura di movimento (media di sotto-fotogrammi) e dell'obiettivo in luce
lineare, curva gamma BT.709, YUV 4:2:0 con crominanza "left", H.264.

Ogni fotogramma del JSON di verità contiene gli angoli veri del **fronte**
della tela, l'angolo di vista (0° = frontale, 90° = di taglio), i pixel di
fronte realmente visibili e la fase: ``retro`` (fronte non visibile),
``taglio`` (oltre 65°), ``rotazione`` (30-65°), ``frontale`` (sotto 30°).

Uso:
    python tools/make_reveal_video.py --out samples/generated/reveal.mp4
    python tools/make_reveal_video.py --out r.mp4 --panel-cm 120x150 --fill 0.9 --direction -1
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_synthetic_video as msv  # noqa: E402
from greenframe.color import from_linear_np  # noqa: E402

# Colori in BGR (valori video 0-255).
SHIRT_BGR = np.array([62.0, 52.0, 48.0])      # maglia grigio scuro
PANTS_BGR = np.array([95.0, 72.0, 58.0])      # pantaloni
HAIR_BGR = np.array([28.0, 38.0, 58.0])       # capelli castano scuro
CANVAS_BGR = np.array([196.0, 212.0, 222.0])  # tela grezza (lati e retro)
WOOD_BGR = np.array([88.0, 128.0, 168.0])     # telaio in legno

# Identificativi degli elementi nella mappa di visibilità (verità di riferimento).
ID_NONE, ID_FRONT, ID_OTHER = 0, 1, 2


def smoothstep(x: float) -> float:
    """Rampa morbida 0->1 (accelerazione e frenata naturali)."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class RevealScene:
    """Persona che si gira con un quadro grande: geometria in funzione del tempo."""

    def __init__(self, width: int, height: int, seconds: float, panel_w: float, panel_h: float,
                 focal: float, seed: int, direction: int = 1, fill: float = 0.65,
                 depth: float = 0.035, green_sides: bool = False,
                 turn: tuple[float, float] = (0.18, 0.62)):
        self.w, self.h = width, height
        self.T = seconds
        self.pw, self.ph = panel_w, panel_h
        self.depth = depth
        self.f = focal
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0
        self.direction = 1 if direction >= 0 else -1
        self.green_sides = green_sides
        self.turn = turn
        rng = np.random.default_rng(seed)
        self.ph_ = rng.uniform(0, 2 * np.pi, size=6)
        # Distanza tale che, frontale, il quadro copra ``fill`` della larghezza.
        self.z0 = self.f * panel_w / (fill * width)
        self.hold = 0.30        # distanza del quadro dall'asse del corpo (m)

    # -- moto -------------------------------------------------------------------
    def theta(self, t: float) -> float:
        """Orientamento della persona: pi = di spalle, 0 = di fronte alla camera."""
        a, b = self.turn
        s = smoothstep((t / self.T - a) / (b - a))
        return np.pi * (1.0 - s)

    def state(self, t: float) -> dict:
        """Posizioni e assi 3D di corpo e quadro al tempo t."""
        p = self.ph_
        th = self.theta(t)
        s = self.direction
        # Leggero ondeggiamento del corpo e del quadro (persona in piedi, a mano).
        body = np.array([0.02 * np.sin(2 * np.pi * 0.31 * t + p[0]),
                         0.012 * np.sin(2 * np.pi * 0.7 * t + p[1]),
                         self.z0 + 0.03 * np.sin(2 * np.pi * 0.23 * t + p[2])])
        f = np.array([s * np.sin(th), 0.0, -np.cos(th)])       # direzione in cui guarda
        u = np.array([np.cos(th), 0.0, s * np.sin(th)])        # asse laterale (destra del quadro)
        v = np.array([0.0, 1.0, 0.0])                          # verso il basso
        # Piccola inclinazione del quadro (beccheggio e rollio di pochi gradi).
        pitch = np.radians(4.0) * np.sin(2 * np.pi * 0.27 * t + p[3])
        roll = np.radians(2.5) * np.sin(2 * np.pi * 0.19 * t + p[4])
        n = f * np.cos(pitch) - v * np.sin(pitch)
        v2 = v * np.cos(pitch) + f * np.sin(pitch)
        u2 = u * np.cos(roll) + v2 * np.sin(roll)
        v3 = v2 * np.cos(roll) - u * np.sin(roll)
        center = body + f * self.hold + np.array([0.0, 0.06, 0.0])
        return {"theta": th, "body": body, "f": f, "u": u2, "v": v3, "n": n, "center": center,
                "u_body": u}

    # -- proiezione -----------------------------------------------------------------
    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.atleast_2d(pts)
        z = pts[:, 2]
        return np.stack([self.f * pts[:, 0] / z + self.cx, self.f * pts[:, 1] / z + self.cy], 1), z

    def front_corners3d(self, st: dict) -> np.ndarray:
        """Angoli 3D TL, TR, BR, BL del fronte della tela (come visti da davanti)."""
        c = st["center"] + st["n"] * self.depth / 2
        a, b = self.pw / 2, self.ph / 2
        u, v = st["u"], st["v"]
        return np.array([c - u * a - v * b, c + u * a - v * b, c + u * a + v * b, c - u * a + v * b])

    def front_corners_at(self, t: float) -> np.ndarray:
        return self.project(self.front_corners3d(self.state(t)))[0]

    def silhouette_at(self, t: float) -> np.ndarray:
        """Proiezione degli 8 vertici della tela (fronte, retro e lati): serve
        al benchmark per cercare verde residuo anche sui lati."""
        st = self.state(t)
        c, u, v, n = st["center"], st["u"], st["v"], st["n"]
        a, b, d = self.pw / 2, self.ph / 2, self.depth / 2
        pts = np.array([c + su * u * a + sv * v * b + sn * n * d
                        for su in (-1, 1) for sv in (-1, 1) for sn in (-1, 1)])
        return self.project(pts)[0]

    def view_info(self, t: float) -> tuple[bool, float]:
        """(il fronte è rivolto alla camera?, angolo di vista in gradi)."""
        st = self.state(t)
        c = st["center"] + st["n"] * self.depth / 2
        to_cam = -c / np.linalg.norm(c)
        cosang = float(st["n"] @ to_cam)
        return cosang > 0, float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

    # -- illuminazione ---------------------------------------------------------------
    @staticmethod
    def shade(normal: np.ndarray) -> float:
        """Luce principale dal lato della camera (alto a sinistra) + luce ambiente."""
        to_light = np.array([-0.35, -0.55, -1.0])
        to_light /= np.linalg.norm(to_light)
        return 0.35 + 0.65 * max(0.0, float(normal @ to_light))


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

def back_texture(tw: int, th: int) -> np.ndarray:
    """Retro della tela: tela grezza con il telaio in legno (bordi e traverse)."""
    img = np.empty((th, tw, 3), np.float32)
    img[:] = CANVAS_BGR
    bar = max(4, int(0.07 * min(tw, th)))
    wood = WOOD_BGR.tolist()
    cv2.rectangle(img, (0, 0), (tw - 1, th - 1), wood, 2 * bar)
    cv2.line(img, (tw // 2, 0), (tw // 2, th - 1), wood, bar)
    cv2.line(img, (0, th // 2), (tw - 1, th // 2), wood, bar)
    return msv.to_linear_np(np.clip(img, 0, 255) / 255.0).astype(np.float32)


def side_texture(green: bool, rng: np.random.Generator) -> np.ndarray:
    """Lati della tela: tela grezza, oppure verdi se dipinti come il fronte."""
    base = msv.GREEN_BGR if green else CANVAS_BGR
    tex = np.empty((32, 256, 3), np.float32)
    tex[:] = base
    tex += rng.normal(0, 3, size=(32, 256, 1)).astype(np.float32)
    return msv.to_linear_np(np.clip(tex, 0, 255) / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Disegno con z-buffer (solo nel rettangolo che contiene ogni elemento)
# ---------------------------------------------------------------------------

class Canvas:
    """Immagine in luce lineare + z-buffer + (facoltativa) mappa di visibilità."""

    def __init__(self, bg: np.ndarray, want_ids: bool):
        self.color = bg.copy()
        self.h, self.w = bg.shape[:2]
        self.depth = np.full((self.h, self.w), np.inf, np.float32)
        self.ids = np.zeros((self.h, self.w), np.uint8) if want_ids else None

    def blend(self, box, alpha, col, dep, ident: int):
        """Disegna un elemento (alpha, colore, profondità) nel rettangolo ``box``."""
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        d_roi = self.depth[y0:y1, x0:x1]
        a = alpha * (dep < d_roi + 1e-4)
        c_roi = self.color[y0:y1, x0:x1]
        if col.ndim == 1:
            col = col[None, None, :]
        c_roi[:] = c_roi * (1 - a[..., None]) + col * a[..., None]
        sel = a > 0.5
        d_roi[sel] = dep[sel] if np.ndim(dep) else dep
        if self.ids is not None:
            self.ids[y0:y1, x0:x1][sel] = ident


def _box(pts: np.ndarray, w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    x0 = int(max(0, np.floor(pts[:, 0].min() - pad)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - pad)))
    x1 = int(min(w, np.ceil(pts[:, 0].max() + pad) + 1))
    y1 = int(min(h, np.ceil(pts[:, 1].max() + pad) + 1))
    return x0, y0, x1, y1


def _plane_depth(scene: RevealScene, box, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Profondità (z) per pixel del piano passante per ``point`` con normale ``normal``."""
    x0, y0, x1, y1 = box
    xs = (np.arange(x0, x1, dtype=np.float32) - scene.cx) / scene.f
    ys = (np.arange(y0, y1, dtype=np.float32) - scene.cy) / scene.f
    den = normal[0] * xs[None, :] + normal[1] * ys[:, None] + normal[2]
    num = float(normal @ point)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = num / den
    return np.where(np.isfinite(z) & (z > 0), z, np.inf).astype(np.float32)


def draw_face(cv: Canvas, scene: RevealScene, corners3: np.ndarray, normal: np.ndarray,
              tex: np.ndarray, ident: int, ss: int = 1):
    """Faccia piana texturizzata (corners3: TL, TR, BR, BL visti dall'esterno)."""
    center = corners3.mean(axis=0)
    if float(normal @ (-center)) <= 0:          # rivolta dall'altra parte: invisibile
        return
    pts, z = scene.project(corners3)
    if (z <= 0.05).any():
        return
    box = _box(pts, cv.w, cv.h, 2)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    th, tw = tex.shape[:2]
    src = np.array([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]], np.float32)
    bw, bh = x1 - x0, y1 - y0
    dst = (pts - [x0, y0]) * ss + (ss - 1) / 2.0
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    col = cv2.warpPerspective(tex, H, (bw * ss, bh * ss), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
    cov = cv2.warpPerspective(np.ones((th, tw), np.float32), H, (bw * ss, bh * ss),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if ss > 1:
        col = cv2.resize(col, (bw, bh), interpolation=cv2.INTER_AREA)
        cov = cv2.resize(cov, (bw, bh), interpolation=cv2.INTER_AREA)
    # col è pre-moltiplicato dalla copertura: si normalizza per il blend.
    col = col / np.maximum(cov, 1e-6)[..., None] * scene.shade(normal)
    dep = _plane_depth(scene, box, center, normal)
    cv.blend(box, np.clip(cov, 0, 1), col, dep, ident)


def draw_capsule(cv: Canvas, scene: RevealScene, p3: np.ndarray, q3: np.ndarray, radius: float,
                 color: np.ndarray, ident: int = ID_OTHER):
    """Segmento 3D spesso (braccio, dito, gamba) con profondità interpolata."""
    pts, z = scene.project(np.array([p3, q3]))
    if (z <= 0.05).any():
        return
    r = scene.f * radius / z.mean()
    box = _box(pts, cv.w, cv.h, r + 2)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    a, b = pts[0] - [x0, y0], pts[1] - [x0, y0]
    cv2.line(mask, tuple(np.round(a * 4).astype(int)), tuple(np.round(b * 4).astype(int)), 1.0,
             thickness=max(1, int(round(2 * r))), lineType=cv2.LINE_AA, shift=2)
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
    ab = b - a
    tt = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / max(float(ab @ ab), 1e-6), 0, 1)
    dep = (z[0] + tt * (z[1] - z[0])).astype(np.float32)
    cv.blend(box, mask, msv.lin(color), dep, ident)


def draw_disc(cv: Canvas, scene: RevealScene, c3: np.ndarray, radius: float, color: np.ndarray,
              aspect: float = 1.0, ident: int = ID_OTHER):
    """Disco (testa, palmo) rivolto alla camera, a profondità costante."""
    pts, z = scene.project(np.array([c3]))
    if z[0] <= 0.05:
        return
    r = scene.f * radius / z[0]
    box = _box(np.array([pts[0] - [r, r * aspect], pts[0] + [r, r * aspect]]), cv.w, cv.h, 2)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    c = pts[0] - [x0, y0]
    cv2.ellipse(mask, tuple(np.round(c * 4).astype(int)), (int(r * 4), int(r * aspect * 4)), 0, 0, 360,
                1.0, -1, cv2.LINE_AA, shift=2)
    cv.blend(box, mask, msv.lin(color), np.float32(z[0]), ident)


def draw_card(cv: Canvas, scene: RevealScene, c3: np.ndarray, half_w: float, y_top: float,
              y_bot: float, color: np.ndarray, ident: int = ID_OTHER):
    """Sagoma (busto, gambe) rivolta alla camera, a profondità costante."""
    pts, z = scene.project(np.array([c3 + [-half_w, y_top, 0], c3 + [half_w, y_bot, 0]]))
    box = _box(pts, cv.w, cv.h, 2)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    p0, p1 = pts[0] - [x0, y0], pts[1] - [x0, y0]
    rad = int(0.35 * (p1[0] - p0[0]))
    poly = cv2.ellipse2Poly((int((p0[0] + p1[0]) / 2), int(p0[1] + rad)),
                            (max(1, int((p1[0] - p0[0]) / 2)), max(1, rad)), 0, 180, 360, 10)
    poly = np.vstack([poly, [[int(p1[0]), int(p1[1])], [int(p0[0]), int(p1[1])]]])
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1.0, cv2.LINE_AA)
    cv.blend(box, mask, msv.lin(color) * 0.8, np.float32(z.mean()), ident)


def render_frame(scene: RevealScene, bg: np.ndarray, front_tex: np.ndarray, t: float,
                 textures: dict, ss: int = 1, want_ids: bool = False):
    """Rende un (sotto)fotogramma in luce lineare; con ``want_ids`` anche la visibilità."""
    st = scene.state(t)
    cv = Canvas(bg, want_ids)
    c, u, v, n = st["center"], st["u"], st["v"], st["n"]
    a, b, d = scene.pw / 2, scene.ph / 2, scene.depth / 2
    body, ub, fdir = st["body"], st["u_body"], st["f"]
    skin = msv.SKIN_BGR

    def face(fc, right, down, hr, hd):
        return np.array([fc - right * hr - down * hd, fc + right * hr - down * hd,
                         fc + right * hr + down * hd, fc - right * hr + down * hd])

    elements = []   # (profondità media, funzione di disegno)
    # Tela: fronte, retro, quattro lati (destra = normale × giù).
    elements.append((float((c + n * d)[2]), lambda: draw_face(
        cv, scene, face(c + n * d, u, v, a, b), n, front_tex, ID_FRONT, ss)))
    elements.append((float((c - n * d)[2]), lambda: draw_face(
        cv, scene, face(c - n * d, -u, v, a, b), -n, textures["back"], ID_OTHER, ss)))
    for sgn in (1, -1):
        nn = u * sgn
        elements.append((float((c + nn * a)[2]), lambda nn=nn: draw_face(
            cv, scene, face(c + nn * a, np.cross(nn, v), v, d, b), nn, textures["side"], ID_OTHER, ss)))
        nv = v * sgn
        elements.append((float((c + nv * b)[2]), lambda nv=nv: draw_face(
            cv, scene, face(c + nv * b, np.cross(nv, n), n, a, d), nv, textures["side"], ID_OTHER, ss)))
    # Persona: gambe, busto, testa (capelli dietro, viso davanti), braccia, mani.
    elements.append((float(body[2]) + 0.02, lambda: draw_card(
        cv, scene, body, 0.17, 0.42, 2.4, PANTS_BGR)))
    half = float(np.hypot(0.21 * np.cos(st["theta"]), 0.12 * np.sin(st["theta"])))
    elements.append((float(body[2]), lambda: draw_card(cv, scene, body, half, -0.45, 0.48, SHIRT_BGR)))
    head = body + np.array([0.0, -0.64, 0.0])
    elements.append((float(head[2]), lambda: draw_disc(cv, scene, head, 0.115, HAIR_BGR, 1.12)))
    face_c = head + fdir * 0.07 + np.array([0.0, 0.02, 0.0])
    elements.append((float(face_c[2]), lambda: draw_disc(cv, scene, face_c, 0.085, skin, 1.2)))
    for sgn in (1, -1):
        shoulder = body + ub * (0.19 * sgn) + np.array([0.0, -0.40, 0.0])
        grip = c + u * (a * sgn) + v * 0.05           # mano sul lato della tela
        elements.append((float((shoulder[2] + grip[2]) / 2), lambda s=shoulder, g=grip: draw_capsule(
            cv, scene, s, g, 0.045, SHIRT_BGR)))
        elements.append((float(grip[2]), lambda g=grip: draw_disc(cv, scene, g, 0.05, skin, 1.3)))
        # Dita sul fronte (visibili quando il quadro guarda la camera) e pollice sul retro.
        fing = grip + n * (d + 0.006)
        elements.append((float(fing[2]), lambda g=fing, sg=sgn: draw_capsule(
            cv, scene, g, g - u * (0.07 * sg) + v * 0.01, 0.014, skin)))
        thumb = grip - n * (d + 0.006)
        elements.append((float(thumb[2]), lambda g=thumb, sg=sgn: draw_capsule(
            cv, scene, g, g - u * (0.05 * sg) - v * 0.02, 0.013, skin)))
    # Ordine dal più lontano al più vicino (pittore) + test di profondità per pixel.
    for _, draw in sorted(elements, key=lambda e: -e[0]):
        draw()
    return cv.color, cv.ids


# ---------------------------------------------------------------------------
# Fotogramma finale e verità di riferimento
# ---------------------------------------------------------------------------

def build_scene(meta: dict):
    """Ricostruisce scena, sfondo e texture verde dai metadati (interfaccia del benchmark)."""
    w, h = meta["width"], meta["height"]
    pw_cm, ph_cm = meta["panel_cm"]
    scene = RevealScene(w, h, meta["seconds"], pw_cm / 100, ph_cm / 100, meta["focal"], meta["seed"],
                        meta.get("direction", 1), meta.get("fill", 0.65), meta.get("depth", 0.035),
                        meta.get("green_sides", False), tuple(meta.get("turn", (0.18, 0.62))))
    rng = np.random.default_rng(meta["seed"])
    bg = msv.make_background(w, h, rng)
    tex = msv.make_panel_texture(800, int(800 * ph_cm / pw_cm), rng)
    scene.textures = {"back": back_texture(400, int(400 * ph_cm / pw_cm)),
                      "side": side_texture(meta.get("green_sides", False), rng)}
    return scene, bg, tex


def synthesize_frame(scene: RevealScene, bg: np.ndarray, texture: np.ndarray, index: int,
                     fps: float, subframes: int = 3, shutter: float = 0.5, noise: float = 2.5,
                     seed: int = 7, ss: int = 1, want_ids: bool = False):
    """Fotogramma BGR in valori video 0-255 (float32). Con ``want_ids`` ritorna anche la
    mappa di visibilità dell'istante centrale."""
    t = index / fps
    offs = (np.arange(subframes) / (subframes - 1) - 0.5) * shutter / fps if subframes > 1 else np.zeros(1)
    acc = np.zeros((scene.h, scene.w, 3), np.float32)
    ids = None
    for k, dt in enumerate(offs):
        mid = want_ids and abs(dt) < 1e-12
        col, idm = render_frame(scene, bg, texture, t + dt, scene.textures, ss, mid)
        acc += col
        if mid:
            ids = idm
    if want_ids and ids is None:
        _, ids = render_frame(scene, bg, texture, t, scene.textures, 1, True)
    frame = cv2.GaussianBlur(acc / len(offs), (0, 0), 0.7)
    frame = from_linear_np(np.clip(frame, 0, None)) * 255.0
    rng = np.random.default_rng([seed, index])
    frame += rng.normal(0, noise, size=frame.shape).astype(np.float32)
    return (frame, ids) if want_ids else frame


def phase_of(facing: bool, angle: float, visible_px: int) -> str:
    """Fase della rivelazione per il fotogramma."""
    if not facing or visible_px < 30:
        return "retro"
    if angle > 65:
        return "taglio"
    if angle > 30:
        return "rotazione"
    return "frontale"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="file .mp4 di uscita (il JSON viene scritto accanto)")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--panel-cm", default="100x120", help="dimensioni del quadro LxA in cm")
    ap.add_argument("--fill", type=float, default=0.65,
                    help="frazione della larghezza coperta dal quadro frontale (>0.95: esce dai bordi)")
    ap.add_argument("--direction", type=int, default=1, help="verso della rotazione: 1 o -1")
    ap.add_argument("--turn", default="0.18,0.62", help="inizio e fine della rotazione (frazioni del video)")
    ap.add_argument("--depth-cm", type=float, default=3.5, help="spessore della tela")
    ap.add_argument("--green-sides", action="store_true", help="lati della tela dipinti di verde")
    ap.add_argument("--subframes", type=int, default=3)
    ap.add_argument("--shutter", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=2.5)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pw_cm, ph_cm = (float(x) for x in args.panel_cm.lower().split("x"))
    w, h = args.width, args.height
    if w % 2 or h % 2:
        raise SystemExit("larghezza e altezza devono essere pari (YUV 4:2:0)")
    if args.subframes > 1 and args.subframes % 2 == 0:
        raise SystemExit("--subframes deve essere dispari (l'istante centrale dà la verità)")
    meta = {
        "generator": "reveal", "width": w, "height": h, "fps": args.fps, "seconds": args.seconds,
        "panel_cm": [pw_cm, ph_cm], "panel_aspect": pw_cm / ph_cm,
        "focal": 0.6 * float(np.hypot(w, h)), "seed": args.seed, "direction": args.direction,
        "fill": args.fill, "depth": args.depth_cm / 100, "green_sides": bool(args.green_sides),
        "turn": [float(x) for x in args.turn.split(",")],
        "subframes": args.subframes, "shutter": args.shutter, "noise": args.noise,
        "linear_light": True, "chroma_location": "left", "lens_sigma": 0.7, "scenarios": ["reveal"],
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
        frame, ids = synthesize_frame(scene, bg, tex, i, args.fps, args.subframes, args.shutter,
                                      args.noise, args.seed, want_ids=True)
        proc.stdin.write(msv.frame_to_yuv420(frame))
        t = i / args.fps
        facing, angle = scene.view_info(t)
        visible = int((ids == ID_FRONT).sum())
        gt.append({
            "frame": i, "corners": scene.front_corners_at(t).round(3).tolist(),
            "facing": facing, "view_angle": round(angle, 2), "visible_px": visible,
            "phase": phase_of(facing, angle, visible),
            # Compatibilità con il benchmark degli altri scenari.
            "corner_occluded": False, "bottom_occluded": False,
        })
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg ha restituito un errore")
    meta["ground_truth"] = gt
    out.with_suffix(".gt.json").write_text(json.dumps(meta))
    phases = {p: sum(g["phase"] == p for g in gt) for p in ("retro", "taglio", "rotazione", "frontale")}
    print(f"Creato {out} ({n} fotogrammi; fasi: {phases})")


if __name__ == "__main__":
    main()
