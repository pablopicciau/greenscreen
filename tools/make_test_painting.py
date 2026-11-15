"""Genera immagini di prova da inserire nel quadro.

* ``--style art``: un "quadro" astratto a pennellate (per le demo);
* ``--style pattern``: griglia, cerchi e angoli etichettati (per verificare
  geometria e proporzioni: se l'aspetto fosse sbagliato i cerchi diventerebbero
  ellissi, se gli angoli fossero scambiati le etichette sarebbero fuori posto).

Uso:
    python tools/make_test_painting.py --out samples/generated/quadro.png --style art
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_art(w: int, h: int, seed: int) -> np.ndarray:
    """Quadro astratto: fondo sfumato + centinaia di pennellate colorate."""
    rng = np.random.default_rng(seed)
    palette = np.array([[40, 60, 180], [30, 140, 230], [60, 200, 240], [150, 90, 40],
                        [200, 160, 70], [90, 50, 120], [230, 230, 220], [40, 40, 50]], np.float32)
    low = palette[rng.integers(0, len(palette), size=(3, 3))]
    img = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(420):
        c = palette[rng.integers(0, len(palette))] * rng.uniform(0.8, 1.15)
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(0.03, 0.25) * w
        p = np.array([x, y]); q = p + length * np.array([np.cos(ang), np.sin(ang)])
        thick = int(rng.uniform(0.004, 0.02) * w) + 1
        cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), c.tolist(), thick, cv2.LINE_AA)
    # Texture da tela.
    canvas = rng.normal(0, 5, size=(h, w)).astype(np.float32)
    img += cv2.GaussianBlur(canvas, (0, 0), 0.8)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_pattern(w: int, h: int) -> np.ndarray:
    """Pattern di controllo geometrico."""
    img = np.full((h, w, 3), 245, np.uint8)
    step = w // 8
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (180, 180, 180), 2, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (180, 180, 180), 2, cv2.LINE_AA)
    r = int(0.3 * min(w, h))
    cv2.circle(img, (w // 2, h // 2), r, (40, 40, 200), max(4, w // 150), cv2.LINE_AA)
    cv2.circle(img, (w // 2, h // 2), r // 2, (200, 120, 30), max(4, w // 150), cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (30, 30, 30), max(6, w // 80))
    font_scale = w / 500
    for txt, (x, y) in (("TL", (0.04, 0.08)), ("TR", (0.84, 0.08)), ("BR", (0.84, 0.96)), ("BL", (0.04, 0.96))):
        cv2.putText(img, txt, (int(x * w), int(y * h)), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (20, 20, 20), max(3, w // 200), cv2.LINE_AA)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", choices=["art", "pattern"], default="art")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args(argv)
    img = make_art(args.width, args.height, args.seed) if args.style == "art" else make_pattern(args.width, args.height)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"Creato {args.out} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
