"""Preparazione dell'immagine da inserire: proporzioni, adattamento, piramide.

Regola fondamentale: l'opera dell'utente **non deve essere deformata**.
L'immagine viene portata alle proporzioni fisiche del pannello (stimate dal
video o indicate dall'utente) prima della proiezione prospettica:

* ``fill`` (riempi): ritaglio centrato, nessuna banda; si perde un po' di bordo;
* ``fit`` (adatta): l'immagine intera, con bande riempite da una versione
  sfocata dell'immagine stessa (stile Instagram) o da un colore;
* ``stretch``: deforma (solo se l'utente lo chiede esplicitamente).

La piramide (mipmap) in luce lineare e il campionamento anti-aliasing sono
in ``render.py`` (sulla GPU).
"""

from __future__ import annotations

import cv2
import numpy as np


def _parse_color(text: str) -> tuple[int, int, int]:
    """'white' / 'black' / '#rrggbb' -> colore BGR."""
    named = {"white": (255, 255, 255), "black": (0, 0, 0), "bianco": (255, 255, 255), "nero": (0, 0, 0)}
    t = text.strip().lower()
    if t in named:
        return named[t]
    if t.startswith("#") and len(t) == 7:
        r, g, b = int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16)
        return (b, g, r)
    raise ValueError(f"Colore non riconosciuto: {text!r} (usa white, black, #rrggbb o blur)")


def fit_to_aspect(img: np.ndarray, aspect: float, mode: str = "fill",
                  background: str = "blur") -> np.ndarray:
    """Porta ``img`` al rapporto larghezza/altezza ``aspect`` senza deformarla."""
    h, w = img.shape[:2]
    img_aspect = w / h
    if mode == "stretch" or abs(np.log(img_aspect / aspect)) < 1e-3:
        return img.copy()

    if mode == "fill":
        # Ritaglio centrato della parte che eccede.
        if img_aspect > aspect:
            new_w = int(round(h * aspect))
            x0 = (w - new_w) // 2
            return img[:, x0:x0 + new_w].copy()
        new_h = int(round(w / aspect))
        y0 = (h - new_h) // 2
        return img[y0:y0 + new_h].copy()

    if mode != "fit":
        raise ValueError(f"Modalità sconosciuta: {mode!r} (usa fill, fit o stretch)")

    # Tela con le proporzioni del pannello che contiene l'immagine intera.
    if img_aspect > aspect:
        cw, ch = w, int(round(w / aspect))
    else:
        cw, ch = int(round(h * aspect)), h
    if background == "blur":
        # Sfondo: l'immagine stessa ingrandita a riempire, sfocata e un po' scurita.
        scale = max(cw / w, ch / h)
        big = cv2.resize(img, (int(np.ceil(w * scale)), int(np.ceil(h * scale))), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (big.shape[0] - ch) // 2, (big.shape[1] - cw) // 2
        canvas = big[y0:y0 + ch, x0:x0 + cw]
        sigma = 0.04 * max(cw, ch)
        canvas = (cv2.GaussianBlur(canvas, (0, 0), sigma) * 0.8).astype(img.dtype)
    else:
        canvas = np.empty((ch, cw, 3), img.dtype)
        canvas[:] = _parse_color(background)
    y0, x0 = (ch - h) // 2, (cw - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def load_image(path: str) -> np.ndarray:
    """Carica un'immagine (anche con caratteri non ASCII nel percorso, su Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {path}")
    return img
