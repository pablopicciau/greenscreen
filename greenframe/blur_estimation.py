"""Stima della nitidezza della camera e del tempo di esposizione dai bordi del pannello.

Perché serve
------------
Il quadro inserito deve essere *sfocato come la scena*: con la stessa
morbidezza dell'obiettivo/compressione e con la stessa sfocatura di
movimento. Un quadro più nitido del resto del video è il primo indizio che
"è finto".

Come funziona
-------------
Durante l'analisi, per ogni lato misurato, si registra la **larghezza del
bordo** (distanza tra i livelli 75% e 25% della copertura, misurata in luce
lineare, vedi ``detector._refine_subpixel``). Il profilo di un bordo è un
gradino convoluto con:

* la sfocatura fissa dell'immagine (obiettivo, sensore, compressione),
  modellata come gaussiana di deviazione standard ``sigma``;
* la sfocatura di movimento: il bordo si sposta durante l'esposizione di
  ``B = e · |v_n|`` pixel, dove ``v_n`` è la velocità del lato lungo la sua
  normale (pixel per fotogramma, nota dalla traccia) ed ``e`` la frazione del
  tempo tra due fotogrammi in cui l'otturatore è aperto ("shutter angle"/360).

La larghezza 25-75% di gaussiana ⊗ rettangolo ha forma chiusa; ``sigma`` ed
``e`` si trovano con una ricerca su griglia che minimizza l'errore robusto
(Huber) su tutti i lati di tutti i fotogrammi. Bordi fermi fissano
``sigma``, bordi in movimento fissano ``e``. È lo stesso principio di
"Video shutter angle estimation using optical flow and linear blur"
(arXiv 2303.10247), applicato ai bordi netti del pannello invece che al
flusso ottico.
"""

from __future__ import annotations

import functools
import math

import numpy as np

# Parte della larghezza misurata dovuta alla misura stessa (integrazione del
# pixel, crominanza 4:2:0, compressione, interpolazione dei profili): non va
# riapplicata al quadro, che viene già campionato "ad area". Calibrata sui
# video sintetici: con un obiettivo da 0.7 px si misura 1.10 px e il
# composito più fedele (PSNR massimo) si ottiene sfocando il quadro di
# 0.7 px -> sqrt(1.10² - 0.7²) = 0.85.
INTRINSIC_SIGMA = 0.85


def _phi(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):
    from scipy.special import ndtr

    return ndtr(z)


def _cdf(x, sigma, box):
    """CDF di N(0, sigma²) + U(-box/2, box/2) (vettoriale)."""
    x = np.asarray(x, np.float64)
    small = box < 1e-6
    b = np.where(small, 1.0, box)
    psi = lambda z: z * _Phi(z) + _phi(z)
    f_box = (psi((x + b / 2) / sigma) - psi((x - b / 2) / sigma)) * sigma / b
    return np.where(small, _Phi(x / sigma), f_box)


@functools.lru_cache(maxsize=4)
def _iqr_table(s_max: float = 3.0, b_max: float = 80.0):
    """Tabella della larghezza 25-75% per sigma (righe) e rettangolo (colonne)."""
    sigmas = np.arange(0.2, s_max + 1e-9, 0.02)
    boxes = np.arange(0.0, b_max + 1e-9, 0.25)
    S, B = np.meshgrid(sigmas, boxes, indexing="ij")
    lo = np.zeros_like(S)
    hi = 0.6745 * S + 0.5 * B + 1.0
    for _ in range(50):                      # bisezione: F(x75) = 0.75
        mid = 0.5 * (lo + hi)
        above = _cdf(mid, S, B) > 0.75
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return sigmas, boxes, 2.0 * 0.5 * (lo + hi)


def edge_iqr(sigma: float, box: np.ndarray) -> np.ndarray:
    """Larghezza 25-75% prevista per un bordo con sfocatura ``sigma`` mosso di ``box`` px."""
    sigmas, boxes, table = _iqr_table()
    i = int(np.clip(np.round((sigma - sigmas[0]) / 0.02), 0, len(sigmas) - 1))
    return np.interp(np.asarray(box, np.float64), boxes, table[i])


def normal_speeds(quads: np.ndarray, t: int) -> np.ndarray | None:
    """Velocità normale mediana (px/fotogramma) di ciascun lato al fotogramma ``t``.

    Si campiona la velocità in 9 punti lungo il lato (con una rotazione la
    velocità varia lungo il lato) e se ne prende la mediana del valore
    assoluto, come per la larghezza misurata (mediana dei profili).
    """
    if t <= 0 or t >= len(quads) - 1:
        return None
    q0, q1, q2 = quads[t - 1], quads[t], quads[t + 1]
    if np.isnan(q0).any() or np.isnan(q1).any() or np.isnan(q2).any():
        return None
    vel = 0.5 * (q2 - q0)                                   # (4, 2) per angolo
    u = np.linspace(0.06, 0.94, 9)[:, None]
    out = np.zeros(4)
    for s in range(4):
        a, b = q1[s], q1[(s + 1) % 4]
        d = b - a
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        v = (1 - u) * vel[s] + u * vel[(s + 1) % 4]
        out[s] = float(np.median(np.abs(v @ n)))
    return out


def estimate_blur(edge_width: np.ndarray, quads: np.ndarray, min_samples: int = 30
                  ) -> tuple[float, float, dict] | None:
    """Stima ``(sigma, shutter, info)`` da larghezze dei bordi (T, 4) e traccia (T, 4, 2).

    ``shutter`` è la frazione 0-1 del tempo tra fotogrammi con otturatore
    aperto. Ritorna None se i dati sono troppo pochi. ``info`` riporta il
    numero di campioni e se l'otturatore è osservabile (servono bordi in
    movimento: se il pannello è quasi fermo per tutto il video l'otturatore
    resta al valore tipico 0.5, che in quel caso non conta).
    """
    ws, vs = [], []
    for t in range(1, len(quads) - 1):
        sp = normal_speeds(quads, t)
        if sp is None:
            continue
        for s in range(4):
            w = edge_width[t, s]
            if np.isfinite(w) and 0.2 < w < 60:
                ws.append(w)
                vs.append(sp[s])
    if len(ws) < min_samples:
        return None
    ws, vs = np.array(ws), np.array(vs)
    sigmas, boxes, table = _iqr_table()
    shutters = np.linspace(0.0, 1.0, 101)
    # Errore robusto (Huber, delta 0.25 px) per ogni coppia (sigma, shutter).
    best = (np.inf, 0.7, 0.5)
    for si, sigma in enumerate(sigmas):
        if sigma > 2.5:
            break
        pred = np.interp(np.outer(shutters, vs), boxes, table[si])     # (E, N)
        r = np.abs(pred - ws[None])
        cost = np.where(r < 0.25, 0.5 * r * r, 0.25 * (r - 0.125)).sum(axis=1)
        k = int(np.argmin(cost))
        if cost[k] < best[0]:
            best = (float(cost[k]), float(sigma), float(shutters[k]))
    _, sigma, shutter = best
    moving = vs > 3.0
    observable = moving.sum() >= 10
    if not observable:
        shutter = 0.5
    return sigma, shutter, {"samples": int(len(ws)), "moving": int(moving.sum()),
                            "shutter_observable": bool(observable)}


def render_blur_sigma(lens_sigma: float | None) -> float:
    """Sfocatura gaussiana da applicare al quadro inserito per eguagliare la scena."""
    if lens_sigma is None:
        return 0.0
    return float(math.sqrt(max(lens_sigma ** 2 - INTRINSIC_SIGMA ** 2, 0.0)))
