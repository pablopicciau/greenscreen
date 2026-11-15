"""Chiave cromatica: dove il pannello verde è visibile e dove c'è un occlusore.

Perché non un semplice chroma key HSV?
-------------------------------------
Usiamo la "differenza di verde" ``d = G - max(R, B)``, la stessa misura dei
keyer professionali (IBK di Nuke, Keylight). Ha tre proprietà utili:

1. è lineare nel mescolamento dei colori: un pixel di bordo dito coperto al 50%
   ha circa metà della ``d`` del pannello -> alpha morbido e corretto;
2. la pelle, il bianco, il grigio e quasi tutti gli sfondi hanno ``d <= 0``;
3. confrontandola con la ``d`` del pannello *pulito nello stesso punto*
   ("screen plate") diventa insensibile a ombre e gradienti di luce.

Lo screen plate è una stima di come apparirebbe il pannello senza mani
davanti: si ottiene interpolando i pixel sicuramente verdi (normalized
convolution multi-scala su un'immagine ridotta, quindi è veloce).
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import KeyParams


def green_difference(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna ``(d, g)`` in float32: d = G - max(R, B) e il canale G."""
    img = img_bgr if img_bgr.dtype == np.float32 else img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    d = g - cv2.max(r, b)
    return d, g


def detection_mask(img_bgr: np.ndarray, kp: KeyParams) -> np.ndarray:
    """Maschera binaria (uint8 0/255) dei pixel "sicuramente verdi".

    Usata per trovare il pannello: richiede sia una differenza assoluta minima
    (scarta il rumore nei neri) sia una differenza relativa (funziona anche in
    ombra, dove tutti i canali calano in proporzione).
    """
    d, g = green_difference(img_bgr)
    mask = (d >= kp.d_min) & (d >= kp.r_min * g)
    return mask.astype(np.uint8) * 255


def screen_plate(
    img_f32: np.ndarray,
    clean_mask: np.ndarray,
    target_px: int = 320,
) -> np.ndarray:
    """Stima il colore del pannello pulito in ogni pixel (anche sotto le mani).

    Parametri
    ---------
    img_f32: immagine BGR float32 (tipicamente una ROI attorno al quad).
    clean_mask: maschera (bool/uint8) dei pixel sicuramente di pannello.
    target_px: lato lungo dell'immagine ridotta su cui si lavora (velocità).
    """
    h, w = clean_mask.shape[:2]
    scale = min(1.0, target_px / max(h, w))
    sw, sh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))

    # Riduzione "ad area": la maschera ridotta diventa una frazione di copertura.
    m_small = cv2.resize(clean_mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    img_small = cv2.resize(img_f32 * clean_mask[..., None].astype(np.float32), (sw, sh),
                           interpolation=cv2.INTER_AREA)
    # Teniamo solo celle quasi interamente verdi (equivale a un'erosione:
    # esclude i pixel misti di bordo, che falserebbero il colore del pannello).
    weight = (m_small > 0.98).astype(np.float32)
    num = img_small * weight[..., None] / np.maximum(m_small, 1e-6)[..., None]

    if weight.sum() < 1:
        # Nessun pixel pulito: usa la media grezza come ripiego.
        mean = img_f32[clean_mask > 0].mean(axis=0) if clean_mask.any() else np.zeros(3)
        return np.broadcast_to(mean.astype(np.float32), img_f32.shape).copy()

    # Normalized convolution multi-scala: le scale fini seguono i gradienti di
    # luce, quelle grosse riempiono i buchi lasciati dalle mani.
    # Ogni scala si calcola a una risoluzione in cui il sigma è di pochi pixel
    # (sfocare a bassa risoluzione e poi ingrandire equivale a una sfocatura
    # ampia, ma costa una frazione): così il keyer resta veloce.
    base_sigma = max(1.5, 0.02 * max(sw, sh))
    result = None
    for sigma in (base_sigma * 16, base_sigma * 4, base_sigma):
        k = min(1.0, 3.0 / sigma)                       # fattore di riduzione
        lw, lh = max(2, int(round(sw * k))), max(2, int(round(sh * k)))
        w_low = cv2.resize(weight, (lw, lh), interpolation=cv2.INTER_AREA)
        n_low = cv2.resize(num, (lw, lh), interpolation=cv2.INTER_AREA)
        den = cv2.GaussianBlur(w_low, (0, 0), sigma * k)
        nume = cv2.GaussianBlur(n_low, (0, 0), sigma * k)
        if k < 1.0:
            den = cv2.resize(den, (sw, sh), interpolation=cv2.INTER_LINEAR)
            nume = cv2.resize(nume, (sw, sh), interpolation=cv2.INTER_LINEAR)
        est = nume / np.maximum(den, 1e-6)[..., None]
        if result is None:
            # La scala più grossa copre tutto (eventuali zeri residui: media globale).
            fallback = num[weight > 0].mean(axis=0)
            est[den < 1e-4] = fallback
            result = est
        else:
            # Le scale più fini sostituiscono le grosse dove hanno abbastanza supporto.
            blend = np.clip(den / 0.35, 0.0, 1.0)[..., None]
            result = blend * est + (1.0 - blend) * result

    return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)


def soft_alpha(d_img: np.ndarray, d_plate: np.ndarray, kp: KeyParams) -> np.ndarray:
    """Alpha morbido del pannello (1 = pannello visibile, 0 = occlusore)."""
    ratio = d_img / np.maximum(d_plate, 4.0)
    alpha = (ratio - kp.low) / max(kp.high - kp.low, 1e-6)
    return np.clip(alpha, 0.0, 1.0)


def despill(img_f32: np.ndarray, strength: float | np.ndarray = 1.0) -> np.ndarray:
    """Toglie il riflesso verde limitando G a max(R, B).

    ``strength`` può essere uno scalare o una mappa (H, W) per applicarlo solo
    vicino al pannello e non alterare il resto della scena.
    """
    b, g, r = cv2.split(img_f32)
    excess = np.maximum(g - cv2.max(r, b), 0.0)
    if isinstance(strength, np.ndarray):
        g = g - excess * strength
    else:
        g = g - excess * float(strength)
    return cv2.merge([b, g, r])


def luminance(img_f32: np.ndarray) -> np.ndarray:
    """Luma Rec.709 (approssimata in spazio gamma) di un'immagine BGR."""
    b, g, r = cv2.split(img_f32)
    return 0.0722 * b + 0.7152 * g + 0.2126 * r
