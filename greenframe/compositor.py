"""Compositing: proiezione dell'immagine sul quadro e gestione delle occlusioni.

Per ogni fotogramma, lavorando solo su un ritaglio (ROI) attorno al quadro:

1. **Proiezione**: l'immagine (già portata alle proporzioni del pannello)
   viene proiettata sul quad con un'omografia (``warpPerspective``),
   scegliendo il livello di piramide adatto per evitare aliasing.
2. **Chiave morbida**: alpha = quanto il pixel è "pannello verde pulito",
   confrontando la sua verdezza con quella dello *screen plate* (pannello
   pulito stimato nello stesso punto). Le mani hanno alpha 0 e restano in
   primo piano; i bordi delle dita (sfocati/mossi) hanno alpha intermedi.
3. **Maschera geometrica**: alpha viene limitato al quad allargato di pochi
   pixel. Così i bordi visibili del quadro li decide la chiave (precisione al
   pixel, segue anche la sfocatura di movimento) e il verde fuori dal quadro
   (piante, oggetti) non viene toccato.
4. **Luce**: la luminosità del pannello (ombre delle mani, gradienti, il
   quadro che si gira verso la luce) viene trasferita all'immagine.
5. **Despill**: il riflesso verde sulle mani vicino al pannello viene rimosso.
6. **Composizione additiva**: ``out = I + alpha · (R - S)`` (I = originale
   senza riflesso, R = immagine illuminata, S = pannello pulito). È la formula
   dei keyer professionali: nei pixel misti (bordo di un dito) sostituisce
   esattamente la quota di verde con la quota di immagine, senza aloni, e
   conserva la grana/rumore della camera sull'immagine inserita.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import geometry as geo
from .config import Config
from .image_fit import SourcePyramid, fit_to_aspect
from .keyer import despill, green_difference, luminance, screen_plate, soft_alpha


def polygon_mask(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Maschera float32 0-1 anti-aliased di un quadrilatero (coordinate sub-pixel)."""
    mask = np.zeros(shape, np.uint8)
    # shift=4: le coordinate sono in 1/16 di pixel.
    pts = np.round(quad * 16).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA, shift=4)
    return mask.astype(np.float32) / 255.0


class Compositor:
    """Inserisce un'immagine nel quadro, fotogramma per fotogramma."""

    def __init__(self, image_bgr: np.ndarray, panel_aspect: float, cfg: Config,
                 reference_luma: float | None = None):
        self.cfg = cfg
        rp = cfg.render
        prepared = fit_to_aspect(image_bgr, panel_aspect, rp.mode, rp.fit_background)
        self.prepared = prepared
        self.pyramid = SourcePyramid(prepared)
        # Luminosità di riferimento del pannello: quella in cui l'immagine
        # appare con la sua luminosità originale.
        self.reference_luma = reference_luma

    def render(self, frame: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
        """Ritorna il fotogramma con il quadro sostituito (copia)."""
        if quad is None or np.isnan(quad).any():
            return frame
        kp, rp = self.cfg.key, self.cfg.render
        h, w = frame.shape[:2]

        # ROI: quad allargato del margine di despill, intersecato col fotogramma.
        outer = geo.offset_quad(quad, rp.despill_margin)
        x0, y0 = np.floor(outer.min(axis=0)).astype(int)
        x1, y1 = np.ceil(outer.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return frame
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        rh, rw = roi.shape[:2]
        q = quad - np.array([x0, y0], dtype=np.float64)

        # --- Maschere geometriche -------------------------------------------
        m_quad = polygon_mask(geo.offset_quad(q, rp.edge_expand), (rh, rw))
        m_inner = polygon_mask(geo.offset_quad(q, -3.0), (rh, rw)) > 0.5
        # Zona di despill sfumata attorno al quadro (calcolata a 1/4 di
        # risoluzione: è una maschera morbida, la precisione non serve).
        sw, sh = max(2, rw // 4), max(2, rh // 4)
        m_spill = polygon_mask(geo.offset_quad(q, 0.6 * rp.despill_margin) / 4.0, (sh, sw))
        m_spill = cv2.GaussianBlur(m_spill, (0, 0), max(0.5, 0.25 * rp.despill_margin / 4.0))
        m_spill = cv2.resize(m_spill, (rw, rh), interpolation=cv2.INTER_LINEAR)

        # --- Chiave ---------------------------------------------------------
        d, g = green_difference(roi)
        clean = (d >= kp.d_min) & (d >= kp.r_min * g) & m_inner
        plate = screen_plate(roi, clean.astype(np.uint8))
        pb, pg, pr = cv2.split(plate)
        d_plate = pg - cv2.max(pr, pb)
        alpha = soft_alpha(d, d_plate, kp) * m_quad

        # --- Immagine proiettata ---------------------------------------------
        src = self.pyramid.level_for(quad)
        H = geo.homography_rect_to_quad(src.shape[1], src.shape[0], q)
        warped = cv2.warpPerspective(src, H, (rw, rh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

        # --- Trasferimento della luce ------------------------------------------
        if rp.lighting > 0 and self.reference_luma:
            light = luminance(plate) / self.reference_luma
            light = np.clip(light, 0.15, 1.3) ** rp.lighting
            warped *= light[..., None]

        # --- Despill e composizione additiva -----------------------------------
        fg = despill(roi, kp.despill * m_spill)
        plate_d = despill(plate, kp.despill)
        a = alpha[..., None]
        residual = fg - plate_d
        # Peso della grana: pieno sui pixel misti (bordi), regolabile sul pannello pieno.
        grain_w = 1.0 - (1.0 - rp.grain) * a * a
        out = (1.0 - a) * fg + a * warped + a * grain_w * residual

        result = frame.copy()
        result[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return result
