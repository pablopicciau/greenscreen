"""Riconoscimento del verde del pannello (maschere per il rilevamento).

Due livelli:

* **verde generico** (:func:`generic_green_mask`): la "differenza di verde"
  ``d = G - max(R, B)``, la stessa misura dei keyer professionali, e la sua
  purezza ``d / G`` (indipendente dalla luminosità, quindi anche in ombra).
  Pelle, bianco, grigio e quasi tutti gli sfondi hanno ``d <= 0``. Serve per
  trovare il pannello la prima volta;
* **colore del pannello** (:class:`PanelColor`): una volta trovato il
  pannello se ne impara la cromaticità esatta in luce lineare. Così un altro
  verde saturo (pianta, maglietta, parete) attaccato al pannello non si fonde
  più con lui.

La chiave per il compositing (alpha morbido, screen plate, modello a due
colori, despill) è in ``render.py`` e lavora in luce lineare sulla GPU.
"""

from __future__ import annotations

import cv2
import numpy as np


def green_maps_u8(img_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Versione veloce (uint8) di ``d = G - max(R, B)`` e della purezza ``d / G``.

    Ritorna ``(d, purezza·255, G)`` come uint8; i valori negativi di ``d``
    diventano 0 (non servono: sono "non verde").
    """
    b, g, r = cv2.split(img_u8)
    d = cv2.subtract(g, cv2.max(r, b))
    purity = cv2.divide(d, g, scale=255)
    return d, purity, g


def generic_green_mask(img_u8: np.ndarray, d_min: float, purity_min: float) -> np.ndarray:
    """Maschera uint8 (0/1) dei pixel "verdi" generici: ``d >= d_min`` e ``d/G >= purity_min``."""
    d, purity, _ = green_maps_u8(img_u8)
    m = cv2.bitwise_and(cv2.compare(d, float(d_min), cv2.CMP_GE),
                        cv2.compare(purity, float(purity_min * 255.0), cv2.CMP_GE))
    return (m > 0).astype(np.uint8)


class PanelColor:
    """Modello del colore del pannello: cromaticità in luce lineare (R/G, B/G).

    In luce lineare un'ombra o un cambio di esposizione moltiplicano i tre
    canali per lo stesso fattore: i rapporti R/G e B/G non cambiano. Il
    pannello occupa quindi un punto preciso (con una piccola dispersione)
    in questo piano, mentre altri verdi (piante, magliette, pareti) cadono
    altrove anche quando sono saturi quanto lui: nei test una pianta di
    verde quasi altrettanto puro sta a ~7 deviazioni standard (soglia 4.5).

    Per velocità la decisione "pannello sì/no" è precalcolata in una tabella
    3D a 64 livelli per canale (262 144 colori): applicarla a un'immagine
    costa un solo accesso per pixel.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray, tau: float = 4.5,
                 noise: float = 0.012, d_min: float = 8.0):
        self.mean = np.asarray(mean, np.float64)
        self.cov = np.asarray(cov, np.float64)
        self.tau = tau
        self.noise = noise
        self.d_min = d_min
        self._lut = self._build_lut()

    # -- costruzione -----------------------------------------------------
    @staticmethod
    def features(bgr_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cromaticità (N, 2) = (R/G, B/G) in luce lineare e G lineare (N,)."""
        from .color import LINEAR_LUT

        px = np.asarray(bgr_u8).reshape(-1, 3)
        lin = LINEAR_LUT[px]
        g = np.maximum(lin[:, 1], 1e-4)
        return np.stack([lin[:, 2] / g, lin[:, 0] / g], axis=1), lin[:, 1]

    @classmethod
    def fit(cls, bgr_u8_pixels: np.ndarray, **kw) -> "PanelColor | None":
        """Stima robusta (mediana e covarianza "rifilata") dai pixel del pannello."""
        f, g = cls.features(bgr_u8_pixels)
        f = f[g > 0.02]
        if len(f) < 50:
            return None
        med = np.median(f, axis=0)
        mad = np.median(np.abs(f - med), axis=0) * 1.4826 + 1e-4
        keep = (np.abs(f - med) / mad).max(axis=1) < 3.0
        f = f[keep]
        if len(f) < 30:
            return None
        mean = f.mean(axis=0)
        cov = np.cov(f.T) + np.eye(2) * 1e-6
        return cls(mean, cov, **kw)

    def updated(self, bgr_u8_pixels: np.ndarray, rate: float = 0.05) -> "PanelColor":
        """Nuovo modello aggiornato lentamente (media mobile) con i pixel del pannello attuale."""
        new = PanelColor.fit(bgr_u8_pixels, tau=self.tau, noise=self.noise, d_min=self.d_min)
        if new is None:
            return self
        mean = (1 - rate) * self.mean + rate * new.mean
        cov = (1 - rate) * self.cov + rate * new.cov
        if np.abs(mean - self.mean).max() < 1e-3 and np.abs(cov - self.cov).max() < 1e-5:
            return self
        return PanelColor(mean, cov, self.tau, self.noise, self.d_min)

    def _build_lut(self) -> np.ndarray:
        from .color import LINEAR_LUT

        levels = np.arange(64) * 4 + 2                      # centro di ogni intervallo
        lin = LINEAR_LUT[levels]
        b, g, r = np.meshgrid(lin, lin, lin, indexing="ij")  # (64, 64, 64) indicizzati b, g, r
        gs = np.maximum(g, 1e-4)
        dr = r / gs - self.mean[0]
        db = b / gs - self.mean[1]
        # Rumore del sensore: la cromaticità di un pixel scuro è più incerta.
        s2 = (self.noise / gs) ** 2
        c00, c01, c11 = self.cov[0, 0] + s2, self.cov[0, 1], self.cov[1, 1] + s2
        det = c00 * c11 - c01 * c01
        d2 = (c11 * dr * dr - 2 * c01 * dr * db + c00 * db * db) / det
        # Verde minimo (valori video) per escludere neri e grigi rumorosi.
        v = levels.astype(np.float64)
        vb, vg, vr = np.meshgrid(v, v, v, indexing="ij")
        green = (vg - np.maximum(vr, vb)) >= self.d_min
        return ((d2 < self.tau ** 2) & green).astype(np.uint8).reshape(-1)

    # -- uso --------------------------------------------------------------
    def mask(self, bgr_u8: np.ndarray) -> np.ndarray:
        """Maschera uint8 (0/1) dei pixel con il colore del pannello."""
        q = (bgr_u8 >> 2).astype(np.int32)
        idx = (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]
        return self._lut[idx]

    def distance(self, bgr_u8_pixels: np.ndarray) -> np.ndarray:
        """Distanza di Mahalanobis (in deviazioni standard) dei pixel dal pannello."""
        f, g = self.features(bgr_u8_pixels)
        s2 = (self.noise / np.maximum(g, 1e-4)) ** 2
        dr, db = f[:, 0] - self.mean[0], f[:, 1] - self.mean[1]
        c00, c01, c11 = self.cov[0, 0] + s2, self.cov[0, 1], self.cov[1, 1] + s2
        det = c00 * c11 - c01 * c01
        return np.sqrt((c11 * dr * dr - 2 * c01 * dr * db + c00 * db * db) / det)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "cov": self.cov.tolist(), "tau": self.tau,
                "noise": self.noise, "d_min": self.d_min}

    @classmethod
    def from_dict(cls, d: dict) -> "PanelColor":
        return cls(np.array(d["mean"]), np.array(d["cov"]), d.get("tau", 4.5),
                   d.get("noise", 0.012), d.get("d_min", 8.0))
