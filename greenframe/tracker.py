"""Tracking temporale del pannello (passata 1) e struttura dati della traccia.

Strategia: "tracking by detection"
---------------------------------
Un pannello verde uniforme non ha texture interna: optical flow e feature
matching (ORB/SIFT/KLT) non hanno nulla da agganciare *sulla* superficie.
Ha invece bordi nettissimi. Per questo ad ogni fotogramma si **ri-misura**
il quad (detector.py, pochi millisecondi), usando la posizione prevista dal
movimento solo come guida (ROI, assegnazione dei punti ai lati, lati nascosti).
Vantaggi: nessuna deriva accumulata, recupero automatico immediato quando il
pannello torna visibile.

Durante il tracking si stima anche il **rapporto d'aspetto reale** del
pannello (e la focale della camera) dai fotogrammi in cui tutti e 4 i lati
sono ben visibili: serve per inserire l'immagine senza deformarla.

La traccia (angoli grezzi, confidenza, angoli lisciati, modello) viene salvata
in JSON: si analizza il video **una volta** e si possono poi generare quanti
video si vuole con quadri diversi, in pochi secondi ciascuno.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import geometry as geo
from .config import Config
from .detector import Detection, detect_quad

TRACK_VERSION = 1

# Stati possibili di un fotogramma nella traccia.
STATUS_OK = "ok"              # misurato
STATUS_MODEL = "ricostruito"  # misurato, con un lato nascosto ricostruito dagli altri tre
STATUS_PREDICTED = "previsto" # non misurato: posizione estrapolata dal movimento
STATUS_LOST = "perso"         # nessuna stima: il fotogramma resta originale


@dataclass
class Track:
    """Risultato dell'analisi di un video."""

    width: int
    height: int
    fps: float
    raw: np.ndarray                     # (T, 4, 2) angoli misurati, NaN se assenti
    confidence: np.ndarray              # (T,) 0-1
    status: list[str]                   # (T,) uno degli STATUS_*
    panel_luma: np.ndarray              # (T,) luminosità del pannello (NaN se assente)
    hidden_side: np.ndarray | None = None  # (T,) lato ricostruito dagli altri tre, -1 = nessuno
    support: np.ndarray | None = None      # (T, 4) copertura visibile di ciascun lato (0-1)
    smooth: np.ndarray | None = None    # (T, 4, 2) angoli finali, NaN = non sostituire
    panel_aspect: float | None = None   # larghezza/altezza fisica del pannello
    aspect_source: str = "sconosciuto"  # 'utente' | 'stimato' | 'predefinito'
    focal: float | None = None          # focale stimata (pixel)
    focal_reliable: bool = False
    green_ratio: float | None = None    # purezza del verde del pannello
    reasons: list[str] = field(default_factory=list)  # motivo per i fotogrammi falliti
    config: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.status)

    # -- serializzazione --------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Salva la traccia in JSON (NaN -> null)."""
        def arr(a):
            if a is None:
                return None
            return [None if np.isnan(np.asarray(x, dtype=float)).any() else np.round(x, 4).tolist()
                    for x in a]

        data = {
            "version": TRACK_VERSION,
            "width": self.width, "height": self.height, "fps": self.fps,
            "raw": arr(self.raw), "smooth": arr(self.smooth),
            "confidence": np.round(self.confidence, 4).tolist(),
            "status": self.status,
            "hidden_side": None if self.hidden_side is None else [int(v) for v in self.hidden_side],
            "support": None if self.support is None else np.round(self.support, 3).tolist(),
            "panel_luma": [None if np.isnan(v) else round(float(v), 2) for v in self.panel_luma],
            "panel_aspect": self.panel_aspect, "aspect_source": self.aspect_source,
            "focal": self.focal, "focal_reliable": self.focal_reliable,
            "green_ratio": self.green_ratio, "reasons": self.reasons, "config": self.config,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "Track":
        """Carica una traccia salvata con :meth:`save`."""
        data = json.loads(Path(path).read_text())

        def arr(lst):
            if lst is None:
                return None
            return np.array([np.full((4, 2), np.nan) if x is None else np.array(x, float) for x in lst])

        return cls(
            width=data["width"], height=data["height"], fps=data["fps"],
            raw=arr(data["raw"]), smooth=arr(data["smooth"]),
            confidence=np.array(data["confidence"], float), status=list(data["status"]),
            panel_luma=np.array([np.nan if v is None else v for v in data["panel_luma"]], float),
            hidden_side=None if data.get("hidden_side") is None else np.array(data["hidden_side"], int),
            support=None if data.get("support") is None else np.array(data["support"], float),
            panel_aspect=data.get("panel_aspect"), aspect_source=data.get("aspect_source", "?"),
            focal=data.get("focal"), focal_reliable=data.get("focal_reliable", False),
            green_ratio=data.get("green_ratio"), reasons=data.get("reasons", []),
            config=data.get("config", {}),
        )

    def summary(self) -> str:
        """Riepilogo leggibile della qualità del tracking."""
        n = self.n_frames
        counts = {s: self.status.count(s) for s in (STATUS_OK, STATUS_MODEL, STATUS_PREDICTED, STATUS_LOST)}
        usable = 0 if self.smooth is None else int((~np.isnan(self.smooth[:, 0, 0])).sum())
        lines = [
            f"Fotogrammi: {n}  ({n / max(self.fps, 1e-6):.1f} s a {self.fps:.3f} fps)",
            f"  misurati: {counts[STATUS_OK]}   con lato ricostruito: {counts[STATUS_MODEL]}   "
            f"previsti: {counts[STATUS_PREDICTED]}   persi: {counts[STATUS_LOST]}",
            f"  fotogrammi con sostituzione: {usable}/{n}",
        ]
        if self.panel_aspect:
            lines.append(f"Rapporto d'aspetto del pannello: {self.panel_aspect:.4f} ({self.aspect_source})")
        if self.focal:
            lines.append(f"Focale stimata: {self.focal:.0f} px ({'affidabile' if self.focal_reliable else 'predefinita/incerta'})")
        valid = self.confidence[[s in (STATUS_OK, STATUS_MODEL) for s in self.status]]
        if valid.size:
            lines.append(f"Confidenza media: {valid.mean():.2f}")
        return "\n".join(lines)


class PanelModelEstimator:
    """Stima incrementale di focale e rapporto d'aspetto del pannello."""

    def __init__(self, image_size: tuple[int, int], user_aspect: float | None = None,
                 min_samples: int = 8):
        self.image_size = image_size
        self.user_aspect = user_aspect
        self.min_samples = min_samples
        diag = float(np.hypot(*image_size))
        self.default_focal = 0.6 * diag   # smartphone ≈ 26 mm equivalenti
        self.focals: list[float] = []
        self.quads: list[np.ndarray] = []

    def add(self, quad: np.ndarray, det: Detection) -> None:
        """Aggiunge un campione se il rilevamento è di ottima qualità."""
        if not det.measured.all() or det.support.min() < 0.5 or det.completed:
            return
        _, focal = geo.estimate_aspect_ratio(quad, self.image_size)
        if focal is not None:
            self.focals.append(focal)
        self.quads.append(quad.copy())
        if len(self.quads) > 300:  # memoria limitata: bastano gli ultimi campioni
            self.quads.pop(0)

    def focal(self) -> tuple[float, bool]:
        """Focale (mediana robusta) e flag di affidabilità."""
        if len(self.focals) >= self.min_samples:
            f = np.array(self.focals)
            med = np.median(f)
            mad = np.median(np.abs(f - med)) * 1.4826
            inl = f[np.abs(f - med) <= 3 * mad + 1e-6]
            med = float(np.median(inl))
            spread = float(np.median(np.abs(inl - med)) * 1.4826 / med)
            return med, spread < 0.12
        return self.default_focal, False

    def aspect(self, focal: float) -> tuple[float | None, str]:
        """Rapporto d'aspetto: quello indicato dall'utente o la mediana stimata."""
        if self.user_aspect is not None:
            return self.user_aspect, "utente"
        if len(self.quads) >= self.min_samples:
            ratios = [geo.estimate_aspect_ratio(q, self.image_size, fixed_focal=focal)[0] for q in self.quads]
            return float(np.median(ratios)), "stimato"
        return None, "sconosciuto"


def _predict(history: list[tuple[int, np.ndarray]], frame_idx: int, damping: float,
             max_gap: int) -> np.ndarray | None:
    """Predizione a velocità costante (smorzata) del quad al fotogramma corrente."""
    if not history:
        return None
    i1, q1 = history[-1]
    gap = frame_idx - i1
    if gap > max_gap:
        return None
    if len(history) >= 2:
        i2, q2 = history[-2]
        if i1 - i2 == 1:
            return q1 + damping * (q1 - q2) * gap
    return q1.copy()


def track_frames(
    frames: Iterable[np.ndarray],
    image_size: tuple[int, int],
    fps: float,
    cfg: Config,
    progress: Callable[[int], None] | None = None,
) -> Track:
    """Passata 1: rileva il pannello in ogni fotogramma.

    ``frames`` è un iterabile di fotogrammi BGR uint8 (letti una sola volta).
    """
    kp, dp, tp = cfg.key, cfg.detect, cfg.track
    est = PanelModelEstimator(image_size, cfg.render.panel_aspect)
    history: list[tuple[int, np.ndarray]] = []
    raws, confs, status, lumas, reasons, hidden, supports = [], [], [], [], [], [], []
    green_ratio: float | None = None
    last_quad: np.ndarray | None = None
    lost = 0

    for i, frame in enumerate(frames):
        pred = _predict(history, i, tp.velocity_damping, tp.max_lost)
        diag: dict = {}
        det = None
        if pred is not None:
            det = detect_quad(frame, kp, dp, prior=pred, green_ratio=green_ratio, diag=diag)
        if det is None:
            # Recupero: ricerca su tutto il fotogramma, senza predizione.
            det = detect_quad(frame, kp, dp, prior=None, green_ratio=green_ratio, diag=diag)

        if det is not None:
            ref = pred if pred is not None else last_quad
            quad = geo.align_to_reference(det.quad, ref) if ref is not None else geo.order_quad(det.quad)
            raws.append(quad)
            confs.append(det.confidence)
            status.append(STATUS_MODEL if det.completed else STATUS_OK)
            lumas.append(det.panel_luma)
            reasons.append("")
            hidden.append(det.completed[0] if det.completed else -1)
            supports.append(np.where(det.measured, det.support, 0.0))
            history = (history + [(i, quad)])[-3:]
            last_quad = quad
            lost = 0
            est.add(quad, det)
            if det.confidence > 0.5:
                # Media mobile della purezza del verde: soglia adattiva più stabile.
                green_ratio = det.green_ratio if green_ratio is None else 0.9 * green_ratio + 0.1 * det.green_ratio
        else:
            lost += 1
            reasons.append(diag.get("reason", "sconosciuto"))
            lumas.append(np.nan)
            hidden.append(-1)
            supports.append(np.zeros(4))
            if pred is not None and lost <= tp.max_lost:
                # Breve perdita: posizione prevista (verrà rifinita dallo smoothing).
                raws.append(pred)
                confs.append(0.0)
                status.append(STATUS_PREDICTED)
                history = (history + [(i, pred)])[-3:]
            else:
                raws.append(np.full((4, 2), np.nan))
                confs.append(0.0)
                status.append(STATUS_LOST)
                history = []
        if progress is not None:
            progress(i)

    focal, f_ok = est.focal()
    aspect, source = est.aspect(focal)
    if aspect is None:
        # Troppi pochi fotogrammi puliti: si stima comunque dai migliori disponibili.
        good = [q for q, s in zip(raws, status) if s == STATUS_OK]
        if good:
            aspect = float(np.median([geo.estimate_aspect_ratio(q, image_size, fixed_focal=focal)[0]
                                      for q in good]))
            source = "stimato (pochi campioni)"
        else:
            aspect, source = 1.0, "predefinito"

    return Track(
        width=image_size[0], height=image_size[1], fps=fps,
        raw=np.array(raws, dtype=np.float64).reshape(-1, 4, 2),
        confidence=np.array(confs, dtype=np.float64),
        status=status,
        panel_luma=np.array(lumas, dtype=np.float64),
        hidden_side=np.array(hidden, dtype=int),
        support=np.array(supports, dtype=np.float64).reshape(-1, 4),
        panel_aspect=aspect, aspect_source=source,
        focal=focal, focal_reliable=f_ok,
        green_ratio=green_ratio,
        reasons=reasons,
        config=cfg.to_dict(),
    )
