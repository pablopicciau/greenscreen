"""Parametri configurabili della pipeline, raggruppati per fase.

I valori di default sono pensati per un pannello verde opaco ripreso con uno
smartphone in Full HD. Ogni parametro è documentato: quando qualcosa non
funziona sul video reale, è qui che si interviene.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class KeyParams:
    """Chiave cromatica (riconoscimento del verde)."""

    # Rilevamento "grezzo" del pannello: un pixel è verde se
    # G - max(R, B) >= d_min  E  (G - max(R, B)) / G >= r_min.
    # Il rapporto rende il test indipendente dalla luminosità (ombre comprese).
    d_min: float = 14.0
    r_min: float = 0.16
    # Chiave morbida (alpha) per il compositing: rapporto tra "verdezza" del
    # pixel e verdezza del pannello pulito nello stesso punto.
    # Sotto 'low' -> alpha 0 (primo piano, es. mano); sopra 'high' -> alpha 1.
    low: float = 0.15
    high: float = 0.70
    # Rimozione del riflesso verde sulle mani (0 = no, 1 = completa).
    despill: float = 1.0


@dataclass
class DetectParams:
    """Riconoscimento del quadrilatero in un singolo fotogramma."""

    # Area minima del pannello rispetto al fotogramma (scarta rumore/piante).
    min_area_frac: float = 0.004
    # Tolleranza (frazione della diagonale del quad) per assegnare i punti di
    # contorno ai lati nelle iterazioni di rifinitura.
    assign_tol: tuple[float, ...] = (0.10, 0.04, 0.015)
    # Soglia RANSAC (pixel) per gli inlier della retta di un lato.
    ransac_px: float = 2.0
    # Copertura minima (0-1) di un lato perché la sua retta sia considerata
    # misurata; sotto questa soglia il lato viene predetto dal movimento.
    min_side_support: float = 0.18
    # Sotto questa copertura un lato ha "poca evidenza": se è l'unico debole
    # viene ricostruito dagli altri tre invece che misurato.
    weak_support: float = 0.35
    # Rifinitura sub-pixel dei bordi campionando la chiave lungo le normali.
    subpixel: bool = True
    subpixel_samples: int = 60
    subpixel_search_px: float = 6.0
    # Margine attorno al quad previsto entro cui cercare (frazione della
    # dimensione del quad). Il fotogramma intero si usa se il tracking è perso.
    roi_margin: float = 0.35


@dataclass
class TrackParams:
    """Tracking temporale e smoothing (pass 1 e 2)."""

    # Fotogrammi consecutivi senza rilevamento prima di considerare perso il tracking.
    max_lost: int = 8
    # Smorzamento della predizione a velocità costante (0 = nessuna velocità).
    velocity_damping: float = 0.8
    # Semi-finestra (fotogrammi) della regressione quadratica locale. 0 = niente smoothing.
    smooth_half_window: int = 2
    # Deviazione (pixel) oltre la quale una misura è considerata anomala.
    outlier_px: float = 6.0


@dataclass
class RenderParams:
    """Inserimento dell'immagine e compositing."""

    # 'fill' = riempi (ritaglia l'immagine), 'fit' = adatta (bande ai lati),
    # 'stretch' = deforma (sconsigliato, cambia le proporzioni dell'opera).
    mode: str = "fill"
    # Rapporto larghezza/altezza fisico del pannello; None = stima automatica.
    panel_aspect: float | None = None
    # Sfondo delle bande in modalità 'fit': 'blur', 'white', 'black' o '#rrggbb'.
    fit_background: str = "blur"
    # Pixel di cui allargare la maschera del quad: i bordi veri li decide la
    # chiave, così piccoli errori geometrici non lasciano filetti verdi.
    edge_expand: float = 3.0
    # Trasferimento dell'illuminazione del pannello all'immagine (0-1).
    lighting: float = 0.7
    # Intensità della grana/rumore della camera trasferita sull'immagine (0-1).
    grain: float = 1.0
    # Margine (pixel) attorno al quad entro cui togliere il riflesso verde.
    despill_margin: float = 40.0


@dataclass
class EncodeParams:
    """Codifica del video finale."""

    # 'libx264' (CPU, massima qualità) oppure 'h264_nvenc' (GPU NVIDIA, veloce).
    codec: str = "libx264"
    crf: int = 17           # qualità x264 (più basso = migliore; 17 ≈ visivamente lossless)
    preset: str = "slow"    # x264: slow; nvenc: p1..p7 (viene convertito)
    keep_audio: bool = True
    audio_bitrate: str = "192k"


@dataclass
class Config:
    """Configurazione completa."""

    key: KeyParams = field(default_factory=KeyParams)
    detect: DetectParams = field(default_factory=DetectParams)
    track: TrackParams = field(default_factory=TrackParams)
    render: RenderParams = field(default_factory=RenderParams)
    encode: EncodeParams = field(default_factory=EncodeParams)

    def to_dict(self) -> dict:
        """Serializzazione (salvata nel file di tracking per riproducibilità)."""
        return asdict(self)
