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
    # pixel e verdezza del pannello pulito nello stesso punto (in luce lineare).
    # Sotto 'low' -> alpha 0 (primo piano, es. mano); sopra 'high' -> alpha 1.
    # Nella fascia di transizione alpha viene poi ricalcolato esattamente con
    # il modello a due colori (vedi render.py).
    low: float = 0.10
    high: float = 0.85
    # Ricalcolo esatto di alpha nella fascia di transizione (modello a due colori).
    unmix: bool = True
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
    # Test della giunzione a T: se il bordo di un lato prosegue dritto oltre
    # l'angolo per almeno questa frazione del prolungamento, il lato è il bordo
    # di un oggetto davanti al pannello e viene ricostruito dagli altri tre.
    impostor_threshold: float = 0.6
    # Oltre questa area (pixel) la ricerca grossolana del pannello si fa a metà
    # risoluzione (più veloce); la precisione finale la dà la rifinitura sub-pixel.
    coarse_half_res_px: int = 600_000
    # Nell'analisi la crominanza 4:2:0 viene resa più nitida con la luminanza
    # (vedi color.yuv420_to_rgb). False = sola interpolazione bilineare.
    chroma_detail: bool = True


@dataclass
class TrackParams:
    """Tracking temporale e smoothing (pass 1 e 2)."""

    # Fotogrammi consecutivi senza rilevamento prima di considerare perso il tracking.
    max_lost: int = 8
    # Smorzamento della predizione a velocità costante (0 = nessuna velocità).
    velocity_damping: float = 0.8
    # Semi-finestra (fotogrammi) della regressione quadratica locale. 0 = niente smoothing.
    smooth_half_window: int = 2
    # Deviazione (pixel) oltre la quale una misura è considerata anomala: il
    # suo peso nello smoothing scende gradualmente e si annulla al doppio di
    # questo valore. Nei movimenti bruschi la soglia si allarga da sola (vedi
    # ``smoothing._outlier_threshold``). Nei video di prova Full HD, rispetto
    # alla vecchia soglia di 6 px, l'errore massimo scende da 2.9 a 1.8 px nel
    # movimento veloce e da 3.4 a 1.4 px con l'oggetto dritto davanti a un lato.
    outlier_px: float = 1.5
    # Controllo di coerenza offline: un fotogramma il cui quad implica un
    # rapporto d'aspetto diverso da quello del pannello (stimato su tutto il
    # video) di più di questa frazione ha un lato sbagliato (tipicamente il
    # bordo di un oggetto davanti al pannello) che viene ricostruito.
    # La soglia effettiva è max(aspect_tolerance, 3 × deviazione mediana).
    aspect_tolerance: float = 0.02
    # I lati visibili per almeno questa frazione non vengono mai messi in dubbio.
    trusted_support: float = 0.9


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
    edge_expand: float = 2.0
    # Trasferimento dell'illuminazione del pannello all'immagine (0-1).
    # In luce lineare 1 è il comportamento fisico (il quadro riceve la stessa
    # luce e le stesse ombre del pannello); valori minori attenuano le ombre.
    lighting: float = 1.0
    # Intensità della grana/rumore della camera trasferita sull'immagine (0-1).
    grain: float = 1.0
    # Margine (pixel) attorno al quad entro cui togliere il riflesso verde dalle mani.
    despill_margin: float = 10.0
    # Dispositivo di calcolo: 'auto' (GPU NVIDIA se disponibile), 'cuda' o 'cpu'.
    device: str = "auto"
    # Sfocatura di movimento del quadro inserito, con l'esposizione stimata dal video.
    motion_blur: bool = True
    # Frazione del tempo tra fotogrammi con otturatore aperto (0-1, 0.5 = 180°).
    # None = stimata dai bordi del pannello durante l'analisi.
    shutter: float | None = None
    # Sfocatura gaussiana (px) applicata al quadro per eguagliare la nitidezza
    # della camera. None = stimata dai bordi del pannello; 0 = nessuna.
    lens_blur: float | None = None
    # Numero massimo di istanti per il motion blur (GPU; sulla CPU al massimo 6).
    motion_samples: int = 16


@dataclass
class EncodeParams:
    """Codifica del video finale."""

    # 'libx264' (CPU, massima qualità) oppure 'h264_nvenc' (GPU NVIDIA, veloce).
    codec: str = "libx264"
    crf: int = 17           # qualità x264 (più basso = migliore; 17 ≈ visivamente lossless)
    preset: str = "slow"    # x264: slow; nvenc: p1..p7 (viene convertito)
    keep_audio: bool = True
    # Audio AAC copiato senza ricodifica (nessuna perdita); altrimenti AAC a questo bitrate.
    copy_audio: bool = True
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
