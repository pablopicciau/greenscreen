"""Colore: luce lineare, conversioni YUV <-> RGB e posizione della crominanza.

Perché serve (precisione dei bordi)
-----------------------------------
1. **Luce lineare.** La fotocamera mescola la luce (sfocatura dell'obiettivo,
   sfocatura di movimento, pixel di bordo) in luce *lineare*, poi applica la
   curva gamma (OETF BT.709). Un bordo verde/sfondo sfocato è quindi una rampa
   lineare solo dopo aver tolto la gamma: se si cerca il punto "a metà" sui
   valori gamma, il bordo risulta spostato verso il lato più scuro, tanto più
   quanto più il bordo è sfocato (fino a 1-2 px con il movimento).
   Anche luce, ombre e motion blur del quadro inserito vanno calcolati in luce
   lineare per comportarsi come nella realtà.
2. **Crominanza 4:2:0.** Telefoni e codec salvano il colore a metà
   risoluzione; ogni campione di crominanza ha una posizione precisa rispetto
   ai pixel di luminanza (``chroma_location``: in H.264/HEVC di solito
   "left", cioè allineato alle colonne pari). Ricostruire il colore
   ignorandola sposta di ~0.25-0.5 px tutti i bordi di colore (proprio quelli
   del pannello verde). Qui la crominanza viene interpolata nella posizione
   giusta e poi resa più nitida trasferendo il dettaglio della luminanza
   (che è a piena risoluzione) dove le due sono correlate localmente.

Le funzioni lavorano su tensori PyTorch (CPU o GPU) e, dove indicato, su
array NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Funzione di trasferimento BT.709 (OETF) e sua inversa
# ---------------------------------------------------------------------------

# Costanti esatte (BT.2020/BT.709 in precisione piena): con i valori arrotondati
# 1.099 / 0.018 la curva ha un piccolo salto e non è esattamente invertibile.
_ALPHA = 1.09929682680944
_BETA = 0.018053968510807
_BETA_V = 4.5 * _BETA  # 0.08124


def to_linear_np(v: np.ndarray) -> np.ndarray:
    """Valori video 0-1 (gamma BT.709) -> luce lineare 0-1 (simmetrica sui negativi)."""
    v = np.asarray(v, dtype=np.float32)
    a = np.abs(v)
    lin = np.where(a < _BETA_V, a / 4.5, ((a + (_ALPHA - 1)) / _ALPHA) ** (1 / 0.45))
    return (np.sign(v) * lin).astype(np.float32)


def from_linear_np(lin: np.ndarray) -> np.ndarray:
    """Luce lineare -> valori video 0-1 (OETF BT.709)."""
    lin = np.asarray(lin, dtype=np.float32)
    a = np.abs(lin)
    v = np.where(a < _BETA, 4.5 * a, _ALPHA * np.power(np.maximum(a, _BETA), 0.45) - (_ALPHA - 1))
    return (np.sign(lin) * v).astype(np.float32)


# Tabella 0-255 -> luce lineare (per immagini uint8).
LINEAR_LUT = to_linear_np(np.arange(256, dtype=np.float32) / 255.0)


def to_linear_u8(img_u8: np.ndarray) -> np.ndarray:
    """Immagine uint8 -> float32 in luce lineare (0-1) tramite tabella."""
    return LINEAR_LUT[img_u8]


def to_linear(v):
    """Come :func:`to_linear_np` ma per tensori PyTorch."""
    import torch

    a = v.abs()
    lin = torch.where(a < _BETA_V, a / 4.5, ((a + (_ALPHA - 1)) / _ALPHA) ** (1 / 0.45))
    return torch.sign(v) * lin


def from_linear(lin):
    """Come :func:`from_linear_np` ma per tensori PyTorch."""
    import torch

    a = lin.abs()
    v = torch.where(a < _BETA, 4.5 * a, _ALPHA * a.clamp_min(_BETA) ** 0.45 - (_ALPHA - 1))
    return torch.sign(lin) * v


# Coefficienti di luminanza (spazio gamma, per Y') e in luce lineare (BT.709).
LUMA_709 = (0.2126, 0.7152, 0.0722)  # R, G, B


# ---------------------------------------------------------------------------
# YUV
# ---------------------------------------------------------------------------

# Coefficienti (Kr, Kb) delle matrici YCbCr.
_MATRICES = {
    "bt709": (0.2126, 0.0722),
    "bt601": (0.299, 0.114),
    "bt2020": (0.2627, 0.0593),
}
_MATRIX_ALIASES = {
    "bt709": "bt709", "": "bt709", "unknown": "bt709", "gbr": "bt709",
    "smpte170m": "bt601", "bt470bg": "bt601", "bt601": "bt601", "fcc": "bt601",
    "bt2020nc": "bt2020", "bt2020c": "bt2020", "bt2020": "bt2020",
}

# Posizione del campione di crominanza (0, 0) in coordinate di luminanza
# (x, y), per ciascun valore di ``chroma_location`` di FFmpeg.
_SITING = {
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "topleft": (0.0, 0.0),
    "top": (0.5, 0.0),
    "bottomleft": (0.0, 1.0),
    "bottom": (0.5, 1.0),
}


def matrix_coeffs(name: str | None) -> tuple[float, float]:
    """(Kr, Kb) per il nome di matrice di FFmpeg (default BT.709)."""
    return _MATRICES[_MATRIX_ALIASES.get((name or "").lower(), "bt709")]


def chroma_siting(location: str | None, rotation_cw: int = 0) -> tuple[float, float]:
    """Offset (ox, oy) del campione di crominanza dopo una rotazione oraria.

    ``location`` è il valore ``chroma_location`` di ffprobe ("left" se non
    specificato, come prevede lo standard H.264/HEVC per il 4:2:0).
    Quando FFmpeg ruota il video (metadato di rotazione degli smartphone)
    traspone anche i piani di crominanza: la posizione dei campioni cambia
    e va ricalcolata, altrimenti i bordi verticali e orizzontali si scambiano
    l'errore.
    """
    ox, oy = _SITING.get((location or "left").lower(), _SITING["left"])
    rot = rotation_cw % 360
    if rot == 90:
        ox, oy = 1.0 - oy, ox
    elif rot == 180:
        ox, oy = 1.0 - ox, 1.0 - oy
    elif rot == 270:
        ox, oy = oy, 1.0 - ox
    return ox, oy


@dataclass
class YuvFormat:
    """Descrizione dei campioni YUV di un video."""

    matrix: str = "bt709"
    full_range: bool = False
    siting: tuple[float, float] = (0.0, 0.5)   # posizione della crominanza (left)


# ---------------------------------------------------------------------------
# Operazioni con PyTorch
# ---------------------------------------------------------------------------

def _sample_at(img, xs, ys):
    """Interpolazione bilineare di ``img`` (C, h, w) nei punti di una griglia.

    ``xs`` (W,) e ``ys`` (H,) sono coordinate continue nel sistema dei pixel
    di ``img`` (centro del pixel = coordinata intera). Bordi replicati.
    """
    import torch

    c, h, w = img.shape
    x = xs.clamp(0, w - 1)
    y = ys.clamp(0, h - 1)
    x0 = x.floor().long().clamp(0, w - 1)
    y0 = y.floor().long().clamp(0, h - 1)
    x1 = (x0 + 1).clamp(max=w - 1)
    y1 = (y0 + 1).clamp(max=h - 1)
    fx = (x - x0.to(x.dtype)).view(1, 1, -1)
    fy = (y - y0.to(y.dtype)).view(1, -1, 1)
    top = img[:, y0][:, :, x0] * (1 - fx) + img[:, y0][:, :, x1] * fx
    bot = img[:, y1][:, :, x0] * (1 - fx) + img[:, y1][:, :, x1] * fx
    return top * (1 - fy) + bot * fy


def _upsample_sited(c_small, height: int, width: int, siting: tuple[float, float]):
    """Porta la crominanza (C, h, w) alla risoluzione di luminanza, nella posizione giusta."""
    import torch

    ox, oy = siting
    dev = c_small.device
    xs = (torch.arange(width, device=dev, dtype=torch.float32) - ox) / 2.0
    ys = (torch.arange(height, device=dev, dtype=torch.float32) - oy) / 2.0
    return _sample_at(c_small, xs, ys)


def _tent_filter(img):
    """Filtro [1, 2, 1]/4 orizzontale e verticale (C, H, W), bordi replicati."""
    import torch.nn.functional as F

    x = F.pad(img[None], (1, 1, 1, 1), mode="replicate")[0]
    x = 0.25 * x[:, :, :-2] + 0.5 * x[:, :, 1:-1] + 0.25 * x[:, :, 2:]
    return 0.25 * x[:, :-2] + 0.5 * x[:, 1:-1] + 0.25 * x[:, 2:]


def _downsample_sited(img, siting: tuple[float, float]):
    """Campiona (C, H, W) nelle posizioni della crominanza 4:2:0 (filtro a tenda)."""
    import torch

    ox, oy = siting
    c, h, w = img.shape
    dev = img.device
    xs = 2.0 * torch.arange((w + 1) // 2, device=dev, dtype=torch.float32) + ox
    ys = 2.0 * torch.arange((h + 1) // 2, device=dev, dtype=torch.float32) + oy
    return _sample_at(_tent_filter(img), xs, ys)


def _box3(img):
    """Media 3x3 (C, h, w) con bordi replicati."""
    import torch.nn.functional as F

    return F.avg_pool2d(F.pad(img[None], (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0]


def yuv420_to_rgb(y, u, v, fmt: YuvFormat, detail: bool = True):
    """Fotogramma YUV 4:2:0 (tensori uint8) -> RGB float32 (3, H, W) in 0-1 (gamma).

    ``detail=True`` rende la crominanza più nitida usando la luminanza: per
    ogni piccola zona si stima la relazione lineare tra crominanza e
    luminanza (a livello dei campioni di crominanza) e si aggiunge alla
    crominanza interpolata il dettaglio di luminanza perso nel 4:2:0, solo
    nella misura in cui le due sono correlate. Su un bordo tra due colori la
    relazione è esatta e il bordo di colore torna a piena risoluzione; dove
    non c'è correlazione resta la normale interpolazione bilineare.
    """
    import torch

    kr, kb = matrix_coeffs(fmt.matrix)
    kg = 1.0 - kr - kb
    Y = y.to(torch.float32)
    U = u.to(torch.float32)
    V = v.to(torch.float32)
    if fmt.full_range:
        Yn, Pb, Pr = Y / 255.0, (U - 128.0) / 255.0, (V - 128.0) / 255.0
    else:
        Yn, Pb, Pr = (Y - 16.0) / 219.0, (U - 128.0) / 224.0, (V - 128.0) / 224.0
    H, W = Yn.shape
    chroma = torch.stack([Pb, Pr])                                  # (2, h, w)
    if chroma.shape[1:] == (H, W):                                  # 4:4:4
        up = chroma
    else:
        up = _upsample_sited(chroma, H, W, fmt.siting)
        if detail:
            y_ds = _downsample_sited(Yn[None], fmt.siting)          # (1, h, w)
            m_y = _box3(y_ds)
            m_c = _box3(chroma)
            cov = _box3(y_ds * chroma) - m_y * m_c
            var = _box3(y_ds * y_ds) - m_y * m_y
            # eps: varianza del rumore di luminanza (~1.5 livelli su 219).
            slope = (cov / (var + (1.5 / 219.0) ** 2)).clamp(-3.0, 3.0)
            slope_up = _upsample_sited(slope, H, W, fmt.siting)
            y_lost = Yn[None] - _upsample_sited(y_ds, H, W, fmt.siting)
            up = up + slope_up * y_lost
    pb, pr = up[0], up[1]
    r = Yn + 2.0 * (1.0 - kr) * pr
    b = Yn + 2.0 * (1.0 - kb) * pb
    g = (Yn - kr * r - kb * b) / kg
    return torch.stack([r, g, b])


def rgb_to_yuv420(rgb, fmt: YuvFormat):
    """RGB float (3, H, W) in 0-1 (gamma) -> piani YUV 4:2:0 uint8 (Y, U, V).

    La crominanza viene filtrata e campionata nella posizione indicata da
    ``fmt.siting`` (normalmente "left", come si aspettano i decoder).
    """
    import torch

    kr, kb = matrix_coeffs(fmt.matrix)
    kg = 1.0 - kr - kb
    r, g, b = rgb[0], rgb[1], rgb[2]
    yn = kr * r + kg * g + kb * b
    pb = (b - yn) / (2.0 * (1.0 - kb))
    pr = (r - yn) / (2.0 * (1.0 - kr))
    chroma = _downsample_sited(torch.stack([pb, pr]), fmt.siting)
    if fmt.full_range:
        Y = yn * 255.0
        C = chroma * 255.0 + 128.0
    else:
        Y = yn * 219.0 + 16.0
        C = chroma * 224.0 + 128.0
    to_u8 = lambda t: (t + 0.5).floor().clamp(0, 255).to(torch.uint8)
    return to_u8(Y), to_u8(C[0]), to_u8(C[1])
