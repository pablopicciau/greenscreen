"""Riconoscimento del quadrilatero verde in un singolo fotogramma.

Idea chiave (robustezza alle occlusioni)
----------------------------------------
Gli angoli **non** vengono cercati direttamente: se una mano copre un angolo
non c'è nulla da vedere. Si stimano invece le **4 rette dei lati** usando solo
i tratti di bordo visibili (fit RANSAC, che scarta il profilo delle dita) e si
ricavano gli angoli come intersezioni di lati adiacenti. Basta che ogni lato
sia visibile anche solo per circa il 20% della sua lunghezza.

Se un lato è completamente coperto, viene **ricostruito dagli altri tre**:
i lati visibili fissano due angoli e le rette degli altri due; le lunghezze
mancanti si prendono, in proporzione, dal fotogramma precedente
(``geometry.complete_by_ratio``). Se mancano due o più lati si usa la
posizione prevista dal movimento.

Nella passata di smoothing (offline) la posizione degli angoli ricostruiti
viene poi corretta interpolando tra il fotogramma prima e quello dopo
l'occlusione, eliminando la deriva.

Un oggetto davanti a un lato può avere tratti di bordo quasi dritti che
vengono scambiati per il lato vero: a fine analisi un controllo di coerenza
globale (``smoothing.flag_inconsistent_sides``) li riconosce dal rapporto
d'aspetto. Limite residuo: un oggetto con un bordo **perfettamente dritto**
che copre un intero lato non sempre viene corretto.

Alternative valutate e scartate:

* ricostruire il lato con i punti di fuga e il rapporto d'aspetto fisico:
  esatto in teoria ma instabile (0.05° di errore sulle direzioni dei lati
  diventano 15-35 px sugli angoli);
* riconoscere il bordo-impostore perché "salta" rispetto alla predizione:
  funziona all'ingresso dell'oggetto ma, se la stima deriva, all'uscita il
  bordo vero viene scambiato a sua volta per un impostore (blocco).

Passi
-----
1. maschera verde (keyer) nella ROI attorno alla posizione prevista;
2. componente connessa del pannello (la più "quadrilatera" se non c'è
   predizione, quella che sovrappone la predizione altrimenti);
3. contorno esterno -> quad iniziale (involucro convesso o predizione);
4. rifinitura iterativa: punti di contorno assegnati ai lati, RANSAC, intersezioni;
5. rifinitura sub-pixel: transizione verde/non-verde campionata lungo le
   normali ai lati, nuovo fit RANSAC;
6. gestione dei lati nascosti (ricostruzione dagli altri tre);
7. validazione geometrica e calcolo della confidenza.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as geo
from .color import LINEAR_LUT
from .config import DetectParams, KeyParams
from .keyer import PanelColor, generic_green_mask


@dataclass
class Detection:
    """Risultato del rilevamento su un fotogramma."""

    quad: np.ndarray          # (4, 2) TL, TR, BR, BL in coordinate del fotogramma
    support: np.ndarray       # (4,) copertura visibile di ciascun lato (0-1)
    measured: np.ndarray      # (4,) bool: lato misurato (True) o dedotto (False)
    confidence: float         # 0-1, sintesi della qualità
    panel_luma: float         # luminosità (percentile 90) del pannello visibile
    green_area: float         # area (px) di verde visibile
    green_ratio: float        # "purezza" mediana del verde del pannello, (G-max(R,B))/G
    completed: list[int] = field(default_factory=list)  # lati ricostruiti dagli altri tre
    # Larghezza dei bordi (px, metà della distanza tra i livelli 75% e 25%), NaN se non misurata.
    edge_width: np.ndarray = field(default_factory=lambda: np.full(4, np.nan))
    # Modello di colore del pannello usato (o trovato) per questo fotogramma.
    color: PanelColor | None = None
    # Probabilità (0-1) che ogni lato sia il bordo di un oggetto davanti (NaN = non misurabile).
    impostor: np.ndarray = field(default_factory=lambda: np.full(4, np.nan))


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def _roi_from_prior(prior: np.ndarray | None, shape: tuple[int, int], margin: float):
    """Rettangolo di ricerca (x0, y0, x1, y1) attorno al quad previsto."""
    h, w = shape
    if prior is None:
        return 0, 0, w, h
    x0, y0 = prior.min(axis=0)
    x1, y1 = prior.max(axis=0)
    pad = margin * max(x1 - x0, y1 - y0) + 16
    return (
        int(max(0, np.floor(x0 - pad))),
        int(max(0, np.floor(y0 - pad))),
        int(min(w, np.ceil(x1 + pad))),
        int(min(h, np.ceil(y1 + pad))),
    )


def _clean(mask: np.ndarray) -> np.ndarray:
    """Pulizia morfologica leggera di una maschera uint8 (0/1)."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _quad_likeness(comp: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Quanto una componente somiglia a un quadrilatero pieno (0-1) e il quad grossolano.

    Si confronta la componente con il quadrilatero che meglio ne approssima
    l'involucro convesso (IoU, calcolata a risoluzione ridotta). Un pannello,
    anche con mani sui bordi, supera 0.85; una pianta o una macchia
    tondeggiante restano sotto 0.7; pannello e pianta fusi insieme pure.
    """
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None
    contour = max(contours, key=cv2.contourArea)
    quad = _initial_quad_from_contour(contour.astype(np.float32))
    if quad is None:
        return 0.0, None
    scale = min(1.0, 400.0 / max(comp.shape))
    small = cv2.resize(comp, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale < 1 else comp
    qmask = np.zeros(small.shape, np.uint8)
    cv2.fillConvexPoly(qmask, np.round(quad * scale).astype(np.int32), 1)
    inter = float(np.count_nonzero(small & qmask))
    union = float(np.count_nonzero(small | qmask))
    return inter / max(union, 1.0), quad


def _initial_quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Quad grossolano: involucro convesso semplificato a 4 vertici.

    La ricerca binaria su epsilon di approxPolyDP trova la semplificazione
    con esattamente 4 vertici; se non esiste si usa il rettangolo ruotato
    di area minima.
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    lo, hi = 0.002 * peri, 0.25 * peri
    approx = None
    for _ in range(30):
        eps = 0.5 * (lo + hi)
        cand = cv2.approxPolyDP(hull, eps, True)
        if len(cand) > 4:
            lo = eps
        elif len(cand) < 4:
            hi = eps
        else:
            approx = cand
            break
    if approx is None:
        approx = cv2.boxPoints(cv2.minAreaRect(hull))
    return geo.order_quad(np.asarray(approx, dtype=np.float64).reshape(-1, 2))


def _candidate_masks(roi: np.ndarray, kp: KeyParams, color: PanelColor | None):
    """Maschere candidate per cercare il pannello senza predizione.

    Con un modello di colore noto basta quello. Altrimenti (primo fotogramma)
    si parte dal verde "generico" e si separano i diversi verdi presenti
    raggruppando i colori (k-means sulla cromaticità in luce lineare, con 1,
    2 e 3 gruppi): un pannello attaccato a una pianta verde finisce in un
    gruppo diverso dalla pianta. Ogni gruppo diventa un modello di colore.
    Si provano anche soglie di purezza più severe (vecchio criterio).
    """
    if color is not None:
        yield color.mask(roi), color
        return
    generic = generic_green_mask(roi, kp.d_min, kp.r_min)
    yield generic, None
    for thr in (0.25, 0.35, 0.45):
        yield generic_green_mask(roi, kp.d_min, thr), None
    px = roi[generic > 0]
    if len(px) < 200:
        return
    rng = np.random.default_rng(0)
    sample = px[rng.choice(len(px), size=min(len(px), 20000), replace=False)]
    feats, g = PanelColor.features(sample)
    ok = g > 0.02
    sample, feats = sample[ok], feats[ok].astype(np.float32)
    if len(sample) < 200:
        return
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    for k in (1, 2, 3):
        _, labels, _ = cv2.kmeans(feats, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        for c in range(k):
            model = PanelColor.fit(sample[labels == c])
            if model is not None:
                yield model.mask(roi), model


def _best_component_without_prior(roi: np.ndarray, kp: KeyParams, min_area: float,
                                  color: PanelColor | None):
    """Cerca il pannello senza sapere dove sia (primo fotogramma o tracking perso).

    Tra tutte le maschere candidate (vedi :func:`_candidate_masks`) e le loro
    componenti connesse più grandi sceglie la più simile a un quadrilatero
    pieno: punteggio = area × IoU⁴ con il quadrilatero approssimante.
    Ritorna ``(componente, modello di colore o None)``.
    """
    best = None  # (score, comp_mask, model)
    for mask, model in _candidate_masks(roi, kp, color):
        mask = _clean(mask)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_lab <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        for k in np.argsort(areas)[::-1][:3]:
            area = float(areas[k])
            if area < min_area:
                break
            comp = (labels == k + 1).astype(np.uint8)
            iou, _ = _quad_likeness(comp)
            score = area * iou ** 4
            if best is None or score > best[0]:
                best = (score, comp, model)
    return None if best is None else (best[1], best[2])


def _run_mask(occ: np.ndarray, min_run: int) -> np.ndarray:
    """Per una matrice booleana (K, bins), marca gli intervalli che fanno parte
    di sequenze di almeno ``min_run`` intervalli occupati consecutivi."""
    K, bins = occ.shape
    win = np.ones((K, bins - min_run + 1), dtype=bool)
    for k in range(min_run):
        win &= occ[:, k:bins - min_run + 1 + k]
    out = np.zeros_like(occ)
    for k in range(min_run):
        out[:, k:bins - min_run + 1 + k] |= win
    return out


def _fit_side_line(
    pts: np.ndarray,
    t: np.ndarray,
    threshold: float,
    iterations: int = 128,
    bins: int = 40,
    min_run: int = 3,
) -> tuple[np.ndarray | None, float]:
    """Retta di un lato con RANSAC valutato sulla **copertura contigua**.

    Un RANSAC classico premia la retta con più punti vicini; qui invece si
    premia quella che spiega i tratti di bordo contigui più lunghi. Così
    punti sparsi allineati per caso (creste di un bordo irregolare, profili
    di dita) non possono vincere contro un tratto di bordo vero anche corto.
    La retta finale è rifittata solo sui punti dei tratti contigui.

    ``t`` è la posizione (0-1) di ciascun punto lungo il lato.
    Ritorna ``(retta, copertura)``; retta ``None`` se non c'è un bordo credibile.
    """
    n = len(pts)
    if n < 2 * min_run:
        return None, 0.0
    rng = np.random.default_rng(12345)  # deterministico
    k = iterations
    i = rng.integers(0, n, size=k)
    j = rng.integers(0, n - 1, size=k)
    j = j + (j >= i)
    p, q = pts[i], pts[j]
    normals = np.stack([p[:, 1] - q[:, 1], q[:, 0] - p[:, 0]], axis=1)
    norms = np.linalg.norm(normals, axis=1)
    ok = norms > 1e-9
    if not ok.any():
        return None, 0.0
    normals = normals[ok] / norms[ok, None]
    c = -(normals * p[ok]).sum(axis=1)
    inl = np.abs(normals @ pts.T + c[:, None]) < threshold          # (K, N)

    # Occupazione degli intervalli lungo il lato per ogni ipotesi.
    bin_idx = np.clip((t * bins).astype(int), 0, bins - 1)
    occ = np.zeros((inl.shape[0], bins), dtype=bool)
    rows, cols = np.nonzero(inl)
    occ[rows, bin_idx[cols]] = True
    runs = _run_mask(occ, min_run)
    scores = runs.sum(axis=1)
    best = int(np.argmax(scores))
    if scores[best] == 0:
        return None, 0.0

    # Rifinitura: solo i punti inlier che cadono in tratti contigui.
    sel = inl[best] & runs[best][bin_idx]
    line = None
    for _ in range(3):
        if sel.sum() < 2:
            break
        line = geo.fit_line_tls(pts[sel])
        inl_now = np.abs(geo.point_line_distance(pts, line)) < threshold
        occ_now = np.zeros((1, bins), dtype=bool)
        occ_now[0, bin_idx[inl_now]] = True
        runs_now = _run_mask(occ_now, min_run)[0]
        sel = inl_now & runs_now[bin_idx]
    if line is None:
        return None, 0.0
    return line, float(_run_mask(np.isin(np.arange(bins), bin_idx[sel])[None], min_run).sum()) / bins


def _project_on_side(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Parametro t della proiezione dei punti sul segmento a->b (0 in a, 1 in b)."""
    ab = b - a
    return ((points - a) @ ab) / max(float(ab @ ab), 1e-9)


def _refine_with_contour(
    quad: np.ndarray,
    pts: np.ndarray,
    prior: np.ndarray | None,
    dp: DetectParams,
    ransac_px: float | None = None,
    max_pts: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Rifinitura iterativa del quad usando i punti del contorno esterno.

    I punti di ogni lato vengono diradati a ``max_pts`` (sono ordinati lungo
    il contorno, quindi la copertura resta uniforme): la precisione finale la
    dà la rifinitura sub-pixel, qui basta una retta entro 1-2 px.
    """
    ransac_px = dp.ransac_px if ransac_px is None else ransac_px
    prior_lines = geo.quad_lines(prior) if prior is not None else None
    support = np.zeros(4)
    measured = np.zeros(4, dtype=bool)
    for tol in dp.assign_tol:
        diag = max(np.linalg.norm(quad[0] - quad[2]), np.linalg.norm(quad[1] - quad[3]))
        lines = geo.quad_lines(quad)
        # Distanza di ogni punto da ogni lato: (N, 4).
        dist = np.abs(np.stack([geo.point_line_distance(pts, ln) for ln in lines], axis=1))
        nearest = np.argmin(dist, axis=1)
        new_lines = []
        for s in range(4):
            a, b = quad[s], quad[(s + 1) % 4]
            sel = (nearest == s) & (dist[:, s] < tol * diag)
            t = _project_on_side(pts[sel], a, b)
            # Esclude le zone vicine agli angoli (spesso arrotondate dal blur)
            # e i punti fuori dal segmento.
            keep = (t > 0.04) & (t < 0.96)
            side_pts, t = pts[sel][keep], t[keep]
            if len(side_pts) > max_pts:
                stride = int(np.ceil(len(side_pts) / max_pts))
                side_pts, t = side_pts[::stride], t[::stride]
            line, cov = _fit_side_line(side_pts, t, ransac_px)
            if line is not None and cov >= dp.min_side_support:
                new_lines.append(line)
                support[s], measured[s] = cov, True
            else:
                # Lato non visibile: retta dalla predizione (se c'è) o dal quad attuale.
                new_lines.append(prior_lines[s] if prior_lines is not None else lines[s])
                support[s], measured[s] = cov, False
        new_quad = geo.quad_from_lines(new_lines)
        if new_quad is None or not geo.is_convex(new_quad):
            return None
        quad = new_quad
    return quad, support, measured


def _first_crossing(p: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni riga di ``p`` (S, K), primo indice (frazionario) dove p scende sotto ``level``.

    Ritorna ``(posizione, valido)``; la posizione è in unità di campione.
    """
    below = p < level
    k = np.argmax(below, axis=1)
    valid = below.any(axis=1) & (k > 0)
    k = np.clip(k, 1, p.shape[1] - 1)
    rows = np.arange(p.shape[0])
    v0, v1 = p[rows, k - 1], p[rows, k]
    frac = (v0 - level) / np.maximum(v0 - v1, 1e-6)
    return k - 1 + np.clip(frac, 0.0, 1.0), valid


def sample_linear(img_u8: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolazione bilineare **in luce lineare** di un'immagine uint8 (H, W, 3).

    I 4 pixel vicini vengono linearizzati (tabella) prima di essere mescolati:
    equivale a ``remap`` dell'immagine lineare, ma tocca solo i pixel che
    servono (pochi millisecondi anche in Full HD). Bordi replicati.
    """
    h, w = img_u8.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = (x - x0)[..., None].astype(np.float32)
    fy = (y - y0)[..., None].astype(np.float32)
    p00 = LINEAR_LUT[img_u8[y0, x0]]
    p01 = LINEAR_LUT[img_u8[y0, x0 + 1]]
    p10 = LINEAR_LUT[img_u8[y0 + 1, x0]]
    p11 = LINEAR_LUT[img_u8[y0 + 1, x0 + 1]]
    top = p00 + (p01 - p00) * fx
    bot = p10 + (p11 - p10) * fx
    return top + (bot - top) * fy


def _edge_profiles(img_u8: np.ndarray, base: np.ndarray, normal: np.ndarray, steps: np.ndarray,
                   offset: tuple[int, int], residual: bool = False):
    """Profili di colore (luce lineare) lungo la normale, normalizzati 1 (dentro) -> 0 (fuori).

    Il bordo vero è dove la copertura del pannello vale 0.5: in luce lineare
    il colore di un pixel di bordo è la media pesata (con la copertura) dei
    due colori, quindi la proiezione del colore sulla direzione
    "fuori -> dentro" è esattamente la copertura. Vale qualunque sia il
    colore dello sfondo (anche un verde diverso da quello del pannello) e
    per qualunque sfocatura simmetrica (obiettivo, movimento).

    Ritorna ``(p, colore_dentro, colore_fuori, contrasto²)``; con
    ``residual=True`` anche il **residuo a due colori** (S, K): distanza di
    ogni campione dalla retta colore_fuori -> colore_dentro, relativa al
    contrasto. In un bordo pulito ogni pixel è una miscela dei due soli
    colori e il residuo è rumore (~0.03); un terzo colore mescolato nella
    transizione (il lato bianco della tela, un dito) lo fa crescere.
    """
    ox, oy = offset
    samples = base[:, None, :] + steps[None, :, None] * normal       # (S, K, 2)
    prof = sample_linear(img_u8, samples[..., 0] - ox, samples[..., 1] - oy)
    n_end = 4                                                        # 2 px a ciascun estremo
    c_in = prof[:, :n_end].mean(axis=1)
    c_out = prof[:, -n_end:].mean(axis=1)
    dvec = c_in - c_out
    dd = (dvec * dvec).sum(axis=1)
    p = ((prof - c_out[:, None]) * dvec[:, None]).sum(axis=2) / np.maximum(dd, 1e-9)[:, None]
    if not residual:
        return p, c_in, c_out, dd
    recon = c_out[:, None] + p[..., None] * dvec[:, None]                # miscela a due colori
    perp = np.linalg.norm(prof - recon, axis=2) / np.sqrt(np.maximum(dd, 1e-9))[:, None]
    return p, c_in, c_out, dd, perp


def _blur_shaped(p: np.ndarray, x25: np.ndarray, x50: np.ndarray, x75: np.ndarray,
                 perp: np.ndarray | None = None, max_residual: float = 0.1) -> np.ndarray:
    """Profili (S, K) che hanno la forma di un bordo netto sfocato.

    Un gradino sfocato da obiettivo e movimento (gaussiana ⊗ rettangolo, nucleo
    simmetrico) dà un profilo **simmetrico** attorno al livello 50% e con un
    rapporto fisso tra larghezza 10-90% e larghezza 25-75%: 1.6 per il solo
    rettangolo, 1.9 per la sola gaussiana (un po' meno con la nitidezza
    artificiale dei telefoni, che rende i bordi più ripidi). Inoltre, oltre
    la zona di transizione, il profilo è **piatto** da entrambe le parti
    (dentro: pannello; fuori: sfondo). Il lato bianco della tela visto di
    sbieco mentre il quadro ruota è una striscia attaccata al bordo che si
    muove con lui: mescola un terzo colore nella sfocatura (la transizione
    sembra più corta) e lascia fuori un secondo gradino (striscia -> sfondo).
    Il secondo gradino è ciò che la tradisce. Anche un dito vicino o uno
    sfondo molto disegnato vengono esclusi: si perdono campioni, non si
    falsa la stima. Le posizioni sono in unità di campione, come quelle di
    :func:`_first_crossing`.

    Con un'esposizione lunga la striscia finisce *dentro* la sfocatura e il
    secondo gradino sparisce: resta però il terzo colore mescolato nella
    transizione. Se si passa ``perp`` (residuo a due colori di
    :func:`_edge_profiles`) si scartano anche i profili il cui residuo medio
    nella transizione supera ``max_residual`` (bordi puliti: ~0.03; accanto al
    lato bianco della tela in rotazione: 0.2-0.8 nei test).
    """
    x90, ok90 = _first_crossing(p, 0.9)
    x10, ok10 = _first_crossing(p, 0.1)
    half = 0.5 * (x25 - x75)             # semi-larghezza 25-75%
    half_out = 0.5 * (x10 - x90)         # semi-larghezza 10-90%
    # Tolleranze: 0.5 campioni (0.25 px) di rumore più il 20% della larghezza.
    sym_in = np.abs(x50 - 0.5 * (x25 + x75)) <= 0.2 * half + 0.5
    sym_out = np.abs(x50 - 0.5 * (x10 + x90)) <= 0.2 * half_out + 0.5
    ratio = half_out / np.maximum(half, 1e-6)
    # Parti piatte: oltre la transizione (più 2 campioni = 1 px di margine)
    # l'escursione del profilo deve restare piccola. Servono almeno 3
    # campioni per parte, altrimenti la piattezza non è verificabile.
    idx = np.arange(p.shape[1])[None, :]
    margin = (np.maximum(half, 2.0) + 2.0)[:, None]
    flat = np.ones(len(p), dtype=bool)
    for zone in (idx > x25[:, None] + margin, idx < x75[:, None] - margin):
        hi = np.where(zone, p, -np.inf).max(axis=1)
        lo = np.where(zone, p, np.inf).min(axis=1)
        flat &= (zone.sum(axis=1) >= 3) & (hi - lo < 0.2)
    ok = ok90 & ok10 & sym_in & sym_out & (ratio > 1.2) & (ratio < 2.4) & flat
    if perp is not None:
        # Residuo medio nella transizione (dal livello 90% al 10%, ±1 campione).
        zone = (idx >= np.floor(x90)[:, None] - 1) & (idx <= np.ceil(x10)[:, None] + 1)
        mean_r = (perp * zone).sum(axis=1) / np.maximum(zone.sum(axis=1), 1)
        ok &= mean_r < max_residual
    return ok


def _refine_subpixel(
    quad: np.ndarray,
    measured: np.ndarray,
    support: np.ndarray,
    img_u8: np.ndarray,
    offset: tuple[int, int],
    dp: DetectParams,
    search_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rifinitura sub-pixel delle rette dei lati (in luce lineare).

    Per ogni lato si campiona il colore lungo segmenti perpendicolari al
    bordo (più o meno uno ogni 4 px di lato) e si trova il punto in cui la
    copertura del pannello vale 0.5 (vedi :func:`_edge_profiles`).
    Si scartano i profili che non hanno l'aspetto di un bordo pulito:
    contrasto basso, lato interno non verde (es. un dito sopra il bordo),
    più attraversamenti (texture, rumore). Un RANSAC elimina il resto.

    Anche i lati non misurati dal contorno vengono provati (con una finestra
    di ricerca doppia): se mostrano un bordo netto diventano misurati.

    Per ogni lato si misura anche la **larghezza del bordo** (distanza tra i
    livelli 75% e 25% della copertura): serve a stimare sfocatura
    dell'obiettivo e tempo di esposizione (motion blur).

    Ritorna ``(quad, measured, support, larghezza_bordi)``.
    """
    centroid = quad.mean(axis=0)
    lines = geo.quad_lines(quad)
    measured = measured.copy()
    support = support.copy()
    widths = np.full(4, np.nan)
    base_search = dp.subpixel_search_px if search_px is None else search_px

    for s in range(4):
        search = base_search * (1.0 if measured[s] else 2.0)
        steps = np.arange(-search, search + 0.01, 0.5)
        a, b = quad[s], quad[(s + 1) % 4]
        n_samples = int(np.clip(np.linalg.norm(b - a) / 4.0, dp.subpixel_samples, 4 * dp.subpixel_samples))
        t_vals = np.linspace(0.06, 0.94, n_samples)
        normal = lines[s][:2].copy()
        # Normale orientata verso l'esterno del quad.
        if np.dot(normal, centroid - a) > 0:
            normal = -normal
        base = a[None] + t_vals[:, None] * (b - a)[None]                 # (S, 2)
        p, c_in, c_out, dd, perp = _edge_profiles(img_u8, base, normal, steps, offset, residual=True)

        # Lato interno "pannello": verde dominante (in luce lineare G >> R, B).
        g_in = c_in[:, 1]
        panel_like = g_in > 1.6 * np.maximum(c_in[:, 0], c_in[:, 2]) + 0.01
        good = panel_like & (np.sqrt(dd) > max(0.04, 0.25 * float(np.median(g_in))))
        x50, ok50 = _first_crossing(p, 0.5)
        x75, ok75 = _first_crossing(p, 0.75)
        x25, ok25 = _first_crossing(p, 0.25)
        good &= ok50 & ok75 & ok25
        # Un solo attraversamento: dopo il bordo il profilo resta "fuori",
        # prima resta "dentro" (esclude dita sul bordo, texture, rumore).
        k50 = np.clip(np.round(x50).astype(int), 0, p.shape[1] - 1)
        idx = np.arange(p.shape[1])[None, :]
        after = np.where(idx > k50[:, None] + 2, p, -np.inf).max(axis=1)
        before = np.where(idx < k50[:, None] - 2, p, np.inf).min(axis=1)
        good &= (after < 0.7) & (before > 0.3)
        if good.sum() < max(8, dp.min_side_support * n_samples):
            continue
        off = steps[0] + 0.5 * x50[good]
        edge_pts = base[good] + off[:, None] * normal
        edge_t = t_vals[good]
        line, coverage = _fit_side_line(edge_pts, (edge_t - 0.06) / 0.88, 0.75)
        if line is None or coverage < dp.min_side_support:
            continue
        # Protezione: la retta rifinita non deve allontanarsi troppo da quella
        # grezza (i punti misurati stanno entro ±search per costruzione; agli
        # estremi del lato, estrapolati, si concede un po' di più).
        shift = np.abs(geo.point_line_distance(np.array([a, b]), line)).max()
        if shift < 1.5 * search:
            lines[s] = line
            if not measured[s]:
                measured[s] = True
            support[s] = max(support[s], coverage)
            inl = np.abs(geo.point_line_distance(edge_pts, line)) < 0.75
            # La larghezza si misura solo sui profili con la forma di una vera
            # sfocatura (vedi _blur_shaped): accanto a un bordo ci può essere
            # una striscia di un altro colore (il lato bianco della tela visto
            # di sbieco mentre il quadro ruota, un dito) che sposta i livelli
            # 25%/75% e falserebbe la stima dell'esposizione.
            shaped = _blur_shaped(p[good], x25[good], x50[good], x75[good], perp[good])
            use = inl & shaped
            if use.sum() >= 8:
                widths[s] = float(np.median(0.5 * (x25[good][use] - x75[good][use])))

    refined = geo.quad_from_lines(lines)
    if refined is None or not geo.is_convex(refined):
        return quad, measured, support, widths
    return refined, measured, support, widths


def _impostor_scores(img_u8: np.ndarray, offset: tuple[int, int], quad: np.ndarray,
                     measured: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Probabilità (0-1) che ciascun lato misurato sia il bordo di un oggetto davanti.

    Test della **giunzione a T**. Il bordo vero del pannello finisce negli
    angoli: oltre l'angolo, lungo la stessa retta, c'è solo sfondo (nessun
    bordo). Il bordo dritto di un oggetto davanti al pannello (tavolo,
    manica, libro) invece *continua* oltre i lati del pannello, perché
    l'oggetto è più grande della parte di pannello che copre.

    Per ogni lato si campionano profili trasversali lungo il prolungamento
    della retta oltre ciascun angolo (dal 3% al 25% della lunghezza del
    lato) e si conta la frazione di profili che mostrano un bordo netto
    esattamente sulla retta (entro 1.5 px). Si tiene il massimo dei due
    prolungamenti (un braccio, ad esempio, prosegue solo da un lato).
    """
    ox, oy = offset
    rh, rw = img_u8.shape[:2]
    fh, fw = frame_shape
    lines = geo.quad_lines(quad)
    centroid = quad.mean(axis=0)
    steps = np.arange(-4.0, 4.01, 0.5)
    scores = np.full(4, np.nan)
    for s in range(4):
        if not measured[s]:
            continue
        a, b = quad[s], quad[(s + 1) % 4]
        length = float(np.linalg.norm(b - a))
        if length < 20:
            continue
        u = (b - a) / length
        normal = lines[s][:2].copy()
        if np.dot(normal, centroid - a) > 0:
            normal = -normal
        fractions = []
        for end, sign in ((a, -1.0), (b, 1.0)):
            dist = np.linspace(0.03 * length + 4.0, 0.25 * length, 24)
            base = end[None] + sign * dist[:, None] * u[None]
            ok = ((base[:, 0] > 6) & (base[:, 0] < fw - 7) & (base[:, 1] > 6) & (base[:, 1] < fh - 7)
                  & (base[:, 0] - ox > 6) & (base[:, 0] - ox < rw - 7)
                  & (base[:, 1] - oy > 6) & (base[:, 1] - oy < rh - 7))
            if ok.sum() < 8:
                continue
            p, _, _, dd = _edge_profiles(img_u8, base[ok], normal, steps, offset)
            x50, valid = _first_crossing(p, 0.5)
            off = steps[0] + 0.5 * x50
            edge = valid & (np.sqrt(dd) > 0.05) & (np.abs(off) < 1.5)
            fractions.append(float(edge.mean()))
        if fractions:
            scores[s] = max(fractions)
    return scores


def _mean_corner_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=1).mean())


def _handle_hidden_sides(
    quad: np.ndarray,
    measured: np.ndarray,
    support: np.ndarray,
    prior: np.ndarray | None,
    dp: DetectParams,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Gestisce i lati non visibili usando il quad previsto come riferimento.

    ``reference`` (se c'è) è l'ultimo quad misurato su tutti e 4 i lati: le
    proporzioni dei lati vengono prese da lì e non dal fotogramma precedente,
    che potrebbe essere a sua volta ricostruito (gli errori si sommerebbero).

    * Esattamente un lato nascosto -> ricostruito dagli altri tre
      (``geometry.complete_by_ratio``).
    * Un solo lato con poca evidenza (copertura < ``dp.weak_support``) mentre
      gli altri tre sono solidi -> trattato come nascosto: un tratto corto
      potrebbe essere il bordo quasi dritto di un occlusore, e la ricostruzione
      (poi corretta offline per interpolazione) è più affidabile.
    * Due o più lati nascosti -> restano le rette previste dal movimento.

    Ritorna il quad e l'elenco dei lati ricostruiti.
    """
    if prior is None:
        return quad, []
    strong = measured & (support >= dp.weak_support)
    if measured.sum() == 3:
        missing = int(np.nonzero(~measured)[0][0])
    elif measured.sum() == 4 and strong.sum() == 3:
        missing = int(np.nonzero(~strong)[0][0])
    else:
        return quad, []
    cand = geo.complete_by_ratio(geo.quad_lines(quad), missing,
                                 reference if reference is not None else prior)
    if cand is None:
        return quad, []
    return cand, [missing]


def _cover_visible_green(quad: np.ndarray, measured: np.ndarray, pts: np.ndarray,
                         tol: float = 1.5) -> np.ndarray:
    """Un lato non misurato non può tagliare il verde visibile del pannello.

    I lati nascosti vengono ricostruiti (dagli altri tre lati o dal movimento):
    se la ricostruzione cade *dentro* la zona verde che si vede, è sicuramente
    sbagliata. Succede con un quadro grande che esce dall'inquadratura mentre
    si gira: le proporzioni prese da un fotogramma di taglio danno un quad
    troppo stretto (nei test: lato sinistro a x=355 invece che a 0). Qui ogni
    lato non misurato viene spostato in fuori, parallelamente, finché il
    contorno verde visibile (anche i punti sul bordo del fotogramma) non è
    tutto dentro il quad.
    """
    if measured.all() or len(pts) == 0:
        return quad
    lines = geo.quad_lines(quad)
    centroid = quad.mean(axis=0)
    moved = False
    for s in np.nonzero(~measured)[0]:
        ln = lines[s].copy()
        if geo.point_line_distance(centroid[None], ln)[0] > 0:
            ln = -ln                                   # normale verso l'esterno
        out = geo.point_line_distance(pts, ln)
        push = float(np.percentile(out, 99.5))        # robusto a pochi punti spuri
        if push > tol:
            ln = ln.copy()
            ln[2] -= push
            lines[s] = ln
            moved = True
    if not moved:
        return quad
    covered = geo.quad_from_lines(lines)
    if covered is None or not geo.is_convex(covered):
        return quad
    return covered


def _fail(diag: dict | None, reason: str) -> None:
    """Registra il motivo di un fallimento (se richiesto) e ritorna None."""
    if diag is not None:
        diag["reason"] = reason
    return None


def detect_quad(
    frame_bgr: np.ndarray,
    kp: KeyParams,
    dp: DetectParams,
    prior: np.ndarray | None = None,
    green_ratio: float | None = None,
    diag: dict | None = None,
    motion_px: float = 0.0,
    color: PanelColor | None = None,
    reference: np.ndarray | None = None,
) -> Detection | None:
    """Rileva il pannello verde in un fotogramma BGR uint8.

    Parametri
    ---------
    prior: quad previsto dal tracking (None = ricerca su tutto il fotogramma).
    green_ratio: purezza del verde del pannello misurata in precedenza; serve
        a scegliere una soglia adattiva che separa il pannello da verdi spenti
        dello sfondo.
    diag: dizionario opzionale in cui viene scritto il motivo di un eventuale
        fallimento (chiave 'reason'), utile per il debug.
    motion_px: spostamento previsto dei bordi in un fotogramma (px): con il
        movimento il bordo è sfocato e la finestra di misura si allarga.
    color: modello di colore del pannello (dal tracking); senza modello il
        pannello si cerca tra i verdi presenti e il modello viene stimato.
    reference: ultimo quad misurato su tutti i lati (proporzioni per
        ricostruire un lato nascosto).
    """
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _roi_from_prior(prior, (h, w), dp.roi_margin)
    roi = frame_bgr[y0:y1, x0:x1]
    # Area minima: severa nella ricerca da zero, permissiva quando si insegue
    # il pannello (può comparire di taglio, come una striscia sottile).
    min_area = (dp.min_area_frac if prior is None else dp.min_area_frac_tracking) * w * h
    # La ricerca grossolana (maschera, componenti, contorno) si fa a metà
    # risoluzione quando la ROI è grande: la posizione esatta dei lati la dà
    # comunque la rifinitura sub-pixel a piena risoluzione (passo 5).
    sc = 2 if roi.shape[0] * roi.shape[1] > dp.coarse_half_res_px else 1
    if sc > 1:
        roi_s = cv2.resize(roi, ((roi.shape[1] + 1) // 2, (roi.shape[0] + 1) // 2),
                           interpolation=cv2.INTER_AREA)
    else:
        roi_s = roi

    def to_small(q: np.ndarray) -> np.ndarray:
        """Coordinate del fotogramma -> pixel della ROI ridotta."""
        return (q - [x0, y0] - (sc - 1) / 2.0) / sc

    def to_full(p: np.ndarray) -> np.ndarray:
        """Pixel della ROI ridotta -> coordinate del fotogramma."""
        return p * sc + (sc - 1) / 2.0 + [x0, y0]

    # 1-2) Maschera del pannello e sua componente connessa.
    if prior is None:
        found = _best_component_without_prior(roi_s, kp, min_area / sc ** 2, color)
        if found is None:
            return _fail(diag, "nessuna zona verde adatta")
        comp, color_used = found
    else:
        color_used = color
        if color is not None:
            mask = _clean(color.mask(roi_s))
        else:
            thr = kp.r_min if green_ratio is None else float(np.clip(0.5 * green_ratio, kp.r_min, 0.45))
            mask = _clean(generic_green_mask(roi_s, kp.d_min, thr))
        # Ritaglia la maschera attorno al quad previsto (allargato del 10% della
        # diagonale): eventuali zone verdi dello sfondo attaccate al pannello
        # vengono separate. Il margine supera la tolleranza di assegnazione dei
        # punti ai lati, quindi i tagli artificiali non vengono scambiati per bordi.
        diag_prior = float(np.linalg.norm(prior[0] - prior[2]))
        clip_quad = geo.offset_quad(prior, 0.10 * diag_prior + 4)
        clip = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(clip, np.round(to_small(clip_quad)).astype(np.int32), 1)
        mask &= clip
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_lab <= 1:
            return _fail(diag, "nessuna zona verde nella ROI")
        # Con una predizione scegliamo la componente che più la sovrappone.
        prior_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(prior_mask, np.round(to_small(prior)).astype(np.int32), 1)
        overlap = np.bincount(labels[prior_mask > 0].ravel(), minlength=n_lab)[1:]
        if overlap.max() <= 0:
            return _fail(diag, "la zona verde non sovrappone la posizione prevista")
        comp = (labels == int(np.argmax(overlap)) + 1).astype(np.uint8)
    green_area = float(comp.sum()) * sc * sc
    if green_area < min_area:
        return _fail(diag, "zona verde troppo piccola")

    # 3) Contorno esterno (i buchi interni causati dalle mani non interessano).
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    pts = to_full(contour.reshape(-1, 2).astype(np.float64))
    # I punti sul bordo del fotogramma non sono bordi del quadro (quadro
    # parzialmente fuori inquadratura): vanno esclusi dal fit.
    edge = sc + 0.5
    on_border = ((pts[:, 0] <= edge) | (pts[:, 1] <= edge)
                 | (pts[:, 0] >= w - 1 - edge) | (pts[:, 1] >= h - 1 - edge))
    pts_fit = pts[~on_border]
    if len(pts_fit) < 16:
        return _fail(diag, "contorno troppo corto")

    # 4) Quad iniziale e rifinitura sui punti di contorno.
    if prior is not None:
        quad0 = prior.copy()
    else:
        quad0 = _initial_quad_from_contour(to_full(contour.reshape(-1, 2).astype(np.float64))
                                           .astype(np.float32))
    if quad0 is None:
        return _fail(diag, "quad iniziale non valido")
    res = _refine_with_contour(quad0, pts_fit, prior, dp, ransac_px=dp.ransac_px * sc)
    if res is None:
        return _fail(diag, "rifinitura dei lati fallita")
    quad, support, measured = res

    # 5) Rifinitura sub-pixel in luce lineare.
    widths = np.full(4, np.nan)
    if dp.subpixel:
        search = max(dp.subpixel_search_px, 0.6 * motion_px + 3.0)
        quad, measured, support, widths = _refine_subpixel(
            quad, measured, support, roi, (x0, y0), dp, search)
    if measured.sum() < 2:
        return _fail(diag, "meno di 2 lati visibili")

    # 5b) Bordi "impostori": un oggetto dritto davanti al pannello.
    impostor = _impostor_scores(roi, (x0, y0), quad, measured, (h, w))
    if prior is not None and measured.sum() == 4:
        sus = np.nan_to_num(impostor, nan=0.0)
        worst = int(np.argmax(sus))
        if sus[worst] >= dp.impostor_threshold:
            # Il bordo prosegue oltre l'angolo: può essere un oggetto davanti
            # oppure un bordo dello sfondo allineato per caso con il lato vero.
            # Decide la geometria: ricostruito dagli altri tre lati (con le
            # proporzioni del fotogramma precedente), il lato vero cade dove è
            # stato misurato; un oggetto davanti invece "accorcia" il pannello.
            lines_now = geo.quad_lines(quad)
            cand = geo.complete_by_ratio(lines_now, worst, reference if reference is not None else prior)
            if cand is not None:
                ends = np.array([cand[worst], cand[(worst + 1) % 4]])
                ln = lines_now[worst]
                if geo.point_line_distance(quad.mean(axis=0)[None], ln)[0] > 0:
                    ln = -ln                                  # normale verso l'esterno
                # Positivo = il lato ricostruito sta *fuori* da quello misurato,
                # cioè il pannello misurato è "accorciato": l'unico caso
                # compatibile con un oggetto davanti (che può solo nascondere).
                disp = float(geo.point_line_distance(ends, ln).min())
                side_len = float(np.linalg.norm(quad[(worst + 1) % 4] - quad[worst]))
                # Tolleranza: errore tipico della ricostruzione, che cresce col movimento.
                if disp > max(3.0, 0.015 * side_len, 0.5 * motion_px):
                    # Il lato viene trattato come nascosto: ricostruito dagli altri
                    # tre (e poi, offline, interpolato tra prima e dopo l'occlusione).
                    measured[worst] = False
                    support[worst] = 0.0

    # 6) Lati nascosti: ricostruzione dagli altri tre o predizione.
    quad, completed = _handle_hidden_sides(quad, measured, support, prior, dp, reference)
    for s in completed:
        measured[s] = False
    # 6b) I lati non misurati non possono tagliare il verde che si vede.
    quad = _cover_visible_green(quad, measured, pts)

    # 7) Validazione geometrica.
    if not geo.is_convex(quad):
        return _fail(diag, "quad non convesso")
    angles = geo.interior_angles(quad)
    min_ang = dp.min_angle if prior is None else dp.min_angle_tracking
    if angles.min() < min_ang or angles.max() > 180.0 - min_ang:
        return _fail(diag, "angoli interni implausibili")
    area = geo.quad_area(quad)
    if area < min_area:
        return _fail(diag, "area troppo piccola")
    if prior is None:
        # Senza predizione il quad deve "spiegare" quasi tutto il verde trovato,
        # altrimenti abbiamo agganciato la forma sbagliata.
        inside = np.zeros(comp.shape, np.uint8)
        cv2.fillConvexPoly(inside, np.round(to_small(geo.offset_quad(quad, 3 * sc))).astype(np.int32), 1)
        outside_frac = float((comp & (1 - inside)).sum()) * sc * sc / max(green_area, 1.0)
        if outside_frac > 0.15:
            return _fail(diag, "il quad non copre la zona verde")

    # Luminosità (luce lineare) e purezza del verde del pannello visibile
    # (campionamento a griglia: bastano poche migliaia di pixel).
    step = max(1, int(np.sqrt(comp.size / 40000)))
    px = roi_s[::step, ::step][comp[::step, ::step] > 0]
    lin = LINEAR_LUT[px]
    luma = 0.0722 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.2126 * lin[:, 2]
    panel_luma = float(np.percentile(luma, 90)) if luma.size else 0.0
    pxf = px.astype(np.float32)
    purity = (pxf[:, 1] - np.maximum(pxf[:, 0], pxf[:, 2])) / np.maximum(pxf[:, 1], 1.0)
    green_ratio_now = float(np.median(purity)) if purity.size else kp.r_min

    confidence = float(np.mean(np.where(measured, support, 0.0)))
    return Detection(
        quad=quad,
        support=support,
        measured=measured,
        confidence=confidence,
        panel_luma=panel_luma,
        green_area=green_area,
        green_ratio=green_ratio_now,
        completed=completed,
        edge_width=widths,
        color=color_used,
        impostor=impostor,
    )
