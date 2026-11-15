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

Limite noto: un oggetto con un bordo **dritto** (tavolo, manica tesa, mobile)
che copre un intero lato produce un contorno rettilineo indistinguibile da un
bordo vero: il quadro risulterà "accorciato" finché l'oggetto è davanti.

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
from .config import DetectParams, KeyParams
from .keyer import green_difference


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


def _clean_mask(d: np.ndarray, ratio: np.ndarray, d_min: float, thr: float) -> np.ndarray:
    """Maschera binaria del verde con pulizia morfologica leggera."""
    mask = ((d >= d_min) & (ratio >= thr)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


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


def _best_component_without_prior(d, ratio, kp: KeyParams, thr0: float, min_area: float):
    """Cerca il pannello senza sapere dove sia (primo fotogramma o tracking perso).

    Prova soglie di purezza del verde crescenti e sceglie la componente più
    simile a un quadrilatero pieno: punteggio = area × riempimento², dove il
    riempimento è area / area dell'involucro convesso. Così un pannello
    "fuso" con una macchia verdastra dello sfondo perde contro il pannello
    isolato da una soglia più severa.
    """
    best = None  # (score, comp_mask, thr)
    for thr in sorted({thr0, 0.25, 0.35, 0.45}):
        mask = _clean_mask(d, ratio, kp.d_min, thr)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_lab <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        for k in np.argsort(areas)[::-1][:3]:
            area = float(areas[k])
            if area < min_area:
                break
            comp = (labels == k + 1).astype(np.uint8)
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
            fill = area / max(hull_area, 1.0)
            score = area * fill ** 2
            if best is None or score > best[0]:
                best = (score, comp, thr)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Rifinitura iterativa del quad usando i punti del contorno esterno."""
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
            line, cov = _fit_side_line(side_pts, t, dp.ransac_px)
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


def _refine_subpixel(
    quad: np.ndarray,
    measured: np.ndarray,
    support: np.ndarray,
    green: np.ndarray,
    offset: tuple[int, int],
    dp: DetectParams,
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rifinitura sub-pixel delle rette dei lati.

    Per ogni lato si campiona la "verdezza" lungo segmenti perpendicolari al
    bordo e si trova, con interpolazione lineare, il punto in cui il profilo
    attraversa la metà tra il valore interno (pannello) e quello esterno.
    La soglia è locale, quindi funziona anche se fuori dal quadro c'è uno
    sfondo verdastro (che nella maschera binaria si fonde col pannello).
    I profili che non hanno l'aspetto di un bordo pulito (es. dita sopra il
    bordo) vengono scartati, e un RANSAC elimina il resto.

    Anche i lati non misurati dal contorno vengono provati (con una finestra
    di ricerca doppia): se mostrano un bordo netto diventano misurati.

    Ritorna ``(quad, measured, support)`` aggiornati.
    """
    ox, oy = offset
    centroid = quad.mean(axis=0)
    lines = geo.quad_lines(quad)
    t_vals = np.linspace(0.06, 0.94, dp.subpixel_samples)
    measured = measured.copy()
    support = support.copy()

    for s in range(4):
        search = dp.subpixel_search_px * (1.0 if measured[s] else 2.0)
        steps = np.arange(-search, search + 0.01, 0.5)
        a, b = quad[s], quad[(s + 1) % 4]
        normal = lines[s][:2].copy()
        # Normale orientata verso l'esterno del quad.
        if np.dot(normal, centroid - a) > 0:
            normal = -normal
        base = a[None] + t_vals[:, None] * (b - a)[None]                 # (S, 2)
        samples = base[:, None, :] + steps[None, :, None] * normal       # (S, K, 2)
        map_x = (samples[..., 0] - ox).astype(np.float32)
        map_y = (samples[..., 1] - oy).astype(np.float32)
        prof = cv2.remap(green, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        inside = prof[:, :3].mean(axis=1)
        outside = prof[:, -3:].mean(axis=1)
        contrast = inside - outside
        good_rows = contrast > 0.35 * np.median(np.maximum(inside, 1e-3))
        edge_pts, edge_t = [], []
        for i in np.nonzero(good_rows)[0]:
            thr = 0.5 * (inside[i] + outside[i])
            below = np.nonzero(prof[i] < thr)[0]
            if len(below) == 0 or below[0] == 0:
                continue
            k = below[0]
            v0, v1 = prof[i, k - 1], prof[i, k]
            frac = (v0 - thr) / max(v0 - v1, 1e-6)
            off = steps[k - 1] + frac * (steps[k] - steps[k - 1])
            edge_pts.append(base[i] + off * normal)
            edge_t.append(t_vals[i])
        if len(edge_pts) < max(8, dp.min_side_support * dp.subpixel_samples):
            continue
        edge_pts, edge_t = np.array(edge_pts), np.array(edge_t)
        # Con 60 campioni su 40 intervalli bastano 3 intervalli contigui.
        line, coverage = _fit_side_line(edge_pts, (edge_t - 0.06) / 0.88, 0.75)
        if line is None or coverage < dp.min_side_support:
            continue
        # Protezione: la retta rifinita non deve allontanarsi troppo da quella grezza.
        shift = np.abs(geo.point_line_distance(np.array([a, b]), line)).max()
        if shift < search:
            lines[s] = line
            if not measured[s]:
                measured[s] = True
            support[s] = max(support[s], coverage)

    refined = geo.quad_from_lines(lines)
    if refined is None or not geo.is_convex(refined):
        return quad, measured, support
    return refined, measured, support


def _mean_corner_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=1).mean())


def _handle_hidden_sides(
    quad: np.ndarray,
    measured: np.ndarray,
    support: np.ndarray,
    prior: np.ndarray | None,
    dp: DetectParams,
) -> tuple[np.ndarray, list[int]]:
    """Gestisce i lati non visibili usando il quad previsto come riferimento.

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
    cand = geo.complete_by_ratio(geo.quad_lines(quad), missing, prior)
    if cand is None:
        return quad, []
    return cand, [missing]


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
    """
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _roi_from_prior(prior, (h, w), dp.roi_margin)
    roi = frame_bgr[y0:y1, x0:x1]
    min_area = dp.min_area_frac * w * h

    # 1) Misure di "verdezza" sulla ROI.
    d, g = green_difference(roi)
    ratio = d / np.maximum(g, 1.0)
    thr = kp.r_min if green_ratio is None else float(np.clip(0.5 * green_ratio, kp.r_min, 0.45))

    # 2) Componente connessa del pannello.
    if prior is None:
        found = _best_component_without_prior(d, ratio, kp, thr, min_area)
        if found is None:
            return _fail(diag, "nessuna zona verde adatta")
        comp, thr = found
    else:
        mask = _clean_mask(d, ratio, kp.d_min, thr)
        # Ritaglia la maschera attorno al quad previsto (allargato del 10% della
        # diagonale): eventuali zone verdi dello sfondo attaccate al pannello
        # vengono separate. Il margine supera la tolleranza di assegnazione dei
        # punti ai lati, quindi i tagli artificiali non vengono scambiati per bordi.
        diag_prior = float(np.linalg.norm(prior[0] - prior[2]))
        clip_quad = geo.offset_quad(prior, 0.10 * diag_prior + 4)
        clip = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(clip, np.round(clip_quad - [x0, y0]).astype(np.int32), 1)
        mask &= clip
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_lab <= 1:
            return _fail(diag, "nessuna zona verde nella ROI")
        # Con una predizione scegliamo la componente che più la sovrappone.
        prior_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(prior_mask, np.round(prior - [x0, y0]).astype(np.int32), 1)
        overlap = np.bincount(labels[prior_mask > 0].ravel(), minlength=n_lab)[1:]
        if overlap.max() <= 0:
            return _fail(diag, "la zona verde non sovrappone la posizione prevista")
        comp = (labels == int(np.argmax(overlap)) + 1).astype(np.uint8)
    green_area = float(comp.sum())
    if green_area < min_area:
        return _fail(diag, "zona verde troppo piccola")

    # 3) Contorno esterno (i buchi interni causati dalle mani non interessano).
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(np.float64) + [x0, y0]
    # I punti sul bordo del fotogramma non sono bordi del quadro (quadro
    # parzialmente fuori inquadratura): vanno esclusi dal fit.
    on_border = (pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] >= h - 2)
    pts_fit = pts[~on_border]
    if len(pts_fit) < 16:
        return _fail(diag, "contorno troppo corto")

    # 4) Quad iniziale e rifinitura sui punti di contorno.
    if prior is not None:
        quad0 = prior.copy()
    else:
        quad0 = _initial_quad_from_contour((contour + [x0, y0]).astype(np.float32))
    if quad0 is None:
        return _fail(diag, "quad iniziale non valido")
    res = _refine_with_contour(quad0, pts_fit, prior, dp)
    if res is None:
        return _fail(diag, "rifinitura dei lati fallita")
    quad, support, measured = res

    # 5) Rifinitura sub-pixel sulla "verdezza" continua.
    if dp.subpixel:
        quad, measured, support = _refine_subpixel(
            quad, measured, support, np.clip(d, 0, None), (x0, y0), dp, prior)
    if measured.sum() < 2:
        return _fail(diag, "meno di 2 lati visibili")

    # 6) Lati nascosti: ricostruzione dagli altri tre o predizione.
    quad, completed = _handle_hidden_sides(quad, measured, support, prior, dp)
    for s in completed:
        measured[s] = False

    # 7) Validazione geometrica.
    if not geo.is_convex(quad):
        return _fail(diag, "quad non convesso")
    angles = geo.interior_angles(quad)
    if angles.min() < 20 or angles.max() > 160:
        return _fail(diag, "angoli interni implausibili")
    area = geo.quad_area(quad)
    if area < min_area:
        return _fail(diag, "area troppo piccola")
    if prior is None:
        # Senza predizione il quad deve "spiegare" quasi tutto il verde trovato,
        # altrimenti abbiamo agganciato la forma sbagliata.
        inside = np.zeros(comp.shape, np.uint8)
        cv2.fillConvexPoly(inside, np.round(geo.offset_quad(quad, 3) - [x0, y0]).astype(np.int32), 1)
        outside_frac = float((comp & (1 - inside)).sum()) / max(green_area, 1.0)
        if outside_frac > 0.15:
            return _fail(diag, "il quad non copre la zona verde")

    # Luminosità e purezza del verde del pannello visibile.
    sel = comp > 0
    luma = 0.7152 * roi[..., 1][sel] + 0.2126 * roi[..., 2][sel] + 0.0722 * roi[..., 0][sel]
    panel_luma = float(np.percentile(luma, 90)) if luma.size else 0.0
    green_ratio_now = float(np.median(ratio[sel])) if luma.size else thr

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
    )
