"""Smoothing temporale degli angoli (passata 2).

Il video viene elaborato *offline*: conosciamo anche i fotogrammi futuri.
Invece di un filtro in tempo reale (Kalman, One-Euro), che introduce ritardo,
si usa una **regressione polinomiale locale pesata** (stile Savitzky-Golay /
LOESS) centrata su ogni fotogramma:

* a fase zero: nessun ritardo rispetto al movimento reale;
* pesata con la confidenza: i fotogrammi con lati nascosti contano meno;
* con rifiuto degli outlier (misura leave-one-out): un singolo fotogramma
  sbagliato non "tira" i vicini;
* i fotogrammi mancanti vengono riempiti dai vicini (interpolazione);
* gli angoli dei lati nascosti (ricostruiti durante il tracking estrapolando
  dal passato, quindi con possibile deriva) vengono ricalcolati
  **interpolando** tra il fotogramma prima e quello dopo l'occlusione;
* prima ancora, un **controllo di coerenza globale** individua i fotogrammi
  in cui un lato "misurato" è in realtà il bordo di un oggetto davanti al
  pannello: il loro quad implica un rapporto d'aspetto diverso da quello del
  pannello, stimato con precisione su tutto il video.

Obiettivo: eliminare il tremolio sub-pixel ("swimming") dell'immagine inserita
senza farla scivolare rispetto al quadro reale nei movimenti rapidi.
"""

from __future__ import annotations

import numpy as np

from . import geometry as geo
from .tracker import STATUS_LOST, STATUS_MODEL, STATUS_OK, STATUS_PREDICTED, Track


def _hidden_ratios(quad: np.ndarray, missing: int) -> tuple[float, float]:
    """Lunghezze dei due lati adiacenti al lato ``missing``, relative al lato opposto."""
    opp = (missing + 2) % 4
    p, q = quad[opp], quad[(opp + 1) % 4]
    base = max(float(np.linalg.norm(q - p)), 1e-9)
    r1 = float(np.linalg.norm(quad[(opp + 2) % 4] - q)) / base
    r2 = float(np.linalg.norm(quad[(opp + 3) % 4] - p)) / base
    return r1, r2


def flag_inconsistent_sides(track: Track, tolerance: float = 0.02,
                            trusted_support: float = 0.9) -> list[int]:
    """Marca come "nascosto" il lato sbagliato dei fotogrammi incoerenti.

    Per ogni fotogramma misurato si calcola il rapporto d'aspetto implicato
    dal quad (con la focale stimata sul video) e lo si confronta con quello
    del pannello. Nei test su video Full HD i fotogrammi corretti deviano al
    massimo dello 0.9%, quelli con un lato sbagliato almeno del 2.3%.

    Tra i lati non pienamente visibili (copertura < ``trusted_support``) si
    sceglie quello la cui ricostruzione dagli altri tre riporta il quad più
    vicino al rapporto corretto. Il fotogramma passa allo stato "ricostruito":
    l'interpolazione offline (:func:`refine_hidden_sides`) ne ricalcola gli angoli.

    Modifica ``track`` (stato e lato nascosto) e ritorna i fotogrammi corretti.
    """
    if (track.support is None or track.hidden_side is None or not track.focal
            or not track.focal_reliable or not track.panel_aspect):
        return []
    size = (track.width, track.height)

    def deviation(quad: np.ndarray) -> float:
        ratio, _ = geo.estimate_aspect_ratio(quad, size, fixed_focal=track.focal)
        return abs(float(np.log(ratio / track.panel_aspect)))

    candidates = [t for t, s in enumerate(track.status)
                  if s == STATUS_OK and track.hidden_side[t] < 0 and not np.isnan(track.raw[t]).any()]
    if not candidates:
        return []
    devs = {t: deviation(track.raw[t]) for t in candidates}
    # Soglia: mai sotto ``tolerance``, più larga se il video è "rumoroso"
    # (distorsione dell'obiettivo, rolling shutter dei telefoni).
    threshold = max(tolerance, 3.0 * float(np.median(list(devs.values()))))
    clean = [t for t in candidates if devs[t] <= threshold]
    if not clean:
        return []
    clean_arr = np.array(clean)

    fixed = []
    for t in candidates:
        if devs[t] <= threshold:
            continue
        weak = [s for s in range(4) if track.support[t, s] < trusted_support]
        if not weak:
            continue
        # Riferimento per le lunghezze: il fotogramma coerente più vicino nel tempo.
        ref = track.raw[clean_arr[np.argmin(np.abs(clean_arr - t))]]
        lines = geo.quad_lines(track.raw[t])
        best = None
        for s in weak:
            cand = geo.complete_by_ratio(lines, s, ref)
            if cand is None:
                continue
            d = deviation(cand)
            if best is None or d < best[0]:
                best = (d, s, cand)
        if best is not None and best[0] < devs[t]:
            track.raw[t] = best[2]
            track.hidden_side[t] = best[1]
            track.status[t] = STATUS_MODEL
            fixed.append(t)
    return fixed


def _place_hidden(quad: np.ndarray, opp: int, r1: float, r2: float) -> np.ndarray:
    """Quad con gli angoli del lato nascosto alle distanze relative r1, r2.

    Gli angoli si muovono lungo le rette (misurate) dei lati adiacenti.
    """
    out = quad.copy()
    p, q = quad[opp], quad[(opp + 1) % 4]
    base = np.linalg.norm(q - p)
    d1 = quad[(opp + 2) % 4] - q
    d2 = quad[(opp + 3) % 4] - p
    out[(opp + 2) % 4] = q + d1 / max(np.linalg.norm(d1), 1e-9) * r1 * base
    out[(opp + 3) % 4] = p + d2 / max(np.linalg.norm(d2), 1e-9) * r2 * base
    return out


def _ratio_slope(track: Track, raw: np.ndarray, frames: np.ndarray, side: int) -> np.ndarray | None:
    """Velocità di variazione (per fotogramma) delle distanze relative r1, r2,
    stimata con una retta ai minimi quadrati sui fotogrammi indicati."""
    if len(frames) < 2:
        return None
    r = np.array([_hidden_ratios(raw[f], side) for f in frames])
    t = frames.astype(np.float64)
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    return np.linalg.lstsq(A, r, rcond=None)[0][0]


def refine_hidden_sides(track: Track, slope_frames: int = 3) -> np.ndarray:
    """Ricalcola gli angoli dei lati nascosti interpolando nel tempo.

    Durante un'occlusione i due angoli del lato nascosto stanno sulle rette
    (misurate) dei lati adiacenti; ignota è solo la loro distanza dal lato
    opposto (relativa al lato opposto: r1, r2). Queste distanze vengono
    interpolate tra l'ultimo fotogramma "pulito" prima dell'occlusione e il
    primo dopo con una **spline cubica di Hermite**, che usa anche la velocità
    di variazione misurata sui ``slope_frames`` fotogrammi puliti a ciascun
    estremo. Rispetto all'interpolazione lineare segue l'andamento curvo
    della prospettiva: in un'occlusione di 25 fotogrammi in Full HD l'errore
    massimo è sceso da 31 a 2.4 px.

    Se l'occlusione tocca l'inizio o la fine del video si usa il valore
    dell'unico estremo disponibile (nessuna estrapolazione: è instabile).
    """
    raw = track.raw.copy()
    hs = track.hidden_side
    if hs is None:
        return raw
    T = len(raw)
    clean = np.array([s == STATUS_OK and h < 0 for s, h in zip(track.status, hs)])
    clean &= ~np.isnan(raw[:, 0, 0])
    clean_idx = np.nonzero(clean)[0]

    t = 0
    while t < T:
        m = int(hs[t])
        if m < 0:
            t += 1
            continue
        # Sequenza contigua di fotogrammi con lo stesso lato nascosto.
        end = t
        while end + 1 < T and hs[end + 1] == m:
            end += 1
        before = clean_idx[clean_idx < t]
        after = clean_idx[clean_idx > end]
        opp = (m + 2) % 4
        if len(before) and len(after):
            a, b = int(before[-1]), int(after[0])
            ra = np.array(_hidden_ratios(raw[a], m))
            rb = np.array(_hidden_ratios(raw[b], m))
            L = float(b - a)
            secant = (rb - ra) / L
            # Pendenze agli estremi: solo fotogrammi puliti e vicini all'estremo.
            fa = before[before > a - 2 * slope_frames][-slope_frames:]
            fb = after[after < b + 2 * slope_frames][:slope_frames]
            sa = _ratio_slope(track, raw, fa, m)
            sb = _ratio_slope(track, raw, fb, m)
            sa = secant if sa is None else sa
            sb = secant if sb is None else sb
            for k in range(t, end + 1):
                u = (k - a) / L
                # Basi di Hermite.
                h00 = 2 * u ** 3 - 3 * u ** 2 + 1
                h10 = u ** 3 - 2 * u ** 2 + u
                h01 = -2 * u ** 3 + 3 * u ** 2
                h11 = u ** 3 - u ** 2
                r = h00 * ra + h10 * L * sa + h01 * rb + h11 * L * sb
                raw[k] = _place_hidden(raw[k], opp, float(r[0]), float(r[1]))
        elif len(before) or len(after):
            ref = int(before[-1]) if len(before) else int(after[0])
            r1, r2 = _hidden_ratios(raw[ref], m)
            for k in range(t, end + 1):
                raw[k] = _place_hidden(raw[k], opp, r1, r2)
        t = end + 1
    return raw


def _local_poly(values: np.ndarray, weights: np.ndarray, t: int, half: int) -> np.ndarray | None:
    """Valore al tempo t di un polinomio locale pesato (fino al 2° grado).

    ``values`` ha forma (T, D): tutte le coordinate vengono stimate insieme.
    """
    T = len(values)
    lo, hi = max(0, t - half), min(T, t + half + 1)
    idx = np.arange(lo, hi)
    x = (idx - t).astype(np.float64)
    # Kernel tricubo: pesa di più i fotogrammi vicini.
    kern = (1 - (np.abs(x) / (half + 1)) ** 3) ** 3
    w = weights[lo:hi] * kern
    ok = w > 1e-9
    n_eff = int(ok.sum())
    if n_eff == 0:
        return None
    # Grado adattivo: con pochi punti si scende a retta o costante.
    deg = 2 if n_eff >= 4 else (1 if n_eff >= 2 else 0)
    X = np.vander(x[ok], deg + 1, increasing=True)        # colonne: 1, x, x²
    sw = np.sqrt(w[ok])[:, None]
    coef, *_ = np.linalg.lstsq(X * sw, values[lo:hi][ok] * sw, rcond=None)
    return coef[0]                                          # valore in x = 0


def smooth_track(track: Track, half_window: int = 2, outlier_px: float = 6.0,
                 max_gap: int = 15) -> np.ndarray:
    """Calcola gli angoli finali (T, 4, 2); NaN dove non si deve sostituire."""
    T = track.n_frames
    raw = refine_hidden_sides(track).reshape(T, 8)
    valid = ~np.isnan(raw[:, 0])

    # Pesi di partenza: confidenza (minimo 0.05 per i misurati), quasi nulli
    # per i previsti (servono solo se mancano misure vicine), zero per i persi.
    w = np.where(valid, np.maximum(track.confidence, 0.05), 0.0)
    for i, s in enumerate(track.status):
        if s == STATUS_PREDICTED:
            w[i] = 0.01
        elif s == STATUS_LOST:
            w[i] = 0.0
    filled = np.where(valid[:, None], raw, 0.0)

    # 1) Pesi di robustezza (come nel LOESS robusto): si liscia, si misurano i
    #    residui e si riducono con la funzione "bisquare" i pesi dei fotogrammi
    #    che si discostano troppo. Ripetendo, un singolo fotogramma sbagliato
    #    perde peso mentre i vicini (inizialmente "tirati" da lui) lo
    #    riacquistano: niente effetto di mascheramento.
    base_w = w.copy()
    robust_half = max(half_window, 3)
    cutoff = 2.0 * outlier_px
    for _ in range(3):
        resid = np.zeros(T)
        for t in np.nonzero(base_w > 0.02)[0]:
            est = _local_poly(filled, w, t, robust_half)
            if est is not None:
                resid[t] = np.linalg.norm((raw[t] - est).reshape(4, 2), axis=1).max()
        u = np.clip(resid / cutoff, 0.0, 1.0)
        w = base_w * (1.0 - u ** 2) ** 2

    # 2) Regressione locale per ogni fotogramma.
    out = np.full((T, 8), np.nan)
    if half_window <= 0:
        out[valid] = raw[valid]
    else:
        for t in range(T):
            est = _local_poly(filled, w, t, half_window)
            if est is not None:
                out[t] = est

    # 3) Buchi più lunghi della finestra: interpolazione lineare tra le misure
    #    più vicine, solo se il buco non è troppo lungo.
    have = ~np.isnan(out[:, 0])
    good_idx = np.nonzero(have)[0]
    if len(good_idx) >= 2:
        for t in np.nonzero(~have)[0]:
            before = good_idx[good_idx < t]
            after = good_idx[good_idx > t]
            if len(before) and len(after) and (after[0] - before[-1]) <= max_gap:
                a, b = before[-1], after[0]
                k = (t - a) / (b - a)
                out[t] = (1 - k) * out[a] + k * out[b]

    # I fotogrammi persi con pesi nulli all'inizio/fine restano NaN: lì il
    # video originale viene lasciato intatto (meglio di un'immagine fuori posto).
    return out.reshape(T, 4, 2)
