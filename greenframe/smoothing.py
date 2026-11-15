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
  **interpolando** tra il fotogramma prima e quello dopo l'occlusione.

Obiettivo: eliminare il tremolio sub-pixel ("swimming") dell'immagine inserita
senza farla scivolare rispetto al quadro reale nei movimenti rapidi.
"""

from __future__ import annotations

import numpy as np

from .tracker import STATUS_LOST, STATUS_OK, STATUS_PREDICTED, Track


def _hidden_ratios(quad: np.ndarray, missing: int) -> tuple[float, float]:
    """Lunghezze dei due lati adiacenti al lato ``missing``, relative al lato opposto."""
    opp = (missing + 2) % 4
    p, q = quad[opp], quad[(opp + 1) % 4]
    base = max(float(np.linalg.norm(q - p)), 1e-9)
    r1 = float(np.linalg.norm(quad[(opp + 2) % 4] - q)) / base
    r2 = float(np.linalg.norm(quad[(opp + 3) % 4] - p)) / base
    return r1, r2


def refine_hidden_sides(track: Track) -> np.ndarray:
    """Ricalcola gli angoli dei lati nascosti interpolando nel tempo.

    Durante un'occlusione i due angoli del lato nascosto stanno sulle rette
    (misurate) dei lati adiacenti; ignota è solo la loro distanza dal lato
    opposto. Questa distanza, relativa al lato opposto, viene interpolata
    linearmente tra l'ultimo fotogramma "pulito" prima dell'occlusione e il
    primo dopo. Se uno dei due manca (occlusione all'inizio o alla fine del
    video) si usa il valore disponibile.
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
        ra = _hidden_ratios(raw[before[-1]], m) if len(before) else None
        rb = _hidden_ratios(raw[after[0]], m) if len(after) else None
        if ra is not None or rb is not None:
            opp = (m + 2) % 4
            for k in range(t, end + 1):
                if ra is not None and rb is not None:
                    u = (k - before[-1]) / (after[0] - before[-1])
                    r1 = (1 - u) * ra[0] + u * rb[0]
                    r2 = (1 - u) * ra[1] + u * rb[1]
                else:
                    r1, r2 = ra if ra is not None else rb
                quad = raw[k]
                p, q = quad[opp], quad[(opp + 1) % 4]
                base = np.linalg.norm(q - p)
                # Le direzioni restano quelle delle rette misurate.
                d1 = quad[(opp + 2) % 4] - q
                d2 = quad[(opp + 3) % 4] - p
                quad[(opp + 2) % 4] = q + d1 / max(np.linalg.norm(d1), 1e-9) * r1 * base
                quad[(opp + 3) % 4] = p + d2 / max(np.linalg.norm(d2), 1e-9) * r2 * base
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
