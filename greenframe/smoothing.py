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
  dal passato, quindi con possibile deriva) vengono ricalcolati combinando
  l'**interpolazione** tra il fotogramma prima e quello dopo l'occlusione con
  la **geometria** del rettangolo ricostruita in ogni fotogramma dai tre lati
  visibili;
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

    imp = track.impostor if track.impostor is not None else np.full((track.n_frames, 4), np.nan)
    fixed = []
    for t in candidates:
        if devs[t] <= threshold:
            continue
        # Sospetti: lati non pienamente visibili e lati il cui bordo prosegue
        # oltre l'angolo (test della giunzione a T: oggetto dritto davanti).
        weak = [s for s in range(4)
                if track.support[t, s] < trusted_support or np.nan_to_num(imp[t, s]) >= 0.3]
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


def _ratio_slope(raw: np.ndarray, frames: np.ndarray, side: int) -> np.ndarray | None:
    """Velocità di variazione (per fotogramma) delle distanze relative r1, r2,
    stimata con una retta ai minimi quadrati sui fotogrammi indicati."""
    if len(frames) < 2:
        return None
    r = np.array([_hidden_ratios(raw[f], side) for f in frames])
    t = frames.astype(np.float64)
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    return np.linalg.lstsq(A, r, rcond=None)[0][0]


# Incertezza (px) di ciascun angolo misurato: serve a propagare l'errore di
# misura alla ricostruzione geometrica del lato nascosto. Con questo valore
# l'errore reale della ricostruzione sui fotogrammi puliti dei video di prova
# è in mediana 0.6-0.8 volte quello previsto (p90: 1.4-1.9 volte).
_CORNER_SIGMA_PX = 0.15
# Errore minimo attribuito a r1, r2 (frazione del lato opposto).
_RATIO_FLOOR = 0.0005


def _rect_ratios(quad: np.ndarray, side: int, focal: float, aspect: float,
                 size: tuple[int, int]) -> np.ndarray | None:
    """r1, r2 del lato ``side`` ricostruito dagli altri tre con il modello del rettangolo."""
    cand = geo.complete_rectangle(geo.quad_lines(quad), side, focal, aspect, size, quad.mean(axis=0))
    if cand is None or not np.isfinite(cand).all():
        return None
    return np.array(_hidden_ratios(cand, side))


def _rect_observation(quad: np.ndarray, side: int, focal: float, aspect: float,
                      size: tuple[int, int]) -> tuple[np.ndarray | None, float]:
    """Ricostruzione geometrica di r1, r2 in un fotogramma e la sua incertezza.

    L'incertezza si ottiene spostando ogni angolo di ``_CORNER_SIGMA_PX`` in x
    e in y e misurando quanto cambia il risultato. Quando le rette dei lati
    sono quasi parallele il punto di fuga è mal determinato e l'incertezza
    cresce: quei fotogrammi contano poco nella stima finale.
    """
    base = _rect_ratios(quad, side, focal, aspect, size)
    if base is None:
        return None, np.inf
    var = np.zeros(2)
    for c in range(4):
        for d in ((_CORNER_SIGMA_PX, 0.0), (0.0, _CORNER_SIGMA_PX)):
            q = quad.copy()
            q[c] += d
            r = _rect_ratios(q, side, focal, aspect, size)
            if r is None:
                return base, np.inf
            var += (r - base) ** 2
    return base, float(np.sqrt(var.max()))


def _dominant_sides(hidden: np.ndarray, half: int = 2) -> np.ndarray:
    """Lato nascosto "di maggioranza" in una finestra di ±``half`` fotogrammi.

    Un fotogramma isolato che indica un lato diverso dai vicini è quasi
    sempre un errore (es. il bordo di un oggetto scambiato per un lato).
    """
    out = hidden.copy()
    for i in range(len(hidden)):
        win = hidden[max(0, i - half): i + half + 1]
        out[i] = int(np.bincount(win, minlength=4).argmax())
    return out


def _hermite_ratios(raw: np.ndarray, clean_idx: np.ndarray, side: int, t: int, end: int,
                    slope_frames: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Interpolazione di r1, r2 nei fotogrammi t..end tra i fotogrammi puliti vicini.

    Spline cubica di Hermite tra l'ultimo fotogramma pulito prima e il primo
    dopo, con le pendenze misurate su ``slope_frames`` fotogrammi puliti a
    ciascun estremo. Se c'è un solo estremo si usa il suo valore (estrapolare
    è instabile). Ritorna i valori (n, 2) e la distanza (fotogrammi) di
    ciascun fotogramma dall'estremo più vicino.
    """
    before = clean_idx[clean_idx < t]
    after = clean_idx[clean_idx > end]
    ks = np.arange(t, end + 1)
    if len(before) and len(after):
        a, b = int(before[-1]), int(after[0])
        ra = np.array(_hidden_ratios(raw[a], side))
        rb = np.array(_hidden_ratios(raw[b], side))
        L = float(b - a)
        secant = (rb - ra) / L
        fa = before[before > a - 2 * slope_frames][-slope_frames:]
        fb = after[after < b + 2 * slope_frames][:slope_frames]
        sa = _ratio_slope(raw, fa, side)
        sb = _ratio_slope(raw, fb, side)
        sa = secant if sa is None else sa
        sb = secant if sb is None else sb
        u = ((ks - a) / L)[:, None]
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        values = h00 * ra + h10 * L * sa + h01 * rb + h11 * L * sb
        return values, np.minimum(ks - a, b - ks)
    if len(before) or len(after):
        ref = int(before[-1]) if len(before) else int(after[0])
        values = np.tile(np.array(_hidden_ratios(raw[ref], side)), (len(ks), 1))
        return values, np.abs(ks - ref)
    return None, None


def _fit_corrections(obs_t: np.ndarray, obs_res: np.ndarray, obs_w: np.ndarray,
                     frames: np.ndarray, prior_w: np.ndarray, half: int,
                     iters: int = 3) -> np.ndarray:
    """Correzione (n, 2) da sommare all'interpolazione nei fotogrammi ``frames``.

    Regressione locale quadratica (kernel tricubo, ±``half`` fotogrammi) dei
    residui "ricostruzione geometrica − interpolazione", pesati con l'inverso
    della varianza. In ogni fotogramma c'è anche una pseudo-osservazione
    "correzione nulla" con peso ``prior_w``: l'interpolazione vale come
    informazione a priori, tanto più forte quanto più si è vicini a un
    fotogramma pulito. Le ricostruzioni anomale (oltre 4 sigma) perdono peso
    con la funzione bisquare.
    """
    ft = frames.astype(np.float64)
    rob = np.ones(len(obs_t))
    sigma = 1.0 / np.sqrt(np.maximum(obs_w, 1e-30))

    def fit(k0: float) -> np.ndarray:
        x = np.concatenate([obs_t - k0, ft - k0])
        y = np.concatenate([obs_res, np.zeros((len(ft), 2))])
        w = np.concatenate([obs_w * rob, prior_w])
        w = w * np.where(np.abs(x) <= half, (1 - (np.abs(x) / (half + 1)) ** 3) ** 3, 0.0)
        ok = w > 0
        deg = 2 if ok.sum() >= 5 else (1 if ok.sum() >= 2 else 0)
        X = np.vander(x[ok], deg + 1, increasing=True)
        sw = np.sqrt(w[ok])[:, None]
        coef, *_ = np.linalg.lstsq(X * sw, y[ok] * sw, rcond=None)
        return coef[0]

    for _ in range(iters):
        resid = np.array([np.abs(obs_res[i] - fit(obs_t[i])).max() for i in range(len(obs_t))])
        u = np.clip(resid / (4.0 * sigma), 0.0, 1.0)
        rob = (1.0 - u ** 2) ** 2
    return np.array([fit(k) for k in ft])


def refine_hidden_sides(track: Track, slope_frames: int = 3, half_window: int = 8,
                        prior_accel: float = 0.002) -> np.ndarray:
    """Ricalcola gli angoli dei lati nascosti combinando geometria e tempo.

    Durante un'occlusione i due angoli del lato nascosto stanno sulle rette
    (misurate) dei lati adiacenti; ignota è solo la loro distanza dal lato
    opposto (relativa al lato opposto: r1, r2). Due fonti di informazione:

    1. **Interpolazione nel tempo** (spline di Hermite tra i fotogrammi puliti
       prima e dopo): ottima per occlusioni brevi, ma non può sapere se
       durante l'occlusione il pannello si inclina e poi torna indietro.
    2. **Geometria del fotogramma**: un rettangolo fisico di proporzioni note,
       visto da una camera di focale nota, è determinato da tre lati
       (:func:`geometry.complete_rectangle`). La ricostruzione di un singolo
       fotogramma è sensibile al rumore quando i lati sono quasi paralleli,
       ma la sua incertezza si può calcolare (:func:`_rect_observation`).
       Prima di usarla, la si verifica sui fotogrammi puliti vicini
       all'occlusione, dove il lato è anche misurato: lo scarto medio
       (distorsione dell'obiettivo, focale imprecisa) viene sottratto e la
       dispersione si aggiunge all'incertezza.

    Le due fonti si combinano con una regressione locale robusta
    (:func:`_fit_corrections`), pesata con le rispettive incertezze; quella
    dell'interpolazione cresce con il quadrato della distanza dal fotogramma
    pulito più vicino (``prior_accel``). In un'occlusione di 44 fotogrammi in
    Full HD, con il pannello che si inclina, l'errore massimo sugli angoli
    nascosti è sceso da 41 a 2.7 px (solo interpolazione: 41 px).

    Nei fotogrammi in cui il rilevamento indica un lato nascosto diverso da
    quello dei vicini (tipicamente un errore) gli angoli diventano NaN:
    :func:`smooth_track` li ricostruisce dai fotogrammi adiacenti.
    """
    raw = track.raw.copy()
    hs = track.hidden_side
    if hs is None:
        return raw
    T = len(raw)
    valid = ~np.isnan(raw[:, 0, 0])
    clean = np.array([s == STATUS_OK and h < 0 for s, h in zip(track.status, hs)]) & valid
    clean_idx = np.nonzero(clean)[0]
    size = (track.width, track.height)
    use_geometry = bool(track.focal) and bool(track.panel_aspect)
    cache: dict[tuple[int, int], tuple[np.ndarray | None, float]] = {}

    def observe(k: int, side: int) -> tuple[np.ndarray | None, float]:
        if (k, side) not in cache:
            cache[(k, side)] = _rect_observation(raw[k], side, track.focal, track.panel_aspect, size)
        return cache[(k, side)]

    t = 0
    while t < T:
        if hs[t] < 0:
            t += 1
            continue
        # Occlusione: fotogrammi consecutivi con un lato nascosto (qualsiasi).
        end = t
        while end + 1 < T and hs[end + 1] >= 0:
            end += 1
        sides = _dominant_sides(hs[t:end + 1])
        s0 = 0
        while s0 < len(sides):
            s1 = s0
            while s1 + 1 < len(sides) and sides[s1 + 1] == sides[s0]:
                s1 += 1
            m = int(sides[s0])
            opp = (m + 2) % 4
            ks = np.arange(t + s0, t + s1 + 1)
            base, dist = _hermite_ratios(raw, clean_idx, m, t, end, slope_frames)
            if base is None:
                s0 = s1 + 1
                continue
            base, dist = base[s0:s1 + 1], dist[s0:s1 + 1].astype(np.float64)
            ratios = base
            if use_geometry:
                ratios = _with_geometry(raw, hs, valid, clean_idx, ks, m, base, dist, t, end,
                                        track.focal_reliable, observe, half_window, prior_accel)
            for j, k in enumerate(ks):
                if hs[k] != m:
                    raw[k] = np.nan
                    continue
                raw[k] = _place_hidden(raw[k], opp, float(ratios[j][0]), float(ratios[j][1]))
            s0 = s1 + 1
        t = end + 1
    return raw


def _with_geometry(raw, hs, valid, clean_idx, ks, side, base, dist, t, end, focal_reliable,
                   observe, half_window, prior_accel) -> np.ndarray:
    """r1, r2 dell'interpolazione ``base`` corretti con le ricostruzioni geometriche."""
    # Verifica sui fotogrammi puliti vicini: scarto sistematico e dispersione.
    ctx = max(30, end - t + 1)
    near = clean_idx[(clean_idx >= t - ctx) & (clean_idx <= end + ctx)]
    resid, sens = [], []
    for k in near:
        r, s = observe(int(k), side)
        if r is not None and np.isfinite(s):
            resid.append(r - np.array(_hidden_ratios(raw[k], side)))
            sens.append(s)
    bias, floor = np.zeros(2), _RATIO_FLOOR
    if len(resid) >= 8:
        resid_a, sens_a = np.array(resid), np.array(sens)
        bias = np.median(resid_a, axis=0)
        var = float(np.median(((resid_a - bias) ** 2).max(axis=1))) / 0.455   # mediana di chi²(1)
        floor = float(np.sqrt(max(var - float(np.median(sens_a ** 2)), _RATIO_FLOOR ** 2)))
    elif not focal_reliable:
        return base       # focale incerta e nessuna verifica possibile: solo interpolazione
    obs_t, obs_res, obs_w = [], [], []
    for j, k in enumerate(ks):
        if hs[k] != side or not valid[k]:
            continue
        r, s = observe(int(k), side)
        if r is None or not np.isfinite(s):
            continue
        obs_t.append(float(k))
        obs_res.append(r - bias - base[j])
        obs_w.append(1.0 / (s ** 2 + floor ** 2))
    if not obs_t:
        return base
    prior_sigma = _RATIO_FLOOR + prior_accel * dist ** 2
    corr = _fit_corrections(np.array(obs_t), np.array(obs_res), np.array(obs_w), ks,
                            1.0 / prior_sigma ** 2, half_window)
    return base + corr


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


MOTION_SCALE_HALF = 4      # semi-finestra (fotogrammi) della scala locale dei residui
MOTION_SCALE_MULT = 6.0    # soglia minima = questo multiplo della mediana dei residui vicini


def _outlier_threshold(resid: np.ndarray, measured: np.ndarray, cutoff: float) -> np.ndarray:
    """Soglia degli outlier per fotogramma: ``cutoff``, allargata nei movimenti bruschi.

    Una misura sbagliata è isolata: i fotogrammi vicini hanno residui piccoli.
    In un movimento brusco, invece, il polinomio locale non segue bene
    *diversi* fotogrammi consecutivi, che hanno residui simili tra loro e non
    vanno scartati. La mediana dei residui nei ±``MOTION_SCALE_HALF``
    fotogrammi distingue i due casi: la soglia diventa almeno
    ``MOTION_SCALE_MULT`` volte quella mediana.
    """
    T = len(resid)
    thr = np.full(T, float(cutoff))
    for t in range(T):
        lo, hi = max(0, t - MOTION_SCALE_HALF), min(T, t + MOTION_SCALE_HALF + 1)
        near = resid[lo:hi][measured[lo:hi]]
        if near.size:
            thr[t] = max(cutoff, MOTION_SCALE_MULT * float(np.median(near)))
    return thr


def smooth_track(track: Track, half_window: int = 2, outlier_px: float = 1.5,
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
    measured = base_w > 0.02
    for _ in range(3):
        resid = np.zeros(T)
        for t in np.nonzero(measured)[0]:
            est = _local_poly(filled, w, t, robust_half)
            if est is not None:
                resid[t] = np.linalg.norm((raw[t] - est).reshape(4, 2), axis=1).max()
        u = np.clip(resid / _outlier_threshold(resid, measured, cutoff), 0.0, 1.0)
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
