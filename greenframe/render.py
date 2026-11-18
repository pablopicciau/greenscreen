"""Rendering su GPU (PyTorch): proiezione del quadro e compositing in luce lineare.

Per ogni fotogramma, su un ritaglio (ROI) attorno al quadro:

1. **Colore**: i piani YUV originali vengono convertiti in RGB rispettando la
   posizione della crominanza (vedi ``color.py``) e portati in luce lineare.
2. **Screen plate** ``P``: come apparirebbe il pannello pulito in ogni pixel
   (anche sotto le mani), per normalized convolution multi-scala dei pixel
   sicuramente di pannello. Fornisce luce e ombre da trasferire al quadro.
3. **Chiave** ``alpha`` (quanto il pixel è pannello):
   * stima iniziale con la differenza di verde ``G - max(R, B)`` rapportata a
     quella dello screen plate;
   * nella fascia di transizione (bordi delle dita, bordi del pannello,
     sfocatura di movimento) alpha viene **ricalcolato esattamente** con il
     modello a due colori in luce lineare: ``I = alpha·P + (1-alpha)·F``,
     dove ``F`` è il colore del primo piano stimato dai pixel vicini sicuramente
     non-pannello. ``alpha = <I-F, P-F> / |P-F|²``. La differenza di verde da
     sola sottostima alpha quando il primo piano ha G < R (pelle, sfondi
     caldi) e "mangia" i bordi: il modello a due colori no.
   * alpha è limitato all'area del quad (allargata di pochi pixel e "spazzata"
     dal movimento): il verde fuori dal quadro (piante) non viene toccato.
     Un verde di tinta diversa dal pannello *attaccato* al bordo (modello di
     colore dell'analisi) conta come primo piano: niente quadro che "sborda"
     sulla pianta e niente despill sul suo verde.
4. **Quadro**: proiettato con la sua omografia, campionato da una piramide
   in luce lineare con filtro trilineare e sovracampionamento 2×2 (niente
   moiré né scalettature) e con **motion blur**: media di più istanti
   all'interno del tempo di esposizione stimato dal video (angoli interpolati
   con un polinomio di secondo grado tra fotogramma precedente e successivo).
   Poi una leggera sfocatura gaussiana uguale a quella della camera.
5. **Luce**: la luminanza (lineare) dello screen plate, rapportata a quella di
   riferimento del pannello, moltiplica il quadro (ombre delle mani, quadro
   che si gira verso la luce).
6. **Composizione** in luce lineare: ``O = I + alpha·(R - P)``, esatta nei
   pixel misti; **despill** del riflesso verde sulle mani vicino al pannello;
   **grana** della camera (luminanza residua del pannello) riportata sul quadro.
7. Uscita in YUV 4:2:0: i pixel non toccati conservano i valori originali
   (nessuna perdita fuori dal quadro).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .blur_estimation import render_blur_sigma
from .color import LUMA_709, YuvFormat, from_linear, rgb_to_yuv420, to_linear, yuv420_to_rgb
from .config import Config
from .image_fit import fit_to_aspect

# ---------------------------------------------------------------------------
# Operazioni di base (tensori (C, H, W) float32)
# ---------------------------------------------------------------------------


def _torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def gaussian_blur(x, sigma: float):
    """Sfocatura gaussiana separabile di un tensore (C, H, W), bordi replicati."""
    torch, F = _torch()
    if sigma < 0.05:
        return x
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[0]
    y = x[None]
    pw = min(r, y.shape[-1] - 1)
    ph = min(r, y.shape[-2] - 1)
    if pw < r or ph < r:          # immagine più piccola del filtro: si riduce il raggio
        r = min(pw, ph)
        if r < 1:
            return x
        t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
        k = torch.exp(-0.5 * (t / sigma) ** 2)
        k = k / k.sum()
    y = F.pad(y, (r, r, 0, 0), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous(), groups=c)
    y = F.pad(y, (0, 0, r, r), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous(), groups=c)
    return y[0]


def _resize(x, h: int, w: int, mode: str):
    torch, F = _torch()
    if mode == "area":
        return F.interpolate(x[None], size=(h, w), mode="area")[0]
    return F.interpolate(x[None], size=(h, w), mode="bilinear", align_corners=False)[0]


def normalized_fill(img, weight, scales: tuple[float, ...], target_px: int = 320):
    """Interpola ``img`` (C, H, W) dove ``weight`` (H, W, 0-1) è nullo (normalized convolution).

    Lavora su una versione ridotta (lato lungo ``target_px``) per velocità.
    ``scales`` sono i sigma relativi al lato lungo ridotto, dal più grosso al
    più fine: le scale fini seguono i dettagli, quelle grosse riempiono i buchi.
    """
    torch, F = _torch()
    c, h, w = img.shape
    scale = min(1.0, target_px / max(h, w))
    sh, sw = max(4, int(round(h * scale))), max(4, int(round(w * scale)))
    w_small = _resize(weight[None], sh, sw, "area")[0]
    n_small = _resize(img * weight[None], sh, sw, "area")
    good = (w_small > 0.98).float()
    if float(good.sum()) < 1:
        tot = float(weight.sum())
        mean = (img * weight[None]).sum(dim=(1, 2)) / max(tot, 1e-6)
        return mean[:, None, None].expand(c, h, w).clone()
    num = n_small * good[None] / w_small.clamp_min(1e-6)[None]
    result = None
    for rel in scales:
        sigma = max(1.0, rel * max(sh, sw))
        k = min(1.0, 3.0 / sigma)
        lh, lw = max(2, int(round(sh * k))), max(2, int(round(sw * k)))
        wl = _resize(good[None], lh, lw, "area")
        nl = _resize(num, lh, lw, "area")
        den = gaussian_blur(wl, sigma * k)
        nume = gaussian_blur(nl, sigma * k)
        if k < 1.0:
            den = _resize(den, sh, sw, "bilinear")
            nume = _resize(nume, sh, sw, "bilinear")
        est = nume / den.clamp_min(1e-6)
        if result is None:
            fallback = (num * good[None]).sum(dim=(1, 2)) / good.sum()
            est = torch.where(den < 1e-4, fallback[:, None, None], est)
            result = est
        else:
            blend = (den / 0.35).clamp(0.0, 1.0)
            result = blend * est + (1.0 - blend) * result
    return _resize(result, h, w, "bilinear")


def quad_coverage(quad: np.ndarray, x, y, expand: float = 0.0):
    """Copertura anti-aliased (0-1) di un quad convesso sulla griglia di pixel (x, y).

    Usa la distanza con segno dai 4 lati (positiva dentro): ``clamp(0.5 + d)``
    è l'area del pixel coperta da un bordo dritto.
    """
    torch, _ = _torch()
    d = signed_inside_distance(quad, x, y)
    return (0.5 + d + expand).clamp(0.0, 1.0)


def signed_inside_distance(quad: np.ndarray, x, y):
    """Distanza (px) dal bordo del quad, positiva all'interno (minimo sui 4 lati)."""
    return _convex_inside_distance(quad, x, y)


def _convex_inside_distance(poly: np.ndarray, x, y):
    """Distanza (px) dal bordo di un poligono convesso, positiva all'interno."""
    torch, _ = _torch()
    centroid = poly.mean(axis=0)
    out = None
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        if np.linalg.norm(q - p) < 1e-9:
            continue
        ln = geo.line_through(p, q)
        if geo.point_line_distance(centroid[None], ln)[0] < 0:
            ln = -ln
        d = float(ln[0]) * x + float(ln[1]) * y + float(ln[2])
        out = d if out is None else torch.minimum(out, d)
    return out


# ---------------------------------------------------------------------------
# Texture del quadro
# ---------------------------------------------------------------------------


class PaintingTexture:
    """Il quadro in luce lineare con la sua piramide (mipmap) sulla GPU."""

    def __init__(self, image_bgr_u8: np.ndarray, device, max_side: int = 3072):
        import cv2

        torch, F = _torch()
        img = image_bgr_u8
        s = min(1.0, max_side / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rgb = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))).to(device)
        lin = to_linear(rgb.float() / 255.0)
        self.h, self.w = lin.shape[1:]
        self.levels = [lin]
        while min(self.levels[-1].shape[1:]) > 8:
            prev = self.levels[-1]
            nh, nw = (prev.shape[1] + 1) // 2, (prev.shape[2] + 1) // 2
            # Riduzione "ad area" in luce lineare: niente aliasing né scurimento.
            self.levels.append(F.interpolate(prev[None], size=(nh, nw), mode="area")[0])

    def sample(self, u, v, lod, lod_range: tuple[float, float]):
        """Campionamento trilineare nei punti (u, v) (coordinate pixel del livello 0).

        ``u``, ``v``, ``lod`` hanno forma (N, H, W) (N campioni per pixel);
        ``lod`` è il livello di dettaglio (log2 dell'impronta del campione in
        pixel del quadro), limitato a ``lod_range`` calcolato sulla CPU (così
        non serve leggere dalla GPU min/max ad ogni fotogramma). Bordi replicati.
        Ritorna (N, 3, H, W).
        """
        torch, F = _torch()
        nlev = len(self.levels)
        lo = int(np.clip(math.floor(lod_range[0]), 0, nlev - 1))
        hi = int(np.clip(math.floor(lod_range[1]) + 1, lo, nlev - 1))
        lod = lod.clamp(float(lo), float(hi))
        n, h, w = u.shape
        # Coordinate normalizzate (align_corners=False): uguali per tutti i livelli.
        grid = torch.stack([(u + 0.5) * (2.0 / self.w) - 1.0, (v + 0.5) * (2.0 / self.h) - 1.0], dim=-1)
        grid = grid.view(1, n * h, w, 2)            # tutti i campioni in un'unica chiamata
        out = None
        for lev in range(lo, hi + 1):
            smp = F.grid_sample(self.levels[lev][None], grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
            smp = smp.view(3, n, h, w).transpose(0, 1)
            if lo == hi:
                return smp
            wgt = (1.0 - (lod - lev).abs()).clamp_min(0.0)[:, None]
            out = smp * wgt if out is None else out + smp * wgt
        return out


def _homography(src_w: int, src_h: int, quad: np.ndarray) -> np.ndarray:
    return geo.homography_rect_to_quad(src_w, src_h, quad).astype(np.float64)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


@dataclass
class RenderResult:
    """Risultato di un fotogramma: piani YUV per l'encoder e (facoltativo) RGB per anteprime."""

    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    bgr: np.ndarray | None = None


class Renderer:
    """Inserisce il quadro nei fotogrammi di un video, usando la traccia dell'analisi."""

    def __init__(self, image_bgr: np.ndarray, track, cfg: Config, device, in_format: YuvFormat):
        torch, _ = _torch()
        self.torch = torch
        self.cfg = cfg
        self.track = track
        self.device = device
        self.in_format = in_format
        self.out_format = YuvFormat("bt709", False, (0.0, 0.5))
        rp = cfg.render
        aspect = rp.panel_aspect or track.panel_aspect or 1.0
        self.prepared = fit_to_aspect(image_bgr, aspect, rp.mode, rp.fit_background)
        self.tex = PaintingTexture(self.prepared, device)
        # Luminanza (lineare) di riferimento del pannello: a questo livello il
        # quadro appare con la sua luminosità originale.
        vals = track.panel_luma[~np.isnan(track.panel_luma)]
        self.reference = float(np.percentile(vals, 95)) if vals.size else None
        # Otturatore e nitidezza: valori dell'utente o stimati dall'analisi.
        self.shutter = rp.shutter if rp.shutter is not None else (track.shutter if track.shutter is not None else 0.5)
        if not rp.motion_blur:
            self.shutter = 0.0
        self.lens_sigma = rp.lens_blur if rp.lens_blur is not None else render_blur_sigma(track.lens_sigma)
        self.max_samples = rp.motion_samples if device.type == "cuda" else min(rp.motion_samples, 6)
        self.sample_chunk = 4
        # Memoria della GPU: PyTorch tiene da parte i blocchi liberati per
        # riusarli, ma il ritaglio cambia dimensione a ogni fotogramma e i
        # blocchi vecchi spesso non vanno bene per quelli nuovi. Senza limiti la
        # riserva cresce fino a riempire la scheda e, su Windows, il driver la
        # estende nella RAM del PC (nei test: 10 GB per un Reel di 15 s).
        # Oltre questa soglia la riserva inutilizzata viene restituita.
        self.cache_limit = 0
        if device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            self.cache_limit = int(min(1.5 * 2**30, 0.4 * total))

    # -- geometria temporale ---------------------------------------------------
    def _quad_at(self, i: int, tau: float) -> np.ndarray:
        """Angoli al tempo ``i + tau`` (tau in fotogrammi) con interpolazione quadratica."""
        sm = self.track.smooth
        q = sm[i]
        prev = sm[i - 1] if i > 0 else None
        nxt = sm[i + 1] if i + 1 < len(sm) else None
        ok_p = prev is not None and not np.isnan(prev).any()
        ok_n = nxt is not None and not np.isnan(nxt).any()
        if ok_p and ok_n:
            vel = 0.5 * (nxt - prev)
            acc = nxt - 2 * q + prev
            return q + tau * vel + 0.5 * tau * tau * acc
        if ok_n:
            return q + tau * (nxt - q)
        if ok_p:
            return q + tau * (q - prev)
        return q

    def _time_samples(self, i: int) -> list[float]:
        """Istanti (in fotogrammi) in cui campionare il quadro per il motion blur."""
        if self.shutter <= 0:
            return [0.0]
        a = self._quad_at(i, -0.5 * self.shutter)
        b = self._quad_at(i, 0.5 * self.shutter)
        extent = float(np.linalg.norm(b - a, axis=1).max())     # px percorsi durante l'esposizione
        n = int(np.clip(math.ceil(extent / 0.75), 1, self.max_samples))
        if n == 1:
            return [0.0]
        return [((k + 0.5) / n - 0.5) * self.shutter for k in range(n)]

    # -- fotogramma ---------------------------------------------------------------
    def render(self, frame, i: int, want_bgr: bool = False) -> RenderResult:
        """Rende il fotogramma ``i`` (``frame``: YuvFrame con i piani originali).

        Si lavora solo sul ritaglio attorno al quadro: i piani YUV del resto
        del fotogramma restano quelli originali (nessuna perdita, e niente
        conversioni a tutto schermo). Con ``want_bgr`` si ricostruisce anche
        l'immagine intera (anteprime, benchmark).
        """
        result = self._render(frame, i, want_bgr)
        if self.cache_limit and self.torch.cuda.memory_reserved(self.device) > self.cache_limit:
            self.torch.cuda.empty_cache()
        return result

    def _render(self, frame, i: int, want_bgr: bool) -> RenderResult:
        torch, F = _torch()
        dev = self.device
        sm = self.track.smooth
        quad = sm[i] if (sm is not None and i < len(sm)) else None
        H, W = frame.y.shape
        if quad is None or np.isnan(quad).any():
            return self._passthrough(frame, want_bgr)

        rp, kp = self.cfg.render, self.cfg.key
        times = self._time_samples(i)
        quads = [self._quad_at(i, t) for t in times]

        # ROI: tutti i quad del tempo di esposizione + margine, a coordinate
        # pari (così il ritaglio della crominanza 4:2:0 resta allineato).
        allq = np.concatenate(quads + [quad])
        margin = rp.despill_margin + 3 * max(1.0, 0.3 * rp.despill_margin) + rp.edge_expand + 8
        x0 = int(max(0, math.floor(allq[:, 0].min() - margin))) // 2 * 2
        y0 = int(max(0, math.floor(allq[:, 1].min() - margin))) // 2 * 2
        x1 = min(W, (int(math.ceil(allq[:, 0].max() + margin)) + 2) // 2 * 2)
        y1 = min(H, (int(math.ceil(allq[:, 1].max() + margin)) + 2) // 2 * 2)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return self._passthrough(frame, want_bgr)
        roi = self._roi_rgb(frame, x0, y0, x1, y1)              # (3, h, w) valori video 0-1
        off = np.array([x0, y0], np.float64)
        lin = to_linear(roi.clamp(-0.1, 1.2))
        rh, rw = lin.shape[1:]
        ys = torch.arange(rh, device=dev, dtype=torch.float32)[:, None].expand(rh, rw)
        xs = torch.arange(rw, device=dev, dtype=torch.float32)[None, :].expand(rh, rw)
        q_roi = quad - off
        quads_roi = [q - off for q in quads]

        # --- Maschere geometriche ------------------------------------------------
        # Zona in cui il quadro può comparire: inviluppo convesso di tutte le
        # posizioni durante l'esposizione (motion blur), allargato di pochi px.
        if len(quads_roi) > 1:
            import cv2

            hull = cv2.convexHull(np.concatenate(quads_roi).astype(np.float32)).reshape(-1, 2)
            m_quad = (0.5 + _convex_inside_distance(hull.astype(np.float64), xs, ys)
                      + rp.edge_expand).clamp(0.0, 1.0)
        else:
            m_quad = quad_coverage(quads_roi[0], xs, ys, rp.edge_expand)
        inner = signed_inside_distance(q_roi, xs, ys) >= 3.0
        spill_zone = gaussian_blur(quad_coverage(q_roi, xs, ys, rp.despill_margin)[None],
                                   max(1.0, 0.3 * rp.despill_margin))[0]

        # --- Chiave iniziale e screen plate ---------------------------------------
        r_, g_, b_ = lin[0], lin[1], lin[2]
        d = g_ - torch.maximum(r_, b_)
        g_lvl = g_.clamp_min(1e-4)
        clean = inner & (d >= 0.25 * g_lvl) & (d > 0.01)
        chroma = tau = None
        if self.track.panel_color:
            tau = float(self.track.panel_color.get("tau", 4.5))
            chroma = self._panel_chroma_distance(lin)
            clean = clean & (chroma < 2.0 * tau)
        plate = normalized_fill(lin, clean.float(), (0.32, 0.08, 0.02))
        d_plate = (plate[1] - torch.maximum(plate[0], plate[2])).clamp_min(1e-3)
        ratio = d / d_plate
        alpha0 = ((ratio - kp.low) / max(kp.high - kp.low, 1e-6)).clamp(0.0, 1.0)

        # Verdi che non sono il pannello (pianta, vestito) accanto al quadro: per
        # la differenza di verde sembrano pannello, ma la loro tinta è diversa da
        # quella misurata durante l'analisi. Si cercano solo dove il quadro non
        # arriva in nessun istante dell'esposizione (oltre ~2.5 px dal bordo):
        # più vicino al bordo i pixel misti pannello/sfondo hanno anch'essi una
        # tinta diversa, ma sono pannello in parte. Contano come primo piano e,
        # vicino a loro, la chiave usa il modello a due colori anche fuori dalla
        # fascia di transizione: il quadro non "sborda" sulla pianta.
        other = near_other = None
        if chroma is not None:
            other = (chroma > tau) & (ratio > 0.2) & (m_quad <= 0.0)
            near_other = F.max_pool2d(other.float()[None, None], 9, stride=1, padding=4)[0, 0] > 0

        # --- Chiave esatta nella fascia di transizione (modello a due colori) -----
        fg = ratio < 0.03
        if other is not None:
            fg = fg | other
        fore = normalized_fill(lin, fg.float(), (0.08, 0.02, 0.006), target_px=640)
        band = (alpha0 > 0.01) & (alpha0 < 0.99)
        band = F.max_pool2d(band.float()[None, None], 5, stride=1, padding=2)[0, 0] > 0
        if near_other is not None:
            band = band | (near_other & ~inner)
        pf = plate - fore
        den = (pf * pf).sum(dim=0)
        a2 = ((lin - fore) * pf).sum(dim=0) / den.clamp_min(1e-6)
        use = band & (den > 0.004)
        if not kp.unmix:
            use = torch.zeros_like(use)
        alpha = torch.where(use, a2.clamp(0.0, 1.0), alpha0) * m_quad

        # --- Quadro proiettato (motion blur + sovracampionamento 2×2) -------------
        painting = self._sample_painting(quads_roi, xs, ys)
        if self.lens_sigma > 0.05:
            painting = gaussian_blur(painting, self.lens_sigma)

        # --- Luce ------------------------------------------------------------------
        if rp.lighting > 0 and self.reference:
            lum = LUMA_709[0] * plate[0] + LUMA_709[1] * plate[1] + LUMA_709[2] * plate[2]
            light = (lum / self.reference).clamp(0.05, 2.0) ** rp.lighting
            painting = painting * light[None]

        # --- Composizione in luce lineare ------------------------------------------
        a = alpha[None]
        out = lin + a * (painting - plate)
        # Despill: il riflesso verde sulla parte "primo piano" dei pixel vicini al pannello.
        excess = (out[1] - torch.maximum(out[0], out[2])).clamp_min(0.0)
        w_sp = kp.despill * spill_zone * (1.0 - alpha)
        if near_other is not None:
            # Accanto a una pianta il verde che resta dopo la composizione è
            # quello della pianta, non un riflesso: si guarda il colore del primo
            # piano stimato (non il pixel, che è in parte pannello). Se è un verde
            # saturo con una tinta diversa da quella del pannello, niente despill.
            f_exc = (fore[1] - torch.maximum(fore[0], fore[2])) / fore[1].clamp_min(1e-4)
            saturated = ((f_exc - 0.25) / 0.15).clamp(0.0, 1.0)
            foreign = ((self._panel_chroma_distance(fore) - 0.75 * tau) / (0.25 * tau)).clamp(0.0, 1.0)
            w_sp = w_sp * (1.0 - saturated * foreign * near_other.float())
        out = torch.stack([out[0], out[1] - excess * w_sp, out[2]])
        out_disp = from_linear(out.clamp(0.0, 1.5))
        # Grana: residuo di luminanza del pannello (rumore, trama) riportato sul quadro.
        if rp.grain > 0:
            plate_disp = from_linear(plate.clamp(0.0, 1.5))
            yr = (LUMA_709[0] * (roi[0] - plate_disp[0]) + LUMA_709[1] * (roi[1] - plate_disp[1])
                  + LUMA_709[2] * (roi[2] - plate_disp[2]))
            out_disp = out_disp + (rp.grain * alpha * alpha * yr)[None]

        # --- Uscita ------------------------------------------------------------------
        changed = (m_quad > 0) | (w_sp > 1e-3)
        new_roi = torch.where(changed[None], out_disp, roi)
        return self._output(frame, new_roi, changed, (x0, y0, x1, y1), want_bgr)

    # -- parti ------------------------------------------------------------------------
    def _upload(self, *arrays):
        """Copia più array uint8 sulla GPU con un unico trasferimento."""
        torch = self.torch
        flat = [np.ascontiguousarray(a).ravel() for a in arrays]
        buf = torch.from_numpy(np.concatenate(flat)).to(self.device)
        out, k = [], 0
        for a in arrays:
            out.append(buf[k:k + a.size].view(a.shape))
            k += a.size
        return out

    def _passthrough(self, frame, want_bgr: bool) -> RenderResult:
        bgr = self._to_bgr(self._full_rgb(frame)) if want_bgr else None
        return RenderResult(frame.y, frame.u, frame.v, bgr)

    def _full_rgb(self, frame):
        y, u, v = self._upload(frame.y, frame.u, frame.v)
        return yuv420_to_rgb(y, u, v, self.in_format, detail=True)

    def _roi_rgb(self, frame, x0: int, y0: int, x1: int, y1: int):
        """RGB del ritaglio (coordinate pari): solo i piani del ritaglio vanno sulla GPU."""
        if frame.u.shape == frame.y.shape:                       # 4:4:4
            cu = (slice(y0, y1), slice(x0, x1))
        else:
            cu = (slice(y0 // 2, (y1 + 1) // 2), slice(x0 // 2, (x1 + 1) // 2))
        y, u, v = self._upload(frame.y[y0:y1, x0:x1], frame.u[cu], frame.v[cu])
        return yuv420_to_rgb(y, u, v, self.in_format, detail=True)

    def _output(self, frame, new_roi, changed, box, want_bgr: bool) -> RenderResult:
        """Piani YUV finali: originali ovunque tranne dove il ritaglio è cambiato."""
        torch, F = _torch()
        x0, y0, x1, y1 = box
        same_layout = (frame.u.shape != frame.y.shape
                       and tuple(self.in_format.siting) == tuple(self.out_format.siting)
                       and not self.in_format.full_range
                       and (self.in_format.matrix or "bt709") == "bt709")
        if not same_layout:
            # Formato d'ingresso diverso da quello d'uscita (es. video ruotato o
            # HDR convertito): si ricodifica l'intero fotogramma.
            full = self._full_rgb(frame)
            full[:, y0:y1, x0:x1] = new_roi
            y_n, u_n, v_n = rgb_to_yuv420(full, self.out_format)
            flat = torch.cat([y_n.reshape(-1), u_n.reshape(-1), v_n.reshape(-1)]).cpu().numpy()
            ny, nc = y_n.numel(), u_n.numel()
            y_o = flat[:ny].reshape(y_n.shape)
            u_o = flat[ny:ny + nc].reshape(u_n.shape)
            v_o = flat[ny + nc:].reshape(v_n.shape)
            bgr = self._to_bgr(full) if want_bgr else None
            return RenderResult(y_o, u_o, v_o, bgr)
        y_r, u_r, v_r = rgb_to_yuv420(new_roi, self.out_format)
        # Luminanza: si sostituiscono solo i pixel cambiati; crominanza: i
        # campioni il cui filtro tocca un pixel cambiato.
        grown = F.max_pool2d(changed.float()[None, None], 3, stride=1, padding=1)[0, 0]
        ch, cw = u_r.shape
        c_mask = F.max_pool2d(F.pad(grown[None, None], (0, cw * 2 - grown.shape[1],
                                                         0, ch * 2 - grown.shape[0])), 2)[0, 0] > 0
        packed = torch.cat([y_r.reshape(-1), u_r.reshape(-1), v_r.reshape(-1),
                            changed.reshape(-1).to(torch.uint8), c_mask.reshape(-1).to(torch.uint8)])
        flat = packed.cpu().numpy()                                # un solo trasferimento
        ny, nc = y_r.numel(), u_r.numel()
        yv = flat[:ny].reshape(y_r.shape)
        uv = flat[ny:ny + nc].reshape(u_r.shape)
        vv = flat[ny + nc:ny + 2 * nc].reshape(v_r.shape)
        m_y = flat[ny + 2 * nc:2 * ny + 2 * nc].reshape(y_r.shape).astype(bool)
        m_c = flat[2 * ny + 2 * nc:].reshape(u_r.shape).astype(bool)
        y_o, u_o, v_o = frame.y.copy(), frame.u.copy(), frame.v.copy()
        np.copyto(y_o[y0:y1, x0:x1], yv, where=m_y)
        np.copyto(u_o[y0 // 2:y0 // 2 + ch, x0 // 2:x0 // 2 + cw], uv, where=m_c)
        np.copyto(v_o[y0 // 2:y0 // 2 + ch, x0 // 2:x0 // 2 + cw], vv, where=m_c)
        bgr = None
        if want_bgr:
            full = self._full_rgb(frame)
            full[:, y0:y1, x0:x1] = new_roi
            bgr = self._to_bgr(full)
        return RenderResult(y_o, u_o, v_o, bgr)

    def _to_bgr(self, rgb) -> np.ndarray:
        torch = self.torch
        return (rgb.flip(0).permute(1, 2, 0) * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

    def _panel_chroma_distance(self, lin):
        """Distanza (in deviazioni standard) dalla cromaticità del pannello stimata nell'analisi.

        Cromaticità in luce lineare (R/G, B/G): non cambia con la luminosità,
        quindi le ombre sul pannello restano vicine. Il termine di rumore
        allarga la tolleranza nei pixel scuri.
        """
        pc = self.track.panel_color
        mean = pc["mean"]
        cov = np.array(pc["cov"])
        g = lin[1].clamp_min(1e-4)
        dr = lin[0] / g - mean[0]
        db = lin[2] / g - mean[1]
        s2 = (pc.get("noise", 0.012) / g) ** 2
        c00, c01, c11 = cov[0, 0] + s2, float(cov[0, 1]), cov[1, 1] + s2
        det = c00 * c11 - c01 * c01
        d2 = (c11 * dr * dr - 2 * c01 * dr * db + c00 * db * db) / det
        return d2.clamp_min(0.0).sqrt()

    def _sample_painting(self, quads_roi: list[np.ndarray], xs, ys):
        """Quadro proiettato: media su istanti di esposizione e posizioni sub-pixel.

        Per ogni campione si calcola l'omografia inversa (pixel -> quadro) e la
        sua jacobiana, da cui il livello di dettaglio della piramide. I pixel
        coperti dal quadro solo in parte del tempo di esposizione ricevono la
        media degli istanti in cui lo sono (il resto lo decide la chiave).
        """
        torch = self.torch
        tex = self.tex
        dev = xs.device
        jitter = [(-0.25, -0.25), (0.25, 0.25), (0.25, -0.25), (-0.25, 0.25)]
        n = max(4, len(quads_roi))
        h_inv, offs, lods = [], [], []
        for k in range(n):
            q = quads_roi[(k * len(quads_roi)) // n]
            hi = np.linalg.inv(_homography(tex.w, tex.h, q))
            h_inv.append(hi.ravel() / hi[2, 2])
            offs.append(jitter[k % 4])
            # Livello di dettaglio agli angoli del quad (gli estremi di una
            # omografia su un quadrilatero convesso stanno ai vertici).
            for c in q:
                den = hi[2, 0] * c[0] + hi[2, 1] * c[1] + hi[2, 2]
                uu = (hi[0, 0] * c[0] + hi[0, 1] * c[1] + hi[0, 2]) / den
                vv = (hi[1, 0] * c[0] + hi[1, 1] * c[1] + hi[1, 2]) / den
                jac = np.array([[hi[0, 0] - hi[2, 0] * uu, hi[0, 1] - hi[2, 1] * uu],
                                [hi[1, 0] - hi[2, 0] * vv, hi[1, 1] - hi[2, 1] * vv]]) / den
                lods.append(math.log2(max(0.5 * float(np.linalg.norm(jac, axis=0).max()), 1e-6)))
        hm = torch.tensor(np.array(h_inv), device=dev, dtype=torch.float32)       # (N, 9)
        jt = torch.tensor(np.array(offs), device=dev, dtype=torch.float32)        # (N, 2)

        def warp(a: int, b: int):
            """Coordinate nel quadro (u, v) dei campioni a..b-1, forma (b-a, H, W)."""
            m = b - a
            p = lambda k: hm[a:b, k].view(m, 1, 1)
            x = xs[None] + jt[a:b, 0].view(m, 1, 1)
            y = ys[None] + jt[a:b, 1].view(m, 1, 1)
            wden = p(6) * x + p(7) * y + p(8)
            wden = torch.where(wden.abs() < 1e-9, torch.full_like(wden, 1e-9), wden)
            return (p(0) * x + p(1) * y + p(2)) / wden, (p(3) * x + p(4) * y + p(5)) / wden, wden, p

        # Jacobiana dell'omografia inversa: impronta del campione (½ px) nel
        # quadro. Varia pochissimo tra gli istanti di esposizione: si calcola
        # una volta sola, sul campione centrale.
        c = n // 2
        u0, v0, w0, p = warp(c, c + 1)
        u0, v0, w0 = u0[0], v0[0], w0[0]
        dux = (p(0)[0] - p(6)[0] * u0) / w0
        duy = (p(1)[0] - p(7)[0] * u0) / w0
        dvx = (p(3)[0] - p(6)[0] * v0) / w0
        dvy = (p(4)[0] - p(7)[0] * v0) / w0
        fp = 0.5 * torch.maximum(dux * dux + dvx * dvx, duy * duy + dvy * dvy).sqrt()
        lod = torch.log2(fp.clamp_min(1e-6))
        del u0, v0, w0, dux, duy, dvx, dvy, fp
        lod_range = (min(lods) - 0.5, max(lods) + 0.5)
        # I campioni si elaborano a gruppi: con 16 istanti di esposizione su
        # un ritaglio Full HD la memoria della GPU servirebbe tutta insieme
        # (~1 GB); a gruppi di ``self.sample_chunk`` ne basta un quarto.
        sum_in = sum_all = cnt = None
        for a in range(0, n, self.sample_chunk):
            b = min(n, a + self.sample_chunk)
            u, v, _, _ = warp(a, b)
            col = tex.sample(u, v, lod[None].expand(b - a, -1, -1), lod_range)       # (m, 3, H, W)
            inside = ((u >= -0.5) & (u <= tex.w - 0.5) & (v >= -0.5) & (v <= tex.h - 0.5)).float()
            del u, v
            s_in = (col * inside[:, None]).sum(dim=0)
            s_all = col.sum(dim=0)
            c_in = inside.sum(dim=0)
            del col, inside
            if sum_in is None:
                sum_in, sum_all, cnt = s_in, s_all, c_in
            else:
                sum_in, sum_all, cnt = sum_in + s_in, sum_all + s_all, cnt + c_in
        mean_in = sum_in / cnt.clamp_min(1e-6)[None]
        return torch.where(cnt[None] > 0, mean_in, sum_all / n)
