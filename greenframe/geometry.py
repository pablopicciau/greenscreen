"""Funzioni geometriche di base: quadrilateri, rette, omografie.

Convenzioni usate in tutto il progetto
--------------------------------------
* Un quadrilatero ("quad") è un array numpy float64 di forma (4, 2) con gli
  angoli in ordine **TL, TR, BR, BL** (alto-sinistra, alto-destra,
  basso-destra, basso-sinistra) in coordinate immagine (x verso destra,
  y verso il basso). L'ordine è quindi orario sullo schermo.
* Il lato ``i`` va dall'angolo ``i`` all'angolo ``i+1``:
  lato 0 = alto, lato 1 = destra, lato 2 = basso, lato 3 = sinistra.
* Una retta è rappresentata in forma omogenea ``(a, b, c)`` con
  ``a*x + b*y + c = 0`` e normale ``(a, b)`` di norma unitaria, così
  ``a*x + b*y + c`` è direttamente la distanza con segno dal punto alla retta.
* Coordinate continue: il centro del pixel (i, j) ha coordinate (i, j).
"""

from __future__ import annotations

import numpy as np

# Nomi leggibili degli angoli, usati nell'overlay di debug.
CORNER_NAMES = ("TL", "TR", "BR", "BL")


# ---------------------------------------------------------------------------
# Rette
# ---------------------------------------------------------------------------

def line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Retta normalizzata passante per due punti."""
    # Il prodotto vettoriale di due punti omogenei dà la retta che li unisce.
    line = np.cross(np.array([p[0], p[1], 1.0]), np.array([q[0], q[1], 1.0]))
    return normalize_line(line)


def normalize_line(line: np.ndarray) -> np.ndarray:
    """Scala la retta in modo che la normale (a, b) abbia norma 1."""
    n = np.hypot(line[0], line[1])
    if n < 1e-12:
        raise ValueError("Retta degenere (normale nulla)")
    return np.asarray(line, dtype=np.float64) / n


def intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    """Punto di intersezione tra due rette, ``None`` se (quasi) parallele."""
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def point_line_distance(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distanza con segno di N punti (N, 2) da una retta normalizzata."""
    return points[:, 0] * line[0] + points[:, 1] * line[1] + line[2]


def fit_line_tls(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Retta ai minimi quadrati totali (ortogonali), opzionalmente pesata.

    È la stima ottima quando il rumore è isotropo, come per i punti di bordo.
    """
    if weights is None:
        weights = np.ones(len(points))
    w = weights / weights.sum()
    centroid = (points * w[:, None]).sum(axis=0)
    d = points - centroid
    # Matrice di covarianza pesata 2x2: l'autovettore con l'autovalore minore
    # è la normale alla retta.
    cov = (d * w[:, None]).T @ d
    _, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    c = -normal @ centroid
    return normalize_line(np.array([normal[0], normal[1], c]))


# ---------------------------------------------------------------------------
# Quadrilateri
# ---------------------------------------------------------------------------

def quad_lines(quad: np.ndarray) -> list[np.ndarray]:
    """Le 4 rette dei lati (alto, destra, basso, sinistra)."""
    return [line_through(quad[i], quad[(i + 1) % 4]) for i in range(4)]


def quad_from_lines(lines: list[np.ndarray]) -> np.ndarray | None:
    """Ricostruisce gli angoli intersecando lati adiacenti.

    L'angolo ``i`` sta tra il lato ``i-1`` e il lato ``i`` (es. TL = sinistra ∩ alto).
    """
    corners = []
    for i in range(4):
        p = intersect(lines[(i - 1) % 4], lines[i])
        if p is None:
            return None
        corners.append(p)
    return np.array(corners, dtype=np.float64)


def order_quad(points: np.ndarray) -> np.ndarray:
    """Ordina 4 punti qualsiasi come TL, TR, BR, BL.

    Ordina per angolo attorno al baricentro (senso orario sullo schermo) e
    fa partire la sequenza dal punto con x+y minima (il più in alto a sinistra).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # con y verso il basso questo è orario a schermo
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def align_to_reference(quad: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotazione ciclica degli angoli che minimizza la distanza da ``ref``.

    Garantisce che "TL" resti lo stesso angolo fisico tra un fotogramma e il
    successivo anche quando il quadro ruota oltre i 45°.
    """
    best, best_cost = quad, np.inf
    for s in range(4):
        cand = np.roll(quad, -s, axis=0)
        cost = np.linalg.norm(cand - ref, axis=1).sum()
        if cost < best_cost:
            best, best_cost = cand, cost
    return best


def quad_area(quad: np.ndarray) -> float:
    """Area con formula di Gauss (positiva per ordine orario a schermo)."""
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex(quad: np.ndarray) -> bool:
    """True se il quadrilatero è convesso e non intrecciato."""
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)
    signs = np.array(signs)
    return bool(np.all(signs > 0) or np.all(signs < 0))


def interior_angles(quad: np.ndarray) -> np.ndarray:
    """Angoli interni in gradi."""
    out = []
    for i in range(4):
        v1 = quad[(i - 1) % 4] - quad[i]
        v2 = quad[(i + 1) % 4] - quad[i]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        out.append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    return np.array(out)


def side_lengths(quad: np.ndarray) -> np.ndarray:
    """Lunghezze dei 4 lati (alto, destra, basso, sinistra)."""
    return np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)


def offset_quad(quad: np.ndarray, distance: float) -> np.ndarray:
    """Allarga (distance > 0) o restringe il quad spostando ogni lato in parallelo."""
    lines = quad_lines(quad)
    centroid = quad.mean(axis=0)
    moved = []
    for ln in lines:
        # Orienta la normale verso l'esterno (il baricentro deve avere distanza negativa).
        if point_line_distance(centroid[None], ln)[0] > 0:
            ln = -ln
        moved.append(np.array([ln[0], ln[1], ln[2] - distance]))
    out = quad_from_lines(moved)
    return quad.copy() if out is None else out


def homography_rect_to_quad(width: int, height: int, quad: np.ndarray) -> np.ndarray:
    """Omografia che porta un'immagine width x height sul quadrilatero.

    I bordi dell'immagine sorgente sono a -0.5 e width-0.5 (bordo esterno dei
    pixel), coerentemente con la convenzione "centro pixel = coordinata intera".
    """
    import cv2  # import locale: il modulo resta usabile anche senza OpenCV

    src = np.array(
        [[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, quad.astype(np.float32))


# ---------------------------------------------------------------------------
# Stima del rapporto d'aspetto reale del pannello
# ---------------------------------------------------------------------------

def estimate_aspect_ratio(
    quad: np.ndarray,
    image_size: tuple[int, int],
    default_focal: float | None = None,
    fixed_focal: float | None = None,
) -> tuple[float, float | None]:
    """Stima larghezza/altezza **fisiche** di un rettangolo visto in prospettiva.

    Implementa il metodo di Zhang & He ("Whiteboard scanning and image
    enhancement", 2007): dai 4 angoli proiettati si ricava la focale della
    camera e quindi il vero rapporto dei lati, indipendentemente
    dall'inclinazione del quadro. Se la configurazione è quasi frontale la
    focale non è osservabile: in quel caso si usa ``default_focal``.

    Se ``fixed_focal`` è indicata, la focale non viene stimata ma imposta
    (serve a verificare la coerenza di un quad con un modello già noto).

    Ritorna ``(rapporto_w_su_h, focale_stimata_o_None)``.
    """
    w_img, h_img = image_size
    u0, v0 = (w_img - 1) / 2.0, (h_img - 1) / 2.0
    if default_focal is None:
        # Camera principale di uno smartphone ≈ 26 mm equivalenti → f ≈ 0.6 × diagonale.
        default_focal = 0.6 * float(np.hypot(w_img, h_img))

    # Notazione del paper: m1=TL, m2=TR, m3=BL, m4=BR (coordinate omogenee).
    tl, tr, br, bl = quad
    m1, m2, m3, m4 = (np.array([p[0], p[1], 1.0]) for p in (tl, tr, bl, br))
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    focal = None
    if fixed_focal is None and abs(n2[2] * n3[2]) > 1e-12:
        f2 = -(
            (n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0)
            + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)
        ) / (n2[2] * n3[2])
        diag = np.hypot(w_img, h_img)
        # Accetta solo focali plausibili (tra 0.25 e 4 volte la diagonale).
        if f2 > 0 and 0.25 * diag < np.sqrt(f2) < 4.0 * diag:
            focal = float(np.sqrt(f2))

    if fixed_focal is not None:
        f = fixed_focal
    else:
        f = focal if focal is not None else default_focal
    a_inv = np.linalg.inv(np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]]))
    # Rapporto tra le lunghezze 3D dei due lati, a meno della scala.
    a2 = a_inv @ n2
    a3 = a_inv @ n3
    ratio = float(np.sqrt(np.dot(a2, a2) / np.dot(a3, a3)))
    return ratio, focal


def intrinsics(focal: float, image_size: tuple[int, int]) -> np.ndarray:
    """Matrice K della camera con punto principale al centro dell'immagine."""
    w_img, h_img = image_size
    return np.array([[focal, 0, (w_img - 1) / 2.0], [0, focal, (h_img - 1) / 2.0], [0, 0, 1.0]])


def complete_by_ratio(
    lines: list[np.ndarray],
    missing: int,
    reference: np.ndarray,
) -> np.ndarray | None:
    """Ricostruisce il lato ``missing`` dagli altri tre e da un quad di riferimento.

    È il metodo usato in pratica. I tre lati visibili determinano due angoli
    (le intersezioni tra loro) e le rette su cui stanno gli altri due; resta da
    stabilire *dove* si trovano lungo quelle rette. Le due lunghezze mancanti si
    ricavano dal quad di riferimento (fotogramma precedente), scalate come il
    lato opposto visibile. L'errore resta confinato lungo le rette misurate e
    non viene amplificato (al contrario di :func:`complete_rectangle`).
    """
    opp = (missing + 2) % 4
    # Angoli noti: estremi del lato opposto (corner opp e opp+1).
    p = intersect(lines[(opp - 1) % 4], lines[opp])
    q = intersect(lines[opp], lines[(opp + 1) % 4])
    if p is None or q is None:
        return None
    ref_len = side_lengths(reference)
    scale = np.linalg.norm(q - p) / max(ref_len[opp], 1e-9)

    quad = np.zeros((4, 2))
    quad[opp], quad[(opp + 1) % 4] = p, q
    # L'angolo opp+2 sta sul lato opp+1 (che parte da q); l'angolo opp+3 sta
    # sul lato opp+3 (che arriva in p). Si percorrono le rette misurate nel
    # verso indicato dal riferimento.
    for corner, start, side in (((opp + 2) % 4, q, (opp + 1) % 4), ((opp + 3) % 4, p, (opp + 3) % 4)):
        ln = lines[side]
        direction = np.array([-ln[1], ln[0]])               # direzione della retta
        ref_vec = reference[corner] - reference[(opp + 1) % 4 if corner == (opp + 2) % 4 else opp]
        if np.dot(direction, ref_vec) < 0:
            direction = -direction
        quad[corner] = start + direction * ref_len[side] * scale
    return quad if is_convex(quad) else None


def complete_rectangle(
    lines: list[np.ndarray],
    missing: int,
    focal: float,
    aspect: float,
    image_size: tuple[int, int],
    interior_point: np.ndarray,
) -> np.ndarray | None:
    """Ricostruisce il lato ``missing`` (non visibile) dagli altri tre.

    ATTENZIONE: metodo valutato e **scartato** per l'uso nella pipeline.
    È esatto su dati ideali, ma numericamente instabile: un errore di soli
    0.05° nella direzione dei lati misurati sposta il punto di fuga e produce
    errori di ~15-35 px. Resta qui come riferimento (e nei test) perché
    documenta perché si usa :func:`complete_by_ratio`.

    Un rettangolo fisico visto in prospettiva è vincolato: i punti di fuga
    delle due direzioni dei lati corrispondono a direzioni 3D ortogonali, e il
    rapporto tra i lati è noto. Con 3 lati misurati, la focale e il rapporto
    d'aspetto, il quarto lato è determinato in modo univoco:

    1. punto di fuga dei due lati adiacenti (intersezione delle loro rette);
    2. direzione 3D del lato opposto: ortogonale alla precedente e contenuta
       nella retta del lato opposto (vincolo dell'immagine della conica assoluta);
    3. normale del piano del quadro e posizione 3D dei due angoli noti;
    4. i due angoli mancanti stanno a distanza (lato noto / aspetto) lungo la
       direzione dei lati adiacenti; si riproiettano nell'immagine.

    ``interior_point`` (es. il baricentro previsto) serve solo a scegliere il
    verso giusto. Ritorna il quad completo o ``None`` se la geometria è degenere.
    """
    K = intrinsics(focal, image_size)
    k_inv = np.linalg.inv(K)
    omega = k_inv.T @ k_inv
    opp = (missing + 2) % 4
    adj1, adj2 = (missing - 1) % 4, (missing + 1) % 4

    # Angoli noti: estremi del lato opposto a quello mancante.
    p_img = intersect(lines[(opp - 1) % 4], lines[opp])
    q_img = intersect(lines[opp], lines[(opp + 1) % 4])
    if p_img is None or q_img is None:
        return None

    v_adj = np.cross(lines[adj1], lines[adj2])          # punto di fuga (anche all'infinito)
    v_opp = np.cross(lines[opp], omega @ v_adj)         # direzione ortogonale sul lato opposto
    d_adj = k_inv @ v_adj
    d_opp = k_inv @ v_opp
    if np.linalg.norm(d_adj) < 1e-12 or np.linalg.norm(d_opp) < 1e-12:
        return None
    d_adj /= np.linalg.norm(d_adj)
    d_opp /= np.linalg.norm(d_opp)
    normal = np.cross(d_opp, d_adj)
    if np.linalg.norm(normal) < 1e-9:
        return None
    normal /= np.linalg.norm(normal)

    # Raggi di vista dei due angoli noti, intersecati con il piano n·X = 1.
    r_p = k_inv @ np.array([p_img[0], p_img[1], 1.0])
    r_q = k_inv @ np.array([q_img[0], q_img[1], 1.0])
    if normal @ r_p < 0:
        normal = -normal
    if normal @ r_p <= 1e-9 or normal @ r_q <= 1e-9:
        return None
    p3 = r_p / (normal @ r_p)
    q3 = r_q / (normal @ r_q)
    known_len = np.linalg.norm(q3 - p3)
    # Il lato opposto è orizzontale (alto/basso) -> è la larghezza.
    adj_len = known_len / aspect if opp in (0, 2) else known_len * aspect

    def project(x3: np.ndarray) -> np.ndarray | None:
        h = K @ x3
        return None if h[2] <= 1e-9 else h[:2] / h[2]

    for sign in (1.0, -1.0):
        p2 = project(p3 + sign * adj_len * d_adj)
        q2 = project(q3 + sign * adj_len * d_adj)
        if p2 is None or q2 is None:
            continue
        # Il verso corretto porta verso l'interno del quadro.
        if np.dot(p2 - p_img, interior_point - p_img) <= 0:
            continue
        quad = np.zeros((4, 2))
        quad[opp] = p_img
        quad[(opp + 1) % 4] = q_img
        quad[(opp + 2) % 4] = q2
        quad[(opp + 3) % 4] = p2
        return quad if is_convex(quad) else None
    return None


def parse_aspect(text: str | None) -> float | None:
    """Converte '4:5', '40x50', '0.8' in un rapporto larghezza/altezza."""
    if text is None or str(text).strip() == "":
        return None
    t = str(text).lower().replace(",", ".").strip()
    for sep in (":", "x", "/"):
        if sep in t:
            a, b = t.split(sep, 1)
            return float(a) / float(b)
    return float(t)
