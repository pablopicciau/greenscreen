# Green Frame Replacer: analisi tecnica

Questo documento raccoglie le valutazioni fatte prima e durante lo sviluppo:
fattibilità, architettura scelta, alternative considerate (anche quelle
scartate, con il motivo e, dove possibile, una misura), dipendenze e licenze,
rischi e risultati misurati.

---

## 1. Sintesi

* **Il progetto è fattibile** con hardware non recente: il cuore del sistema
  (riconoscimento, tracking, compositing) gira **interamente su CPU** con
  OpenCV e NumPy, senza modelli di intelligenza artificiale e senza GPU.
  La RTX 2060 serve solo come acceleratore facoltativo (codifica NVENC) e
  resta disponibile per le fasi future (SAM 2).
* Il pannello **verde** è la scelta giusta: il verde stesso fa da "sensore"
  sia per la geometria (bordi nettissimi) sia per le occlusioni (le mani non
  sono verdi). Con un pannello bianco nessuna delle due cose è affidabile.
* La scelta tecnica più importante è stata **non** usare il tracking classico
  (optical flow, feature): una superficie verde uniforme non ha punti da
  inseguire. Si usa invece *tracking by detection*: il quad viene ri-misurato
  in ogni fotogramma dai bordi, con la posizione prevista solo come guida.
* Gli angoli non vengono mai cercati direttamente: si stimano le **4 rette dei
  lati** dai tratti visibili e gli angoli sono le loro intersezioni. Così un
  angolo completamente coperto da una mano viene comunque ricostruito con
  ~1 px di errore.
* Le mani restano in primo piano grazie a una **chiave cromatica morbida con
  screen plate** (la tecnica dei keyer professionali), che sui pixel misti
  (bordi delle dita, sfocatura di movimento) è più precisa di qualunque
  segmentazione AI e non costa nulla.

## 2. Architettura

```
                ┌──────────────────── ANALISI (una volta per video) ───────────────────┐
 video.mp4 ──►  │ FFmpeg (fps costante,  ──► rilevatore per fotogramma ──► smoothing    │ ──► traccia.json
                │ HDR→SDR, rotazione)        (keyer + 4 rette RANSAC        offline      │
                │                             + sub-pixel + lati nascosti)  (LOESS       │
                │                             guidato dalla predizione      robusto +    │
                │                                                            interpol.)  │
                └───────────────────────────────────────────────────────────────────────┘
                ┌──────────────────── RENDERING (una volta per quadro) ────────────────┐
 quadro.jpg ──► │ proporzioni (fill/fit) ──► piramide ──► per ogni fotogramma, in      │ ──► reel.mp4
 traccia.json ─►│                                         parallelo: omografia, chiave │     (H.264 + audio
 video.mp4 ───► │                                         morbida, luce, despill,      │      originale)
                │                                         composizione additiva        │
                └───────────────────────────────────────────────────────────────────────┘
```

Moduli (`greenframe/`):

| Modulo | Ruolo |
|---|---|
| `geometry.py` | rette, intersezioni, omografie, stima del rapporto d'aspetto (Zhang-He), ricostruzione di un lato |
| `keyer.py` | differenza di verde, maschera di rilevamento, screen plate, alpha morbido, despill |
| `detector.py` | quad in un fotogramma: componente verde, fit dei lati, sub-pixel, lati nascosti |
| `tracker.py` | passata 1: rilevamento guidato dalla predizione, recupero, stima delle proporzioni, struttura `Track` (JSON) |
| `smoothing.py` | passata 2: regressione locale robusta, interpolazione dei lati nascosti e dei buchi |
| `image_fit.py` | fill / fit / stretch, piramide anti-aliasing, caricamento con EXIF |
| `compositor.py` | inserimento dell'immagine e compositing |
| `video_io.py` | lettura/scrittura con FFmpeg, metadati (ffprobe) |
| `pipeline.py` | analisi, rendering (multi-thread), anteprima, video di debug |
| `cli.py` | interfaccia a riga di comando |

## 3. Alternative valutate

### 3.1 Riconoscimento del verde

| Opzione | Esito | Motivo |
|---|---|---|
| Soglie HSV (chroma key "da tutorial") | scartata | dipende dalla luminosità; nei pixel misti (bordi) non dà un alpha proporzionale |
| **Differenza di verde `G − max(R,B)`** e rapporto con `G` | **scelta** | lineare nel mescolamento, indipendente dalle ombre se rapportata allo screen plate; pelle/bianco/grigio hanno valore ≤ 0 |
| Segmentazione AI (SAM 2) | rimandata (fase 4) | non necessaria per un pannello verde; utile per casi speciali (vedi 3.8) |

Il rilevamento iniziale (senza predizione) prova soglie di purezza crescenti e
sceglie la componente più "quadrilatera" (area × riempimento² dell'involucro
convesso), così il pannello non si fonde con verdi spenti dello sfondo.
Durante il tracking la soglia è adattiva (metà della purezza misurata sul
pannello) e la maschera viene ritagliata attorno alla posizione prevista.

### 3.2 Angoli del quadro

| Opzione | Esito | Motivo |
|---|---|---|
| `approxPolyDP` / `minAreaRect` sul contorno | solo come stima iniziale | se una mano copre un angolo l'angolo trovato è sbagliato |
| Rilevatori di angoli (Harris, Shi-Tomasi) | scartata | gli angoli coperti non si vedono; gli angoli delle dita creano falsi positivi |
| **4 rette dei lati (RANSAC) + intersezioni** | **scelta** | un angolo coperto viene ricostruito dai tratti visibili dei due lati: ~1 px di errore misurato |
| Rifinitura **sub-pixel** lungo le normali ai lati | **aggiunta** | soglia locale (metà tra interno ed esterno): funziona anche con sfondi verdastri; errore medio per angolo 0.74 px |

Il RANSAC dei lati è modificato: le ipotesi sono valutate sulla **copertura
contigua** del lato (tratti di almeno il 7.5% della lunghezza) e non sul
numero di punti. Un RANSAC classico si faceva ingannare da punti sparsi ma
allineati per caso (creste di un bordo irregolare, profili di dita):
nei test l'errore sul lato coperto è sceso da 21 a 2 px (mediana).

### 3.3 Tracking temporale

| Opzione | Esito | Motivo |
|---|---|---|
| Optical flow (Lucas-Kanade) / feature (ORB, SIFT, AKAZE) | scartata | sul verde uniforme non ci sono punti da inseguire; i punti sulle mani si muovono rispetto al quadro; deriva accumulata |
| Planar tracker a intensità (ESM, ECC di OpenCV) | scartata | richiede texture sulla superficie |
| CoTracker / TAPIR (point tracking neurale) | scartata | pesante, e CoTracker ha licenza **non commerciale** (CC-BY-NC) |
| SAM 2 video | rimandata | pesante per un dato che i bordi forniscono già con precisione sub-pixel |
| **Tracking by detection** guidato dalla predizione | **scelta** | nessuna deriva, recupero immediato, ~20-30 ms per fotogramma su CPU |

La predizione (velocità costante smorzata) serve solo a: ritagliare la zona
di ricerca, assegnare i punti di contorno ai lati giusti, fornire le rette
dei lati quando due o più sono nascosti, e mantenere l'ordine degli angoli
anche con rotazioni oltre i 45°.

### 3.4 Lato completamente nascosto

| Opzione | Esito | Misura |
|---|---|---|
| Punti di fuga + rapporto d'aspetto fisico + focale (geometria proiettiva esatta) | **scartata** | esatta su dati ideali (errore 1e-10 px), ma **instabile**: 0.05° di errore sulle direzioni dei lati → 16 px di errore mediano; focale sbagliata del 20% → 33 px |
| Retta prevista a velocità costante | scartata come metodo principale | deriva rapidamente se il movimento cambia |
| Riconoscimento del "bordo impostore" (un lato che salta rispetto alla predizione) | **scartata** | funziona all'ingresso dell'oggetto, ma se la stima deriva il bordo vero, quando riappare, viene a sua volta scartato: blocco con errori fino a 100 px |
| Ricostruzione dai 3 lati visibili + lunghezze dal fotogramma precedente (durante il tracking) | **scelta** come stima provvisoria | l'errore resta confinato lungo le rette misurate, ma si accumula nel tempo |
| Interpolazione offline **lineare** tra il fotogramma prima e quello dopo | superata | in un'occlusione di 25 fotogrammi (Full HD) errore massimo 31 px: la prospettiva non varia linearmente |
| Interpolazione lineare + riscalatura per rispettare il rapporto d'aspetto noto | **scartata** | peggiora (23.8 px): l'errore dell'interpolazione è di *forma* (inclinazione del lato), non di scala |
| **Interpolazione offline cubica di Hermite** (usa anche le velocità misurate ai due estremi) | **scelta** | stessa occlusione: errore massimo **2.4 px** |

Durante il tracking, inoltre, il bordo di un oggetto davanti al pannello può
contenere tratti abbastanza dritti da essere scambiati per il lato vero.
A fine analisi si esegue quindi un **controllo di coerenza globale**: il
rapporto d'aspetto implicato da ogni quad (con la focale stimata sul video)
viene confrontato con quello del pannello. Nei test Full HD i fotogrammi
corretti deviano al massimo dello 0.9%, quelli con un lato sbagliato almeno
del 2.3%: soglia 2% (o 3 volte la deviazione mediana, se il video è più
rumoroso). Il lato colpevole viene scelto tra quelli non pienamente visibili
(copertura < 90%) come quello la cui ricostruzione ripristina meglio la
coerenza. Considerare candidati tutti e quattro i lati è stato provato e
scartato: peggiora (da 1.1 a 4.1 px di errore medio sul lato coperto).

Effetto complessivo sul lato inferiore coperto per 1.5 s (Full HD): errore
massimo da **84 px** (solo tracking) a **7.8 px**, medio da 9.0 a **1.2 px**.

### 3.5 Smoothing

| Opzione | Esito | Motivo |
|---|---|---|
| Kalman / One-Euro filter | scartata | sono filtri in tempo reale: introducono ritardo o lasciano tremolio |
| **Regressione polinomiale locale pesata e robusta (LOESS, offline)** | **scelta** | a fase zero (nessun ritardo), pesata con la confidenza, pesi *bisquare* contro gli outlier; tremolio ridotto da 0.23 a 0.08 px |

### 3.6 Occlusioni (mani)

| Opzione | Esito | Motivo |
|---|---|---|
| MediaPipe Hands | scartata | fornisce punti chiave della mano, non una maschera al pixel |
| Robust Video Matting | scartata | licenza GPL-3.0 e pensato per persone intere, non per mani su un pannello |
| SAM 2 | rimandata (fase 4) | maschere binarie, bordi meno precisi di una chiave sui pixel misti; costo GPU |
| **Chiave morbida con screen plate** | **scelta** | alpha continuo e corretto su dita sfocate o mosse; gestisce ombre e gradienti di luce; costo trascurabile |

### 3.7 Compositing

* **Composizione additiva** `out = I + α·(R − S)` invece di `α·R + (1−α)·I`:
  nei pixel misti sostituisce esattamente la quota di verde, niente aloni.
* **Despill** (G limitato a max(R,B)) solo in una fascia sfumata attorno al
  quadro: il riflesso verde sulle mani sparisce, il resto della scena non
  viene toccato.
* **Trasferimento della luce**: la luminosità dello screen plate, normalizzata
  sul valore di riferimento del video, moltiplica l'immagine. Le ombre delle
  mani e il quadro che si gira verso la luce si vedono anche sull'opera.
* **Grana**: il rumore della camera viene riportato sull'immagine inserita
  (che altrimenti apparirebbe "troppo pulita").
* **Maschera geometrica allargata di 3 px**: i bordi visibili li decide la
  chiave, quindi piccoli errori geometrici non lasciano filetti verdi.
* **Piramide (mipmap)**: l'immagine viene campionata dal livello di
  risoluzione appena superiore al quad, niente moiré sui dettagli fini.

### 3.8 Pannello bianco

Non supportato in questa versione, per scelta:
il bianco ha crominanza nulla come pareti, riflessi, maniche chiare e le zone
sovraesposte della pelle. Senza un colore "firma" la chiave non può separare le
mani e il rilevamento dei bordi si confonde con lo sfondo.
Alternative possibili, in ordine di affidabilità:
1. usare il verde (consigliato);
2. pannello bianco con **marcatori ArUco/AprilTag** stampati su una cornice
   esterna (geometria affidabile) + **SAM 2** per le mani;
3. pannello bianco puro con SAM 2 inizializzato con un clic: possibile ma con
   bordi delle dita meno precisi e costo GPU elevato.

### 3.9 Marcatori ArUco / AprilTag (modalità assistita)

OpenCV include già i dizionari ArUco e AprilTag (nessuna dipendenza in più).
Vincolo pratico: i marcatori non verdi **dentro** l'area verde verrebbero letti
come "mani" e resterebbero visibili. Vanno quindi su una **cornice esterna**
(che resterà visibile nel video) oppure dentro l'area verde con una gestione
dedicata (posizioni note → forzati come "pannello" nella chiave).
Ogni marcatore visibile dà da solo 4 punti per l'omografia, quindi il tracking
resta robusto anche con occlusioni estese. È pianificato come modalità
alternativa per scene difficili (vedi roadmap).

### 3.10 Accelerazione e GPU (RTX 2060, 6 GB)

| Opzione | Esito | Motivo |
|---|---|---|
| OpenCV compilato con CUDA | scartata | su Windows richiede di compilare OpenCV da sorgente; il guadagno sarebbe limitato perché il collo di bottiglia non è il calcolo |
| **CPU multi-thread** per il rendering | **scelta** | i fotogrammi sono indipendenti: rendering 4× più veloce su 4 core, output identico bit per bit |
| **NVENC** (`--codec h264_nvenc`) | **opzione** | codifica sulla GPU, utile per video lunghi; libx264 resta il default per la qualità |
| SAM 2.1 (tiny/small) su GPU | fase 4 | entra nei 6 GB in fp16; da usare solo per inizializzazione/recupero, non per ogni fotogramma |

### 3.11 Lettura/scrittura video

| Opzione | Esito | Motivo |
|---|---|---|
| `cv2.VideoWriter` | scartata | niente audio, codec limitati, conversione colore BT.601 |
| **FFmpeg in pipe** | **scelta** | frame rate costante (i telefoni registrano a frame rate variabile: senza conversione l'audio va fuori sincrono), HDR→SDR, rotazione, BT.709 esplicito, audio originale, `+faststart` |

### 3.12 Interfaccia grafica (fase 5)

PySide6 (LGPL-3.0) resta la scelta proposta per un'app Windows nativa:
anteprima con slider dei fotogrammi, trascinamento manuale degli angoli per
correggere i fotogrammi difficili, coda di più quadri, esportazione.
Alternativa più rapida da sviluppare: Gradio (Apache-2.0) come interfaccia web
locale. In questa fase si è data priorità al motore e alla riga di comando.

### 3.13 Estensione generativa (facoltativa)

Per la modalità `fit` oggi le bande sono riempite con una versione sfocata
dell'immagine (o un colore). Un'estensione "generativa" locale è possibile con
LaMa (Apache-2.0, leggera) o Stable Diffusion inpainting (licenze OpenRAIL,
~4-5 GB di VRAM in fp16). Per opere d'arte è sconsigliata come default:
inventa parti del quadro che l'artista non ha dipinto.

## 4. Dipendenze e licenze

| Componente | Licenza | Note per la distribuzione |
|---|---|---|
| Python | PSF | libera |
| NumPy | BSD-3-Clause | libera |
| OpenCV ≥ 4.5 (opencv-contrib-python) | Apache-2.0 | i wheel includono FFmpeg (LGPL-2.1) e, nella versione con GUI, Qt (LGPL-3.0) |
| FFmpeg | LGPL-2.1+ / **GPL** se compilato con libx264 | le build Windows comuni (gyan.dev) sono GPL: uso personale libero; per *ridistribuire* l'eseguibile servono gli obblighi GPL (o una build LGPL + NVENC) |
| imageio-ffmpeg | BSD-2-Clause | scarica un binario FFmpeg GPL (stesse note) |
| PySide6 (fase 5) | LGPL-3.0 | ok con collegamento dinamico |
| SAM 2 (fase 4) | Apache-2.0 | codice e checkpoint |
| PyTorch (fase 4) | BSD-3-Clause | |
| LaMa (facoltativo) | Apache-2.0 | |
| Da evitare | | CoTracker (CC-BY-NC, non commerciale), Ultralytics YOLO (AGPL-3.0), Robust Video Matting (GPL-3.0) |
| DaVinci Resolve | proprietario | solo come riferimento o strumento esterno, nessun codice estratto |

Nessun abbonamento, nessuna API a pagamento, nessun servizio cloud: dopo
l'installazione tutto funziona offline.

## 5. Rischi tecnici e mitigazioni

| Rischio | Gravità | Mitigazione implementata | Stato |
|---|---|---|---|
| Superficie verde uniforme senza texture | alta | tracking by detection sui bordi | risolto |
| Angolo coperto da una mano | alta | rette dei lati + intersezione | risolto (~1 px) |
| Mani sopra i bordi | alta | RANSAC a copertura contigua, sub-pixel con rifiuto dei profili anomali | risolto |
| Lato intero coperto da un oggetto irregolare | media | ricostruzione, controllo di coerenza globale, interpolazione di Hermite | risolto: 1.2 px medio, 7.8 px massimo (Full HD) |
| Lato intero coperto da un oggetto con **bordo dritto** | media | controllo di coerenza (riconosce i fotogrammi ma non sempre il lato colpevole) | **limite dichiarato**: fino a ~33 px (test 540p) |
| Verde saturo dello sfondo attaccato al pannello | media | soglia adattiva, ritaglio sulla predizione, punteggio di forma | parziale; consigliati riprese pulite o marcatori |
| Quadro parzialmente fuori inquadratura | media | punti sul bordo del fotogramma esclusi, lato ricostruito | risolto |
| Tremolio dell'immagine inserita | media | LOESS robusto offline | risolto (0.08 px) |
| Aloni verdi sui bordi | media | composizione additiva + despill + maschera allargata | risolto |
| Video HDR dell'iPhone | media | rilevamento con ffprobe, tone mapping con zscale | gestito (non testato su HDR reale) |
| Frame rate variabile dei telefoni | media | conversione a frame rate costante in lettura | gestito |
| Foto del quadro ruotata (EXIF) | bassa | caricamento che applica l'EXIF (testato) | risolto |
| Sfocatura di movimento | bassa | la chiave segue il bordo mosso; l'immagine inserita resta nitida | da migliorare (motion blur sintetico) |
| Pannello lucido con riflessi | media | — | da evitare in ripresa (guida) |

## 6. Risultati misurati

Non avendo ancora un video reale, il sistema è stato misurato su video
sintetici con **verità di riferimento** (`tools/make_synthetic_video.py`):
pannello 40×50 cm in movimento 3D (rotazioni fino a 32°, avvicinamenti,
un movimento brusco), mani che lo reggono con ombra, un braccio che copre
completamente un angolo, un oggetto irregolare che nasconde il lato
inferiore, sfocatura di movimento, rumore del sensore, compressione H.264.

**Reel Full HD verticale (1080×1920, 15 s, 450 fotogrammi)**:

| Scenario | Fotogrammi | Errore medio per angolo | Peggior angolo: mediana / p95 / massimo |
|---|---|---|---|
| mani ai lati, movimento libero (anche fuori inquadratura) | 308 | **0.58 px** | 1.2 / 1.8 / 2.1 px |
| un angolo completamente coperto da un braccio | 98 | **0.78 px** | 1.4 / 2.4 / 2.7 px |
| lato inferiore coperto da un oggetto irregolare | 44 | **1.24 px** | 1.7 / 7.7 / 7.8 px |
| tutti | 450 | **0.69 px** | 1.3 / 2.3 / 7.8 px |

* Rapporto d'aspetto stimato dal video: **0.8000** (vero 0.800).
* Focale stimata: 1315 px (vera 1322 px, −0.5%).
* Tremolio (variazione seconda dell'errore): **0.07 px**.
* Nessun fotogramma perso.
* Tempi (4 core CPU in cloud, senza GPU): analisi **30 s**, rendering con
  codifica x264 "slow" **47 s**. File finale H.264 BT.709 + audio AAC, 18 MB.

Video 540×960 (180 fotogrammi), stessi scenari: errore medio 0.79 px;
lato coperto 1.1 px medio (4.1 px massimo). Con un oggetto a **bordo dritto**
davanti al lato inferiore: 8.3 px medio, ~33 px massimo (limite dichiarato).

## 7. Roadmap

| Fase | Contenuto | Stato |
|---|---|---|
| 1 | rilevamento del rettangolo verde, angoli, omografia, warpPerspective | **fatto** |
| 2 | tracking temporale, prospettiva, recupero, smoothing, quad fuori inquadratura, lati nascosti | **fatto** |
| 3 | occlusioni: chiave morbida, despill, luce, composizione additiva | **fatto** |
| 3b | modalità assistita ArUco/AprilTag | da fare |
| 3c | motion blur sintetico sull'immagine inserita | da fare |
| 4 | SAM 2 per pannello bianco, inizializzazione con clic e recupero | da fare (facoltativa) |
| 5 | interfaccia PySide6, correzione manuale degli angoli, coda di esportazione | da fare |

**Prossimo passo consigliato:** provare il programma su un video reale e
tarare i parametri sui risultati, prima di aggiungere funzionalità.
