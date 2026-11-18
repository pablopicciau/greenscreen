# Green Frame Replacer: analisi tecnica

Questo documento raccoglie le valutazioni fatte prima e durante lo sviluppo:
fattibilità, architettura scelta, alternative considerate (anche quelle
scartate, con il motivo e, dove possibile, una misura), dipendenze e licenze,
rischi e risultati misurati.

La versione 0.2 è una revisione completa della precisione e della qualità
dell'immagine, con il calcolo per-pixel spostato sulla GPU. Le sezioni
marcate **(v0.2)** descrivono cosa è cambiato e perché, con le misure.

---

## 1. Sintesi

* **Il progetto è fattibile** con hardware non recente. Il riconoscimento del
  quadro gira su CPU (OpenCV + NumPy); conversione del colore, chiave e
  compositing girano su **GPU NVIDIA con PyTorch/CUDA** (RTX 2060: ~15
  fotogrammi Full HD al secondo), con ripiego automatico sulla CPU.
* Il pannello **verde** è la scelta giusta: il verde stesso fa da "sensore"
  sia per la geometria (bordi nettissimi) sia per le occlusioni (le mani non
  sono verdi). Con un pannello bianco nessuna delle due cose è affidabile.
* **Tracking by detection**: una superficie verde uniforme non ha punti da
  inseguire, quindi il quad viene ri-misurato in ogni fotogramma dai bordi.
  Gli angoli sono le intersezioni delle **4 rette dei lati**, così un angolo
  coperto da una mano viene comunque ricostruito.
* **(v0.2) Precisione delle linee.** L'errore misurato aveva due cause
  sistematiche, non casuali: la crominanza 4:2:0 ricostruita nella posizione
  sbagliata dalla conversione rapida di FFmpeg (+0.4 px in orizzontale) e la
  soglia del bordo calcolata sui valori gamma invece che in luce lineare
  (bordi "ristretti" di 0.25 px, fino a 1-2 px con il motion blur). Corrette
  entrambe alla radice: errore medio sugli angoli da **0.98 a 0.16 px**.
* **(v0.2) Qualità dell'immagine.** Il compositing è ora in luce lineare, con
  motion blur ricavato dal tempo di esposizione stimato dal video, nitidezza
  uguale a quella della camera, campionamento anti-aliasing e chiave esatta a
  due colori sui bordi delle dita: rispetto al "composito perfetto", nei
  Reel Full HD di prova il PSNR passa da **10-21 a 35 dB** (errore
  d'immagine da 5 a 17 volte più piccolo).
* **(v0.2) Problemi noti risolti** (Reel Full HD da 15 s): verde saturo
  attaccato al pannello (errore massimo da 1383 px a 1.3 px, e la pianta non
  viene più "invasa" dal quadro), oggetto con bordo dritto davanti a un lato
  (da 84 a 1.3 px), lato coperto per 1.5 s mentre il pannello si inclina (da
  200 a 2.7 px), sfocatura di movimento assente, HDR non verificato. Il
  rendering usa ~2.7 GB di RAM (un errore di gestione della memoria della GPU
  arrivava a 12.5 GB).

## 2. Architettura

```
                ┌──────────────────────── ANALISI (una volta per video) ────────────────────────┐
 video.mp4 ──►  │ FFmpeg: piani YUV    ──► GPU: YUV→RGB con  ──► CPU: rilevatore    ──► offline:   │ ──► traccia.json
                │ originali (fps       │   crominanza nella      (modello colore,     coerenza,   │
                │ costante, rotazione, │   posizione giusta      4 rette RANSAC,      Hermite,    │
                │ HDR→SDR libplacebo)  │                         sub-pixel in luce    LOESS,      │
                │                      │                         lineare, giunzione   esposizione │
                │                      │                         a T, lati nascosti)  e nitidezza │
                └───────────────────────────────────────────────────────────────────────────────┘
                ┌──────────────────────── RENDERING (una volta per quadro) ─────────────────────┐
 quadro.jpg ──► │ proporzioni ──► GPU: piramide in luce lineare; per ogni fotogramma, solo nel   │ ──► reel.mp4
 traccia.json ─►│ (fill/fit)      ritaglio attorno al quadro: screen plate, chiave a due colori, │     (H.264/HEVC,
 video.mp4 ───► │                 quadro con motion blur + anti-aliasing, luce, despill, grana,  │      audio copiato)
                │                 composizione in luce lineare → YUV (resto del fotogramma intatto)│
                └───────────────────────────────────────────────────────────────────────────────┘
```

Moduli (`greenframe/`):

| Modulo | Ruolo |
|---|---|
| `color.py` | **(v0.2)** luce lineare (BT.709), YUV↔RGB su GPU, posizione della crominanza (anche dopo la rotazione) |
| `device.py` | **(v0.2)** scelta GPU/CPU |
| `geometry.py` | rette, intersezioni, omografie, rapporto d'aspetto (Zhang-He), ricostruzione di un lato |
| `keyer.py` | verde generico e **(v0.2)** modello di colore del pannello |
| `detector.py` | quad in un fotogramma: maschera, rette, sub-pixel in luce lineare, **(v0.2)** giunzione a T, lati nascosti |
| `tracker.py` | passata 1: predizione, recupero, modello di colore, proporzioni, struttura `Track` (JSON) |
| `smoothing.py` | passata 2: coerenza globale, lati nascosti (interpolazione + **(v0.2)** geometria del rettangolo), LOESS robusto con soglia adattiva |
| `blur_estimation.py` | **(v0.2)** nitidezza della camera e tempo di esposizione dai bordi del pannello |
| `render.py` | **(v0.2)** rendering su GPU (sostituisce `compositor.py`) |
| `image_fit.py` | fill / fit / stretch, caricamento con EXIF |
| `video_io.py` | FFmpeg: piani YUV in lettura/scrittura, metadati, HDR, encoder |
| `pipeline.py` | analisi, rendering (lettura, GPU e codifica in parallelo), anteprima, video di debug |
| `cli.py` | riga di comando |

## 3. Alternative valutate

### 3.1 Riconoscimento del verde

| Opzione | Esito | Motivo |
|---|---|---|
| Soglie HSV (chroma key "da tutorial") | scartata | dipende dalla luminosità; nei pixel misti (bordi) non dà un alpha proporzionale |
| **Differenza di verde `G − max(R,B)`** e purezza `d/G` | **scelta** per il primo aggancio | indipendente dalle ombre; pelle/bianco/grigio hanno valore ≤ 0 |
| **(v0.2) Modello di colore del pannello** (cromaticità `R/G`, `B/G` in luce lineare, distanza di Mahalanobis) | **scelto** durante il tracking | un'ombra moltiplica i tre canali per lo stesso fattore: la cromaticità non cambia. Un altro verde saturo (pianta) cade a ~7 deviazioni standard dal pannello (soglia 4.5) mentre per `G − max(R,B)` era indistinguibile |
| Segmentazione AI (SAM 2) | non necessaria | vedi 3.7 |

Il primo aggancio prova il verde generico a più soglie e, in più,
**raggruppa i verdi presenti** (k-means a 1, 2 e 3 gruppi sulla cromaticità):
ogni gruppo diventa un modello di colore candidato. Vince la componente più
"quadrilatera" (area × IoU⁴ con il quadrilatero approssimante: un pannello
supera 0.85, una pianta o un pannello fuso con una pianta restano sotto 0.7).
Il modello si aggiorna lentamente durante il video (cambi di luce graduali)
e, se il pannello si perde, si riparte dal verde generico. La decisione
"pannello sì/no" è una tabella 3D precalcolata (64³ colori): un accesso per pixel.

Misura (scenario "verde_sfondo", pianta satura che tocca il bordo destro del
pannello): errore medio sugli angoli **da 72 px a 0.16 px** (540×960).

### 3.2 Angoli del quadro e precisione delle linee

| Opzione | Esito | Motivo |
|---|---|---|
| `approxPolyDP` / `minAreaRect` sul contorno | solo come stima iniziale | se una mano copre un angolo l'angolo trovato è sbagliato |
| Rilevatori di angoli (Harris, Shi-Tomasi) | scartata | gli angoli coperti non si vedono; gli angoli delle dita creano falsi positivi |
| **4 rette dei lati (RANSAC a copertura contigua) + intersezioni** | **scelta** | un angolo coperto viene ricostruito dai tratti visibili dei due lati |
| Sub-pixel a metà tra interno ed esterno della "verdezza" `G − max(R,B)` | **sostituita (v0.2)** | il `max` e il taglio dei valori negativi rendono il profilo non lineare: bordo spostato verso l'interno (~0.25 px) in modo diverso per ogni sfondo |
| Sub-pixel sulla luminanza (piena risoluzione) con soglia "lineare" | scartata | esatta senza compressione (0.02 px), ma con la compressione H.264 la luminanza dei bordi a basso contrasto si deforma: ±0.9 px a seconda dello sfondo |
| **(v0.2) Sub-pixel in luce lineare: proiezione del colore sulla direzione "fuori→dentro"** | **scelta** | in luce lineare il colore di un pixel di bordo è la media pesata dei due colori: la proiezione *è* la copertura, e il bordo è dove vale 0.5, per qualunque sfondo e qualunque sfocatura simmetrica (obiettivo, movimento). Bias residuo per lato ≤ 0.06 px |

La rifinitura campiona un profilo ogni ~4 px di lato (prima: 60 profili in
tutto), scarta i profili con lato interno non verde (dita sopra il bordo) o
con più attraversamenti, e misura anche la **larghezza del bordo** (serve
alla stima dell'esposizione, §3.9). La ricerca grossolana (maschera,
componenti, contorno) si fa a metà risoluzione quando il ritaglio supera
600 000 pixel: la precisione la dà comunque il sub-pixel a piena risoluzione.

### 3.3 Colore e crominanza (v0.2)

I telefoni salvano il colore a metà risoluzione (YUV 4:2:0) e ogni campione
di crominanza ha una posizione precisa (`chroma_location`, di solito "left":
allineato alle colonne pari).

| Opzione | Esito | Misura (bordo verde/sfondo, H.264 crf 20) |
|---|---|---|
| FFmpeg `-pix_fmt bgr24` (conversione rapida: un campione di crominanza replicato su 2×2 pixel) | **scartata** | bordo spostato di **+0.38 px** in orizzontale, +0.16 in verticale |
| FFmpeg con `full_chroma_int+accurate_rnd` | scartata | interpola, ma non legge la posizione della crominanza dal file |
| **Piani YUV originali + interpolazione nella posizione giusta (GPU)** | **scelta** | bias ≤ 0.03 px senza compressione |
| + **dettaglio della crominanza dalla luminanza** (regressione locale crominanza/luminanza, il dettaglio di luminanza perso nel 4:2:0 viene aggiunto dove le due sono correlate) | **scelta** | errore medio degli angoli 0.16 invece di 0.22 px; bordi di colore più nitidi per la chiave |

La rotazione dei video verticali degli smartphone (metadato) fa trasporre
a FFmpeg anche i piani di crominanza: la loro posizione cambia ("left" →
"top"/"bottom") e viene ricalcolata (`color.chroma_siting`, testato con
video ruotati).

### 3.4 Tracking temporale

| Opzione | Esito | Motivo |
|---|---|---|
| Optical flow (Lucas-Kanade) / feature (ORB, SIFT, AKAZE) | scartata | sul verde uniforme non ci sono punti da inseguire; deriva accumulata |
| Planar tracker a intensità (ESM, ECC di OpenCV) | scartata | richiede texture sulla superficie |
| CoTracker / TAPIR (point tracking neurale) | scartata | pesante, e CoTracker ha licenza **non commerciale** |
| **Tracking by detection** guidato dalla predizione | **scelta** | nessuna deriva, recupero immediato |

La predizione (velocità costante smorzata) serve solo a ritagliare la zona di
ricerca, assegnare i punti ai lati e fornire le rette dei lati nascosti.
**(v0.2)** La velocità si usa solo se gli ultimi due fotogrammi sono stati
misurati su tutti i lati: un lato ricostruito può avere qualche pixel di
errore e la velocità lo amplificherebbe (effetto a catena osservato nei test).

### 3.5 Lato completamente nascosto

| Opzione | Esito | Misura |
|---|---|---|
| Punti di fuga + rapporto d'aspetto fisico + focale, da sola | scartata | esatta su dati ideali, ma sensibile al rumore: con lati quasi paralleli alcuni fotogrammi sbagliano di oltre 30 px |
| Interpolazione offline lineare | superata | errore massimo 31 px su 25 fotogrammi |
| Ricostruzione dai 3 lati visibili + interpolazione offline di Hermite | superata (v0.2) | 2.4 px su 25 fotogrammi, ma **41 px** su 44 fotogrammi con il pannello che si inclina |
| **(v0.2)** proporzioni prese dall'**ultimo fotogramma misurato su tutti i lati** invece che dal precedente | **scelta** | niente accumulo di errori nelle occlusioni lunghe |
| **(v0.2) Interpolazione di Hermite + geometria del rettangolo in ogni fotogramma, pesate con la loro incertezza** | **scelta** | stessa occlusione di 44 fotogrammi: errore massimo **2.7 px** |

**Occlusioni lunghe (v0.2).** Il benchmark Full HD da 15 s ha mostrato un
caso che i video brevi non coprivano: un oggetto copre il lato inferiore per
1.5 s e intanto il pannello si inclina e torna indietro. L'interpolazione nel
tempo conosce solo il prima e il dopo, e a metà occlusione sbagliava di 41 px.
Ora ogni fotogramma dà anche una stima *geometrica*: un rettangolo fisico di
proporzioni note, visto da una camera di focale nota, è determinato dai tre
lati visibili. Da sola questa stima è fragile (quando i lati sono quasi
paralleli il punto di fuga è mal determinato), ma la sua incertezza si può
calcolare fotogramma per fotogramma, spostando di 0.15 px ogni angolo misurato
e osservando quanto cambia il risultato. Sui fotogrammi puliti l'errore reale
è in mediana 0.6-0.8 volte quello previsto. Prima di usarla la stima viene
verificata sui fotogrammi puliti vicini all'occlusione, dove il lato è anche
misurato: lo scarto sistematico (focale imprecisa, distorsione dell'obiettivo)
viene sottratto e la dispersione si aggiunge all'incertezza. Le due fonti si
combinano con una regressione locale robusta: l'interpolazione pesa molto
vicino ai fotogrammi puliti, la geometria a metà di un'occlusione lunga.
Un fotogramma isolato che indica un lato nascosto diverso dai vicini (il bordo
di un oggetto preso per un lato) viene ricostruito dai fotogrammi adiacenti.
Se la focale non è affidabile e la verifica non è possibile, resta la sola
interpolazione.

**Oggetto con bordo dritto davanti a un lato (v0.2).** Era il limite
dichiarato: il bordo dritto dell'oggetto copre tutto il lato, sembra un lato
*perfettamente visibile* e quindi affidabile, e il controllo di coerenza
incolpava i lati vicini (in 5 casi su 6 errori di 40-60 px). Soluzione
geometrica, il **test della giunzione a T**: il bordo vero del pannello
finisce negli angoli, mentre il bordo di un oggetto davanti *prosegue dritto
oltre gli angoli*, sopra lo sfondo. Per evitare falsi allarmi (un bordo dello
sfondo allineato per caso) il lato viene ricostruito solo se, in più, il
pannello misurato risulta *accorciato* rispetto alla ricostruzione dagli
altri tre lati: un oggetto davanti può solo nascondere, mai allungare.
Il punteggio della giunzione entra anche nel controllo di coerenza globale.
Misura su 6 varianti della scena: errore massimo **da 60.7 a 2.5 px**,
lato giusto individuato in tutti i casi.

### 3.6 Smoothing

| Opzione | Esito | Motivo |
|---|---|---|
| Kalman / One-Euro filter | scartata | filtri in tempo reale: ritardo o tremolio |
| **Regressione polinomiale locale pesata e robusta (LOESS, offline)** | **scelta** | a fase zero, pesata con la confidenza, pesi *bisquare* contro gli outlier |

**(v0.2) Soglia degli outlier adattiva.** Con la precisione attuale delle
misure (~0.15 px) la vecchia soglia (peso nullo oltre 12 px) lasciava passare
errori di misura da 3-6 px, che lo smoothing poi spalmava sui vicini. Una
soglia fissa più stretta, però, scarterebbe anche i fotogrammi di un
movimento brusco, che il polinomio locale non segue perfettamente. La
differenza è che una misura sbagliata è isolata, mentre in un movimento
brusco *diversi* fotogrammi consecutivi hanno residui simili. La soglia è ora
di 3 px, ma diventa almeno 6 volte la mediana dei residui nei ±4 fotogrammi
vicini. Nei video Full HD l'errore massimo scende da 2.9 a 1.8 px nel
movimento veloce e da 3.4 a 1.4 px con l'oggetto dritto davanti a un lato,
senza effetti sugli altri casi.

### 3.7 Occlusioni (mani)

| Opzione | Esito | Motivo |
|---|---|---|
| MediaPipe Hands | scartata | punti chiave della mano, non una maschera al pixel |
| Robust Video Matting | scartata | licenza GPL-3.0, pensato per persone intere |
| SAM 2 | non necessario | maschere binarie: bordi delle dita meno precisi di una chiave sui pixel misti |
| CorridorKey (keyer neurale di Corridor Digital, 2026) | **valutato, non integrato** | ottimo su capelli e motion blur, ma i **modelli hanno licenza CC BY-NC-SA** (uso non commerciale: un Reel per vendere quadri è commerciale), pesante per 6 GB di VRAM, e richiede comunque una chiave grezza in ingresso |
| Chiave morbida con screen plate (spazio gamma) | sostituita (v0.2) | con la sola differenza di verde, nei pixel misti pelle/pannello alpha è sottostimato (la pelle ha G < R): bordi delle dita "mangiati" |
| **(v0.2) Chiave in luce lineare + modello a due colori nella fascia di transizione** | **scelta** | `I = α·P + (1−α)·F` con `P` = screen plate e `F` = colore del primo piano stimato dai pixel vicini: `α = ⟨I−F, P−F⟩/‖P−F‖²` è esatto per qualunque colore delle mani e anche nella sfocatura di movimento. Bordi del pannello **+4.4 dB**, bordi delle mani **+2.8 dB** |

### 3.8 Compositing (v0.2)

* **Luce lineare**: luce, ombre, motion blur e pixel di bordo si mescolano
  come nella fotocamera. **Composizione additiva** `O = I + α·(R − P)`:
  esatta nei pixel misti, nessun alone.
* **Quadro**: piramide (mipmap) in luce lineare, campionamento trilineare con
  livello di dettaglio per pixel (dalla jacobiana dell'omografia) e
  sovracampionamento 2×2: niente moiré né scalettature anche con il quadro
  molto inclinato.
* **Motion blur**: media di più istanti nel tempo di esposizione (fino a 16,
  uno ogni ~0.75 px di movimento); angoli interpolati con un polinomio di
  secondo grado tra il fotogramma precedente e il successivo. La chiave usa
  l'area "spazzata" dal quadro durante l'esposizione.
* **Nitidezza**: sfocatura gaussiana pari a quella misurata sui bordi del
  pannello, meno la parte dovuta alla misura stessa (calibrata: con un
  obiettivo da 0.7 px il composito più fedele si ottiene esattamente a 0.7 px).
* **Luce**: luminanza lineare dello screen plate / riferimento del pannello.
  Con `--lighting 1` (predefinito) il quadro riceve esattamente luce e ombre
  del pannello.
* **Despill** solo sulla parte "primo piano" dei pixel vicini al pannello
  (non tocca il quadro né gli oggetti verdi lontani).
* **(v0.2) Verdi estranei attaccati al quadro** (una pianta, un vestito). Per
  la differenza di verde sembrano pannello: il quadro "sbordava" di 2-3 px
  sulla pianta e il despill la scoloriva per una decina di pixel (fascia
  esterna dei bordi a 12.6 dB). Ora si usa il modello di colore del pannello
  stimato nell'analisi. I pixel verdi con una tinta diversa, dove il quadro
  non arriva (oltre ~2.5 px dal bordo), sono "primo piano". Vicino a loro la
  chiave a due colori stima quanta parte di ogni pixel è pannello e quanta
  pianta. E se il colore del primo piano stimato è un verde estraneo, il
  verde che resta dopo la composizione è della pianta e non viene tolto.
  Fascia esterna dei bordi accanto alla pianta da 12.6 a 20.3 dB. Il resto
  della differenza è la crominanza 4:2:0 del video, che sfuma il confine tra
  i due verdi su 2 px. Nessun effetto dove non ci sono verdi estranei.
* **Grana**: la luminanza residua del pannello (rumore, trama) viene
  riportata sul quadro, che altrimenti apparirebbe "troppo pulito".
* **Uscita in YUV**: i piani del resto del fotogramma restano quelli
  originali (bit per bit): nessuna perdita fuori dal quadro.

### 3.9 Motion blur ed esposizione (v0.2)

Il tempo di esposizione dei telefoni cambia con la luce (1/30 s in casa,
1/1000 s al sole) e va stimato ripresa per ripresa. Il profilo di un bordo è
un gradino convoluto con la sfocatura fissa dell'immagine (gaussiana `σ`) e
con lo spostamento del bordo durante l'esposizione (rettangolo largo
`e·|v_n|`, con `v_n` velocità normale del lato, nota dalla traccia, ed `e`
frazione del tempo tra fotogrammi con otturatore aperto). La larghezza
25-75% di gaussiana ⊗ rettangolo ha forma chiusa; `σ` ed `e` si trovano con
una ricerca su griglia (errore di Huber) su tutti i lati di tutti i
fotogrammi: i bordi fermi fissano `σ`, quelli in movimento fissano `e`.
Stesso principio di *Video shutter angle estimation using optical flow and
linear blur* (arXiv 2303.10247), applicato ai bordi netti del pannello.
Misura: otturatore vero 0.50 (180°), stimato **0.53** nel video con
movimento veloce e **0.48** in un video lento con esposizione simulata da 15
istanti. Nei video lenti del benchmark la stima vale ~0.7: è un limite dei
video di prova, non del metodo. Lì l'esposizione è simulata con soli 3
istanti, e 3 copie del bordo distribuite sull'intervallo allargano il profilo
come un'esposizione continua √2 volte più lunga (0.5 · √2 = 0.71). Una vera
fotocamera espone in modo continuo. Se il pannello resta
quasi fermo l'otturatore non è osservabile (e il motion blur non serve):
si usa 180°. Opzioni `--shutter` e `--lens-blur` per forzare i valori.

### 3.10 Pannello bianco

Non supportato, per scelta: il bianco ha crominanza nulla come pareti,
riflessi, maniche chiare e le zone sovraesposte della pelle. Alternative, in
ordine di affidabilità: usare il verde (consigliato); pannello bianco con
marcatori ArUco/AprilTag su una cornice esterna + segmentazione AI per le
mani; pannello bianco puro con SAM 2 inizializzato con un clic.

### 3.11 Marcatori ArUco / AprilTag (modalità assistita)

OpenCV include già i dizionari ArUco e AprilTag. I marcatori non verdi
*dentro* l'area verde verrebbero letti come "mani": vanno su una cornice
esterna. Ogni marcatore visibile dà da solo 4 punti per l'omografia. Resta
pianificato per scene difficili; con le correzioni della v0.2 i casi che lo
rendevano necessario (verde nello sfondo, bordi dritti) sono gestiti.

### 3.12 Accelerazione e GPU (RTX 2060, 6 GB)

| Opzione | Esito | Motivo |
|---|---|---|
| OpenCV compilato con CUDA | scartata | su Windows richiede di compilare OpenCV; copre solo alcune funzioni |
| CuPy | scartata | ottimo per NumPy su GPU, ma servono kernel scritti a mano per il campionamento |
| **(v0.2) PyTorch + CUDA** | **scelta** | `grid_sample`, convoluzioni, operazioni vettoriali su GPU; ruote ufficiali per Windows; stesso codice su CPU se la GPU manca |
| Triton / `torch.compile` | rimandata | su Windows non è supportato ufficialmente |
| **NVENC** (`h264_nvenc`, `hevc_nvenc`) | **opzione** | codifica sulla GPU; libx264 resta il default per la qualità |
| **libplacebo** (Vulkan) per l'HDR | **scelta** se disponibile | tone mapping BT.2390 sulla GPU, metadati Dolby Vision; ripiego `zscale` |

Cosa gira dove: decodifica e codifica in FFmpeg (processi separati), colore,
chiave e compositing su GPU, rilevamento su CPU (contorni, RANSAC: lavoro
irregolare che la GPU non accelera). Lettura, GPU e codifica procedono in
parallelo. Ottimizzazioni misurate: niente `pin_memory` per fotogramma
(costava 25 ms), campioni del motion blur elaborati a gruppi di 4 con una
sola chiamata per gruppo, lavoro solo sul ritaglio attorno al quadro.

**Memoria (v0.2).** Il benchmark Full HD si era interrotto per memoria
esaurita. Causa: PyTorch tiene da parte la memoria della GPU liberata per
riusarla, ma il ritaglio attorno al quadro cambia dimensione a ogni
fotogramma e i blocchi vecchi spesso non vanno bene per quelli nuovi. La
riserva cresceva fino a riempire i 6 GB della scheda e, su Windows, il driver
la estendeva nella RAM del PC: **12.5 GB** di memoria del processo per un
Reel di 15 s, anche se ne servivano 1.3. Correzioni: i 16 istanti del motion
blur si elaborano a gruppi di 4 (picco da 1.3 a 0.44 GB), e oltre 1.5 GB di
riserva quella inutilizzata viene restituita. Risultato: 2.7 GB in tutto
(contesto CUDA compreso), stessa velocità.

### 3.13 Lettura/scrittura video

| Opzione | Esito | Motivo |
|---|---|---|
| `cv2.VideoWriter` | scartata | niente audio, codec limitati, conversione colore BT.601 |
| **FFmpeg in pipe, piani YUV nativi (v0.2)** | **scelta** | frame rate costante (audio in sincrono), rotazione, HDR→SDR, crominanza nella posizione giusta, BT.709 e `chroma_location` dichiarati, audio AAC copiato senza ricodifica, `+faststart`, tag `hvc1` per HEVC (compatibile Apple/Instagram) |

### 3.14 Interfaccia grafica

PySide6 (LGPL-3.0) resta la scelta proposta per un'app Windows nativa;
alternativa più rapida da sviluppare: Gradio (Apache-2.0). In questa fase si è
data priorità al motore e alla riga di comando.

### 3.15 Estensione generativa (facoltativa)

Per la modalità `fit` le bande sono riempite con una versione sfocata
dell'immagine (o un colore). Un'estensione generativa locale è possibile
(LaMa, Stable Diffusion inpainting) ma è sconsigliata come default: inventa
parti del quadro che l'artista non ha dipinto.

## 4. Dipendenze e licenze

| Componente | Licenza | Note per la distribuzione |
|---|---|---|
| Python | PSF | libera |
| NumPy, SciPy | BSD-3-Clause | libere |
| OpenCV ≥ 4.8 (opencv-contrib-python) | Apache-2.0 | i wheel includono FFmpeg (LGPL-2.1) e, nella versione con GUI, Qt (LGPL-3.0) |
| **PyTorch** | BSD-3-Clause | i wheel CUDA includono le librerie NVIDIA (ridistribuibili secondo la licenza CUDA) |
| FFmpeg | LGPL-2.1+ / **GPL** se compilato con libx264 | le build Windows comuni (gyan.dev) sono GPL: uso personale libero; per *ridistribuire* l'eseguibile servono gli obblighi GPL |
| imageio-ffmpeg | BSD-2-Clause | scarica un binario FFmpeg GPL |
| Pillow (solo test) | MIT-CMU | |
| Da evitare | | CoTracker (CC-BY-NC), **modelli CorridorKey (CC BY-NC-SA)**, Ultralytics YOLO (AGPL-3.0), Robust Video Matting (GPL-3.0) |

Nessun abbonamento, nessuna API a pagamento, nessun servizio cloud: dopo
l'installazione tutto funziona offline.

## 5. Rischi tecnici e mitigazioni

| Rischio | Gravità | Mitigazione | Stato |
|---|---|---|---|
| Superficie verde uniforme senza texture | alta | tracking by detection sui bordi | risolto |
| Angolo coperto da una mano | alta | rette dei lati + intersezione | risolto |
| Precisione delle linee (crominanza 4:2:0, gamma) | alta | crominanza nella posizione giusta, sub-pixel in luce lineare | **risolto (v0.2)**: 0.16 px |
| Lato intero coperto da un oggetto irregolare | media | ricostruzione, coerenza globale, Hermite + **(v0.2)** geometria del rettangolo | **risolto (v0.2)**: 1.5 s di occlusione con il pannello che si inclina, da 41 a 2.7 px |
| Memoria che cresce durante il rendering (GPU con riserva estesa nella RAM, Windows) | alta | campioni a gruppi, riserva limitata | **risolto (v0.2)**: da 12.5 a 2.7 GB |
| Lato intero coperto da un oggetto con **bordo dritto** | media | giunzione a T + vincolo "solo accorciare" + coerenza | **risolto (v0.2)**: ≤ 2.5 px |
| Verde saturo dello sfondo attaccato al pannello | media | modello di colore del pannello, raggruppamento dei verdi; nel rendering chiave a due colori e despill guidati dalla tinta | **risolto (v0.2)** (resta irrisolvibile un verde *identico* al pannello) |
| Sfocatura di movimento | media | motion blur con esposizione stimata | **risolto (v0.2)** |
| Quadro più nitido/pulito della scena | media | nitidezza stimata, grana | **risolto (v0.2)** |
| Bordi delle dita "mangiati" o con alone | media | chiave a due colori in luce lineare, despill | **risolto (v0.2)** |
| Video HDR dell'iPhone | media | libplacebo/zscale | **gestito e testato (v0.2)** su un video HLG generato; da provare su un file reale |
| Video verticali (rotazione nei metadati) | media | rotazione e posizione della crominanza | **testato (v0.2)** |
| Frame rate variabile dei telefoni | media | frame rate costante in lettura | gestito |
| Pannello lucido con riflessi | media | — | da evitare in ripresa (guida) |
| Pannello curvo (cartoncino che si flette) | bassa | la chiave decide i bordi visibili (maschera allargata di 2 px) | da verificare su riprese reali |

## 6. Risultati misurati

Il sistema è misurato su video sintetici con **verità di riferimento**
(`tools/make_synthetic_video.py`), ora con formazione dell'immagine
**fisicamente corretta**: scena, sfocatura dell'obiettivo e di movimento
composte in luce lineare, poi curva gamma BT.709, YUV 4:2:0 con crominanza
"left" e compressione H.264 (crf 20). Il generatore precedente mescolava i
colori sui valori gamma: premiava proprio la soglia sbagliata e nascondeva
metà dell'errore reale.

`tools/benchmark.py` confronta anche il risultato finale con il **composito
perfetto**: lo stesso fotogramma sintetico con il quadro "stampato" sul
pannello (stessa luce, mani, ombre, motion blur e grana; quadro reso con
anti-aliasing 3×3). PSNR in dB, più alto è meglio.

### Reel Full HD verticali da 15 s (1080×1920, 30 fps)

Quattro scene da 450 fotogrammi, **versione originale (v0.1) → versione
attuale**. Qualità del composito su un fotogramma ogni 5 (90 per scena).
Portatile con i7-9750H e RTX 2060 (6 GB), codifica libx264.

| Scena | errore medio angoli | errore massimo | lato coperto (max) | PSNR composito | PSNR bordi | PSNR bordi mani |
|---|---|---|---|---|---|---|
| standard (mani, angolo e lato coperti) | 7,40 → **0,19** px | 200,1 → **2,7** px | 200,1 → **2,7** px | 19,1 → **34,8** dB | 20,3 → **29,2** dB | 18,5 → **26,4** dB |
| oggetto dritto davanti a un lato | 5,62 → **0,17** px | 83,8 → **1,3** px | 83,8 → **1,3** px | 20,2 → **35,2** dB | 20,7 → **29,2** dB | 19,6 → **26,5** dB |
| pianta verde attaccata al pannello | 383,5 → **0,16** px | 1382,7 → **1,3** px | — | 10,2 → **35,0** dB | 13,0 → **27,5** dB | 14,8 → **26,3** dB |
| movimento veloce (2.5×) | 1,61 → **0,26** px | 7,6 → **1,8** px | — | 20,8 → **35,0** dB | 20,4 → **29,1** dB | 21,1 → **26,4** dB |

| Scena | tremolio | aspetto (vero 0,8) | otturatore (vero 0,5) | analisi | rendering + codifica |
|---|---|---|---|---|---|
| standard | 0,187 → **0,082** px | 0,7999 | 0,69 | 77 → **31 s** | 99 → **38 s** |
| oggetto dritto | 0,407 → **0,072** px | 0,8000 | 0,69 | 65 → **27 s** | 96 → **38 s** |
| pianta verde | 1,022 → **0,073** px | 0,8000 | 0,71 | 51 → **27 s** | 88 → **38 s** |
| movimento veloce | 0,300 → **0,110** px | 0,7998 | 0,54 | 62 → **29 s** | 104 → **41 s** |

Note:

* **errore massimo**: l'angolo più sbagliato dell'intero video. **Lato
  coperto**: 1.5 s (44 fotogrammi) con il lato inferiore nascosto mentre il
  pannello si inclina; nella scena "oggetto dritto" il bordo dell'oggetto è
  perfettamente rettilineo (il caso più difficile per il rilevamento).
* **PSNR**: rispetto al composito perfetto. Nella fascia di ±3 px attorno ai
  lati il limite pratico è ~29 dB, dato dalla compressione H.264 e dalla
  crominanza 4:2:0 del video di partenza. Accanto alla pianta verde (27.5 dB)
  la crominanza 4:2:0 sfuma su 2 px anche il confine tra due verdi.
* **Otturatore**: nelle scene lente i video di prova simulano l'esposizione
  con 3 istanti, che allargano il bordo come un'esposizione continua √2 volte
  più lunga (0.5 · √2 = 0.71): la stima ~0.7 è corretta *per quei video*.
  Scena veloce (9 istanti): 0.54. Esposizione continua simulata con 15
  istanti: 0.48 (vedi 3.9).
* **Memoria**: ~2.7 GB di RAM del processo durante il rendering (contesto
  CUDA compreso); prima della correzione descritta in 3.12, 12.5 GB.
* La versione originale lavorava solo sulla CPU; i tempi dipendono anche dal
  carico del PC durante la misura.

### Video brevi 540×960 da 4 s (usati durante lo sviluppo)

Stesse quattro scene, 120 fotogrammi; l'occlusione del lato dura solo 11
fotogrammi (per questo i video brevi non mostravano il problema delle
occlusioni lunghe). Versione originale → versione attuale.

| Scena | errore medio angoli | errore massimo | PSNR composito | PSNR bordi | PSNR bordi mani |
|---|---|---|---|---|---|
| standard | 0,98 → **0,16** px | 2,5 → **0,4** px | 21,6 → **34,1** dB | 21,2 → **29,7** dB | **28,5** dB |
| oggetto dritto | 0,95 → **0,16** px | 2,5 → **0,5** px | 21,5 → **34,1** dB | 21,2 → **29,7** dB | **28,6** dB |
| pianta verde | 72,2 → **0,16** px | 473,3 → **0,6** px | 11,8 → **33,7** dB | 14,6 → **28,0** dB | **28,6** dB |
| movimento veloce | 1,05 → **0,17** px | 2,9 → **1,0** px | 21,2 → **34,5** dB | 21,0 → **29,4** dB | **28,4** dB |

(La versione originale del benchmark non misurava i bordi delle mani.)

## 7. Roadmap

| Fase | Contenuto | Stato |
|---|---|---|
| 1 | rilevamento del rettangolo verde, angoli, omografia | fatto |
| 2 | tracking temporale, prospettiva, recupero, smoothing, lati nascosti | fatto |
| 3 | occlusioni: chiave morbida, despill, luce, composizione additiva | fatto |
| 3c | motion blur sintetico sull'immagine inserita | **fatto (v0.2)**, con esposizione stimata |
| 3d | precisione: crominanza, luce lineare, modello di colore, giunzione a T | **fatto (v0.2)** |
| 3e | rendering su GPU, chiave a due colori, nitidezza, HDR con libplacebo | **fatto (v0.2)** |
| 3b | modalità assistita ArUco/AprilTag | da fare (facoltativa) |
| 4 | segmentazione AI per pannello bianco | da fare (facoltativa) |
| 5 | interfaccia PySide6, correzione manuale degli angoli, coda di esportazione | da fare |

**Prossimo passo consigliato:** provare il programma su una ripresa reale
(con il video di debug) e tarare i parametri sui risultati.
