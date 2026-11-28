# Green Frame Replacer

Sostituisce automaticamente un **pannello verde tenuto in mano** in un video con
l'immagine di un tuo quadro: l'immagine segue la prospettiva del pannello
mentre lo giri e lo inclini, le **mani restano in primo piano**, la luce, le
ombre, la nitidezza e la **sfocatura di movimento** della scena vengono
trasferite sul quadro. Pensato per Reel Instagram da 15 secondi.

* Funziona **in locale e offline**, senza abbonamenti né API a pagamento.
* Usa la **GPU NVIDIA** (PyTorch + CUDA) per colore, chiave e compositing;
  senza GPU lo stesso codice gira sulla CPU, più lentamente.
* Si analizza il video **una volta** e poi si generano tanti Reel quanti sono
  i quadri da promuovere.

Novità della versione 0.2 (dettagli e misure in `docs/ANALISI_TECNICA.md`):

* **precisione delle linee 6 volte migliore** (errore medio sugli angoli da
  ~1 px a 0.16 px): colore letto nel formato nativo del video, con la
  crominanza nella posizione giusta, e bordi misurati in luce lineare;
* **motion blur** sul quadro, con il tempo di esposizione stimato dal video;
* nitidezza e grana uguali a quelle della camera, bordi delle dita esatti
  (modello a due colori), compositing in luce lineare sulla GPU;
* risolti i limiti noti: **verde saturo vicino al pannello** e, nelle scene
  di prova, **oggetto con bordo dritto** davanti a un lato (vedi però i limiti
  noti); HDR convertito sulla GPU e testato;
* **marcatori di tracking** (crocette), macchie e piccoli riflessi sul pannello
  vengono sostituiti insieme al pannello (provato su video reali);
* un **lato coperto per secondi** mentre inclini il pannello viene
  ricostruito dalla geometria degli altri tre lati (errore da 41 a ~2-3 px);
* memoria sotto controllo: ~3 GB di RAM per il rendering di un Reel Full HD.

Documentazione:

* [`docs/GUIDA_RIPRESE.md`](docs/GUIDA_RIPRESE.md): come preparare pannello,
  luce e fotocamera (leggila prima di girare!);
* [`docs/ANALISI_TECNICA.md`](docs/ANALISI_TECNICA.md): architettura,
  alternative valutate, licenze, rischi, risultati misurati.

## Installazione (Windows)

1. **Python 3.11-3.13** da <https://www.python.org/downloads/>
   (durante l'installazione spunta *Add python.exe to PATH*).
2. **FFmpeg**: apri PowerShell e scrivi

   ```bash
   winget install Gyan.FFmpeg
   ```

   poi chiudi e riapri PowerShell.
3. Nella cartella del progetto crea un ambiente virtuale e attivalo:

   ```bash
   py -3.13 -m venv .venv
   ```

   ```bash
   .venv\Scripts\activate
   ```

4. **PyTorch con CUDA** (per la GPU NVIDIA; ~3 GB). Con driver NVIDIA recenti:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu132
   ```

   Con driver più vecchi usa `cu130` o `cu128` al posto di `cu132`. Senza GPU
   NVIDIA basta `pip install torch`.
5. Le altre dipendenze:

   ```bash
   pip install -r requirements.txt
   ```

6. Verifica che tutto funzioni (crea video sintetici e controlla la precisione):

   ```bash
   python -m pytest
   ```

## Uso

Tutto in un comando (analisi + video finale):

```bash
python -m greenframe run --video ripresa.mp4 --image quadro.jpg --out reel.mp4
```

Flusso consigliato quando hai più quadri da promuovere con la stessa ripresa:

```bash
python -m greenframe analyze --video ripresa.mp4 --debug-video controllo.mp4
```

```bash
python -m greenframe render --video ripresa.mp4 --image quadro1.jpg quadro2.jpg quadro3.jpg --out reel/
```

Anteprima veloce di un fotogramma (PNG con originale e risultato affiancati):

```bash
python -m greenframe preview --video ripresa.mp4 --image quadro.jpg --frame 120 --out anteprima.png
```

Opzioni principali:

| Opzione | Significato |
|---|---|
| `--aspect 40x50` | proporzioni reali del pannello (consigliato se le conosci; altrimenti vengono stimate dal video) |
| `--mode fill` / `fit` / `stretch` | riempi (ritaglia l'immagine), adatta (bande ai lati), deforma (sconsigliato) |
| `--fit-bg blur` | sfondo delle bande in modalità `fit`: `blur`, `white`, `black`, `#rrggbb` |
| `--lighting 1` | quanto luce e ombre del pannello si trasferiscono sul quadro (0 = per niente, 1 = come nella realtà) |
| `--grain 1` | quanta grana della camera viene riportata sul quadro |
| `--shutter auto` | otturatore in gradi per il motion blur (`auto` = stimato dal video, es. `180`) |
| `--no-motion-blur` | niente sfocatura di movimento sul quadro |
| `--lens-blur auto` | sfocatura (px) del quadro per eguagliare la nitidezza della camera (`auto` = stimata, `0` = nessuna) |
| `--device auto` | `cuda` (GPU NVIDIA), `cpu` o `auto` |
| `--codec libx264` | `libx264` (qualità massima), `h264_nvenc` / `hevc_nvenc` (GPU NVIDIA, più veloce), `libx265` |
| `--crf 17` | qualità di codifica (più basso = migliore e più pesante) |
| `--no-audio` | non copiare l'audio originale |
| `--keep-markers` | non sostituire le piccole zone non verdi interne al pannello (marcatori di tracking, macchie, riflessi) |
| `--debug-video file.mp4` | video con il contorno del tracking: verde = misurato, arancio = lato ricostruito, rosso = perso |

L'analisi salva la traccia in `ripresa.track.json` accanto al video: i comandi
`render` e `preview` la riusano automaticamente (una traccia creata da una
versione precedente viene rifatta).

## Come funziona (in breve)

1. **Colore**: FFmpeg decodifica il video senza convertirlo; la GPU ricostruisce
   il colore rispettando la posizione dei campioni di crominanza 4:2:0.
2. **Rilevamento**: il pannello viene trovato per colore (il programma impara
   la tinta esatta del tuo pannello, così distingue altri verdi); i 4 lati
   sono stimati con un RANSAC robusto sui tratti visibili e rifiniti al
   sub-pixel **in luce lineare**; gli angoli sono le intersezioni dei lati,
   quindi un angolo coperto da una mano viene comunque ricostruito. Un lato
   coperto da un oggetto dritto viene riconosciuto (il suo bordo prosegue
   oltre gli angoli) e ricostruito dagli altri tre.
3. **Smoothing offline**: controllo di coerenza delle proporzioni, lati
   nascosti ricostruiti combinando l'interpolazione nel tempo con la
   geometria del rettangolo vista in ogni fotogramma, regressione locale
   robusta delle traiettorie (niente tremolio, niente ritardo; le misure
   sbagliate isolate vengono scartate). Dai bordi si stimano anche la
   nitidezza della camera e il tempo di esposizione.
4. **Rendering (GPU)**: il quadro (alle proporzioni del pannello, mai
   deformato) viene proiettato con anti-aliasing e motion blur; la chiave a
   due colori mantiene le mani davanti con bordi esatti; luce, ombre, grana e
   nitidezza della scena vengono trasferite sul quadro; tutto in luce lineare.
   Fuori dal quadro i pixel restano quelli originali.

## Prestazioni

Misurate su Reel di prova Full HD verticali da 15 s (1080×1920, 30 fps),
generati con la posizione vera del pannello, su un portatile con i7-9750H e
RTX 2060 (6 GB). Quattro scene: mani e lati coperti, oggetto dritto davanti a
un lato, pianta verde attaccata al pannello, movimento veloce. Dettagli in
[`docs/ANALISI_TECNICA.md`](docs/ANALISI_TECNICA.md) (sezione 6).

| | Versione originale | Versione attuale |
|---|---|---|
| Errore medio sugli angoli | 1,6-7,4 px (383 px con la pianta verde) | **0,16-0,26 px** |
| Errore massimo nell'intero video | 7,6-200 px (1383 px con la pianta) | **1,3-2,7 px** |
| Fedeltà al "composito perfetto" (PSNR) | 10-21 dB | **35 dB** |
| Analisi di un Reel da 15 s | 51-77 s | **27-31 s** |
| Rendering di un quadro, con codifica | 88-104 s | **38-41 s** |

In pratica: l'analisi richiede circa mezzo minuto e ogni versione con un
quadro diverso circa 40 secondi. Durante il rendering il programma usa circa
3 GB di RAM.

### Reveal: di spalle, ti giri e mostri il quadro

Lo scenario tipico dei Reel degli artisti ha una scena di prova dedicata
(`tools/make_reveal_video.py`: persona che ruota, tela con spessore, retro con
telaio, quadri grandi anche fuori inquadratura) e preset nel benchmark:

```bash
python tools/benchmark.py --scenarios rivelazioni --render
```

Risultati sulla scena di prova (tela 100×120 cm, Full HD, `--aspect 100x120`):

* il quadro viene agganciato **dal primo fotogramma in cui compare**, anche se
  prima, mentre sei di spalle, nella stanza c'è qualcosa di verdastro;
* **nessun lampo verde** e nessuna modifica mentre si vede il retro della tela;
* errore medio sugli angoli **0.36 px durante la rotazione** e **0.17 px**
  quando il quadro è frontale (7 px nei 2-3 fotogrammi quasi di profilo,
  dove il quadro è una striscia sottile);
* il **motion blur** del giro è stimato correttamente anche con una tela
  spessa: il lato della tela visto di sbieco viene riconosciuto ed escluso
  dalla misura (prima l'esposizione risultava più corta del 30%).

Prima di queste correzioni il quadro veniva agganciato a una macchia
verdastra dello sfondo (errore oltre 400 px). Il benchmark copre anche la
rotazione nell'altro verso, il giro veloce (~1 s) e i quadri grandi 120×150 e
150×100 cm che escono dall'inquadratura; dettagli e limiti in
[`docs/ANALISI_TECNICA.md`](docs/ANALISI_TECNICA.md) (sezione 3.17).

## Struttura del progetto

```
greenframe/           pacchetto Python
  color.py            luce lineare, YUV <-> RGB, posizione della crominanza
  device.py           scelta GPU/CPU
  geometry.py         rette, omografie, proporzioni
  keyer.py            verde generico e modello di colore del pannello
  detector.py         quad in un singolo fotogramma
  tracker.py          tracking temporale e traccia JSON
  smoothing.py        coerenza, lati nascosti, smoothing robusto offline
  blur_estimation.py  nitidezza e tempo di esposizione dai bordi
  render.py           rendering su GPU (chiave, quadro, motion blur, compositing)
  image_fit.py        fill/fit, caricamento immagini
  video_io.py         FFmpeg (piani YUV, HDR, encoder), metadati
  pipeline.py         analisi, rendering, anteprima, debug
  cli.py              riga di comando
tools/
  make_synthetic_video.py   video di prova con verità di riferimento
  make_reveal_video.py      scena "reveal": di spalle, ti giri e mostri un quadro grande
  make_test_painting.py     immagini di prova
  benchmark.py              precisione e qualità rispetto al composito perfetto
tests/                test automatici (pytest)
docs/                 analisi tecnica e guida alle riprese
```

## Limiti noti

* Pannello **bianco** non supportato (serve il verde: vedi l'analisi tecnica).
* Un verde **identico** a quello del pannello attaccato al pannello stesso non
  può essere separato per colore.
* Il quadro inserito è piano: un pannello che si flette (cartoncino sottile)
  avrà bordi leggermente curvi che il quadro non segue.
* Un lato coperto viene ricostruito dagli altri tre; con **due lati coperti
  insieme** la posizione si può solo prevedere dal movimento (per pochi
  fotogrammi va bene, a lungo no).
* Un oggetto con un **bordo dritto** che copre un intero lato è riconosciuto
  nelle scene del benchmark, ma non in tutti i casi: con un oggetto
  orizzontale davanti al lato inferiore di un pannello inclinato l'errore
  arriva a ~45 px (vedi l'analisi tecnica, sezione 3.16).
* Una crocetta (marcatore) **toccata da un dito** si unisce al dito e resta
  visibile.
* Reveal con un quadro che alla fine esce dall'inquadratura da **entrambi i
  lati**: la sua posizione orizzontale non si può misurare (proporzioni
  esatte, ma la parte visibile del dipinto può spostarsi del 2-3% della
  larghezza). Tieni almeno un bordo verticale nell'inquadratura.
* **Lati della tela verdi**: restano strisce verdi mentre il quadro è di
  spalle o di taglio. Colora di verde solo il fronte.
* Provato su video sintetici (anche il reveal) e su tre clip reali (telefoni
  con schermo verde tenuti in mano): manca ancora la prova su una tua
  ripresa con il pannello.
