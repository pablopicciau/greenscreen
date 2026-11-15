# Green Frame Replacer

Sostituisce automaticamente un **pannello verde tenuto in mano** in un video con
l'immagine di un tuo quadro: l'immagine segue la prospettiva del pannello
mentre lo giri e lo inclini, le **mani restano in primo piano**, la luce e le
ombre della scena vengono trasferite sul quadro. Pensato per Reel Instagram
da 15 secondi.

* Funziona **in locale e offline**, senza abbonamenti né API a pagamento.
* Gira su **CPU** (OpenCV + NumPy); la GPU NVIDIA è usata solo, se vuoi, per
  la codifica video (NVENC).
* Si analizza il video **una volta** e poi si generano in pochi secondi tanti
  Reel quanti sono i quadri da promuovere.

Documentazione:

* [`docs/GUIDA_RIPRESE.md`](docs/GUIDA_RIPRESE.md): come preparare pannello,
  luce e fotocamera (leggila prima di girare!);
* [`docs/ANALISI_TECNICA.md`](docs/ANALISI_TECNICA.md): architettura,
  alternative valutate, licenze, rischi, risultati misurati.

## Installazione (Windows)

1. **Python 3.10 o superiore** da <https://www.python.org/downloads/>
   (durante l'installazione spunta *Add python.exe to PATH*).
2. **FFmpeg**: apri PowerShell e scrivi

   ```bash
   winget install Gyan.FFmpeg
   ```

   poi chiudi e riapri PowerShell (in alternativa basta il pacchetto
   `imageio-ffmpeg` installato al punto 3).
3. Nella cartella del progetto crea un ambiente virtuale e installa le dipendenze:

   ```bash
   python -m venv .venv
   ```

   ```bash
   .venv\Scripts\activate
   ```

   ```bash
   pip install -r requirements.txt
   ```

4. Verifica che tutto funzioni (crea video sintetici e controlla la precisione):

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
| `--lighting 0.7` | quanto la luce del pannello si trasferisce sul quadro (0 = per niente) |
| `--grain 1` | quanta grana della camera viene riportata sul quadro |
| `--codec h264_nvenc` | codifica sulla GPU NVIDIA (più veloce); default `libx264` (qualità massima) |
| `--crf 17` | qualità di codifica (più basso = migliore e più pesante) |
| `--no-audio` | non copiare l'audio originale |
| `--debug-video file.mp4` | video con il contorno del tracking: verde = misurato, arancio = lato ricostruito, rosso = perso |

L'analisi salva la traccia in `ripresa.track.json` accanto al video: i comandi
`render` e `preview` la riusano automaticamente.

## Come funziona (in breve)

1. **Rilevamento**: la "differenza di verde" `G − max(R,B)` individua il
   pannello; i suoi 4 lati vengono stimati con un RANSAC robusto sui tratti
   visibili e rifiniti al sub-pixel; gli angoli sono le intersezioni dei lati,
   quindi un angolo coperto da una mano viene comunque ricostruito.
2. **Tracking**: il quad viene ri-misurato in ogni fotogramma, guidato dalla
   posizione prevista dal movimento; i lati completamente nascosti vengono
   ricostruiti dagli altri tre.
3. **Smoothing offline**: regressione locale robusta sulle traiettorie degli
   angoli (niente tremolio, niente ritardo).
4. **Compositing**: l'immagine (alle proporzioni del pannello, mai deformata)
   viene proiettata con un'omografia; una chiave morbida mantiene le mani
   davanti; luce, ombre e grana della camera vengono trasferite sull'immagine.

## Prestazioni

Misurate su un Reel sintetico Full HD verticale (1080×1920, 15 s, 450
fotogrammi) con verità di riferimento, su una macchina cloud con 4 core CPU e
**senza GPU** (su un PC con più core i tempi scendono):

| Misura | Risultato |
|---|---|
| Analisi (tracking + smoothing) | 30 s |
| Rendering + codifica x264 "slow" | 47 s (con `--codec h264_nvenc` la codifica passa alla GPU) |
| Errore medio sugli angoli, mani ai lati | 0.58 px (massimo 2.1 px) |
| Errore medio, un angolo coperto da un braccio | 0.78 px (massimo 2.7 px) |
| Errore medio, lato coperto da un oggetto | 1.2 px (massimo 7.8 px) |
| Tremolio dell'immagine inserita | 0.07 px |
| Rapporto d'aspetto stimato dal video | 0.8000 (vero 0.800) |

Dettagli e confronto delle alternative in `docs/ANALISI_TECNICA.md`.

## Struttura del progetto

```
greenframe/           pacchetto Python
  geometry.py         rette, omografie, proporzioni
  keyer.py            chiave cromatica, screen plate, despill
  detector.py         quad in un singolo fotogramma
  tracker.py          tracking temporale e traccia JSON
  smoothing.py        smoothing robusto offline
  image_fit.py        fill/fit, piramide, caricamento immagini
  compositor.py       inserimento e compositing
  video_io.py         FFmpeg (lettura/scrittura), metadati
  pipeline.py         analisi, rendering, anteprima, debug
  cli.py              riga di comando
tools/
  make_synthetic_video.py   video di prova con verità di riferimento
  make_test_painting.py     immagini di prova
tests/                test automatici (pytest)
docs/                 analisi tecnica e guida alle riprese
```

## Limiti noti

* Pannello **bianco** non supportato (serve il verde: vedi l'analisi tecnica).
* Un oggetto con un **bordo dritto** che copre un **intero lato** del pannello
  può essere scambiato per il bordo vero: il quadro risulta un po' deformato
  finché l'oggetto è davanti (fino a ~30 px nei test). Dita, mani e oggetti
  irregolari sono invece gestiti.
* Verdi saturi nello sfondo **a contatto** con il pannello possono disturbare
  il rilevamento.
* Con movimenti molto rapidi l'immagine inserita resta più nitida della scena
  (manca ancora il motion blur sintetico).
* Finora testato su video sintetici: il prossimo passo è la prova su riprese
  reali.
