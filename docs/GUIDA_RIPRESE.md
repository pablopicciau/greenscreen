# Guida alle riprese

Il software funziona tanto meglio quanto più il video è "pulito". Queste
indicazioni nascono dalle prove fatte durante lo sviluppo: rispettarle vale più
di qualsiasi parametro.

## Il pannello

| Consiglio | Perché |
|---|---|
| **Verde saturo e opaco** (cartoncino verde chroma, tessuto chroma, vernice verde opaca) | La chiave usa la "differenza di verde" `G − max(R,B)` e il colore esatto del pannello: più il verde è puro, più netta è la separazione da pelle e sfondo |
| Superficie **liscia e rigida**, senza pieghe | Pieghe e riflessi diventano ombre sul quadro inserito; un cartoncino che si flette ha bordi curvi, mentre il quadro inserito è piano |
| **Bordi dritti e netti** | I quattro lati sono la base del tracking (precisione misurata: 0.16 px) |
| Se il pannello è una **tela con spessore**, colora di verde **solo il fronte**: lati bianchi o di un colore neutro | Quando giri il quadro i lati si vedono: se sono verdi si fondono con il fronte (bordo falsato fino a 58 px nei test) e restano come strisce verdi durante la rotazione, che nessun programma sa con cosa sostituire |
| Evita superfici lucide | I riflessi della luce sono bianchi e vengono letti come "mano" davanti al quadro |
| Proporzioni del pannello uguali a quelle dei tuoi quadri (es. 40×50) | Il quadro inserito non viene mai deformato: se le proporzioni sono diverse verrà ritagliato (`fill`) o avrà bande (`fit`) |

Il pannello **bianco** non è supportato: il bianco si confonde con riflessi,
pareti, maniche chiare e non permette di separare le mani in modo affidabile.
Vedi `ANALISI_TECNICA.md` per le alternative.

## La scena

* **Altri verdi vicino al pannello** (piante, magliette, pareti verdine): dalla
  versione 0.2 il programma impara il colore esatto del tuo pannello e li
  distingue anche quando lo toccano. Resta da evitare solo un verde
  *identico* al pannello (per esempio un secondo telo chroma alle spalle).
* Sfondo con un buon contrasto rispetto al verde (pareti neutre, legno, ecc.).
* Mani nude o maniche non verdi.
* Coprire angoli e parti dei lati con le dita va benissimo. Anche un oggetto
  davanti a un intero lato (tavolo, mobile, braccio con manica tesa, anche
  con un **bordo dritto**) viene riconosciuto e il lato ricostruito dagli
  altri tre, pure se resta davanti per qualche secondo mentre inclini il
  pannello (nei test: 1.5 s, errore massimo ~2-3 px in Full HD). Evita solo
  di coprire **due lati** insieme: con due lati nascosti la posizione del
  quadro si può soltanto stimare dal movimento.

## Il reveal: di spalle, ti giri e mostri il quadro

È il Reel classico degli artisti ed è stato provato apposta, con una scena
sintetica che lo riproduce (persona che ruota, tela 100×120, 120×150 e
150×100 cm con spessore, retro con il telaio, mani sui lati): con le
correzioni il quadro viene agganciato fin da quando compare di taglio, senza
"lampi verdi", e seguito durante tutta la rotazione (errore medio 2-4 px in
rotazione, sotto 0.4 px quando è frontale, nella scena 100×120).

Consigli specifici:

* **Lati della tela non verdi** (vedi sopra): è la cosa più importante per
  questo tipo di video.
* **Camera fissa** (cavalletto o appoggio): è il caso misurato. Un movimento
  di camera a mano va bene, ma la camera ferma rende tutto più stabile.
* **Blocca esposizione e bilanciamento del bianco**: quando un grande
  pannello verde entra in scena, l'automatismo del telefono tende a cambiare
  luminosità e colori di tutta l'immagine proprio durante la rivelazione.
* **Gira a velocità naturale**: un mezzo giro in 1-3 secondi va bene (provato
  anche in ~1 secondo); la sfocatura di movimento viene misurata e riprodotta
  sul quadro.
* **Quadri grandi e inquadratura**: se alla fine il quadro esce
  dall'inquadratura su **entrambi i lati** (destra e sinistra, o alto e
  basso), la sua posizione in quella direzione non si può misurare: le
  proporzioni restano esatte (il quadro non viene deformato) ma la porzione
  visibile può spostarsi del 2-3% della larghezza. Se puoi, tieni almeno un
  bordo verticale del quadro dentro l'inquadratura; in ogni caso indica le
  misure del quadro con `--aspect`, ad esempio `--aspect 150x100`.
* **Indica sempre le misure del quadro** (`--aspect LARGHEZZAxALTEZZA`): nel
  reveal il pannello passa molto tempo di taglio o fuori campo, e conoscere
  le proporzioni esatte aiuta a ricostruire ciò che non si vede.
* Piante o oggetti verdi nella stanza: mentre sei di spalle il programma non
  li scambia più per il pannello (verificato), ma è comunque meglio tenerli
  fuori dall'inquadratura.

## Luce

* Luce **morbida e uniforme** sul pannello (finestra con tenda, luce diffusa).
* Le ombre delle mani sul pannello vengono trasferite sul quadro inserito
  (è voluto: aumenta il realismo). Ombre molto scure però riducono la qualità
  della chiave. Con `--lighting 0.6` le ombre sul quadro si attenuano.

## Impostazioni della fotocamera

* **HDR video**: viene convertito automaticamente in SDR (sulla GPU, con
  libplacebo), ma i colori sono più fedeli se si parte da SDR (iPhone:
  Impostazioni → Fotocamera → Registra video → HDR video: off).
* **1080p a 30 fps** è il formato ideale per i Reel (4K funziona ma è più lento
  e non serve: Instagram ricomprime a 1080).
* Blocca esposizione e messa a fuoco (tieni premuto sullo schermo sul pannello):
  evita "pompaggi" di luminosità durante il movimento.
* **Movimenti**: puoi muoverti in modo naturale. La sfocatura di movimento del
  tuo video viene misurata dai bordi del pannello (tempo di esposizione) e
  riprodotta sul quadro inserito. Movimenti molto bruschi restano comunque i
  più difficili da tracciare.

## Verifica prima di pubblicare

1. Lancia l'analisi con il video di debug:
   `python -m greenframe analyze --video ripresa.mp4 --debug-video controllo.mp4`
2. Guarda `controllo.mp4`: il contorno colorato deve seguire il pannello.
   Verde = misurato, arancio = un lato ricostruito (coperto), rosso = perso.
3. Se qualcosa non va, una riga di testo in alto indica il motivo
   (es. "nessuna zona verde adatta").
4. Il riepilogo dell'analisi riporta anche l'esposizione stimata (gradi di
   otturatore) e la nitidezza: se ti sembrano sbagliati puoi forzarli con
   `--shutter 180` e `--lens-blur 0.7`.
