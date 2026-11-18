# Guida alle riprese

Il software funziona tanto meglio quanto più il video è "pulito". Queste
indicazioni nascono dalle prove fatte durante lo sviluppo: rispettarle vale più
di qualsiasi parametro.

## Il pannello

| Consiglio | Perché |
|---|---|
| **Verde saturo e opaco** (cartoncino verde chroma, tessuto chroma, vernice verde opaca) | La chiave usa la "differenza di verde" `G − max(R,B)` e il colore esatto del pannello: più il verde è puro, più netta è la separazione da pelle e sfondo |
| Superficie **liscia e rigida**, senza pieghe | Pieghe e riflessi diventano ombre sul quadro inserito; un cartoncino che si flette ha bordi curvi, mentre il quadro inserito è piano |
| **Bordi dritti e netti**; se il pannello ha uno spessore, colora di verde anche i lati oppure tienilo frontale | I quattro lati sono la base del tracking (precisione misurata: 0.16 px) |
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
