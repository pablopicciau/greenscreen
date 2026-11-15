# Guida alle riprese

Il software funziona tanto meglio quanto più il video è "pulito". Queste
indicazioni nascono dalle prove fatte durante lo sviluppo: rispettarle vale più
di qualsiasi parametro.

## Il pannello

| Consiglio | Perché |
|---|---|
| **Verde saturo e opaco** (cartoncino verde chroma, tessuto chroma, vernice verde opaca) | La chiave usa la "differenza di verde" `G − max(R,B)`: più il verde è puro, più netta è la separazione da pelle e sfondo |
| Superficie **liscia**, senza pieghe | Pieghe e riflessi diventano ombre sul quadro inserito |
| **Bordi dritti e netti**; se il pannello ha uno spessore, colora di verde anche i lati oppure tienilo frontale | I quattro lati sono la base del tracking |
| Evita superfici lucide | I riflessi della luce sono bianchi e vengono letti come "mano" davanti al quadro |
| Proporzioni del pannello uguali a quelle dei tuoi quadri (es. 40×50) | Il quadro inserito non viene mai deformato: se le proporzioni sono diverse verrà ritagliato (`fill`) o avrà bande (`fit`) |

Il pannello **bianco** non è supportato in questa versione: il bianco si
confonde con riflessi, pareti, maniche chiare e non permette di separare le
mani in modo affidabile. Vedi `ANALISI_TECNICA.md` per le alternative.

## La scena

* **Niente verde vicino al pannello**: piante, magliette verdi, pareti verdine.
  Un verde saturo che tocca il pannello può fondersi con esso.
* Sfondo con un buon contrasto rispetto al verde (pareti neutre, legno, ecc.).
* Mani nude o maniche non verdi.
* Un oggetto con un **bordo dritto** (tavolo, mobile, braccio con manica tesa)
  che copre **un intero lato** del pannello è il caso più difficile: può essere
  scambiato per il bordo vero e il quadro risulterà un po' deformato finché
  l'oggetto è davanti. Coprire angoli e parti di lati con le dita invece va bene.

## Luce

* Luce **morbida e uniforme** sul pannello (finestra con tenda, luce diffusa).
* Le ombre delle mani sul pannello vengono trasferite sul quadro inserito
  (è voluto: aumenta il realismo). Ombre molto scure però riducono la qualità
  della chiave.

## Impostazioni della fotocamera

* **Disattiva l'HDR video** (iPhone: Impostazioni → Fotocamera → Registra video
  → HDR video: off). I video HDR vengono convertiti, ma i colori sono più
  fedeli se si parte da SDR.
* **1080p a 30 fps** è il formato ideale per i Reel (4K funziona ma è più lento
  e non serve: Instagram ricomprime a 1080).
* Blocca esposizione e messa a fuoco (tieni premuto sullo schermo sul pannello):
  evita "pompaggi" di luminosità durante il movimento.
* Movimenti fluidi: il programma gestisce rotazioni e inclinazioni, ma un
  movimento molto rapido produce sfocatura di movimento sul pannello e il
  quadro inserito resterà più nitido del resto della scena.

## Verifica prima di pubblicare

1. Lancia l'analisi con il video di debug:
   `python -m greenframe analyze --video ripresa.mp4 --debug-video controllo.mp4`
2. Guarda `controllo.mp4`: il contorno colorato deve seguire il pannello.
   Verde = misurato, arancio = un lato ricostruito, rosso = perso.
3. Se qualcosa non va, una riga di testo in alto indica il motivo
   (es. "nessuna zona verde adatta").
