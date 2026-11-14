# Llista de mesures per maximitzar la conversió (TEAM 32)

Fonts: les vostres regles (R1–R8), `calypso_4_1/MEDIDAS_CONVERSION.md` (M1–M8, D1–D11), `calypso_4_1/README_4_1.md`, i mesures addicionals meves (X1–X12).

> **Què és dada i què no.** Les xifres citades de `MEDIDAS_CONVERSION.md` surten d'un log de 25.000 peticions i 386 compres. Aquest log no té catàleg, carretó ni historial d'usuari, així que **categoria, marca, preu i carretó no estan mesurats**. Les mesures X1–X12 són propostes meves, no resultats. No he revisat el codi de `recommender.py` ni `business_rules.py` per veure què fa avui cada regla; ho indico a la columna "Comprovar".

---

## 1. Les vostres regles

| # | Regla | Encaix amb el document 4_1 | Com implementar-la | Comprovar |
|---|---|---|---|---|
| R1 | **Producte ja vist: recomanar-lo si és pertinent** | Coherent. "Vist" multiplica la conversió per ~10 (+2,30 al logit). Però un vist de cua (puesto >300) gairebé no compra: "vist" equival a dividir el puesto de vendes entre ~28. | Bonus fix de +2,30 (M3) sumat a la taxa base. No fixar-lo mai per força: ha de competir. Treure el ×0,65 de l'últim vist a la pàgina de producte (D6). | Que no hi hagi `self_cvr × 5 × 1,85` |
| R2 | **Prioritzar molt la mateixa categoria; el top 1 ha de ser de la categoria que s'explora** | No mesurat al log (la consulta S6 ho mesura però no està executada). Risc: el document diu que el puesto de vendes domina; una regla dura al top 1 pot desplaçar un superventes d'una altra categoria. | "Categoria explorada" = categoria de l'últim `recent_views` (només hi ha dades en `product`/`category`, no en `home`). **Regla dura només al slot 1** (filtre sobre el pool de candidats), i bonus fort a la resta. Decidir si "categoria" és `category` (10 valors) o `subcategory` (90): jo faria dura la categoria i bonus la subcategoria. | Executar S6 abans de fixar el pes del bonus |
| R3 | **Treure tot producte ja comprat** | Coherent amb les bases ("un comprador que ja ha comprat tendeix a no voler-lo"). **Conflicte** amb 4.1: `CALYPSO_V4_RECURRENT=1` garanteix els recurrents (comprats ≥2 cops). | `CALYPSO_V4_RECURRENT=0`. Mantenir `CALYPSO_V4_LIVE_EVENT_PURCHASES=1` (compres orgàniques de `shop.events` i del feedback també compten). | Doble comptatge events/feedback (usar `max`, no suma) |
| R4 | **Sense estoc → no pot aparèixer** | Coherent. Un sol producte esgotat invalida tota la llista i va al fallback (0 punts). | Filtre d'estoc just abans de l'enviament, no només al principi del cicle. Refresc d'estoc cada 1,5 s ja existeix; tractar també `back_in_stock`. | Latència entre `out_of_stock` i el filtre |
| R5 | **Si es treu un producte, no afegir-ne un d'aleatori: pujar els següents per prioritat** | Coherent amb el document: cal ordenar per puntuació i tallar els 10 *després* de filtrar. | Filtrar **abans** de fer el tall top-10, sobre un pool de 120–150 candidats ja puntuats (D3, M3.4). El farciment ha de ser el següent per puntuació, mai popularitat ni aleatori. | Que el pool no es quedi curt amb 2 vistes o més (D3) |
| R6 | **Producte al carretó (o acabat d'afegir): no a les 3 primeres posicions** | **Conflicte** amb 4.1 (M1 la posa als primers slots) i amb M6 del document (bonus +2,3 a +3). El document tampoc ho ha mesurat, i l'ablació sintètica ja dona negatiu per al carretó oblidat. | `CALYPSO_V4_CART_PRIORITY=0` i `CALYPSO_V4_FORGOTTEN_CART=0`. Afegir una regla de postprocés: si un candidat és al carretó, es col·loca com a mínim al slot 4. | Executar S7 per saber si els del carretó convertien al log; si no, es pot excloure del tot |
| R7 | **Dos productes molt semblants i prioritat equivalent → primer el més barat** | **Conflicte parcial** amb 4.1: `CALYPSO_V4_PRICE_ASC=1` ordena tot per preu i costa ~25 % de DCG a l'ablació sintètica. La vostra regla és només un desempat, i això és compatible. | `CALYPSO_V4_PRICE_ASC=0`. Desempat: només si la diferència de puntuació és < ε (p. ex. 1–2 % de la puntuació) i els productes són de la mateixa subcategoria. | Triar ε amb dades; començar amb un valor petit |
| R8 | **Nou client → popularitat de la regió** | Compatible amb la taxa base del document. **Conflicte** amb D7: ara `cold_gate = 0` i exploració UCB des del slot 1, i els slots 1–3 concentren el 63 % de les compres. | Rànquing per `(regió)` amb finestra recent (60 min, M4) i encongit cap al rànquing global quan la regió té poques dades (Bayes jeràrquic). Exploració només als slots 8–10. | Que hi hagi prou volum per regió |

---

## 2. Interruptors per alinear Calypso 4.1 amb les vostres regles

| Variable | Valor | Per què |
|---|---|---|
| `CALYPSO_V4_PRICE_ASC` | `0` | R7 és desempat, no ordre global |
| `CALYPSO_V4_CART_PRIORITY` | `0` | R6 |
| `CALYPSO_V4_FORGOTTEN_CART` | `0` | R6, sense evidència |
| `CALYPSO_V4_RECURRENT` | `0` | R3 |
| `CALYPSO_V4_LIVE_EVENT_PURCHASES` | `1` | R3 |
| `CALYPSO_V3_BG_PENALTY` | `0`, i treure també el camí `bg_score *= 0.25` al codi | D4 |

Les regles R2, R5, R6 (mínim slot 4) i R7 (desempat) **no són interruptors**: cal escriure-les com a postprocés.

---

## 3. Mesures de `MEDIDAS_CONVERSION.md`

| Mesura | Què ataca | Esforç | Evidència |
|---|---|---|---|
| **M1** Treure l'ordre per preu | Ordre de la llista | 1 min | Log real + ablació pròpia |
| **M2** Comprovar que supereu el baseline (A/B 50 % per hash de `request_id`) | Diagnòstic | 30 min | Estimació del model 1.5 |
| **M3** Puntuar com a taxa base × relació (`−0,69·ln(puesto+1) + 2,30·vist + 0,61·relació`) | Quins productes surten | 2–3 h | Log real (1.2, 1.4) |
| **M4** Popularitat en viu per a la final (finestra de 60 min, productes nous amb prior dels veïns) | Final amb tendències noves | 1–2 h | Bases + pes de la taxa base |
| **M5** Ordre estricte; exploració només als slots 8–10; corba d'atenció mesurada | Slots 1–5 | 30 min | Log real (1.3) |
| **M6** Carretó i recurrents com a senyals, no posicions fixes | Slots 1–3 | 30 min | Falta d'evidència (vegeu R3/R6 per a la vostra versió) |
| **M7** `home`: només superventes en viu, sense MMR | 4 % de les compres | 15 min | Log real (1.1) |
| **M8** Re-ranker entrenat de debò (250.000 sessions, sense retallar als priors) | Afinar pesos | 2–3 h | — |

**Fets del log que justifiquen l'ordre:**
- El 96 % de les compres són a `product` i `category`; `home` n'aporta el 4 %.
- La conversió cau aproximadament com `puesto^-0,87`.
- El slot 1 concentra el 34 % de les compres i els slots 1–3 el 63 %.
- Una llista dominada per co-visites queda per sota del baseline (0,20× amb sessió).
- El document diu: *si només hi ha temps per a tres, M1, M3 i M4.*

**Problemes detectats (D1–D11) que cal vigilar:** ordre per preu (D1), popularitat massa comprimida (D2), pool de candidats que exclou els superventes (D3), penalització de fons (D4), falta de taxa base al lift (D5), l'últim vist penalitzat (D6), `cold_gate = 0` (D7), tendències en viu sense finestra temporal (D8), re-ranker entrenat amb només ~46 compres (D9), carretó fixat sense mesurar (D10), MMR a `home` (D11).

---

## 4. Mesures addicionals meves (no mesurades)

| # | Mesura | Per què |
|---|---|---|
| X1 | **Guardrail final únic**: exactament 10 IDs, sense duplicats, actius, sense comprats, i ordre respectat; si falten, omplir amb el següent per puntuació | Una llista invàlida val 0 i tot el que heu fet abans no compta (R4/R5) |
| X2 | **Mesurar l'efecte de cada regla amb A/B simultani** en un sol procés (`hash(request_id) % N`, `model_version` diferent per variant) | Evita barrejar atribucions al feedback; el dissabte no puntua, així que és el moment |
| X3 | **Mostra mínima per decidir**: amb ~2.700 peticions per branca a 30 min només es veuen diferències de CTR de 2–3 punts; no decidiu amb la conversió per compra en finestres curtes | Les compres són poques; evita decisions sobre soroll |
| X4 | **Executar S6 i S7** (categoria/marca de l'última vista; carretó i comprats previs) | Són les dues regles vostres (R2, R6) que el document no ha pogut mesurar |
| X5 | **Congelar la configuració abans de les 10:00** de diumenge i deixar el mateix model sense canvis durant la final; tenir un *rollback* a punt | Un error a les 10:05 costa molt i no hi ha leaderboard per notar-ho (només el semàfor) |
| X6 | **Decaïment del que s'ha après al practice** quan comença la final (braços del bandit, popularitat per producte), mantenint els pesos generals (preu, subcategoria, posició) | Productes, usuaris i tendències són nous; les estadístiques per producte no es traslladen |
| X7 | **Reservar els slots de dalt per al millor candidat** i fer servir el desempat de preu només per a empats reals | Combina R7 amb M5 |
| X8 | **Decaïment temporal dels `recent_views`** (més pes als més recents) | Intenció de sessió; ja hi ha un decaïment al lift, comprovar-lo |
| X9 | **No diversificar** (excepte la regla de categoria al slot 1): el document diu que les probabilitats se sumen sense saturar fins a ~9 % per llista | Omplir amb productes forts suma; diversificar canvia superventes per productes que no es compren |
| X10 | **Productes nous**: prior de puesto igual al mitjà dels seus veïns per contingut, corregit per les compres de `shop.events` | Palanca per al cold start (10 punts) |
| X11 | **Canvis de preu**: provar si una baixada de preu (`price_change < 0`) afegeix conversió; en cas negatiu, ignorar-ho | Hipòtesi de les bases; no hi ha cap dada que ho recolzi |
| X12 | **Fiabilitat**: PC endollat sense suspensió, watchdog actiu, `ON TIME` ≥ 99 % | 10 punts directes, i cada petició que cau al fallback val 0 en conversió |

---

## 5. Ordre d'execució suggerit

1. **Interruptors de la secció 2** (minuts). Això alinea 4.1 amb R3, R6 i R7.
2. **R5 + X1** (postprocés de filtre i farciment per puntuació). Protegeix de llistes invàlides.
3. **M3 + M4** (taxa base × relació, popularitat en viu): és el que el document dona més pes.
4. **R2 + R8** (categoria al top 1; regió per a clients nous), amb **X4** per calibrar el pes.
5. **X2/M2**: A/B contra el baseline per saber si esteu per sobre.
6. **X5**: congelar abans de les 10:00.

## 6. Què no he comprovat
- No he executat `analisis_conversion.py` ni cap consulta S6/S7.
- No he llegit el codi de `recommender.py` ni `business_rules.py`, així que no sé quines d'aquestes regles ja estan implementades.
- Les mesures X1–X12 són judici meu, no resultats.
