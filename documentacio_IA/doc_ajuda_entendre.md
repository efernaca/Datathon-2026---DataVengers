# Guia Conceptual i Operativa del Sistema (`calypso-2.1` · Etapes 1–4)

Document de referència ràpida amb les respostes essencials sobre com funciona el recomanador, com interactua amb RawTree, com aprèn en temps real i fins a on el podem portar abans de la Ronda Final de diumenge a les 10:00 CET.

---

## 1. Què canvia en considerar cada model (segons prioritats sobre el comprador)?

L'entrada sempre és una petició de comprador (`user_id`, `is_new_user`, `context = {device, region, page}`, `recent_views`) i la sortida són **10 `product_id` actius i ordenats** (els primers slots concentren la major part de clics i compres). Cada arquitectura canvia **quina funció d'utilitat prioritza**:

| Enfocament / Etapa | Prioritat principal | Què col·loca als primers slots | Trade-off |
| :--- | :--- | :--- | :--- |
| **Warm Precision** (`etapa_1`, `etapa_2`) | Compradors coneguts amb intenció clara | Carretó abandonat (`intent_score`), últim producte vist (`self_cvr`), transicions seqüencials (*Lift*) i afinitat de subcategoria/marca. | Molt bona conversió en usuaris recurrents, però **0% d'encert** en productes nous (*Cold Start*). |
| **Cold-Start / Bandit agressiu** (`etapa_3`) | Compradors nous i productes acabats d'entrar | Pont semàntic TF-IDF + SVD (`ColdStartIndex`) i bonificació d'incertesa UCB (`PositionAwareBandit`). | Puja molt la mètrica *Cold Start*, però si posa ítems nous sense validar als slots `0..3` degrada els usuaris *Warm*. |
| **Revenue-Maximizer** (`€ / request`) | Maximitzar valor esperat ($\text{P(compra)} \times \text{preu}$) | Productes de preu més alt compatibles amb el comprador (`price_compat`, `log_price_norm`). | Ajuda al desempat (`€/req`), però forçar preus alts redueix la conversió si el comprador busca preu moderat. |
| **Híbrid Slot-Gated** (`etapa_4` · `calypso-2.1`) | Adapta l'estratègia segons sessió i slot ([recommender.py](../etapa_4/src/recommender.py)) | **Sessió *Warm*:** blinda `0..6` amb precisió ML/IPS i coherència de subcategoria, deixa entrar *cold winners* confirmats a `4..9` i explora UCB a `7..9` (`màx. 2` nous). **Sessió *Cold Start*:** activa pont semàntic i UCB des del slot `0`. | Equilibra els **50 punts de Conversió** i els **10 punts de Cold Start** sense sacrificar l'un per l'altre. |

---

## 2. Què fa el model amb les *queries* (consultes SQL)?

El sistema **no llança consultes SQL pesades mentre un comprador espera**: això superaria el límit d'SLA de `3.000 ms`. Per això separa l'execució en dos bucles ([serve.py](../etapa_4/src/serve.py)):

1. **Bucle de Fons (*Slow / Live Loop*):**
   - Executa SQL sobre RawTree per agregar popularitat, transicions seqüencials, perfils, estoc, esdeveniments en viu i feedback propi.
   - Reentrena el re-ranker IPS ([reranker.py](../etapa_4/src/reranker.py)), actualitza el bandit ([bandits.py](../etapa_4/src/bandits.py)) i empaqueta totes les estructures en un **`RecommenderSnapshotV4`** immutable a la memòria RAM ([state.py](../etapa_4/src/state.py)).
2. **Bucle Ràpid (*Fast Loop*):**
   - Cada `0.12 s` fa **una sola consulta SQL ultralleugera** a `shop.requests` per llegir els compradors dels últims `2.400 ms` (`LIMIT 25`).
   - Puntua tots els compradors del lot **íntegrament a la RAM** en `~0.4–0.5 ms/req` (sense cap SQL addicional per comprador) i envia totes les respostes juntes en **un sol `POST` (*batch*)** a `team_32.submissions`.

---

## 3. On es guarden les *queries*, les dades i els resultats?

- **Al servidor de RawTree (ClickHouse):**
  - **Taules compartides (`shop.*`, lectura):** `shop.requests`, `shop.events`, `shop.catalog`, `shop.catalog_updates`, `shop.history_sessions` (~2M files) i `shop.history_events` (~12M files).
  - **Taules privades del nostre equip (`team_32.*`):**
    - `team_32.submissions`: recomanacions que enviem (`request_id`, `items`, `model_version`).
    - `team_32.feedback`: resultat real escrit per l'evaluador (`served_by`, `fallback_reason`, `latency_ms`, `clicked`, `purchased`, `revenue_eur`).
    - `team_32.scratch`: memòria JSON sense esquema on pugem decisions de `CalypsoAgent`, pesos del model, estat del bandit i veïns *Cold Start* ([agent_scratch.py](../etapa_4/src/agent_scratch.py)).
- **En local al repositori:**
  - Consultes SQL: [`etapa_4/sql/etapa4_queries.sql`](../etapa_4/sql/etapa4_queries.sql) (i carpetes `sql/` de cada etapa).
  - Caché atòmica a disc ([disk_cache.py](../etapa_4/src/disk_cache.py)): `etapa_4/cache/static_history_v4.pkl` i `etapa_4/cache/snapshot_v4_latest.pkl`.

---

## 4. Quantes *queries* màximes per unitat de temps podem fer? Es relaciona amb els compradors que entren?

Les dues ràtios estan relacionades però **no són 1 a 1**:

- **Ritme d'entrada de compradors:** el simulador genera ~**3 compradors/segon** de mitjana (`~180/min`), cadascun amb un límit màxim de **3 segons** des del seu `ts`.
- **Concepte d'"autobús llançadora" (*batching*):** no fem 1 consulta SQL per comprador. Una única consulta SQL cada `0.12 s` (~8/s) recull en un paquet tots els compradors nous de la finestra de `2.400 ms` (`0`, `3` o `15` de cop), els calcula a la RAM i els puja en 1 sol `POST`. **Si entren més compradors, pugen més compradors al mateix autobús sense fer més *queries*.**
- **Per què limitem la freqüència i separem les *queries*:** perquè totes les peticions comparteixen l'API key sobre ClickHouse. Si llancem consultes pesades de 12M de files repetidament (com passava cada `18 s` a l'Etapa 3), el servidor triga més a contestar les lectures en viu i les respostes passen dels `3.000 ms` (`ON TIME` baix).

Pressupost segur implementat a l'Etapa 4 ([config.py](../etapa_4/src/config.py)):

| Tipus de tasca | Cadència | Objectiu |
| :--- | :--- | :--- |
| **Polling ràpid (`shop.requests`)** | Cada `0.12 s` | Llegir finestra de `2.400 ms` (`LIMIT 25`) + `POST` agrupat |
| **Control d'estoc (`out_of_stock`)** | Cada `1.5 s` | Eliminar a l'instant productes esgotats per no invalidar enviaments |
| **Actualització *live* (`events`, `feedback`, `catalog_updates`)** | Cada `15.0 s` | Tendències recents, compres vives, productes nous i reentrenament |
| **Històric pesat (60 dies / 12M files)** | `1` cop + cada `900 s` | Cachejat a `static_history_v4.pkl`; no bloqueja arrencades ni el bucle ràpid |

---

## 5. Com podem gestionar diferents models? Podem executar subcarpetes diferents alhora?

### Regla fonamental: **NO executeu dos `serve.py` independents alhora contra `team_32`**
Segons les regles ([README.md](../datathon-fme-2026-main/README.md)), només compta una resposta per `request_id`. Llançar dues terminals paral·leles (`etapa_2` i `etapa_4`) duplica consultes SQL, consumeix ample de banda sense millorar l'SLA i barreja l'atribució a `team_32.feedback`.

### Les 3 maneres correctes de gestionar i comparar models

1. **Benchmark offline simultani (`etapa_1`..`etapa_4`):** [`etapa_4/src/eval.py`](../etapa_4/src/eval.py) importa els recomanadors de totes quatre subcarpetes i els avalua alhora sobre les mateixes peticions de test sense tocar producció.
2. **Proves en viu per finestres de temps (`model_version`):** 1 sol procés actiu enviant el nom del model (`calypso-2.1`, etc.). El leaderboard mostra `BEST MODEL` per `model_version` i [`monitor.py`](../etapa_4/src/monitor.py) (`--watch 10`) desglossa `on_time%`, `p50_ms`, `p95_ms`, `CTR%`, `CVR%` i `€/req` per cada versió.
3. **A/B Testing simultani dins d'UN sol procés:** si voleu comparar 2 variants en directe a la vegada, un únic `serve.py` llegeix cada `request_id` un sol cop i un enrutador determinista (p. ex. per `hash(request_id) % 2`) decideix quin model respon i amb quina etiqueta `model_version`, enviant exactament **1 sola resposta per comprador**.

---

## 6. Què fa el servidor de RawTree i com aprèn el nostre algoritme?

### Què fa RawTree (`api.rawtree.com`)
1. **Botiga i evaluador en viu:** emmagatzema les peticions (`shop.requests`), els enviaments (`team_32.submissions`) i el retorn real (`team_32.feedback`).
2. **Motor d'agregació SQL al núvol (5 punts del jurat — *Use of RawTree*):** processa els 12M d'esdeveniments i 2M de sessions als seus servidors ClickHouse i ens retorna només taules resumides.
3. **Memòria persistent (`team_32.scratch`):** desa en format JSON les decisions del nostre agent i l'estat après.

### Els 5 mecanismes d'aprenentatge continu de `calypso-2.1` (cada 15 segons)

| Mecanisme | Fitxer | Què aprèn i què millora |
| :--- | :--- | :--- |
| **1. Bandit conscient de la posició** | [bandits.py](../etapa_4/src/bandits.py) | Descompta el biaix d'exposició per slot ($(k+1)^{-0.75}$) sobre `team_32.feedback`, premiant ítems que realment converteixen i validant *cold winners*. |
| **2. Re-Ranker ML IPS (14 pesos)** | [reranker.py](../etapa_4/src/reranker.py) | Reentrena una Regressió Logística ponderada per *Inverse Propensity Scoring* i ajusta l'equilibri entre seqüència, carretó, afinitat, preu i context. |
| **3. Senyals en viu de `shop.events`** | [serve.py](../etapa_4/src/serve.py) | Capta tendències calentes recents (`live_trend_norm`) i exclou productes ja comprats pel mateix usuari (`live_user_bought`). |
| **4. Pont semàntic + estoc 1.5 s** | [cold_start.py](../etapa_4/src/cold_start.py) | Connecta productes nous sense vendes (`new_product`) amb productes *warm* similars per descripció, marca, subcategoria i preu. |
| **5. Agent autònom `CalypsoAgent`** | [agent_scratch.py](../etapa_4/src/agent_scratch.py) | Monitoritza `CTR`, `CVR` i volum de productes nous per adaptar el règim (`nominal`, `cold_surge`, `high_velocity_ml`, `conservative_fallback`) sense intervenció humana. |

---

## 7. Persistència dels pesos, consell de l'organitzador i fins a on podem portar el sistema

### On es guarden els pesos i si parteix de zero en reiniciar
- **No parteix de zero.** Té persistència doble:
  1. **Local (`etapa_4/cache/snapshot_v4_latest.pkl`):** es desa atòmicament (`os.fsync` + `os.replace`) cada `15 s`; en reiniciar, `build_bootstrap_snapshot()` arrenca immediatament carregant l'últim estat (~`15.8 ms` al `--dry-run`).
  2. **Al núvol (`team_32.feedback` i `team_32.scratch`):** tot l'històric de clics i compres segueix a RawTree; encara que es perdés la caché local, en 15 segons es reconstrueix tot l'aprenentatge des de ClickHouse.

### Què volia dir l'organitzador amb *"doneu-li canya al sistema de queries"*?
- ClickHouse pot calcular agregacions complexes en segons directament al servidor (joins de sessió, perfils de preu, `argMax`, `countIf`, `LIMIT N BY`), estalviant baixar dades crues al PC i puntuant directament en **Use of RawTree (5 pts)** i **Learning from the stream (10 pts)**.
- El secret és donar-li canya **amb intel·ligència**: enriquir les consultes sobre dades en viu (`shop.events`, `shop.catalog_updates`, `team_32.feedback`) dins el bucle de `15 s`, sense carregar el bucle ràpid de `0.12 s` ni repetir càlculs pesats de 12M de files.

### Què cal fer fins demà diumenge a les 10:00 i fins a on podem escalar-lo
1. **Operativa immediata (dissabte nit):**
   - Desactivar la suspensió (*Sleep*) de Windows amb el PC endollat.
   - Mantenir actiu `powershell -ExecutionPolicy Bypass -File etapa_4/run_watchdog.ps1` i comprovar `on_time%` (>= 95–100%) i `CVR%` amb `python etapa_4/src/monitor.py --watch 10`. No cal cap entrenament manual extern: `serve.py` aprèn automàticament cada `15 s`.
2. **Ronda Final (diumenge 10:00–12:00 CET):**
   - El *stream* canvia a **usuaris nous, productes nous i noves tendències** i compta el 70% de la nota automàtica (`50` Conversió, `10` Cold Start, `10` Fiabilitat SLA + desempat per `€/req`, amb indicador verd si `ON TIME > 90%` als últims 5 min).
3. **Evolucions recomanades per portar el sistema encara més amunt:**
   - **3 noves *queries* SQL en viu sobre `shop.events` (cada 15 s):**
     1. `live_user_intent`: carretons abandonats i vistes recents de la botiga orgànica del mateix matí (vital quan arribin compradors nous a les 10:00).
     2. `live_transitions`: co-compres i co-vistes formades en viu durant l'última hora (captant relacions reals dels **productes nous** de diumenge).
     3. `live_context_pop`: rànquing viu per regió, dispositiu i pàgina sobre `shop.events`.
   - **Mode `--ab-test` dins un sol `serve.py`:** per comparar variants d'exploració o pesos de preu en directe durant la pràctica sense duplicar respostes ni trencar l'SLA.
