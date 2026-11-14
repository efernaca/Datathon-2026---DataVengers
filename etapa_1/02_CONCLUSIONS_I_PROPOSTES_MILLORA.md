# Conclusions i Nou Full de Ruta de Millores (Etapes 1, 2 i 3)

**Projecte:** Datathon FME 2026 — RawTree Real-Time Recommendation Challenge  
**Fonts integrades:** Auditoria real del codi a `etapa_1/src/` i `etapa_1/sql/`, `01_PLANIFICACIO_ETAPA_1.md`, `03_FUNCIONAMENT_ACTUAL.md`, `04_REVISION_Y_ENFOQUES_ALTERNATIVOS.md`, `plan_collaborative_filtering_user_item_hibrido.md`, `documentacio_IA/` i les bases oficials (`datathon-fme-2026-main/README.md`).

---

## 1. Diagnòstic de l'Estat Real (Codi vs. Documentació)

La base arquitectònica de l'Etapa 1 (*Fast/Slow Loop* + `RecommenderSnapshot` immutable + `guardrails.py` deterministes + interfície única `recommend(request, snapshot)`) és correcta i **no s'ha de refer**. Tanmateix, en contrastar els documents amb el codi implementat actualment:

* **Ja resolt al codi:**
  * [`src/config.py`](src/config.py) ja carrega tant `.env` com `group_API.env` (`filenames = (".env", "group_API.env")`), així que l'avís de credencials no trobades està solucionat.
* **Punts crítics pendents abans d'avançar a models nous:**
  1. **Latència i fiabilitat del bucle ràpid ([`src/serve.py`](src/serve.py)):** `step()` executa lectura (`1.2s`) + `push` (`1.4s`) + reintent (`1.4s`) de forma seqüencial (pitjor cas `> 4s` per sobre de l'SLA de `3s`), marca `_seen` *abans* de confirmar el `push`, i llegeix `ORDER BY ts DESC LIMIT 50` sense calcular `age_ms`.
  2. **Refresc bloquejant i lentitud en Cold Start ([`src/serve.py`](src/serve.py)):** `refresh_full()` recalcula els 60 dies històrics cada `180s` bloquejant l'actualització d'estoc mentre dura, i els `new_product` triguen fins a 3 minuts a entrar al snapshot (`catalog_refresh_sec` i `trends_refresh_sec` estan declarats a `config.py` però no s'usen).
  3. **Senyals incompletes ([`src/recommender.py`](src/recommender.py)):** `user_subcat_rows` no es passa mai a `build_snapshot_from_records`, `user_bought` no incorpora compres en viu de `shop.events` ni `team_NN.feedback`, i `SQL_TRANSITIONS` no filtra `a.ts < b.ts` ni descompta popularitat de destí.
  4. **Validació sintètica circular ([`src/eval.py`](src/eval.py)):** encara no hi ha un carregador connectat a `shop.history_sessions` reals.

### Nota sobre volums de trànsit reals
* **`shop.requests` (`~3/s`):** Correcte i consistent a tota la documentació (`~260.000` files/dia, `~21.600` peticions en les 2 hores de la final).
* **`shop.events` (`~50/s` als esquemes):** **No és una dada oficial de les bases.** El `README.md` només indica una proporció històrica de `~12M` d'events orgànics enfront de `~2M` de sessions (`6:1`, equivalent a `~18 events/s` si es mantingués a `3 req/s`). Cal mesurar la taxa real en viu via SQL abans de dimensionar finestres o bandits sobre `shop.events`.

---

## 2. Millores a Implementar (Prioritzades i Consolidades)

### Bloc A · Blindatge de Fiabilitat i Cold Start Immediat (Prerequisit)
1. **Desacoblar lectura i enviament a `FastServingLoop`:**
   * Consultar `shop.requests` amb `age_ms` (`ts::DateTime64(3) > now64(3) - INTERVAL 4 SECOND`) per prioritzar peticions que caduquen abans i descartar les ja fora d'SLA.
   * Enviar `push_submissions` en un pool de fils separat i marcar `_seen` només quan el push es confirmi (reencolant peticions si `age_ms < 2.2s`).
2. **Separar el `BackgroundUpdater` en 3 cadències reals + caché a disc:**
   * **Arrencada des de disc:** calcular les agregacions dels 60 dies històrics **una sola vegada**, desar-les en disc (`parquet`/`npz`/`pickle`) i carregar-les en `< 5–10s` en reiniciar.
   * **Fil d'estoc i catàleg (cada `1–2s`):** consultar només `out_of_stock` i `shop.catalog_updates` incrementals (`WHERE ts > last_ts`), calculant a l'instant els veïns TF-IDF de qualsevol `new_product` nou via `ColdStartIndex.neighbors_for_batch()`.
   * **Fil de senyals en viu (cada `15–30s`):** refrescar incrementalment `shop.events` (tendències amb decaïment exponencial + suavitzat tipus Poisson/Empirical Bayes) i llegir `team_NN.feedback` (creuat amb una caché local `request_id -> user_id`) per actualitzar `user_bought` i mètriques en temps real.
3. **Supervisor i alarma externa:** executar el servei sota un bucle de reinici automàtic en màquina sense suspensió i un monitor periòdic sobre `team_NN.feedback` si `served_by = 'team'` baixa del `98%` o `p95 > 2.000 ms`.

### Bloc B · EDA Experimental sobre RawTree Real (`sql/eda_queries.sql` + proves de simulador)
Executar i desar els resultats numèrics de:
* Corba real de *position bias* (`γ_k` per slot `1..10`) ponderada per `1 / propensity`.
* Taxa real de compra sobre productes ja dins `recent_views` (per calibrar si convé re-engagement o cross-sell).
* Elasticitat al preu i descompte relatiu dins la subcategoria.
* Canibalització/dependència entre ítems d'una mateixa llista (decideix si cal diversitat activa o només ordenar per probabilitat).
* Percentatge d'`is_new_user = true` i `recent_views = []`, i comparativa de conversió entre històric i stream de dissabte.

### Bloc C · Evolució del Model de Rànquing (Conciliació entre propostes)
* **Descartar User-CF pur** (de `plan_collaborative_filtering_user_item_hibrido.md`): és costós d'escalar, no aporta valor en `is_new_user = true` i perd eficàcia si la final introdueix compradors nous.
* **Substituir les transicions heurístiques per EASE (Item-Item en forma tancada):**
  * Sobre `~5.000` productes, invertir $(X^\top X + \lambda I)^{-1}$ en NumPy triga segons, elimina el biaix cap a productes ultra-populars amb un únic hiperparàmetre ($\lambda$), i permet puntuar tot el catàleg en microsegons.
  * Per a `new_product`, aprendre una projecció lineal des del vector de contingut (`TF-IDF` + subcategoria + marca + $\log(1+\text{preu})$) cap a les files d'EASE i validar-ho ocultant un 5–10% de productes coneguts.
* **Substituir els 12 pesos manuals de `config.py` per un Re-Ranker Supervisat:**
  * Entrenar LightGBM o Regressió Logística sobre `shop.history_sessions` (2M impressions $\times$ 10 slots) ponderades per `1 / propensity` (retallat), incloent `slot` en entrenament i fixant `slot = 1` en inferència per eliminar el biaix de posició.
  * Com que la petició arriba a ~3/s i disposem de `> 1s` de marge, podem puntuar o bé els top ~150 candidats o directament els ~5.000 productes en pocs mil·lisegons, combinant features estructurals (coincidència amb `recent_views`, score EASE, descompte, tendència en viu, preu relatiu, context) i entrenant una variant sense IDs per generalitzar millor a la final.

### Bloc D · Aprenentatge en Viu, Diversitat i `team_NN.scratch`
* **Repartiment realista de l'aprenentatge en viu (evitant sobreparametritzar `feedback`):**
  * En 2 hores de final (`~21.600` peticions, ~400 compres si la CVR ronda el 2%) **no hi ha mostra per mantenir un bandit per a cadascun dels 5.000 productes** a `team_NN.feedback`.
  * **Popularitat per producte ← `shop.events`:** actualitzar comptadors amb decaïment exponencial suavitzats cap a un *prior* (històric per productes coneguts; mitjana de veïns de contingut per `new_product`).
  * **Selecció d'estratègia ← `team_NN.feedback`:** aplicar Thompson Sampling sobre pocs braços (3–5 mescles de pes viu vs. històric per segment `warm/cold × page`) utilitzant principalment els **clics** com a senyal més dens, i reservar exploració acotada als slots `7–10` per a `new_product` o tendències emergents.
* **Diversitat i desempat per ingressos:**
  * Aplicar diversificació voraz / MMR (slots `4–10`) principalment quan l'intent és ambigu (`recent_views` buit a `home`) o si l'EDA confirma dependència entre ítems mostrats.
  * Usar `price_eur` només com a **desempat suau** entre candidats amb probabilitat de compra gairebé idèntica (sense sacrificar conversió).
* **Ús de `team_NN.scratch` i Side Challenge:**
  * Desar a `team_NN.scratch` resultats de l'EDA, embeddings/matrius EASE, pesos versionats i telemetria d'experiments per cobrir els 5 punts de *Use of RawTree* del jurat i habilitar un agent analista separat del *Fast Loop* (amb *rollback* automàtic si cau el CVR).

---

## 3. Repercussions en la Resta d'Etapes del Projecte

Implementar aquest pla transforma les etapes següents en quatre aspectes clau:

1. **Impacte en [`src/state.py`](src/state.py) (`RecommenderSnapshot`):**
   * El contracte `recommend(request, snapshot)` es manté estable (zero reescriptura del servidor ni dels guardrails).
   * En canvi, `RecommenderSnapshot` haurà d'evolucionar de diccionaris dispersos (`transitions` top-30) a incloure matrius NumPy compactes en memòria (matriu EASE `5000 × 5000` en `float32`, ~100 MB, + matriu de features estàtiques d'ítem) i permetre que `with_fast_updates()` actualitzi també `user_bought` incrementalment.
2. **Impacte en [`src/eval.py`](src/eval.py) (Validació Offline):**
   * El simulador sintètic queda únicament com a banc de tests unitaris i de latència.
   * Cal afegir un carregador sobre `shop.history_sessions` reals amb tall temporal estricte (`train < CUT <= test`), mètriques per parell `(petició, slot)` (Log-loss, AUC, calibració i valor de llista $\sum_k e_k \hat{p}_k$, perquè `SNIPS` a l'slot 1 tindrà pocs `slot1_matches` quan personalitzem), i un **escenari de simulació de Ronda Final** (forçant `is_new_user = True` i ocultant productes com a `new_product`).
3. **Impacte en Desplegament i Experimentació (Dissabte):**
   * Amb la caché històrica a disc, passar d'una versió de model a una altra deixarà de provocar desenes de segons de *fallback* en arrencar.
   * Les proves A/B en viu durant dissabte s'han de repartir per `hash(request_id)` amb `model_version` diferenciat i comparar-les principalment per clics a `team_NN.feedback` i lectures del leaderboard (que usa probabilitats esperades sense soroll de moneda).
4. **Impacte en la Ronda Final (Diumenge 10:00–12:00) i el Jurat (30 punts):**
   * **Reliability (10 pts) i Cold Start (10 pts):** desacoblar el `push`, filtrar per `age_ms` i ingerir `new_product` + `out_of_stock` en `< 2s` protegeix directament 20 dels 70 punts automàtics.
   * **Jurat (30 pts):** congelar la versió final abans de les 08:00 CET i documentar les taules de l'EDA, la validació offline real i l'evolució de `model_version` a `team_NN.scratch` proporciona tota l'evidència per a *Method* (10p), *Learning from the stream* (10p) i *Use of RawTree* (5p).

---

## 4. Criteris d'Acceptació abans de Pujar Canvis a Producció

Cap millora d'Etapa 2 o 3 s'ha de desplegar al procés principal sense superar:
1. **Validació Offline Real:** millora consistent sobre `shop.history_sessions` tant en segment `ALL` com en `COLD_START` i en l'escenari simulat de final (`is_new_user = True`).
2. **Pressupost de Latència:** inferència local `< 5 ms/petició` en lots i latència extrem-a-extrem `p95 < 1.500 ms` a `team_NN.feedback`.
3. **Fiabilitat 100%:** `served_by = 'team'` amb 10 IDs únics, actius i en estoc sota proves de fallades de xarxa i reinicis.
