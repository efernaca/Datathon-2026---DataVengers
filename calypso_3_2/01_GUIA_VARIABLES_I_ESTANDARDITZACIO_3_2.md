# Calypso 3.2 · Mapa de Variables Modificables, Priorització i Estandardització per Conversió

**Objectiu:** Estandarditzar el model `calypso_3_2` per saber exactament quines variables tocar per orientar-lo cap a la **conversió a compra efectiva** (prioritzant productes més barats, tendències per perfil d'usuari i perfils de producte) i definir com refactoritzar l'arquitectura per fer iteracions ràpides i segures sobre els pesos.

---

## 1. Diagnòstic ràpid: què hi ha ara i on està fragmentat

Ara mateix el model `calypso_3_2` té les palanques repartides en **7 punts** de `etapa_4/src/`, amb dues inconsistències importants detectades al codi:

1. **`ConfigV4` vs `DEFAULT_PRIOR_WEIGHTS` (camps desconnectats):** [config.py](./etapa_4/src/config.py) defineix pesos (`self_view_weight=1.80`, `intent_weight=0.50`, `affinity_weight=0.25`, `price_compat_weight=0.20`, `ctx_bayes_weight=0.30`, `global_pop_weight=0.22`, `live_trend_weight=0.25`), però [recommender.py](./etapa_4/src/recommender.py#L999-L1006) agafa `w0..w13` de `PositionDebiasedRanker` ([reranker.py](./etapa_4/src/reranker.py#L31-L49)), ignorant aquells 7 camps de `ConfigV4`. Només s'hi injecten directament `weight_price_drop`, `revenue_tie_weight`, `weight_forward_cold` i `weight_reverse_cold_graft` (aquests 3 darrers sobreescrits per `CalypsoAgent`).
2. **Biaix actual cap a ingressos i clics (no conversió pura):**
   - `reranker.py` etiqueta `clicked` sense compra com a positiu (`y=1`, pes `ips_w * 0.45`).
   - `bandits.py` premia el clic (`click_reward = 0.25` vs `purchase_reward = 1.0`).
   - `log_price_norm` té prior positiu (`+0.03`), i el desempat N9 de `recommender.py` (línia 1197: `if pb_p > pa_p`) **puja el producte més car**, no el més barat ni el de més conversió.
   - `discount_pct` (V6) suma a `bg_score` via `weight_price_drop`, però **no** és dins les 14 *features* de `FEATURE_NAMES`, de manera que el re-ranker ML no n'aprèn el pes.

---

## 2. Inventari complet de variables modificables per fitxer

### A. [etapa_4/src/config.py](./etapa_4/src/config.py) (`ConfigV4`)
- **Ritme d'actualització i SLA:** `poll_interval_s` (`0.12`), `stock_refresh_interval_s` (`1.5`), `live_refresh_interval_s` (`15.0`), `static_refresh_interval_s` (`900.0`), `max_request_age_ms` (`2400`), `max_requests_per_batch` (`25`), `candidate_pool_size` (`64`).
- **Seqüència i decaïment temporal:** `recency_decay` (`0.72` per posició a `recent_views`), `recent_halflife_sec` (`624.0` s per N13), `lift_beta` (`0.45`, penalitza popularitat en transicions), `lift_prior_pop` (`30.0`), `ctx_kappa` (`50.0`, suavitzat jeràrquic N12).
- **Preu i *Cold Start*:** `weight_price_drop` (`0.18`), `weight_forward_cold` (`0.48`), `weight_reverse_cold_graft` (`0.30`), `reverse_min_sim` (`0.45`), `reverse_same_subcat` (`True`), `max_cold_neighbors` (`8`), `max_reverse_cold_per_warm` (`3`).
- **ML Re-Ranker i Bandit (*Slot-Gating*):** `ml_max_blend_alpha` (`0.45`), `ml_min_positives` (`25`), `ml_l2_c` (`1.2`), `bandit_prior_strength` (`20.0`), `bandit_ucb_alpha` (`0.14`), `bandit_cold_ucb_multiplier` (`1.8`), `bandit_exam_power` (`0.75`), `cold_explore_start_slot_warm` (`7`), `max_cold_explore_in_warm` (`2`).
- **Diversitat (MMR N10) i Desempat (N9):** `mmr_start_slot_session` (`5`), `mmr_start_slot_home` (`3`), `mmr_subcat_lambda_session` (`0.0`), `mmr_subcat_lambda_home` (`0.15`), `mmr_brand_lambda` (`0.04`), `revenue_eps` (`0.015`), `revenue_tie_weight` (`0.03`).

### B. [etapa_4/src/reranker.py](./etapa_4/src/reranker.py) (`PositionDebiasedRanker`)
- **`DEFAULT_PRIOR_WEIGHTS` (`w0..w13`):** `[1.05, 1.85, 0.52, 0.25, 0.20, 0.30, 0.22, 0.25, 0.02, 0.32, 0.48, 0.03, 0.10, 0.04]` corresponents a `(seq_lift_score, self_cvr_view, intent_score, user_affinity, price_compat, ctx_bayes_score, global_pop_norm, live_trend_norm, is_new_product, reverse_graft_score, forward_cold_score, log_price_norm, same_top_subcat, same_top_brand)`.
- **Límits d'estabilitat ML:** `_MIN_PRIOR_RATIO` (línies 51–69), sostre `max(prior * 2.2, 0.30)` (línia 140), rampa `(n_pos - min_positives) / 350.0` (línia 143). *(Nota crítica: `np.clip` assumeix priors no negatius; si poses `w11 < 0` per afavorir barats sense ajustar aquest retall, el model el neutralitzarà a `0.0`).*
- **Recompensa d'entrenament (`build_ips_training_dataset`, línies 259–270):** `purchased` (`y=1`, pes `ips_w * 2.0`), `clicked` (`y=1`, pes `ips_w * 0.45`), no interacció (`y=0`, pes `exam_p`), límit `max_ips_weight=3.5`, `max_sessions=3000`.

### C. [etapa_4/src/bandits.py](./etapa_4/src/bandits.py) (`PositionAwareBandit`)
- `click_reward` (`0.25`), `purchase_reward` (`1.0`), `warm_ucb_multiplier` (`0.20`), `cold_ucb_multiplier` (`1.8`).
- Rangs interns fixats: `exploit_delta` retallat a `[-0.18, 0.35]` amb pendent `2.5` (línia 129); `ucb_bonus` capat a `0.38` (línia 133).

### D. [etapa_4/src/recommender.py](./etapa_4/src/recommender.py)
- **Mescla seqüencial i EASE:** `lam=250.0`, `top_k=35` (línia 615) i combinació `0.75 * lift + 0.25 * ease` (línia 627).
- **5 segments de sessió (línies 944–955):** `seq_scale=1.06` (`nuevo_sesion`), `intent_scale=1.15` (`conocido_sin_sesion`), `ctx_scale=1.25/1.10`, `pop_scale=1.15` (`nuevo_sin_sesion`), `cold_scale=1.25` (`producto_nuevo`).
- **Caps, esmorteïment i compatibilitat:** caps `s_seq <= 1.8` i `s_self * 5.0 <= 1.2` (línies 1027–1031); fons reduït a `0.18` (si diferent subcategoria) o `0.45` (mateixa subcategoria) quan hi ha sessió (línia 1068); dispersió mínima de preu d'usuari `std_lp >= 0.45` (línia 440); pesos jeràrquics de context `1.0 / 0.70 / 0.45` (línies 419–421).
- **Desempat N9 (línies 1183–1198):** intercanvia parells dins `revenue_eps` si `pb_p > pa_p`.

### E. [etapa_4/src/cold_start.py](./etapa_4/src/cold_start.py) (`ColdStartIndexV4`)
- **Perfil semàntic i d'atributs de producte (línies 92–118):**
  - Text: `0.65 * sparse_cos + 0.35 * dense_cos` (TF-IDF 1–2 grames `max_features=4096` + SVD `64`).
  - Atributs: `+ 0.35 * (subcat ==)` + `0.15 * (cat ==)` + `0.12 * (brand ==)` + `0.15 * exp(-|Δlog_price|)`.

### F. [etapa_4/src/serve.py](./etapa_4/src/serve.py) (Consultes SQL ClickHouse, línies 99–193)
- `SQL_POPULARITY`: `5.0*purchase + 2.0*cart + 0.2*view` i `self_cvr = (purchases+0.5)/(views+25.0)`.
- `SQL_TRANSITIONS`: `6.0*purchase + 2.5*cart + 1.0*view`, `LIMIT 60 BY src_product`.
- `SQL_USER_INTENT`: `3.0*cart + 0.5*view` amb semivida de `7.0` dies (N11), `LIMIT 10 BY user_id`.
- `SQL_USER_AFFINITY`: `5.0*purchase + 2.0*cart + 0.3*view`, `LIMIT 8 BY user_id`.
- `SQL_LIVE_TRENDS`: `5.0*purchase + 2.0*cart + 0.2*view`, `LIMIT 200` (global).

### G. [etapa_4/src/agent_scratch.py](./etapa_4/src/agent_scratch.py) (`CalypsoAgent`)
- Regles que sobreescriuen paràmetres en viu segons llindars (`cvr < 0.015` $\rightarrow$ `ml_scale=0.45`; `new_share >= 0.06` o `n_new >= 25` $\rightarrow$ `cold_surge`; `cvr >= 0.045` $\rightarrow$ `ml_scale=1.15`).

---

## 3. Què tocar per a cada prioritat de negoci

### Prioritat 0 (Absoluta): Maximitzar conversió a compra efectiva
1. **Eliminar el soroll dels clics sense compra:**
   - A `reranker.py` (`build_ips_training_dataset`): baixar el pes de `clicked` de `0.45` a `0.0` (entrenar `y=1` només sobre `purchased`) o a `0.05` si hi ha molt pocs positius.
   - A `bandits.py`: posar `click_reward = 0.0` (o `0.05`) i `purchase_reward = 1.0` perquè productes cridaners que no venen no acumulin `exploit_delta` positiu.
   - A les SQL de `serve.py`: apujar la ràtio de `purchase` front `view` (p. ex. `10.0 * purchase + 2.5 * cart + 0.05 * view`) i reduir `0.2` / `1.0` de les vistes.
2. **Protegir els slots alts amb certesa de compra:**
   - Mantenir o endurir `cold_explore_start_slot_warm` (`7` o `8`), reduir `bandit_ucb_alpha` (`0.14` $\rightarrow$ `0.06–0.10`) i `warm_ucb_multiplier` (`0.20` $\rightarrow$ `0.05`) quan ja hi ha trànsit estable.
   - Apujar `w0` (`seq_lift`), `w1` (`self_cvr_view`) i `w2` (`intent_score`), i pujar el sostre de `s_self` si les recompres/intencions recents converteixen molt bé.
3. **Treure la priorització per ingrés si entra en conflicte amb convertir:**
   - Posar `revenue_tie_weight = 0.0` i `revenue_eps = 0.0` (o substituir-lo per desempat per preu més baix / més descompte, vegeu avall).

### Prioritat 1: Productes més barats (sense enfonsar la conversió)
1. **Revertir els 3 impulsos pro-preu alt:**
   - `DEFAULT_PRIOR_WEIGHTS[11]` (`log_price_norm`) de `+0.03` a **negatiu** (ex. `-0.10` a `-0.25`), o substituir la *feature* per `cheapness_norm = 1.0 - lp_norm` amb prior positiu (així no trenca el `np.clip(..., floors, ceilings)` de `reranker.py`!).
   - `ConfigV4.revenue_tie_weight = 0.0` (i eliminar `+ cfg.revenue_tie_weight * lp_s` a `fallback_scores`).
   - A `recommender.py` (línia 1197): canviar el desempat N9 de `if pb_p > pa_p` a `if pb_p < pa_p` (amb `pb_p > 0`), o desactivar-lo (`revenue_eps = 0.0`).
2. **Potenciar rebaixes i asimetria del pressupost d'usuari:**
   - Pujar `ConfigV4.weight_price_drop` (`0.18` $\rightarrow$ `0.35–0.50`) i incloure `discount_pct` dins `FEATURE_NAMES` perquè el ML aprengui l'impacte real del descompte en la compra.
   - Modificar `price_compat` (`recommender.py`, línia 1048): actualment penalitza igual passar-se per dalt que per baix (`z^2`). Fer-la **asimètrica**: si l'ítem és més barat que `mean_lp` (`z < 0`), penalització suau o nul·la (`0.2 * z^2`); si és més car (`z > 0`), penalització forta (`1.5 * z^2`).

### Prioritat 2: Tendències segons perfils d'usuari
Actualment `SQL_LIVE_TRENDS` i `live_trend_norm` (`w7 = 0.25`) són **globals** (idèntics per a tothom) i s'esmorteeixen a `0.18x` quan hi ha sessió. Per personalitzar-les:
1. **Interacció tendència $\times$ perfil d'usuari:** crear la *feature* `user_trend_match = live_trend_norm * (0.4 + 0.6 * affinity_subcat_brand)` i `ctx_live_trend` per `(region, device, page)`.
2. **Enriquir el perfil d'usuari:**
   - A `SQL_USER_AFFINITY` (`serve.py`): apujar `LIMIT 8` a `15–20` i aplicar decaïment temporal (com a `SQL_USER_INTENT`) perquè pesin més les preferències recents de marca/subcategoria.
   - A `reranker.py`: apujar `w3` (`user_affinity`, ara `0.25`), `w2` (`intent_score`, ara `0.50`) i reduir l'esmortiment del bloc contextual/tendència quan hi ha `same_sc` (línia 1068).

### Prioritat 3: Perfils de producte
1. **Similitud d'atributs i text (`cold_start.py`):** ajustar els pesos de `_similarity_matrix` (`0.35` subcategoria, `0.15` categoria, `0.12` marca, `0.15` preu) i la barreja `0.65 * sparse_cos + 0.35 * dense_cos`. Si marca/subcategoria manen en la compra, pujar-los i reduir el soroll de descripcions llargues.
2. **Coherència de producte en sessió (`recommender.py`):** ajustar `w12` (`same_top_subcat`, ara `0.10`), `w13` (`same_top_brand`, ara `0.04`), `mmr_brand_lambda` (`0.04`) i la ràtio `0.75 * lift + 0.25 * ease` (més EASE afavoreix productes substitutius del mateix perfil; més Lift afavoreix complements seqüencials).

---

## 4. Proposta d'estandardització arquitectònica per iterar ràpidament

Per poder optimitzar pesos de forma **iterativa, reproduïble i amb un sol objectiu (conversió)** sense tocar 6 fitxers cada vegada:

1. **Unificar configuració i *features* en un `PolicyProfile` extern (`weights_profile.json`):**
   - Moure *tots* els pesos (`w0..wN`, `weight_price_drop`, multiplicadors de 5 segments, `lift/ease ratio`, `click_reward`, límits `bandit` i paràmetres MMR/Slot-Gate) a un únic `dataclass` + JSON (`calypso_3_2/weights_profile.json`).
   - Eliminar els camps duplicats de `ConfigV4` perquè `CalypsoRecommenderV4`, `PositionDebiasedRanker` i `eval.py` llegeixin el **mateix vector de pesos** sense discrepàncies.
2. **Vector de *features* declaratiu orientat a compra (`FEATURE_NAMES` v2):**
   - Incloure al vector únic del re-ranker: `discount_pct`, `cheapness_norm` (`1 - lp_norm`, evitant problemes de signe al `np.clip`), `price_over_budget` (asimetria per sobre del preu de l'usuari), `user_aff_x_trend` i `ctx_trend`, eliminant termes manuals fora de `score_matrix`.
   - Afegir un flag `conversion_only: bool = True` (`click_weight = 0.0`) tant a `build_ips_training_dataset` com a `PositionAwareBandit`.
3. **Recarregat en calent (*hot-reload*) sense reiniciar el servei:**
   - `Stage4BackgroundUpdater` comprova al cicle de `15s` el *mtime* de `weights_profile.json`; si canvia i passa validació d'esquema, reconstrueix el `RecommenderSnapshotV4` i el commuta atòmicament a `StateStore` sense perdre ni una petició.
4. **Optimitzador automàtic offline + bucle en viu (`etapa_4/src/tuner.py`):**
   - Registrar mostres lleugeres `(request_id, segment, shown, features_matrix)` (patró ja implementat a `calypso_3_2/etapa_2/src/learner.py` amb `signals_*.jsonl`) i creuar-les amb `team_32.feedback`.
   - Llançar un *tuner* periòdic (Coordinate Search / Optuna / LogisticRegression IPS amb partició de validació per hash de `request_id`) que maximitzi **CVR / `hit_rate_buy@10` / SNIPS CVR** subjecte a latència `< 1.0 ms/req`. Només escriu un nou `weights_profile.json` si supera els pesos vigents en validació.
