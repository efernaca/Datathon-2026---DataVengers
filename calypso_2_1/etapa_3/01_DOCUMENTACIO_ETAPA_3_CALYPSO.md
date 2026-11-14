# Etapa 3 · `calypso-2.1` — Recomanador Híbrid ML amb Bandits Conscients de Posició i Agent Autònom

**Versió del model:** `calypso-2.1`  
**Objectiu:** Fer evolucionar el motor heurístic seqüencial de l'Etapa 2 (`etapa2_seq_mmr`) cap a una arquitectura adaptativa de **bucle tancat (Closed-Loop ML + Bandits + Agent Scratch)** que aprèn directament dels eslogs d'exposició (`shop.history_sessions`) i del `feedback` en viu (`team_NN.feedback`), corregeix el biaix de posició (IPS), optimitza l'exploració *Cold Start* amb UCB i utilitza `team_32.scratch` com a memòria operativa de decisions.

---

## 1. Per què passem de l'Etapa 2 a `calypso-2.1`?

En el simulador de RawTree hi ha dues realitats que un model amb pesos fixos no pot capturar sol:

1. **Biaix de posició en les impressions històriques i en directe:** Els usuaris examinen el slot 1 molt més que el slot 10 (`P(exam | k) ≈ (k + 1)^{-0.75}`). Si entrenem o actualitzem popularitats sense ponderar per l'exposició del slot, sobrerecompensem productes que simplement apareixien primers.
2. **Canvis de règim en directe (*Concept Drift* i onades de *Cold Start*):** Durant l'avaluació final de diumenge (10:00–12:00 CET), el simulador pot injectar nous productes sense historial (`is_new_product = true`), canviar preus, marcar articles fora d'estoc o desplaçar la conversió per dispositiu/regió.

`calypso-2.1` resol tots dos problemes amb **quatre capes coordinades**:

| Capa | Mòdul | Funció principal | Pressupost de latència |
| :--- | :--- | :--- | :--- |
| **1. Recuperació & Fusió Seqüencial** | [recommender.py](./src/recommender.py) | Genera ~70–100 candidats d'alta recall (transicions Lift, `self_cvr`, intencions no comprades, *Reverse Cold Start Grafting* i Bayes contextual). | `< 0.20 ms` / petició |
| **2. Re-Ranker ML Debiasat (IPS)** | [reranker.py](./src/reranker.py) | Vectoritza **14 característiques** per candidat i puntua amb `w^T x_std + b` entrenat per `LogisticRegression` amb pesos IPS `(k+1)^{0.75}` + regularització cap a priors monòtons. | `< 0.25 ms` / petició |
| **3. Bandit UCB Conscient de Posició** | [bandits.py](./src/bandits.py) | Estima la taxa de conversió empírica per producte ponderada per examinació de slot i afegeix un bonus d'exploració UCB especialment calibrat per a productes *Cold Start*. | `< 0.08 ms` / petició |
| **4. Agent Autònom + `scratch`** | [agent_scratch.py](./src/agent_scratch.py) | En cada cicle del *Slow Loop*, diagnostica la salut del sistema (CTR, CVR, onada Cold Start, efectivitat *Reverse Graft*) i ajusta hiperparàmetres en calent desant la traça a `team_32.scratch`. | Només al *Slow Loop* (0 ms en *Fast Loop*) |

---

## 2. Les 14 característiques del `PositionDebiasedRanker`

Tots els candidats actius d'una petició es vectoritzen en una única matriu NumPy `(N_cand, 14)` i s'avaluen amb producte matricial `X_norm @ coef_ + intercept_`:

1. `seq_lift_score`: Afinitat amb els `recent_views` ponderada per decaïment temporal i correcció Bayesiana de popularitat (*Lift*).
2. `self_cvr_view`: Auto-conversió empírica si el producte ja forma part dels `recent_views` actuals.
3. `intent_score`: Senyal de cistella abandonada (`add_to_cart`) o múltiples vistes recents no comprades de l'usuari.
4. `user_affinity`: Afinitat històrica `(usuari, subcategoria, marca)`.
5. `price_compat`: Compatibilitat gaussiana amb la mitjana i desviació de preu logarítmic de l'usuari.
6. `ctx_bayes_score`: Popularitat contextual Bayesiana jeràrquica `(regió, dispositiu, pàgina)`.
7. `global_pop_norm`: Popularitat global ponderada (amb impuls de tendències en viu).
8. `live_trend_norm`: Activitat recent a `shop.events`.
9. `is_new_product`: Indicador binari `1.0` si és un article nou (*Cold Start*).
10. `reverse_graft_score`: Similitud semàntica TF-IDF + SVD des d'un ítem *warm* de sessió cap a aquest producte nou.
11. `forward_cold_score`: Puntuació transferida des de veïns *warm* quan el producte vist en sessió és nou o rar.
12. `log_price_norm`: Logaritme normalitzat del preu (capacitat de desempat per `revenue_eur`).
13. `same_top_subcat`: Coincidència exacta amb la subcategoria de l'últim ítem vist en la sessió.
14. `same_top_brand`: Coincidència exacta amb la marca de l'últim ítem vist en la sessió.

### Fusió segura ML + Priors monòtons (`ml_blend_alpha`)
Per evitar que un nombre baix d'impressions o soroll inicial inverteixi el signe d'un senyal clar (com `seq_lift_score` o `intent_score`), els pesos apresos per `LogisticRegression` amb ponderació IPS s'ancoren a un vector de **priors monòtons de domini**:
$$w_{\text{final}} = \alpha_{\text{ML}} \cdot w_{\text{IPS}} + (1 - \alpha_{\text{ML}}) \cdot w_{\text{prior}}$$
on $\alpha_{\text{ML}}$ escala suaument de `0.0` (0 mostres) fins a `0.65` (quan el sistema supera 500 interaccions positives en `feedback`/`history_sessions`).

---

## 3. Memòria operativa a `team_32.scratch`

`calypso-2.1` converteix `team_32.scratch` en una memòria estructurada amb quatre tipus de registres (`kind`):
- `agent_decision`: Decisions del `CalypsoAgent` (per exemple, augmentar `ucb_alpha` i `weight_reverse_cold_graft` quan detecta més del 5% de productes nous actius, o reduir `ml_blend_alpha` si detecta caiguda anòmala de CTR).
- `bandit_arm`: Estat agregat dels braços superiors del `PositionAwareBandit` (impressions efectives ajustades per posició, compres, clics i bonus UCB).
- `reranker_weights`: Pesos actuals `coef` i nombre de mostres amb què s'ha calibrat el `PositionDebiasedRanker`.
- `cold_start_neighbors`: Mapatges semàntics `new -> warm` perquè l'equip pugui auditar en directe quins productes nous hereten tràfic.

---

## 4. Transició sense temps d'inactivitat (*Zero-Downtime Handover*)

Com que RawTree selecciona l'última resposta per `request_id` dins la finestra de 3 segons i totes les versions escriuen amb `model_version`, la migració de l'Etapa 2 cap a `calypso-2.1` és neta:

1. **Verificar en local dins el projecte abans de tocar la terminal en producció:**
   ```powershell
   python -m unittest discover -s etapa_3/tests -v
   python etapa_3/src/eval.py
   python etapa_3/src/serve.py --dry-run
   ```
2. **En una segona terminal** (o aturant amb `Ctrl+C` el bucle anterior just quan acabi un lot), arrenca `calypso-2.1`:
   ```powershell
   python etapa_3/src/serve.py --poll-interval 0.20 --slow-interval 18.0
   ```
3. La primera vegada trigarà uns segons a construir l'*snapshot* inicial (SQL + TF-IDF/SVD + fit IPS del Re-Ranker) i a continuació respondrà cada lot en `< 1 ms/petició` sota `model_version = "calypso-2.1"`.
