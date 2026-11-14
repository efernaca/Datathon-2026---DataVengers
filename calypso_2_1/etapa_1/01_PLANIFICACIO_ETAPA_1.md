# Planificació Tècnica de l'Etapa 1: Infraestructura, Validació Offline i Arquitectura Base

**Projecte:** Datathon FME 2026 — RawTree Real-Time Recommendation Challenge  
**Objectiu de l'Etapa 1:** Construir un sistema de recomanació en temps real **desacoblat, resilient, testable offline sense biaix de popularitat i preparat per iterar models avançats** garantint un **0% de fallbacks** i una latència $p_{95} < 950\text{ ms}$ (molt per sota de l'SLA de $3.000\text{ ms}$).

---

## 1. Objectius i SLA d'Enginyeria de l'Etapa 1

El repte avalua tres pilars automàtics en la ronda final de diumenge (10:00–12:00 CET): **Conversió (50/70 punts)**, **Cold Start (10/70 punts)** i **Fiabilitat / Reliability (10/70 punts)**, amb desempat per **Ingrés Esperat (`revenue_eur`)**. Qualsevol resposta que superi els $3.000\text{ ms}$ des del `ts` de creació a `shop.requests`, contingui menys de 10 productes, tingui IDs duplicats o inclogui un sol producte `out_of_stock` / `is_active = false`, és descartada i substituïda pel *fallback* (puntuació = 0).

### 1.1. Pressupost de Latència (*Latency Budget*)

Considerant que el servidor de RawTree fa *polling* o mesura el temps des de la marca temporal `ts` de la petició i que hi arriben ~3 peticions/segon (~50 esdeveniments/segon a `shop.events`), establim un pressupost estricte per petició:

| Fase del Cicle | Pressupost Màxim ($p_{95}$) | Estratègia Tècnica |
|---|---|---|
| **Espera de Polling (`shop.requests`)** | $\le 250\text{ ms}$ | Bucle ràpid amb `sleep(0.20s)` adaptatiu (sense esperes innecessàries si s'han trobat peticions). |
| **Query HTTP de Lectura (`POST /v1/query`)** | $\le 250\text{ ms}$ | Connexió persistent (`requests.Session` + `HTTPAdapter` keep-alive TLS), lectura incremental per finestra `ts` + filtre `seen`. |
| **Inferència Local i Guardrails (lot de $\le 50$ reqs)** | $\le 15\text{ ms}$ total ($< 0.5\text{ ms/req}$) | Estructures $O(1)$ en memòria (diccionaris Python / arrays NumPy precomputats), zero SQL i zero I/O dins la ruta crítica. |
| **Inserció Batch (`POST /v1/tables/submissions`)** | $\le 250\text{ ms}$ | Enviament únic en bloc JSON per totes les peticions noves del cicle, amb `timeout` curt i reintent segur. |
| **Marge de Seguretat + Jitter de Xarxa** | $\sim 200\text{ ms}$ | **Latència total esperada $p_{95} \approx 750\text{–}950\text{ ms}$** (marge $> 2.000\text{ ms}$ respecte al límit de $3.000\text{ ms}$). |

### 1.2. Arquitectura *Zero-Fallback* (Fast Loop / Slow Loop)

Per evitar que cap consulta analítica pesada (co-compres, tendències o actualització de catàleg) bloquegi mai la resposta a `shop.requests`, el servei es divideix en dos plans d'execució totalment aïllats:

1. **Fast Serving Loop (Fil Principal):**
   - Llegeix únicament una referència atòmica a un objecte **`RecommenderSnapshot` immutable** en memòria.
   - Mai espera cap `Lock` de llarga durada ni fa consultes pesades a RawTree.
   - Si falla qualsevol component del scorer, salta directament a la capa de `guardrails` amb la llista prefiltrada d'actius (*graceful degradation* local).
   - Manté un buffer `seen` amb finestra limitada (`deque` + `set`, p. ex. últimes 20.000 peticions) per evitar fuites de memòria durant les 24h+ d'execució contínua.

2. **Slow Background Updater (Fil Secundari Daemon):**
   - Executa periòdicament les consultes de refresc contra RawTree a intervals diferenciats segons la volatilitat de cada dada:
     - **Cada $2\text{ s}$:** Estat d'estoc (`out_of_stock` / `back_in_stock`) i canvis de preu (`price_change`) a `shop.catalog_updates`.
     - **Cada $30\text{ s}$:** Nous productes (`new_product`), càlcul de veïns de Cold Start (`cold_start.py`) i detector de tendències en viu sobre `shop.events`.
     - **Cada $120\text{ s}$:** Refresc incremental de popularitat contextual i compres recents d'usuaris.
   - Construeix un nou `RecommenderSnapshot` complet en segon pla i fa un **intercanvi atòmic de punter** (`self._snapshot = new_snapshot`), garantint consistència absoluta sense bloquejar el fil principal.

---

## 2. Disseny de Dades i Consultes SQL en RawTree (Motor ClickHouse)

Les taules de RawTree són *schemaless* (columnes de tipus `Dynamic`). Per maximitzar la velocitat d'execució a ClickHouse i prevenir errors de tipus en `JOIN` o agregacions, totes les consultes de `etapa_1/sql/` segueixen aquests principis:

1. **Càsting explícit obligatori en JOINs i agregacions:** `product_id::String`, `user_id::String`, `session_id::String`, `price_eur::Float64`, `ts::DateTime64(3)`.
2. **Push-down de còmput pesat a RawTree:** En lloc de descarregar 12 milions de files de `shop.history_events` per xarxa, executem l'agregació i el `LIMIT K BY` directament dins RawTree, reduint el trànsit de xarxa a uns pocs megabytes de taules preagregades.
3. **Separació de 3 famílies de consultes:**
   - **EDA (`sql/eda_queries.sql`):** Validació empírica de les lleis del simulador (corba de *position bias* per slot 1..10, taxa de recompra per usuari, conversió per `page`/`device`/`region`, lift de transició `view -> purchase` i `add_to_cart -> purchase`).
   - **Features (`sql/feature_queries.sql`):**
     - *Catàleg consolidat:* `shop.catalog` fusionat mitjançant `argMax(..., ts)` amb `shop.catalog_updates` per obtenir preu actual, estat `is_active` i data d'entrada de nous productes.
     - *Matrius dirigides de sessió:* Parells $(A \to B)$ a `shop.history_events` distingint `view -> view` i `view/cart -> purchase` dins la mateixa sessió, filtrant productes inactius.
     - *Popularitat suavitzada (Empirical Bayes / Beta-Binomial):* Per evitar sobreajustar productes amb 1 vista i 1 compra ($100\%$ artificial), la taxa de conversió empírica es calcula com:
       $$\widehat{\text{CVR}}(i \mid c) = \frac{\text{purchases}_{i,c} + \alpha \cdot \bar{p}_c}{\text{views}_{i,c} + \alpha}$$
       on $\bar{p}_c$ és la taxa mitjana de conversió del context/categoria i $\alpha$ és la força del *prior* (p. ex. $\alpha = 20$).
     - *Historial de compres per usuari:* Conjunt de `product_id` ja comprats per cada `user_id` per aplicar la penalització de saturació (*repeat-purchase suppression*).
   - **Monitorització (`sql/monitoring.sql`):** Lectura contínua de `team_NN.feedback` per controlar `served_by`, `fallback_reason`, latència `p50`/`p95`, CTR, CVR i `revenue_per_request` per `model_version`.

---

## 3. Motor d'Avaluació Offline Fidedigne (`src/eval.py`)

El quadern original (`starter/offline_eval.ipynb`) té dues limitacions crítiques: només avalua clics (`clicked`) i no separa els casos de **Cold Start**. El motor de l'Etapa 1 corregeix això amb total rigor estadístic:

### 3.1. Partició Temporal Sense Fuites (*Zero Data Leakage*)
- Es pren una data de tall `CUT` (p. ex., els últims 7 dies de `shop.history_sessions` i `shop.history_events`).
- Totes les estadístiques, matrius de co-ocorrència i perfils d'usuari es construeixen **estrictament amb dades `< CUT`**.
- L'avaluació s'executa sobre una mostra representativa de `shop.history_sessions` amb `ts >= CUT`.

### 3.2. Formulació de Mètriques Alineades amb el Simulador

Sigui $\mathcal{D}_{\text{test}}$ el conjunt d'impressions de test, on per a cada petició $t$ tenim la llista mostrada pel sistema antic $S_t = (s_{t,1}, \dots, s_{t,10})$, la propensió $p_{t,1} = P_{\text{old}}(s_{t,1} \mid x_t)$ de l'slot 1, el producte clicat $c_t$, el producte comprat $b_t$, i l'ingrés $r_t$. Sigui $\pi(x_t) = (a_{t,1}, \dots, a_{t,10})$ la recomanació del nostre model:

1. **HitRate@10 (Click i Purchase):**
   $$\text{HR}_{\text{click}}@10 = \frac{\sum_{t: c_t \neq \varnothing} \mathbb{I}[c_t \in \pi(x_t)]}{|\{t : c_t \neq \varnothing\}|}, \qquad \text{HR}_{\text{buy}}@10 = \frac{\sum_{t: b_t \neq \varnothing} \mathbb{I}[b_t \in \pi(x_t)]}{|\{t : b_t \neq \varnothing\}|}$$
2. **Position-Weighted Recall@10 (DCG Proxy):**
   Pondera els encerts segons la posició dins el nostre rànquing ($1 / \log_2(\text{rank} + 1)$), reflectint que els primers slots reben molta més atenció.
3. **IPS i SNIPS (*Self-Normalized Inverse Propensity Scoring*) a l'Slot 1:**
   Per eliminar el biaix de popularitat del log històric, per a qualsevol recompensa $Y_t \in \{\mathbb{I}[c_t = s_{t,1}],\, \mathbb{I}[b_t = s_{t,1}],\, r_t \cdot \mathbb{I}[b_t = s_{t,1}]\}$:
   $$w_t = \frac{\mathbb{I}[a_{t,1} = s_{t,1}]}{\max(p_{t,1}, \epsilon)}$$
   $$\widehat{V}_{\text{IPS}}(\pi) = \frac{1}{N} \sum_{t=1}^N w_t Y_t, \qquad \widehat{V}_{\text{SNIPS}}(\pi) = \frac{\sum_{t=1}^N w_t Y_t}{\max\left(\sum_{t=1}^N w_t, 10^{-9}\right)}$$
   Reportem sempre **`slot1_matches`** ($\sum_t \mathbb{I}[a_{t,1} = s_{t,1}]$) i l'Error Estàndard de SNIPS per mesurar la fiabilitat de l'estimador.

### 3.3. Avaluació Estratificada (Overall, Warm, Cold Start)
Tota política es reporta desglossada en 3 segments per evitar millores que degradin el Cold Start:
- **`ALL`:** Totalitat del conjunt de validació.
- **`WARM`:** Usuaris coneguts (`is_new_user = false`) amb tots els productes de `recent_views` presents al catàleg previ a `CUT`.
- **`COLD_START`:** Petició amb `is_new_user = true` **o** on algun producte de `recent_views` hagi aparegut/estat creat després de `CUT` (`new_product`).

---

## 4. Arquitectura Modular i Contractes d'Interfície (`etapa_1/src/`)

Cada mòdul té una responsabilitat única i un contracte estable per permetre evolucionar cap a les Etapes 2 i 3 sense reescriure la infraestructura:

```text
etapa_1/
├── 01_PLANIFICACIO_ETAPA_1.md          # Aquest document
├── 02_PROPOSTES_MILLORA_FUTURES.md     # Full de ruta d'innovacions per Etapes 2-3
├── requirements.txt                    # Dependències Python
├── sql/
│   ├── eda_queries.sql                 # Anàlisi del simulador i biaixos
│   ├── feature_queries.sql             # Queries de producció per al Slow Loop
│   └── monitoring.sql                  # Salut en temps real (team_NN.feedback)
├── src/
│   ├── config.py                       # Configuració centralitzada i càrrega segura de .env
│   ├── rawtree_client.py               # Client HTTP amb pooling, retries i batch insert
│   ├── state.py                        # Snapshot immutable + intercanvi atòmic (StateStore)
│   ├── cold_start.py                   # Indexador TF-IDF + atributs per a productes nous
│   ├── guardrails.py                   # Garantia de 10 IDs únics, actius i en estoc
│   ├── recommender.py                  # Política híbrida Etapa 1 (co-ocorrència + context + cold start)
│   ├── eval.py                         # Avaluador offline (IPS/SNIPS) + simulador sintètic
│   └── serve.py                        # Orquestrador Fast/Slow Loop amb mode --dry-run
└── tests/
    └── test_stage1.py                  # Suite de verificació de guardrails, cold start i rendiment
```

### Contracte Principal de Recomanació
Qualsevol model present o futur implementa una interfície pura i sense efectes secundaris:

```python
def recommend(request: dict[str, Any], snapshot: RecommenderSnapshot) -> list[str]:
    """Retorna exactament 10 product_id vàlids, ordenats per probabilitat de compra esperada."""
```

Això garanteix que:
1. El **mateix codi exacte** s'executa a l'avaluador offline (`eval.py`) i al servidor en viu (`serve.py`).
2. Els **guardrails** actuen sempre com a última barrera determinista abans d'enviar el lot a RawTree.
3. Podem provar i verificar el rendiment, la memòria i la lògica sencera **abans de rebre la clau API**, gràcies al simulador sintètic integrat a `eval.py` i als tests unitaris.
