# Propostes de Millora Futures (Etapes 2 i 3)

**Projecte:** Datathon FME 2026 — RawTree Real-Time Recommendation Challenge  
**Propòsit:** Recopilar, prioritzar i especificar tècnicament les evolucions del sistema un cop l'Etapa 1 estigui desplegada i monitoritzada contra les dades reals de RawTree.

---

## Resum Prioritzat de Millores

| Prioritat | Proposta | Impacte Esperat al Marcador | Dificultat / Risc | Moment d'Implementació |
|---|---|---|---|---|
| **P0** | **1. Matrius Seqüencials Multi-Event (`view` vs `cart` vs `buy`) + Time-Decay** | +15–25 pts Conversió | Baixa / Molt segur | Etapa 2 (Dissabte matí) |
| **P0** | **2. Cold Start Semàntic + *Graph Grafting* per `new_product`** | +6–9 pts Cold Start | Mitjana / Segur | Etapa 2 (Dissabte migdia) |
| **P1** | **3. Detector de Tendències en Viu i Elasticitat per Baixada de Preu** | +5–12 pts Conversió (Ronda Final) | Baixa / Segur | Etapa 2 (Dissabte tarda) |
| **P1** | **4. Re-Ranker ML (LightGBM / Logistic) entrenat amb `history_sessions` + `feedback`** | +10–15 pts Conversió | Mitjana | Etapa 3 (Dissabte vespre) |
| **P2** | **5. Optimització Multi-Objectiu (Expected Revenue) i Diversitat MMR** | Desempat + +3–5 pts | Baixa | Etapa 3 (Dissabte nit) |
| **P2** | **6. Contextual Bandits en Viu (Thompson Sampling sobre `team_NN.feedback`)** | +5–10 pts Stream Learning (Jurat + Viu) | Mitjana-Alta | Etapa 3 (Nit / Diumenge) |
| **Bonus** | **7. Bucle Agèntic Autònom sobre `team_NN.scratch` (Side Challenge)** | Premi Side Challenge + 5 pts RawTree | Baixa-Mitjana | Dissabte tarda – Diumenge |

---

## Proposta 1: Matrius de Transició Seqüencial Multi-Event i Decaïment Temporal

### Diagnòstic del Problema
El *baseline* de co-vistes només compta si dos productes apareixen a la mateixa sessió, tractant igual un `view` passatger que un `add_to_cart` o una compra (`purchase`). A més, no distingeix si el producte vist actualment porta a comprar **el mateix producte** (retenció en sessió) o un **producte complementari** (cross-sell).

### Solució Tècnica
1. **Triple Senyal dins de `shop.history_events`:**
   Calcular per a cada parell dirigit $(A \to B)$ dins una mateixa `session_id` amb $t_A \le t_B$:
   - **Re-engagement / Direct Conversion:** Quina probabilitat hi ha que l'usuari compri el mateix producte $A$ que acaba de veure? ($P(\text{buy } A \mid \text{recent\_view } A)$). Si el simulador premia mostrar el producte que l'usuari ja estava considerant comprar, col·locar els últims `recent_views` (no comprats encara) als primers slots pot disparar la conversió.
   - **View-to-Purchase ($A \to_{\text{buy}} B$):** Probabilitat condicionada que després de veure $A$, el comprador acabi comprant $B$.
   - **Cart-to-Purchase / Substituts ($A \leftrightarrow B$):** Productes de la mateixa subcategoria i rang de preu similar amb alt *Lift*:
     $$\text{Lift}(A \to B) = \frac{P(B \mid A)}{P(B)^\beta}, \quad \beta \in [0.5, 0.8]$$
     El factor $P(B)^\beta$ evita que productes ultra-populars dominin totes les associacions i fa aflorar associacions específiques de nínxol.
2. **Decaïment temporal dins la sessió i del passat:**
   - Dins `recent_views`, ponderar per recència temporal real ($e^{-\lambda \Delta t}$ en minuts) i posició ($0.75^k$).
   - Al càlcul històric a RawTree, ponderar més els últims 14 dies que els primers 45 dies (`exp(-dateDiff('day', ts, now) / 21)`).

---

## Proposta 2: Cold Start Avançat mitjançant *Graph Grafting* (Empelt de Graf)

### Diagnòstic del Problema
A la ronda final de diumenge (10:00–12:00 CET) apareixen **nous productes (`new_product`)** sense cap historial previ a `shop.history_events` ni `shop.history_sessions`, i **nous compradors (`is_new_user`)**. Això val **10 dels 70 punts** automàtics.

### Solució Tècnica
1. **Representació Multimodal Híbrida del Catàleg:**
   - Combinar text (`description` + `brand` + `subcategory`) mitjançant TF-IDF de paraules i n-grames de caràcters (robust a paraules clau de marca/material) amb **TruncatedSVD** (64–128 dimensions) + preu log-normalitzat ($\log(1 + \text{price\_eur})$).
   - Escriure aquests embeddings a **`team_NN.scratch`** per complir també el criteri d'avaluació d'ús avançat de RawTree.
2. **Graph Grafting (Transferència d'Associacions):**
   - Quan el *Slow Loop* detecta un `new_product` $p_{\text{new}}$ a `shop.catalog_updates`:
     1. Troba els $K=5$ productes històrics més similars $\{q_1, \dots, q_5\}$ dins la mateixa `subcategory` (o `category` si no n'hi ha prou).
     2. **Enllaç sortint (si el comprador mira $p_{\text{new}}$):** Recomana el mateix producte $p_{\text{new}}$ (si és rellevant) juntament amb la combinació ponderada de les llistes de co-compra de $\{q_1, \dots, q_5\}$.
     3. **Enllaç entrant (quan es recomana $p_{\text{new}}$):** Si $p_{\text{new}}$ surt amb un descompte o pertany a una subcategoria en tendència, hereta temporalment un *prior* de conversió basat en la mitjana de $\{q_1, \dots, q_5\}$ i s'injecta amb un **bonus d'exploració controlada** (p. ex. slot 4–6) en usuaris afins a aquella subcategoria.

---

## Proposta 3: Detector de Tendències en Viu i Elasticitat de Preu

### Diagnòstic del Problema
A la Ronda Final el simulador introdueix **tendències noves** visibles únicament a `shop.events` (~50 esdeveniments/segon) i **canvis de preu** a `shop.catalog_updates`. Un model estàtic basat només en els 60 dies històrics reaccionarà massa tard.

### Solució Tècnica
1. **Score de *Momentum* en Finestres Mòbils (`shop.events`):**
   Cada 20–30 segons, consultar en SQL l'acceleració recent:
   $$\text{Velocity}(i) = \frac{\text{events}_{15\text{m}}(i) + 1}{\max(\text{avg\_events}_{15\text{m\_prev\_3h}}(i),\, 0.5)}$$
   ponderant més `purchase` ($\times 5$) i `add_to_cart` ($\times 3$) que `view` ($\times 1$).
2. **Detecció de Baixades de Preu (`price_change`):**
   Mantenir el `initial_price` i el `current_price` a l'estat en memòria. Quan $\text{discount}(i) = \frac{p_{\text{prev}} - p_{\text{curr}}}{p_{\text{prev}}} > 0.05$, aplicar un multiplicador d'elasticitat al score de compra (calibrat en EDA mirant com responen els usuaris a diferents preus dins la mateixa categoria i per `region` / `device`).

---

## Proposta 4: Re-Ranker Supervisat (LightGBM / Logistic Regression) amb Correció de Biaix

### Diagnòstic del Problema
Combinar heurísticament pesos (`w_co_buy`, `w_context`, `w_trend`, `w_price`) és subòptim perquè la importància relativa canvia segons el `page` (`home` vs `product` vs `category`) o si la sessió té `recent_views` buit.

### Solució Tècnica
1. **Arquitectura en 2 Fases (Retrieve $\to$ Re-Rank):**
   - **Fase 1 (Candidate Retrieval — $< 2\text{ ms}$):** Seleccionar els ~60–100 millors candidats únics d'entre:
     1. Últims `recent_views` + veïns de `co_purchase` i `co_view`.
     2. Productes similars per *Cold Start* (si hi ha `new_product`).
     3. Top populars del context `(region, device, page)` i de les subcategories vistes/històriques de l'usuari.
     4. Top tendències en viu (`shop.events`) i productes amb descompte recent.
   - **Fase 2 (Vectorized Scorer — $< 8\text{ ms}$ per a 100 candidats):**
     Un model lleuger (Regressió Logística L2 o LightGBM amb $\le 100$ arbres i profunditat $\le 5$, compilat/executat via arrays NumPy o `treelite` / inferència matricial directa) que prediu $P(\text{purchase} \mid u, i, \text{context})$.
2. **Features Clau del Re-Ranker:**
   - *Match directe:* El producte $i$ és dins `recent_views`? Quants cops? Fa quants segons?
   - *Associació:* Scores màxim i mitjà de `co_purchase`, `co_view` i `cold_sim` respecte als `recent_views`.
   - *Afinitat d'Usuari:* L'usuari ha comprat ja aquesta `category` / `subcategory` / `brand`? Distància del preu al preu mitjà històric de l'usuari (`price_zscore_user`). **Ja l'ha comprat abans?** (penalització forta).
   - *Context & Preu:* `page` (`home`, `product`, `category`), `device`, `region`, `price_eur`, `discount_pct`, popularitat contextual i *velocity* en viu.
3. **Entrenament sense biaix de posició:**
   Entrenar sobre `shop.history_sessions` + `team_NN.feedback` utilitzant **Inverse Propensity Weighting (IPW)** o incloent la posició `slot` com a variable durant l'entrenament i fixant `slot = 1` en inferència (*Position-Bias Debiasing*).

---

## Proposta 5: Optimització d'Ingressos (`Expected Revenue`) i Diversitat Intra-Llista (MMR)

### Diagnòstic del Problema
L'empat es resol per **ingrés esperat per petició (`revenue_eur`)** i el comprador pot ignorar 10 productes quasi idèntics de la mateixa marca o subcategoria si cap d'ells li encaixa.

### Solució Tècnica
1. **Rànquing conscient del marge/preu:**
   Un cop estimada $\hat{p}_i = P(\text{purchase} \mid u, i, c)$, ordenar els candidats per una utilitat lleugerament ajustada per preu sense sacrificar taxa de conversió:
   $$\text{Score}(i) = \hat{p}_i \cdot (1 + \alpha_{\text{rev}} \cdot \min(\text{price\_eur}_i / \bar{P}_{\text{cat}},\, 2.0))$$
   amb $\alpha_{\text{rev}}$ petit (p. ex. $0.03\text{–}0.08$, validat offline amb SNIPS Revenue vs SNIPS CVR).
2. **Selecció Seqüencial tipus MMR (*Maximal Marginal Relevance*) per als slots 4–10:**
   - Slots 1–3: Màxima probabilitat pura de conversió (màxima atenció del comprador).
   - Slots 4–10: Aplicar una penalització suau si ja hem inclòs més de 2 productes de la mateixa `subcategory` o `brand`, augmentant la probabilitat que almenys 1 dels 10 productes capturi la intenció del comprador.

---

## Proposta 6: Aprenentatge en Viu amb Contextual Bandits sobre `team_NN.feedback`

### Diagnòstic del Problema
La Ronda Final dura 2 hores (10:00–12:00 CET) amb **usuaris nous, productes nous i tendències noves** i el leaderboard públic està apagat. Durant aquestes 2 hores rebem **~21.600 files de feedback privat** (`team_NN.feedback`) en quasi temps real (~5 segons després de cada petició).

### Solució Tècnica
1. **Position-Aware Thompson Sampling per a Nous Productes i Tendències:**
   Per a cada producte $i$ (especialment `new_product` i ítems en tendència), mantenir una distribució Beta actualitzada en línia a partir de `team_NN.feedback`:
   $$\alpha_i \leftarrow \alpha_i + \mathbb{I}[\text{purchased}=i] + 0.2\,\mathbb{I}[\text{clicked}=i], \qquad \beta_i \leftarrow \beta_i + \sum_{k=1}^{10} \mathbb{I}[\text{shown}_k = i \land \text{clicked} \neq i] \cdot \gamma_k$$
   on $\gamma_k \in (0, 1]$ és el factor d'examen de l'slot $k$ (mesurat a l'EDA de `shop.history_sessions`, p. ex. $\gamma_1 = 1.0, \gamma_2 = 0.65, \dots, \gamma_{10} = 0.15$). Això evita penalitzar un producte només perquè ha aparegut a l'slot 9 i ningú l'ha vist.
2. **A/B Testing / Canary Automàtic en Dissabte:**
   Durant la jornada de pràctica de dissabte, assignar un petit percentatge del trànsit (o alternar versions) etiquetant `model_version` (p. ex. `etapa2-seq-v1` vs `etapa2-seq-v2`) i comparar `conversion_pct` i `revenue_per_request` directament a `team_NN.feedback` sense dependre únicament dels 10 minuts del leaderboard.

---

## Proposta 7: Estratègia per al Side Challenge (*"Best Agentic Use of RawTree"*)

Les bases inclouen un premi específic per a l'ús agèntic de RawTree. Proposem construir un **Agent Analista i Auto-Calibrador** separat del *Fast Loop* (per no posar mai en risc la latència de 3s):
1. Cada 5–10 minuts, l'agent consulta `team_NN.feedback` i `shop.events` via RawTree SQL/MCP.
2. Detecta anomalies o segments febles (p. ex., *"caiguda de conversió a `page='home'` en `device='ios'`"* o *"nous productes amb 0 clics"*).
3. Escriu taules de configuració dinàmica, pesos del model, clusters de tendències i registres d'experiments a **`team_NN.scratch`** amb un camp `experiment_id` i `reasoning`.
4. El *Slow Loop* del recomanador llegeix automàticament els nous pesos validats de `team_NN.scratch` (amb *rollback* automàtic si el CVR en `team_NN.feedback` baixa respecte a la finestra anterior).

---

## Criteris d'Acceptació abans de Pujar qualsevol Millora a Producció

Cap proposta d'aquesta llista passarà al servidor en viu sense complir 3 filtres:
1. **Validació Offline:** Millora estadísticament consistent en **SNIPS Purchase CVR** i **HitRate@10** a `eval.py` tant en `ALL` com en `COLD_START`.
2. **Benchmark de Latència:** Temps d'inferència local $< 2\text{ ms/petició}$ en lots de 50 peticions.
3. **Zero Regressió de Fiabilitat:** $100\%$ de respostes amb 10 IDs distints, actius i en estoc sota tests de càrrega i errors simulats.
