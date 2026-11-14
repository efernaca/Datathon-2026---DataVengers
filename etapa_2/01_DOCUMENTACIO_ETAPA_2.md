# Etapa 2 · Recomanador Seqüencial Híbrid, Personalització Profunda i *Cold Start* Bidireccional

**Versió del model:** `etapa2-seq-mmr-v1`  
**Objectiu:** Fer un salt qualitatiu sobre el model base de l'Etapa 1 atacant les quatre fonts principals de conversió del simulador ocult: **intenció seqüencial real (amb correcció per popularitat *Lift*)**, **memòria profunda d'usuari (cistelles abandonades, marca i sensibilitat al preu)**, **descobriment actiu de productes nous (*Bidirectional Graph Grafting*)** i **diversificació conscient d'ingressos (*Revenue-Aware MMR*)**.

---

## 1. Millores Clau respecte a l'Etapa 1

| Component | Etapa 1 (`etapa1-hybrid-v1`) | Etapa 2 (`etapa2-seq-mmr-v1`) |
|---|---|---|
| **Transicions en sessió** | Co-ocorrència bruta `transitions[A]` | **Lift debiassat** ($s_{A\to B} / \text{pop}(B)^\beta$) + **Self-Conversion** ($A \to_{\text{self}} A$) + **Decaïment dual** (ordre + $\Delta t$ en segons) |
| **Memòria d'usuari** | Només llista de compres `user_bought` i afinitat bàsica | **`UserProfile` complet:** recuperació de `add_to_cart` / vistes repetides no comprades, afinitat de **marca**, ajust de **rang de preu ($\mu_{\log p}, \sigma_{\log p}$)** i supressió en viu des de `team_NN.feedback` |
| **Context (`recent_views = []`)** | 1 nivell `(region, device, page)` o `(page)` | **Suavitzat jeràrquic Empirical Bayes:** `(region, device, page)` $\to$ `(region, page)` $\to$ `("", "", page)` |
| **Cold Start (`new_product`)** | Només TF-IDF cap endavant (`new -> warm`) | **TF-IDF + TruncatedSVD + *Graph Grafting* bidireccional:** `new -> warm` quan l'usuari mira un ítem nou **i `warm -> new`** quan mira un ítem càlid afí |
| **Rànquing i Desempat** | Ordenació lineal simple | **Revenue Tie-Breaker** (preu relatiu a la subcategoria) + **MMR als slots 4–10** per evitar redundància i maximitzar probabilitat d'encert total |
| **Ús avançat de RawTree** | Només lectura i `submissions` | Persistència d'embeddings/veïns de *Cold Start* i telemetria a **`team_NN.scratch`** + lectura de compres en viu a **`team_NN.feedback`** |

---

## 2. Formulació Matemàtica dels Nous Components

### 2.1. Transicions amb *Lift* Suavitzat i Decaïment Temporal Dual
Per evitar que els productes ultra-populars dominin totes les llistes de transició (sense amplificar coincidències d'1 sol esdeveniment), cada aresta dirigida $(A \to B)$ es repondera amb un *Lift* amb suavitzat bayesià:
$$\text{LiftScore}(A \to B) = \frac{\text{co\_purchases}\cdot 6.0 + \text{co\_carts}\cdot 2.5 + \text{co\_views}\cdot 1.0}{\left(\text{weighted\_pop}(B) + 30.0\right)^{\beta}}, \qquad \beta = 0.45$$
A més, a `recent_views`, el pes de cada producte vist combina la posició inversa $k$ i el temps transcorregut $\Delta t_k$ respecte al `ts` de la petició:
$$w_k = 0.72^k \cdot \left(0.5 + 0.5\,\exp\left(-\frac{\Delta t_k}{900\text{ s}}\right)\right)$$

### 2.2. Recuperació d'Intenció i Elasticitat de Preu Personal
Dins de `shop.history_events`, si un usuari va fer `add_to_cart` o múltiples `view` d'un producte $i$ sense haver-lo comprat encara, entra a `profile.intent_items` amb pes proporcional a la força de la intenció. A més, si l'usuari té historial de compres, estimem la seva mitjana i desviació estàndard en log-preu $(\mu_u, \sigma_u)$ i apliquem un factor gaussià d'encaix de pressupost:
$$\text{PriceFit}(u, i) = \exp\left(-\frac{1}{2}\left(\frac{\ln(1 + p_i) - \mu_u}{\max(\sigma_u, 0.35)}\right)^2\right)$$

### 2.3. *Cold Start* Bidireccional (*Forward & Reverse Grafting*)
`ColdStartIndexV2` combina similitud cosinus de TF-IDF (precisió lèxica/marca) i projecció densa **TruncatedSVD** ($d \le 64$, similitud semàntica) juntament amb categoria, subcategoria, marca i preu:
* **Forward (`new -> warm`):** Si `recent_views` conté un producte nou, rep el perfil i les transicions dels seus $K=8$ veïns càlids més propers.
* **Reverse (`warm -> new`):** Quan un comprador mira un producte càlid $v$, injectem els productes nous (`is_new_product = true`) que són veïns propers de $v$ dins dels candidats, permetent que els productes introduïts durant la Ronda Final de diumenge rebin clics i compres des del primer minut.

### 2.4. Optimització d'Ingressos i MMR Adaptatiu
Un cop calculat el score de conversió base $S_{\text{conv}}(i)$, apliquem un multiplicador suau per optimitzar el criteri de desempat (`revenue_eur`):
$$S_{\text{rev}}(i) = S_{\text{conv}}(i) \cdot \left(1 + \alpha_{\text{rev}} \min\left(\frac{p_i}{\bar{P}_{\text{subcat}(i)}},\, 2.0\right)\right), \qquad \alpha_{\text{rev}} = 0.03$$
Finalment, apliquem selecció voraç **MMR** adaptada al context de la sessió:
* Quan l'usuari ja té `recent_views`, preservem intactes els **slots 1–5** i apliquem una diversificació suau només de marca ($\lambda_{\text{brand}} = 0.04$) a partir de l'slot 6 per no perdre rellevància dins la subcategoria activa.
* Quan `recent_views` és buit (entrada freda a `home`), preservem els **slots 1–3** i diversifiquem subcategoria ($\lambda_{\text{subcat}} = 0.15$) i marca ($\lambda_{\text{brand}} = 0.04$) als slots 4–10 per maximitzar el cobriment d'interessos.

---

## 3. Execució i Verificació

```powershell
# 1. Passar tots els tests de l'Etapa 2 (guardrails, cold start bidireccional, perfils, MMR, benchmark)
python -m unittest discover -s etapa_2/tests -v

# 2. Comparar offline Baseline vs Etapa 1 vs Etapa 2
python etapa_2/src/eval.py

# 3. Provar el bucle ràpid en local (--dry-run)
python etapa_2/src/serve.py --dry-run

# 4. Llançar l'Etapa 2 en producció contra RawTree (fent servir group_API.env -> team_32)
python etapa_2/src/serve.py
```
