# Comparativa de Models i Afinament de Paràmetres: Calypso 6.2b

> **Data:** 15 de novembre de 2026  
> **Model base:** `calypso_6_2_b` (`model_version = "calypso-6.2b"`)  
> **Objectiu:** Comparativa empírica de rendiment modificant únicament els paràmetres més rellevants per maximitzar la conversió i el cold-start abans de la Ronda Final (10:00–12:00 CET).

---

## 1. Recompte de Paràmetres de Calypso 6.2b

El model `calypso_6_2_b` disposa exactament de:

* **190 paràmetres directes de configuració** (definits com a camps estructurats a la dataclass `WeightsProfile` i al fitxer `weights_profile.json`).
* **309 valors escalars totals** si es desglossen els vectors, tuples i matrius internes (com `min_prior_ratios` de 64 dimensions, `position_discount_curve`, vectors de ponderació per segment d'usuari i per tipus de pàgina).
* **67 hiperparàmetres actius** dins l'espai d'optimització acotat (`OPTIMIZABLE_PARAMS` a `tuner.py`).

---

## 2. Definició dels Tres Models Comparats

Per avaluar l'impacte real dels canvis, s'han definit tres configuracions que actuen sobre els eixos d'arquitectura més crítics (pesos prior, bloqueig de posicions, penalització de fons, afinació de vendes i exploració cold-start):

### Model 1: `conversion_first` (Base de Referència per defecte)
* **Concepte:** Prioritat a la conversió històrica amb conservadorisme: bloqueig de carret actiu als primers slots (`v4_cart_priority = True`), penalització del fons de catàleg (`v3_bg_penalty = True`) i forta dependència del rànquing de vendes històric (`sales_rank_blend_weight = 0.95`).
* **Paràmetres clau:**
  * `w_global_pop = 0.80`
  * `w_self_cvr_view = 2.2702`
  * `sales_rank_blend_weight = 0.95`
  * `v4_cart_priority = True`
  * `v3_bg_penalty = True`
  * `bandit_ucb_alpha = 0.12`

### Model 2: `calibrated_m3` (Conversió Calibrada sense Bloqueig de Slots)
* **Concepte:** Inspirat en les mesures empíriques de conversió. Elimina les restriccions rígides que forcen productes de carret als slots 1–3 i suprimeix la penalització artificial de fons (`v3_bg_penalty = False`), deixant que el model predigui la millor combinació per puntuació natural. Suavitza lleugerament el biaix de supervendes perquè les transicions de sessió tinguin espai.
* **Paràmetres modificats respecte a la Base:**
  * `v4_cart_priority = False` (el carret competeix per puntuació mitjançant `cart_active_score_bonus = 1.55`)
  * `v3_bg_penalty = False` (sense penalització de `0.48 / 0.85`)
  * `w_global_pop = 0.72` (baixa de `0.80` per no tapar els ítems de sessió)
  * `w_self_cvr_view = 2.30` (puja de `2.27` per afavorir auto-conversió de productes vistos)
  * `sales_rank_blend_weight = 0.90` (suavitza la dependència exclusiva dels més venuts)

### Model 3: `product_profile_focus` (Afinitat Semàntica i Cold-Start Avançat)
* **Concepte:** Dissenyat per sobresortir en la Ronda Final quan entren usuaris nous i productes nous. Amplia l'herència de veïns de contingut (TF-IDF + TruncatedSVD), obre la porta a l'exploració UCB controlada des del slot 5 en sessions WARM i bonifica fortament la coincidència de subcategoria i marca.
* **Paràmetres modificats respecte a la Base:**
  * `w_forward_cold = 0.62` (puja de `0.52`: major pes al pont de veïns per a productes nous)
  * `w_reverse_graft = 0.46` (puja de `0.38`: recomana productes nous afins quan es mira un ítem conegut)
  * `w_same_top_subcat = 0.26` (puja de `0.1856`: reforça la intenció de subcategoria)
  * `w_same_top_brand = 0.14` (puja de `0.1101`)
  * `cold_explore_start_slot_warm = 5` (avança l'exploració del slot 7 al slot 5)
  * `max_cold_explore_in_warm = 3` (permet fins a 3 candidats cold validats)
  * `bandit_ucb_alpha = 0.16` (puja de `0.12`: exploració d'incertesa més activa)
  * `cold_subcat_match_bonus = 0.48` (puja de `0.38`)

---

## 3. Taula Comparativa de Paràmetres Clau

| Paràmetre | `conversion_first` (Base) | `calibrated_m3` | `product_profile_focus` | Impacte Funcional |
| :--- | :---: | :---: | :---: | :--- |
| `v4_cart_priority` | `True` | **`False`** | `True` | Si és `False`, el carret no segresta els slots 1–3 de forma rígida |
| `cart_active_score_bonus` | `1.60` | **`1.55`** | `1.60` | Bonificació de competició per puntuació del carret |
| `v3_bg_penalty` | `True` | **`False`** | `True` | Si és `False`, elimina multiplicadors artificials de descompte |
| `w_global_pop` | `0.80` | **`0.72`** | `0.80` | Pes de la popularitat global de vendes |
| `sales_rank_blend_weight` | `0.95` | **`0.90`** | `0.95` | Ponderació de la corba de rànquing de vendes |
| `w_self_cvr_view` | `2.2702` | **`2.3000`** | `2.2702` | Reforç a recomanar el producte que l'usuari ja està mirant |
| `w_forward_cold` | `0.52` | `0.52` | **`0.62`** | Transferència de puntuació de veïns càlids a productes nous |
| `w_reverse_graft` | `0.38` | `0.38` | **`0.46`** | Injecció de productes nous semblants a la sessió activa |
| `w_same_top_subcat` | `0.1856` | `0.1856` | **`0.2600`** | Coherència amb la subcategoria de l'última vista |
| `cold_explore_start_slot_warm`| `7` | `7` | **`5`** | Slot a partir del qual s'introdueixen candidats nous |
| `max_cold_explore_in_warm` | `2` | `2` | **`3`** | Màxim de productes nous explorats en sessions conegudes |
| `bandit_ucb_alpha` | `0.12` | `0.12` | **`0.16`** | Coeficient de bonificació per incertesa UCB |

---

## 4. Resultats Empírics del Benchmark Offline

Avaluació estratificada sobre **1.200 sessions de test** (distribuint compradors coneguts, nous usuaris i productes acabats d'arribar) amb el motor d'avaluació estandarditzat de Calypso 6.2b:

| Model | Objectiu Combinat (`obj`) | HitRate@10 Total (`ALL`) | DCG@10 Compra | HitRate@10 (`WARM`) | HitRate@10 (`COLD START`) | SNIPS CVR Slot-1 | Latència Mitjana |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`product_profile_focus`** | **`0.71874`** | **`0.7441` (74.4%)** | **`0.4463`** | `0.7095` (71.0%) | **`0.8261` (82.6%)** | **`0.2709`** | **0.81 ms** |
| **`calibrated_m3`** | `0.71308` | `0.7333` (73.3%) | `0.4436` | **`0.7339` (73.4%)** | `0.7319` (73.2%) | `0.2492` | 1.08 ms |
| **`conversion_first` (Base)**| `0.69535` | `0.7032` (70.3%) | `0.4296` | `0.7278` (72.8%) | `0.6449` (64.5%) | `0.2541` | 0.82 ms |

> *Nota de referència:* La variant `cheap_boost` (que prioritza productes barats) assoleix un `obj = 0.67715` i `DCG = 0.4077`, confirmant que forçar preus baixos perjudica el rànquing top-1 respecte als models orientats a rellevància pura.

---

## 5. Quin Model Ofereix Millor Rendiment? (Anàlisi i Veredicte)

### El guanyador global és **`product_profile_focus`** (amb suport directe de **`calibrated_m3`**):

1. **Màxima puntuació global (`obj = 0.71874` vs `0.69535` de la base):**
   - Aconsegueix un **74.41% d'encert a compra** al top-10 (`+4.09%` per sobre de `conversion_first`) i el millor **SNIPS CVR al slot 1 (`0.2709`)**.
2. **Rendiment extraordinari en Cold Start (`82.61%` vs `64.49%` de la base):**
   - La millora de `+18.12%` en el segment fred és decisiva: a la **Ronda Final de les 10:00 CET**, l'organització canvia el flux cap a compradors i productes nous que ningú ha vist. Aquest segment val **10 punts directes** a la nota final.
3. **Equilibri WARM:**
   - Si bé `calibrated_m3` obté la millor taxa en usuaris recurrents coneguts (`WARM = 73.39%` gràcies a desblocar les posicions de carret), `product_profile_focus` manté un excel·lent `70.95%` en WARM alhora que dispara el rendiment dels slots 1–5 en productes afins.
4. **Eficiència extrema:**
   - La inferència és la més ràpida del banc de proves (**0.81 ms/petició**), deixant més de 2.000 ms de marge respecte a l'SLA de 3.000 ms.

---

## 6. Com Activar o Provar aquests Models en Viu

Calypso 6.2b disposa de **recàrrega en calent (*hot-reload*)**, per la qual cosa es pot canviar de preset o ajustar paràmetres sense reiniciar el servei ni perdre cap petició a la cua:

```powershell
# 1. Comprovar els presets existents
python calypso_6_2_b/tuner.py --list-presets

# 2. Carregar el model guanyador (product_profile_focus)
python calypso_6_2_b/tuner.py --preset product_profile_focus

# 3. O carregar la variant calibrada M3
python calypso_6_2_b/tuner.py --preset calibrated_m3

# 4. Validar el servei localment en mode dry-run
python calypso_6_2_b/serve.py --self-check
python calypso_6_2_b/serve.py --dry-run
```
