<<<<<<< HEAD
# Datathon FME 2026 · DataVengers (Team 32) — RawTree Challenge

Sistema de recomanació en temps real per al **RawTree Real-Time Recommendation Challenge** ([enunciat original](datathon-fme-2026-main/README.md)), desenvolupat per **DataVengers (Team 32)** durant la Datathon FME 2026 (14–15 de novembre de 2026, UPC Barcelona).

Objectiu del repte: per a cada petició d'usuari procedent de `shop.requests`, seleccionar i retornar en **menys de 3 segons** un rànquing ordenat de **10 productes** del catàleg (~5.000 ítems) maximitzant clics, compres i ingressos en un flux en viu amb canvis de preus, productes nous (*cold-start*) i retroalimentació en temps real (`team_32.feedback`).

---

## Què hem acabat fent

El sistema final evoluciona des d'una arquitectura estadística lleugera amb correcció de biaix posicional fins a un motor híbrid d'aprenentatge seqüencial i neuronal en temps real blindat per a producció:

1. **Canal de dades en streaming i blindatge SLA (`< 3.000 ms`)**:
   - **Arrancada instantània (`< 0,4 s`)**: càrrega atòmica des de disc de l'historial estàtic (`static_history_v4.pkl`) i el darrer snapshot (`snapshot_v4_latest.pkl`) sense recalcular els ~12M d'esdeveniments històrics en reiniciar.
   - **Bucle dual (`FastLoop` + `BackgroundUpdater`)**: un fil ràpid serveix peticions amb consultes SQL estrictament de cua fresca (`now() - 25 min` amb límit de seguretat) i un fil en segon pla actualitza incrementalment estadístiques en viu (cada ~10 s) i reentrena pesos sense bloquejar la inferència.
   - **Enviament asíncron i deduplicació**: escriptura en segon pla cap a `team_32.submissions` i conjunt `_seen` actualitzat post-push per evitar reprocessar peticions o perdre respostes prop del límit de 3 segons.
2. **Motor de recuperació multicapa + *Cold-Start* bidireccional**:
   - Combina transicions seqüencials de sessió (`P(j|i)`), co-visitació, *SVD* de 32 dimensions, *EASE*, popularitat contextual (`device × region × page`) i afinitat d'usuari.
   - **Cold-Start**: enllaça productes nous sense historial mitjançant similitud TF-IDF + SVD (`forward_cold` i *reverse graft* cap a ítems calents) més compatibilitat per categoria, subcategoria, marca i preu.
3. **Regles de negoci i gestió modular (`weights_profile.json` + `tuner.py`)**:
   - Recàrrega en calent (*hot-reload*) de pesos i regles per a 5 segments d'usuari, amb senyals explícits per **carret actiu / carret oblidat** i **compres recurrents**.
   - Presets intercanviables (`conversion_first`, `calibrated_m3`, `product_profile_focus`, `cheap_boost`, etc.).
4. **Doble capçal i *Transfer Learning* al model final (`calypso_6_2_b`)**:
   - **Capçal Slot-1** dedicat a maximitzar la probabilitat de compra al primer lloc del carrusel (`64` features) + rànquing IPS/MMR per a les posicions 2–10.
   - **Destil·lació Teacher → Student**: `HistGradientBoostingClassifier` entrenat sobre compres i destil·lat a una xarxa **MLP `(80, 40)` en NumPy pur**, amb fase d'adaptació *Synthetic → Real* via *Ridge* sobre feedback real i velocitat de CVR en viu.

---

## Estructura neta del repositori

S'ha reorganitzat el repositori mitjançant `.gitignore` per mantenir exclusivament l'enunciat original del repte i les tres fites clau de l'evolució del model:

| Carpeta | Funció |
|---|---|
| [`datathon-fme-2026-main/`](datathon-fme-2026-main/) | Documentació oficial del repte, esquema de taules RawTree, *starter code* i notebook d'avaluació base. |
| [`calypso_2_1/`](calypso_2_1/) | **Fita base estable**: model estadístic lleuger encapsulat amb IPS ranker, bandit posicional UCB, MMR i blindatge operatiu de producció. |
| [`calypso_4_1_e/`](calypso_4_1_e/) | **Fita modular i regles de negoci**: separació total de pesos a [`weights_profile.json`](calypso_4_1_e/weights_profile.json), recàrrega en calent, gestió CLI amb [`tuner.py`](calypso_4_1_e/tuner.py), regles de carret/recurrència i protecció del `DCG@10`. |
| [`calypso_6_2_b/`](calypso_6_2_b/) | **Model final de producció**: doble capçal (*Slot-1 Precision* + posicions 2–10), 64 features, MLP `(80, 40)` destil·lat amb *Transfer Learning* en 2 fases i velocitat de conversió en viu. |

---

## Comparativa dels models conservats

| Aspecte | `calypso_2_1` | `calypso_4_1_e` | `calypso_6_2_b` |
|---|---|---|---|
| **Rol** | Baseline ràpid i estable | Arquitectura modular amb regles de negoci | Model final de producció |
| **Representació / Model** | Scoring lineal (14 features) + IPS logístic + Bandit UCB + MMR | Scoring modular per 5 segments d'usuari + IPS amb pesos signats + carret/recurrència | Doble capçal: *Slot-1* (64 features, MLP `(80, 40)` destil·lat + *Transfer Ridge*) + Rànquing 2–10 |
| **Senyals clau** | Seqüència 1r ordre, CVR propi, afinitat, popularitat, cold-start TF-IDF/SVD | `2.1` + detecció carret actiu/oblidat, recurrència, asimetria de preu i desempat configurable | `4.1.e` + `EASE`, SVD 32d, `hop2`, co-visitació, `cross_subcat_lift`, consens multicapa i velocitat CVR en viu |
| **Configuració i tuning** | Variables d'entorn / codi | [`weights_profile.json`](calypso_4_1_e/weights_profile.json) recarregable en calent + [`tuner.py`](calypso_4_1_e/tuner.py) | [`weights_profile.json`](calypso_6_2_b/weights_profile.json) ampliat + presets multi-objectiu + [`tuner.py`](calypso_6_2_b/tuner.py) |

---

## Resultats empírics verificats

> Les xifres següents provenen dels logs d'ablació guardats al repositori, del benchmark estratificat sobre `calypso-6.2b` i de la telemetria observada al panell en directe durant la competició. No es barregen escenaris diferents en un mateix rànquing perquè fan servir conjunts de sessió diferents.

### 1. Ablacions offline a 5 llavors (`calypso_2_1` vs `calypso_4_1 / 4.1_e`)

Mitjana sobre 5 llavors registrades a [`ablation_results.csv`](calypso_4_1_e/etapa_4/ablation_results.csv) i [`ablation_v4_results.csv`](calypso_4_1_e/etapa_4/ablation_v4_results.csv):

| Variant / Configuració | Font | `hit_buy` | `dcg_buy` | `cold_item` | Lectura principal |
|---|---|---:|---:|---:|---|
| `A_2.1_equivalente` | `ablation_results.csv` | `77,25%` | `0,4749` | `85,87%` | Baseline sòlid de `calypso_2_1` |
| `2.1 + penaliz_fondo` (`3.1b`) | `ablation_results.csv` | `79,04%` | `0,4835` | `85,76%` | Penalitzar fons repetit millora el hit |
| `B_4.1_completo` (`precio_asc=True`) | `ablation_v4_results.csv` | `78,72%` | `0,3941` | `82,45%` | Reordenar estrictament per preu destrueix `-18,8%` el `dcg_buy` |
| `4.1 - precio_asc` (base `4.1.e conversion_first`) | `ablation_v4_results.csv` | `78,72%` | `0,4799` | `82,45%` | Manté la conversió de `4.1` recuperant el `dcg_buy` |

### 2. Benchmark offline estratificat per presets (`calypso_6_2_b`, 1.200 sessions)

Avaluació comparativa sobre `calypso_6_2_b` amb 1.200 sessions estratificades (càlcul pur de motor ~`0,8–1,1 ms/req`):

| Preset | Objectiu compost | `ALL_hit_buy` | `ALL_dcg_buy` | `WARM_hit_buy` | `COLD_hit_buy` | `COLD_item_hit` | `SNIPS_s1` | Latència motor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `calibrated_m3` | `0,7223` | `75,05%` | `0,4485` | `75,23%` | `74,64%` | `80,49%` | `0,2422` | `1,076 ms` |
| `product_profile_focus` | `0,6922` | `70,32%` | `0,4191` | `65,44%` | `81,88%` | `93,90%` | `0,2344` | `0,807 ms` |
| `conversion_first` (actiu) | `0,6901` | `69,25%` | `0,4282` | `70,03%` | `67,39%` | `69,51%` | `0,2266` | `0,818 ms` |

### 3. Telemetria observada en viu (`calypso-6.2b` en producció)

Lectura directa del monitor del leaderboard sobre el flux real de RawTree per a `calypso-6.2b`:

| Versió en producció | Peticions respostes | `On-Time` (`<= 3 s`) | Latència `p50 / p95` | `CTR / CVR` | `EUR / req` |
|---|---:|---:|---:|---:|---:|
| `calypso-6.2b` | `534 (99,4%)` | `96,82%` | `1199 / 2274 ms` | `3,37% / 1,12%` | `0,2976 €` |

---

## Com executar i verificar

Cal definir les credencials de RawTree al fitxer `.env` (a partir de [`datathon-fme-2026-main/.env.example`](datathon-fme-2026-main/.env.example)) abans d'executar en viu. Per a verificació local de qualsevol dels tres models conservats (per exemple `calypso_6_2_b`):

```powershell
# Self-check del model final (sense connexió en viu requerida per als components locals)
python calypso_6_2_b/serve.py --self-check

# Dry-run i comparativa de presets
python calypso_6_2_b/serve.py --dry-run
python calypso_6_2_b/tuner.py --eval-presets

# Tests unitaris de l'etapa 4
python -m unittest discover -s calypso_6_2_b/etapa_4/tests -p "test_*.py" -v
python -m unittest discover -s calypso_4_1_e/etapa_4/tests -p "test_*.py" -v
```
=======
# 🏋️‍♂️ DataVengers — Datathon FME 2026 (UPC Barcelona)

**Sistema de recomanació en temps real per al [RawTree Challenge](https://datathon.rawtree.dev)**

📅 14–15 Novembre 2026 · UPC Barcelona

---

## 📋 Què és aquest projecte?

Un sistema automàtic que, cada vegada que un usuari entra a una botiga online, li **mostra 10 productes recomanats en menys de 3 segons**. L'objectiu és maximitzar les **comandes** (conversió a compra) en un entorn en temps real.

El sistema llegeix peticions de la base de dades compartida (`shop.requests`), analitza el comportament de l'usuari i genera un rànquing personalitzat de 10 productes que escriu a la taula `team_NN.submissions`.

---

## 🗂️ Estructura del projecte

```
Datathon-2026---DataVengers/
├── datathon-fme-2026-main/       # Challenge base (README, starter kit, queries)
├── calypso_2_1/                  # Model base 2.1 — Hybrid + Sequential + IPS Re-ranker
├── calypso_3_1/                  # Intent d'agressivitat (74.5% hit rate)
├── calypso_3_1b/                 # Millor versió abans de 4.1 (79.0% hit rate)
├── calypso_3_2/                  # Afinament final (en desenvolupament)
├── calypso_4_1/                  # Regles de negoci: carrito, recurrents, preu
├── calypso_4_1_e/                # ★ 4.1.e — Modular + Optimització iterativa (ACTIU)
├── calypso_4_2/                  # Versió experimental 4.2
├── calypso_5_1/                  # Versió experimental 5.1
├── calypso_6_2_b/                # 6.2b — Dual-Head + Transfer Learning (64 features)
├── odyseus/                      # Alternativa Odyseus
├── odyseus_2_0/                  # Odyseus v2
├── odyseus_3_0/                  # Odyseus v3
├── gestor_models/                # Manager per canviar/llançar models
├── Presentacion/                 # Materials de presentació
├── Informes/                     # Informes i anàlisis
├── documentacio_IA/              # Documentació IA
├── offline_eval_cache/           # Dades d'avaluació offline (parquet, JSON)
├── comparativa_models_llenguatge_natural.md  # Comparativa senzilla dels models
├── MEDIDAS_IMPLEMENTADAS.md      # Totes les mesures implementades (E1→4.1)
├── .active_model.json            # Model actiu actual (calypso 4.1.e)
├── model_manager.py              # Entry point del gestor
└── run_offline_eval_official.py  # Avaluació oficial offline
```

---

## 🧠 Evolució dels models Calypso

| Model | Hit Rate Buy@10 | Descripció |
|-------|-----------------|------------|
| **2.1** | 77.2% | Model base: transicions, auto-conversió, intenció, afinitat, preu, context, popularitat |
| **3.1** | 74.5% | 4 canvis agressius (va pitjor) |
| **3.1b** | **79.0%** | Només la penalització de fons funciona (+1.8%) |
| **4.1** | — | Regles de negoci: carrito actiu/oblidat, productes recurrents, ordre de preu |
| **4.1.e** | — | ★ **Model actiu**: arquitectura modular, perfils JSON, optimització iterativa |
| **6.2b** | — | Dual-head Slot-1, 64 features, transfer learning, CVR velocity |

> Consulta `calypso_*/README_*.md` i `MEDIDAS_IMPLEMENTADAS.md` per al detall de cada model.

---

## ⚡ Com s'executa?

### 1. Configura les credencials
```bash
cp group_API.env .env   # omple amb el TEAM_ID i RAWTREE_API_KEY
```

### 2. Model actiu
El model actiu es defineix a `.active_model.json`. Actualment és **calypso 4.1.e**.

Per canviar de model:
```powershell
python model_manager.py --set calypso_4_1_e
python calypso_4_1_e/serve.py --watchdog
```

### 3. Avaluació offline
```bash
python run_offline_eval_official.py
```

### 4. Optimització de pesos (4.1.e)
```bash
cd calypso_4_1_e
python tuner.py --optimize --iters 20 --save-best
python tuner.py --eval-presets
```

---

## 📊 Measures clau del sistema

- **Velocitat de resposta**: < 3 segons per petició
- **10 productes** per sol·licitud
- **5 segments d'usuari**: conegut_sensio, conegut_sense_sensio, nou_sensio, nou_sense_sensio, producte_nou
- **Hot-reload**: els pesos es poden canviar sense reiniciar el servei
- **Slot-gating**: slots 1-6 per precisió, slots 7-10 per exploració
- **Bandit UCB**: exploració equilibrada amb explotació
- **Guardrails**: sempre 10 IDs vàlids, sense repeticions de compra

---

## 🏆 Categories de l'avaluació

| Criteri | Punts | Descripció |
|---------|-------|------------|
| Mètode | 10 | Enfoquament fonamentalment justificat i validat offline |
| Aprenentatge del stream | 10 | Ús de feedback i esdeveniments en viu |
| Ús de RawTree | 5 | Consultes SQL, embeddings, pipeline |
| Comunicació | 5 | Entenedor per a un PM |

---

## 📜 Regles

- Tot generat per un simulador amb llavor secreta
- Feedback privat per equip
- Recomanacions per codi, no manualment
- Sistema sense vigilància
- Equips del top share el repositori

---

## 👥 Equip DataVengers

Equip de la **Facultat de Matemàtiques i Estadística (FME), UPC Barcelona**.

---

## 📚 Documentació addicional

- [Challenge base](datathon-fme-2026-main/README.md)
- [Comparativa de models](comparativa_models_llenguatge_natural.md)
- [Mesures implementades](MEDIDAS_IMPLEMENTADAS.md)
- [README Model 4.1.e](calypso_4_1_e/README_4_1_e.md)
- [README Model 6.2b](calypso_6_2_b/README_6_2_b.md)
>>>>>>> cbe71fec6826253cc30a383015bc169527273ef9
