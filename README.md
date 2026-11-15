# Datathon FME 2026 · DataVengers (Team 32) — RawTree Challenge

> Motor de recomanació en temps real per al [**RawTree Real-Time Recommendation Challenge**](datathon-fme-2026-main/README.md) · **Datathon FME 2026** (14–15 de novembre de 2026, UPC Barcelona).

---

## Resum executiu

En un comerç electrònic amb ~5.000 productes repartits en 10 categories i 90 subcategories, cada petició entrant a `shop.requests` requereix retornar una llista ordenada de **10 recomanacions** a `team_32.submissions` en **≤ 3.000 ms**, adaptant-se en directe a canvis de preu (`shop.catalog_updates`), productes nous (*cold-start*) i retroalimentació real (`team_32.feedback`).

**DataVengers (Team 32)** hem dissenyat i portat a producció la família de recomanadors **Calypso**, evolucionant des d'un rànquer estadístic lleuger corregit per propensitat (`calypso_2_1`) cap a una arquitectura modular recarregable en calent (`calypso_4_1_e`) i culminant en un model híbrid de doble capçal amb *Transfer Learning* (`calypso_6_2_b`) capaç d'inferir en mil·lisegons i sostenir el flux de producció sota un SLA estricte.

---

## Arquitectura del sistema

```mermaid
flowchart LR
    subgraph RawTree["Stream RawTree (ClickHouse)"]
        H["shop.history_sessions / events\n(~2M sessions · ~12M esdeveniments)"]
        Q["shop.requests + shop.events\n+ catalog_updates (cua fresca)"]
        F["team_32.feedback\n(clics i compres reals)"]
    end

    subgraph Bg["BackgroundUpdater (asíncron · cada ~10 s)"]
        S["Snapshots atòmics en disc\nstatic_history_v4.pkl / snapshot_v4_latest.pkl"]
        T["Models multicapa\nSeq · Co-visit · EASE · SVD 32d · TF-IDF\n+ Teacher GBDT → MLP (80,40) + Ridge"]
        P["weights_profile.json\n(hot-reload)"]
    end

    subgraph Fast["FastLoop (inferència en temps real)"]
        C["Recuperació multicapa\n+ Cold-Start bidireccional"]
        R["Capçal Slot-1 (64 features)\n+ IPS / UCB / MMR (slots 2–10)\n+ Regles de negoci i guardrails"]
        A["Enviament asíncron + deduplicació _seen"]
    end

    H --> S --> T
    Q --> Bg
    F --> Bg
    T --> R
    P -.-> R
    Q --> C --> R --> A -->|"team_32.submissions (< 3 s)"| RawTree
```

### Pilars tècnics implementats

1. **Blindatge operatiu i compliment d'SLA (`< 3.000 ms`)**
   - **Arrencada instantània (`< 0,4 s`)**: càrrega atòmica des de disc de l'historial estàtic (`static_history_v4.pkl`) i el darrer snapshot (`snapshot_v4_latest.pkl`), evitant recalcular ~12M d'esdeveniments en cada reinici.
   - **Desacoblament `FastLoop` / `BackgroundUpdater`**: la inferència consulta únicament cua fresca recent mentre el fil en segon pla actualitza estadístiques en viu i reentrena sense bloquejar el servei.
   - **Escriptura asíncrona i deduplicació post-push**: l'enviament a `team_32.submissions` no frena la següent petició i marca `request_id` a `_seen` només després de confirmar l'entrega.

2. **Recuperació híbrida multicapa i *Cold-Start* bidireccional**
   - Combina transicions seqüencials (`P(j|i)`), co-visitació, factorització **EASE**, **SVD** (32 dimensions), popularitat contextual (`device × region × page`), afinitat d'usuari i tendència en directe.
   - **Cold-Start**: connecta productes nous sense historial mitjançant similitud TF-IDF + SVD (*forward cold* i *reverse graft* cap a ítems amb trànsit) i compatibilitat de categoria, subcategoria, marca i preu.

3. **Arquitectura modular i regles de negoci (`weights_profile.json`)**
   - Separació completa de pesos, llindars i paràmetres en un perfil JSON recarregable en calent sense reiniciar el procés ni perdre peticions, gestionat per CLI mitjançant `tuner.py`.
   - Segmentació en 5 perfils d'usuari, detecció de **carret actiu / carret oblidat**, **compres recurrents** i calibració de recompensa compra vs. clic per protegir el `DCG@10`.

4. **Doble capçal *Slot-1 Precision* + *Transfer Learning* (`calypso_6_2_b`)**
   - **Capçal dedicat al Slot 1 (`64` features)**: optimitzat específicament per a conversió al primer lloc del carrusel amb regularització de risc seqüencial, separat del rànquing IPS/MMR de les posicions 2–10.
   - **Destil·lació Teacher → Student i *Transfer Learning* en 2 fases**: `HistGradientBoostingClassifier` destil·lat a una xarxa **MLP `(80, 40)` en NumPy pur**, amb adaptació *Sintètic → Real* mitjançant *Ridge* sobre la capa oculta amb `team_32.feedback` real i senyal de **Live-CVR Velocity**.

---

## Estructura del repositori

El repositori s'ha depurat mitjançant [`.gitignore`](.gitignore) per conservar exclusivament l'enunciat del repte i les tres fites representatives de l'evolució del sistema:

| Ruta | Descripció |
|---|---|
| [`datathon-fme-2026-main/`](datathon-fme-2026-main/) | Documentació oficial del repte, esquema de taules RawTree, *starter code* i notebook d'avaluació base. |
| [`calypso_2_1/`](calypso_2_1/) | **Fita base estable (`calypso-2.1`)**: motor estadístic lleuger encapsulat en 4 etapes amb IPS ranker, bandit UCB, MMR i blindatge SLA. |
| [`calypso_4_1_e/`](calypso_4_1_e/) | **Fita modular (`calypso-4.1e`)**: desacoblament total de pesos a [`weights_profile.json`](calypso_4_1_e/weights_profile.json), recàrrega en calent, [`tuner.py`](calypso_4_1_e/tuner.py) i regles de negoci. |
| [`calypso_6_2_b/`](calypso_6_2_b/) | **Model final (`calypso-6.2b`)**: doble capçal *Slot-1 Precision*, 64 features, destil·lació MLP `(80, 40)` + *Transfer Ridge*, *Live-CVR Velocity* i vectorització FastLoop. |
| [`README.md`](README.md) · [`.gitignore`](.gitignore) | Documentació general del projecte i política d'exclusió d'artefactes temporals, caches i experiments intermedis. |

---

## Comparativa dels models conservats

| Dimensió | `calypso_2_1` (`calypso-2.1`) | `calypso_4_1_e` (`calypso-4.1e`) | `calypso_6_2_b` (`calypso-6.2b`) |
|---|---|---|---|
| **Rol evolutiu** | Baseline de producció estable | Arquitectura modular orientada a conversió | Model final de producció |
| **Motor de scoring** | Model lineal (14 features) + IPS logístic + Bandit UCB + MMR | Scoring modular per 5 segments d'usuari + IPS amb pesos signats | **Dual-Head**: Slot-1 (64 features, MLP `(80, 40)` + *Transfer Ridge*) + rànquing IPS/MMR (slots 2–10) |
| **Senyals principals** | Seqüència 1r ordre, CVR propi, afinitat, popularitat contextual, cold-start TF-IDF/SVD | Base `2.1` + carret actiu/oblidat, compres recurrents, asimetria de preu i control de `DCG@10` | Base `4.1.e` + `EASE`, SVD 32d, `hop2`, co-visitació, `cross_subcat_lift`, consens multicapa i *Live-CVR Velocity* |
| **Configuració i CLI** | `serve.py`, `monitor.py`, `eval.py` | `+` [`weights_profile.json`](calypso_4_1_e/weights_profile.json) i [`tuner.py`](calypso_4_1_e/tuner.py) (hot-reload) | `+` [`weights_profile.json`](calypso_6_2_b/weights_profile.json) ampliat, presets multi-objectiu i [`tuner.py`](calypso_6_2_b/tuner.py) |

---

## Resultats empírics

> Les taules següents separen explícitament cada protocol d'avaluació (ablacions offline a 5 llavors, benchmark estratificat per presets i telemetria en producció real) per garantir rigor comparatiu.

### 1. Ablacions offline a 5 llavors (`calypso_2_1` vs `calypso_4_1 / 4.1_e`)

Mitjana sobre 5 llavors procedent de [`ablation_results.csv`](calypso_4_1_e/etapa_4/ablation_results.csv) i [`ablation_v4_results.csv`](calypso_4_1_e/etapa_4/ablation_v4_results.csv):

| Configuració avaluada | Font | `hit_buy` | `dcg_buy` | `cold_item` | Conclusió clau |
|---|---|---:|---:|---:|---|
| `A_2.1_equivalente` | `ablation_results.csv` | `77,25%` | `0,4749` | `85,87%` | Baseline sòlid de la família `2.1` |
| `2.1 + penaliz_fondo` (`3.1b`) | `ablation_results.csv` | `79,04%` | `0,4835` | `85,76%` | Penalitzar fons repetit millora el hit i el DCG |
| `B_4.1_completo` (`precio_asc=True`) | `ablation_v4_results.csv` | `78,72%` | `0,3941` | `82,45%` | Ordenar estrictament per preu degrada `-18,8%` el `dcg_buy` |
| `4.1 - precio_asc` (base `4.1.e conversion_first`) | `ablation_v4_results.csv` | `78,72%` | `0,4799` | `82,45%` | Conserva el `hit_buy` de `4.1` i recupera el `dcg_buy` |

### 2. Benchmark offline estratificat per presets (`calypso_6_2_b`, 1.200 sessions)

Comparativa sobre el banc estratificat de 1.200 sessions de `calypso_6_2_b` (càlcul de motor al voltant d'`1 ms/req`):

| Preset | Objectiu | `ALL_hit_buy` | `ALL_dcg_buy` | `WARM_hit_buy` | `COLD_hit_buy` | `COLD_item_hit` | `SNIPS_s1` | Latència motor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `calibrated_m3` | `0,7223` | `75,05%` | `0,4485` | `75,23%` | `74,64%` | `80,49%` | `0,2422` | `1,076 ms` |
| `product_profile_focus` | `0,6922` | `70,32%` | `0,4191` | `65,44%` | `81,88%` | `93,90%` | `0,2344` | `0,807 ms` |
| `conversion_first` (actiu) | `0,6901` | `69,25%` | `0,4282` | `70,03%` | `67,39%` | `69,51%` | `0,2266` | `0,818 ms` |

### 3. Telemetria observada en producció real (`calypso-6.2b`)

Mètriques directes del panell en viu de RawTree durant l'execució de `calypso-6.2b`:

| Model en directe | Peticions respostes | `On-Time` (`≤ 3 s`) | Latència `p50 / p95` | `CTR / CVR` | `EUR / req` |
|---|---:|---:|---:|---:|---:|
| `calypso-6.2b` | `534 (99,4%)` | `96,82%` | `1199 / 2274 ms` | `3,37% / 1,12%` | `0,2976 €` |

---

## Posada en marxa i verificació

### Requisits previs

1. Python 3.10+ amb les dependències de l'etapa corresponent (`etapa_4/requirements.txt`).
2. Configurar les credencials d'accés a RawTree en un fitxer `.env` a partir de [`datathon-fme-2026-main/.env.example`](datathon-fme-2026-main/.env.example) per a execucions en viu.

### Ordres principals (exemple sobre `calypso_6_2_b` / `calypso_4_1_e`)

```powershell
# Verificació d'integritat local del model (self-check)
python calypso_6_2_b/serve.py --self-check
python calypso_4_1_e/serve.py --self-check
python calypso_2_1/serve.py --self-check

# Simulació local i avaluació/gestió de presets
python calypso_6_2_b/serve.py --dry-run
python calypso_6_2_b/tuner.py --eval-presets
python calypso_4_1_e/tuner.py --list-presets

# Execució en producció amb supervisió (watchdog) i monitoratge
python calypso_6_2_b/serve.py --watchdog
python calypso_6_2_b/monitor.py
```

---

## Equip · DataVengers (Team 32)

Projecte desenvolupat per l'equip **DataVengers (Team 32)** per al repte **RawTree** a la **Datathon FME 2026** (Facultat de Matemàtiques i Estadística, UPC).
