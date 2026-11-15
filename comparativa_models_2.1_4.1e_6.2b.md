# Comparativa de Models: 2.1, 4.1.e, 6.2.b

## Models Calypso de DataVengers - Datathon FME 2026

---

## Resum Ràpid

| Aspecte | **2.1** | **4.1.e** | **6.2.b** |
|---------|---------|-----------|-----------|
| **Hit Rate Buy@10** | 77.2% | ~79%+ (tunnejat) | En validacio |
| **Arquitectura** | Monolitica (serve.py) | Modular (JSON hot-reload) | Dual-head (Slot-1 dedicat) |
| **Features** | 14 principals | 14 + segment multipliers | **64 features** d'alta resolucio |
| **ML** | Re-ranker logistic IPS | Re-ranker IPS + bandit UCB | MLP (80,40) destil·lat de GradientBoosting |
| **Cold Start** | TF-IDF + SVD | TF-IDF + SVD parametritzat | SVD 32 dims + similitud avançada |
| **Transfer Learning** | No | No | **Si** (Sintetic -> Real, 2 fases) |
| **Optimitzacio** | Manual (poc a poc) | Iterativa automatica (tuner.py) | Multi-source IPS + consensus |
| **Pesos** | Hardcoded al codi | **JSON hot-reloadable** (130+ params) | JSON amb presets multiples |
| **Live CVR** | No | No | **Velocity CVR en viu** |
| **EASE** | No | No (0.0) | **Si** (0.32) + covisit + hop2 |
| **Carrito regles** | No | **Si** (actiu + oblat + recurrents) | No (desactivades per defecte) |
| **Diversity (MMR)** | Si (slots 1-5 / 3 home) | Si (lambdas configurables) | Si (mes agressiu) |
| **Speed** | ~1.5-2ms/req | ~2ms/req | **< 2.8ms/req** (FastLoop) |

---

## 2.1 - El Punt de Partida (77.2%)

### Qu es?
El primer model amb intel-ligencia artificial real de l'equip. Base de totes les iteracions posteriors.

### Senyals utilitzades
Cada producte rep un **puntatge** basat en:

| Senyal | Pes | Descripcio |
|--------|-----|------------|
| Transicions seqüencials | Alt | Si vas per A -> compres B, pesa molt |
| Auto-conversio | Alt | Si ja has vist el producte i en vols mes |
| Intencio | Moderat | Carrets abandonats o moltes vistes sense comprar |
| Afinitat històrica | Baix | Subcategoria i marca preferida |
| Preu adequat | Baix | S'ajusta al pressupost habitual |
| Context | Moderat | Què compra gent de la teva region/dispositiu |
| Popularitat global | Baix | Què es ven mes |
| Tendencies en viu | Moderat | Què esta de moda ara |
| Productes nous | Molt petit | Bonus minim |
| Similaritat de contingut | Moderat | Veins tematics (TF-IDF) |

### Com tria els 10 productes
1. **Puntuacio** -> cada candidat rep un score compost
2. **Re-ranker IPS** -> correccio de sesg de posicio (14 variables)
3. **Bandit UCB** -> exploracio equilibrada per slot
4. **MMR diversity** -> evita repeticions a partir del slot 5

### Avantatges
- Funciona be en tots els escenaris
- Simple i facil d'entendre
- No es massa agressiu

### Desavantatges
- No penalitza prou els productes "fantasma"
- No té en compte el temps de forma precisa
- Pesos hardcoded -> cal recompilar per cambiar
- Només 14 features

---

## 4.1.e - Modular + Optimitzacio Iterativa (ACTIU)

### Qu es?
Evolucio de Calypso 4.1 amb **arquitectura modular completa**. Tots els pesos (130+ parametres) es poden modificar **en calent sense reiniciar**. Ideal per a optimitzacio automatica.

### Millores respecte 2.1

| Millora | Descripcio |
|---------|------------|
| **Hot-reload** | weights_profile.json es recarrega automaticament cada ~60s |
| **Tuner CLI** | python tuner.py --optimize optimitza automaticament |
| **Presets** | 5 perfils predefinits: conversion_first, cheap_boost, user_trend_focus, etc. |
| **5 segments d'usuari** | Cada segment té multiplicadors specifics per cada senyal |
| **Regles de negoci** | Carrito actiu/oblat, productes recurrents, ordre de preu |
| **Tie-break configurable** | conversion / cheap / revenue |
| **Desigualtat de preu** | price_cheap_asymmetry per preferir productes economics |

### Pesos optimitzats (conversion_first)

| Pes | Valor | Descripcio |
|-----|-------|------------|
| w_seq_lift | 1.4 | Lift en transicions seqüencials |
| w_self_cvr_view | 1.7 | Auto-conversio per vistes |
| w_intent_score | 0.6 | Intencio de compra (carrets, moltes vistes) |
| w_user_affinity | 0.28 | Afinitat historica (subcat, marca) |
| w_price_compat | 0.22 | Compatibilitat de preu |
| w_ctx_bayes | 0.42 | Context bayesiano (regio, dispositiu, pagina) |
| w_global_pop | 0.15 | Popularitat global |
| w_live_trend | 0.28 | Tendencies en viu |
| w_reverse_graft | 0.32 | Reverse graft (conegut -> nou) |
| w_forward_cold | 0.48 | Forward cold (nou -> conegut) |
| w_log_price_norm | -0.1 | Preferencia per preus baixos |
| w_same_top_subcat | 0.14 | Mateixa subcategoria principal |
| w_same_top_brand | 0.1 | Mateixa marca principal |

### Regles de negoci (v4)

| Regla | Descripcio |
|-------|------------|
| Carrito actiu | Max 3 slots per productes al carrito (<30 min) |
| Carrito oblat | Max 2 slots per productes al carrito (>30 min, <30 dies) |
| Productes recurrents | Max 2 slots per productes comprats >=2 vegades |
| No repetir comprats | Comprat una vegada -> no es recomana (excepte recurrents) |
| Preu ascendent | Cada bloc s'ordena de mes barat a mes car (apagable) |

### Presets disponibles

| Preset | Objectiu | Pesos clau |
|--------|----------|------------|
| conversion_first | Max. conversions | ips_click_weight=0.10, tie_break=conversion |
| cheap_boost | Productes economics | w_log_price_norm=-0.18, price_cheap_asymmetry=0.45 |
| cheap_hard_sort | Preu estricte | v4_price_ascending=True |
| user_trend_focus | Tendencies d'usuari | Amplifica user_affinity, intent, live_trend |
| product_profile_focus | Similitud de producte | Amplifica cold_start features |

### Avantatges
- 130+ parametres optimitzables automaticament
- Hot-reload sense perdre peticions
- Regles de negoci avancades
- 5 segments d'usuari amb multiplicadors
- Validat amb --self-check i tests automatics

### Desavantatges
- Mes complex que 2.1
- Pot overfitjar si el tuning es massa agressiu
- Regles de negoci poden interferir amb el model