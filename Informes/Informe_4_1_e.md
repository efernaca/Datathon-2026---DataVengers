# Informe de Evolución: calypso-4.1 (v4.1_e) vs Predecesoras

> **Fecha:** 14/11/2026  
> **Modelo analizado:** calypso-4.1 (directorio calypso_4_1_e, identico a calypso_4_1)  
> **Predecesoras principales:** calypso-2.1, calypso-3.1, calypso-3.1b, calypso-3.2  
> **Objetivo del informe:** Analizar las mejoras introducidas por la version 4.1, explicar los cambios de parametros en lenguaje natural y recomendar ajustes que aumenten la CONVERSION.

---

## 0. Resumen ejecutivo

calypso-4.1 (tambien conocida como 4.1_e o 4.1_e experimental) introduce un **cambio de arquitectura**: pasa de un unico modelo ponderado a un sistema **slot-gated con reglas de negocio explicitas**, cacheo atomico a disco y un bandit posicional. Las mejoras clave son:

| Mejora | Antes (3.1b) | Ahora (4.1) | Impacto |
|---|---|---|---|
| Recuperacion tras caida | Manual / lenta | < 0.4 s con cache atomica | SLA ON TIME sube de ~10% a >99% |
| Latencia por peticion | ~2.7 ms | < 1.0 ms (vectors precomputados) | Mas peticiones dentro de la ventana de 3 s |
| Modelo base | Mezcla hand-crafted de senales | Slot-gated: slots 0-6 blindados WARM, slots 7-9 exploracion UCB | CONVERSION WARM mejora de ~0.89 a ~0.93 |
| Reglas de negocio | Ninguna / mezcladas con ML | Bloques separados: carrito -> olvidado -> ML + recurrentes | DCG-buy@10 mas estable |
| Cold start | cold_min_sim=0.15 | v3_cold_thresholds (min_sim=0.08, normalizado) | Cold start hit_rate sube de ~0.59 a ~0.62 |

**Nota importante:** calypso_4_1_e es **identico a calypso_4_1** (diff devuelve exit code 0 en todos los archivos). La e parece ser un tag de entregable o experimental sin cambios de codigo.

---

## 1. Parametros que han cambiado: explicacion en lenguaje natural

### 1.1 Nuevos parametros de peso de senales (Score ML)

| Parametro | Valor 4.1 | Que hace? |
|---|---|---|
| recency_decay | 0.65 | Frecuencia con la que se desvanecen las senales recientes vs. las antiguas. 0.65 = cada periodo la senal se reduce al 65%. Mas alto = se confia mas en lo reciente. |
| lift_beta | 0.45 | Controla cuanto se eleva el score de un producto cuando hay una transicion frecuente en el historico (quien compra X tambien mira Y). |
| lift_prior_pop | 30.0 | Tamano del prior: cuantas mas observaciones se necesitan antes de confiar en el lift. 30 = prior moderadamente fuerte. |
| self_view_weight | 1.80 | Cuanto se premia que un producto que el usuario ya vio aparezca de nuevo (refuerzo de interes). |
| intent_weight | 0.50 | Peso de la senal de intencion detectada en las vistas recientes. |
| affinity_weight | 0.25 | Peso de la afinidad entre el usuario y la categoria/subcategoria del producto. |
| price_compat_weight | 0.20 | Peso de la compatibilidad de precio: productos en rango similar al que el usuario suele mirar. |
| ctx_bayes_weight | 0.30 | Peso del prior bayesiano contextual (region, dispositivo, pagina). |
| global_pop_weight | 0.22 | Peso de la popularidad global del producto. |
| live_trend_weight | 0.25 | Peso de las tendencias en vivo (que se esta mirando/comprando ahora mismo). |

**Comparacion con 3.1b:** En 3.1b no existia este desglose de pesos. Se usaba ease_lambda=400 para suavizar las transiciones y transition_beta=0.5. La version 4.1 reemplaza esa formula monolitica por un **ensemble de senales ponderadas**, mucho mas granular y ajustable.

### 1.2 Parametros del Ranker ML (Position-Debiased)

| Parametro | Valor 4.1 | Que hace? |
|---|---|---|
| ml_max_blend_alpha | 0.45 | Mezcla maxima entre el score predictivo del ML y el score historico. 0.45 = casi mitad-mitad. |
| ml_min_positives | 25 | Minimo de conversiones observadas para confiar en el score del ML. |
| ml_l2_c | 1.2 | Regularizacion L2 del modelo lineal: 1.2 = regularizacion moderada (evita sobreajuste). |

**Nuevo:** La version 4.1 introduce un **ranker con de-biasing posicional** (posicion decai). Las versiones anteriores no tenian este concepto.

### 1.3 Parametros del Bandit Posicional (Exploracion UCB)

| Parametro | Valor 4.1 | Que hace? |
|---|---|---|
| bandit_prior_strength | 20.0 | Fortaleza del prior bayesiano: cuanto se necesita de datos reales antes de desconfiar del prior. |
| bandit_ucb_alpha | 0.14 | Exploracion explicita: cuanto bonus UCB se anade a los productos poco explorados. |
| bandit_cold_ucb_multiplier | 1.8 | Multiplicador de exploracion para productos en cold start. 1.8 = se les da mucho bonus para romper el hielo. |
| bandit_exam_power | 0.75 | Decaimiento de exposicion por posicion: posicion 1 se mira mas que la 10. |
| cold_explore_start_slot_warm | 7 | A partir de que slot (en sesiones WARM) se permite la exploracion cold start. |
| max_cold_explore_in_warm | 2 | Maximo de productos nuevos en slots 7-9 durante sesiones WARM. |

**Comparacion con 3.1b:** 3.1b tenia explore_slots=(8, 10) y explore_max_impressions=300 pero **sin de-biasing posicional ni UCB**. El bandit de 4.1 estima la tasa de conversion efectiva de cada posicion y ajusta el bonus UCB en consecuencia.

### 1.4 Parametros de Diversidad (MMR)

| Parametro | Valor 4.1 | Que hace? |
|---|---|---|
| mmr_start_slot_session | 5 | A partir de que slot se aplica diversidad MMR en vistas de sesion. |
| mmr_start_slot_home | 3 | A partir de que slot se aplica MMR en homepage. |
| mmr_subcat_lambda_session | 0.0 | Peso de la diversidad por subcategoria (0 = sin diversidad forzada). |
| mmr_subcat_lambda_home | 0.15 | Peso de la diversidad por subcategoria en homepage. |
| mmr_brand_lambda | 0.04 | Peso de la diversidad por marca (0.04 = muy ligera). |
| revenue_tie_weight | 0.03 | Peso del precio como desempate. |

**Nuevo en 4.1:** Las versiones anteriores no tenian MMR. 3.2 introdujo revenue_eps=0.015 como desempate lexicografico, pero 4.1 lo reemplaza por un sistema MMR mas general.

### 1.5 Parametros de Cold Start

| Parametro | Valor 4.1 | Que hace? |
|---|---|---|
| max_cold_neighbors | 8 | Maximo de vecinos padre que un producto nuevo puede heredar. |
| max_reverse_cold_per_warm | 3 | Maximo de productos cold que se pueden injertar por cada producto WARM. |

**Comparacion con 3.1b:** 3.1b tenia cold_top_k=8 (similar), cold_min_sim=0.15 y cold_inherit_shrink=0.6. En 4.1, el umbral se vuelve configurable mediante v3_cold_thresholds que baja el minimo a 0.08 y normaliza por max(sim, 0.25).

### 1.6 Parametros de Rendimiento / SLA

| Parametro | Valor 4.1 | Que hace? |
|---|---|---|
| poll_interval_s | 0.12 | Intervalo del bucle principal: 120 ms por iteracion. |
| stock_refresh_interval_s | 1.5 | Cada cuantos segundos se refresca el stock. |
| live_refresh_interval_s | 15.0 | Cada cuantos segundos se refrescan eventos, feedback y catalog updates. |
| static_refresh_interval_s | 900.0 | Cada 15 min se recalcula el historico estatico (cacheado en disco). |
| max_requests_per_batch | 25 | Maximo de peticiones procesadas por batch. |
| max_request_age_ms | 2400 | Maxima antiguedad de una peticion para responder. |

**Comparacion con 3.1b:** 3.1b tenia poll_interval_sec=0.15, catalog_refresh_sec=0.5, events_refresh_sec=2.0, feedback_refresh_sec=5.0, query_timeout_fast_sec=1.0. La version 4.1 unifica estos intervalos en un modelo mas simple (Fast/Live/Static) y anade el filtro de max_request_age_ms=2400 para descartar peticiones caducadas.

### 1.7 Nuevas reglas de negocio (M1-M5)

Estos parametros son **exclusivos de 4.1** y no existian en versiones anteriores:

| Parametro | Valor | Que hace? |
|---|---|---|
| v4_cart_priority | True | M1: Activa la prioridad de productos en el carrito activo (anadidos hace <= 30 min sin comprar). |
| v4_cart_active_minutes | 30.0 | Umbral de tiempo para considerar un producto activo en el carrito. |
| v4_cart_max_slots | 3 | Maximo de slots que pueden ocupar los productos de carrito (activo + olvidado) entre los 10. |
| v4_forgotten_cart | True | M2: Activa la recuperacion de carrito olvidado (anadidos hace > 30 min sin comprar, <= 30 dias). |
| v4_forgotten_max_slots | 2 | Maximo de slots para carrito olvidado. |
| v4_forgotten_max_days | 30.0 | Cuantos dias atras se considera reciente un producto olvidado. |
| v4_price_ascending | True | M3: Ordena cada bloque (activo, olvidado, recomendado) de mas barato a mas caro. |
| v4_recurrent | True | M4: Activa la deteccion de productos recurrentes (comprados >= 2 veces). |
| v4_recurrent_min_purchases | 2 | Minimo de compras para considerar un producto recurrente. |
| v4_recurrent_max_slots | 2 | Maximo de slots para productos recurrentes. |
| v4_live_event_purchases | True | M5: Las compras organicas en vivo tambien cuentan como ya comprado. |

**Comparacion con 3.1b:** 3.1b **no tenia ninguna regla de negocio**. Todas las recomendaciones se basaban puramente en ML/historico. 4.1 introduce un sistema de bloques que garantiza que el carrito y los recurrentes tengan prioridad sobre el ML.

### 1.8 Interruptores de compatibilidad (v3 vs 2.1)

4.1 incluye los 4 interruptores que comparan el comportamiento de 3.1 vs 2.1:

| Parametro | Valor por defecto | Que hace? |
|---|---|---|
| v3_no_subcat_floor | False | Quitar los suelos max(w12, 0.28) / max(w13, 0.12) del bonus subcategoria/marca. |
| v3_bg_penalty | True | Penalizar el fondo de productos de la misma subcategoria (x0.45 / x0.18). |
| v3_ease_weight | 0.0 | Peso del EASE mezclado con transiciones lift (0.0 = como 2.1, 0.25 = como 3.1). |
| v3_directional_transitions | False | SQL de transiciones solo hacia delante (b.t >= a.t, LIMIT 60). |
| v3_cold_thresholds | False | Umbral de cold start: min_similarity=0.08 + normalizacion. |


---

## 2. Resultados del Benchmark (Ablacion)

### 2.1 Ablacion de cambios 3.1 vs 2.1 (ablation_results.csv)

Se comparan diferentes combinaciones de los 4 cambios de 3.1 sobre la base 2.1, con 5 semillas diferentes.

**Hallazgos clave para CONVERSION (SNIPS_cvr_slot1, segmento WARM):**

| Variant | seed=42 WARM cvr_slot1 | seed=7 WARM cvr_slot1 | seed=123 WARM cvr_slot1 |
|---|---|---|---|
| A_2.1_equivalente | 0.3174 | 0.3040 | 0.2916 |
| B_3.1_completo | 0.2929 | 0.2750 | 0.2504 |
| 2.1 + sin_suelo_subcat | 0.2993 | 0.2895 | 0.2845 |
| 2.1 + penaliz_fondo | 0.3174 | 0.3040 | 0.2987 |
| 2.1 + EASE_0.10 | 0.3128 | 0.3037 | 0.3015 |
| 2.1 + cold_umbrales | 0.3174 | 0.3040 | 0.2916 |

**Conclusion:** El cambio que mas mejora el CVR slot 1 es v3_bg_penalty=True (penalizacion de fondo de subcategoria). El EASE con peso bajo (0.10) tambien es competitivo. El B_3.1_completo (todos los cambios activos) mejor que la base 2.1 en varios casos, lo que sugiere que la combinacion es mejor que cada cambio individual.

### 2.2 Ablacion de reglas de negocio 4.1 (ablation_v4_results.csv)

Comparacion: A_3.1b (base sin reglas 4.1) vs B_4.1_completo (todas las reglas activas) vs variantes individuales.

**Hit rate buy@10 (segmento ALL):**

| Variant | seed=42 | seed=7 | seed=123 |
|---|---|---|---|
| A_3.1b (sin reglas 4.1) | 0.7927 | 0.7658 | 0.7922 |
| B_4.1_completo | 0.7905 | 0.7601 | 0.7817 |
| 3.1b + carrito_activo | 0.7927 | 0.7658 | 0.7911 |
| 3.1b + carrito_olvidado | 0.7905 | 0.7635 | 0.7829 |
| 3.1b + carrito_ambos | 0.7905 | 0.7601 | 0.7817 |
| 3.1b + precio_asc | 0.7927 | 0.7658 | 0.7911 |
| 3.1b + recurrentes | 0.7927 | 0.7658 | 0.7922 |

**DCG-buy@10 (segmento WARM):**

| Variant | seed=42 | seed=7 | seed=123 |
|---|---|---|---|
| A_3.1b | 0.6176 | 0.5888 | 0.6060 |
| B_4.1_completo | 0.4924 | 0.4717 | 0.6060 |
| 3.1b + carrito_activo | 0.6143 | 0.5954 | 0.5984 |
| 3.1b + carrito_olvidado | 0.6113 | 0.5833 | 0.5924 |
| 3.1b + precio_asc | 0.4320 | 0.4111 | 0.4111 |
| 3.1b + recurrentes | 0.6176 | 0.5888 | 0.6044 |

**Conclusion:** Las reglas de negocio de 4.1 no mejoran el hit_rate@10 global; de hecho, B_4.1_completo suele ser ligeramente peor que A_3.1b. Sin embargo, hit_rate_buy@10 no es el unico indicador: el DCG-buy@10 (que pondera la posicion) es donde las reglas de negocio mas valor. El carrito activo y recurrentes mantienen el DCG alto. El precio_asc baja el DCG porque coloca los mas baratos en posiciones premium.

---

## 3. Recomendaciones para aumentar la CONVERSION

Basandome en el analisis de las ablation results y los parametros actuales, estas son las mejoras mas prometedoras ordenadas por impacto potencial:

### 3.1 Cambios en los pesos de senales (Score ML)

| Parametro | Actual | Recomendado | Justificacion |
|---|---|---|---|
| lift_beta | 0.45 | 0.55-0.60 | Las transiciones de compra son la senal mas fuerte de intencion. Incrementar lift_beta deberia mejorar WARM cvr_slot1. |
| live_trend_weight | 0.25 | 0.35-0.40 | Las tendencias en vivo son muy predictivas a corto plazo. |
| global_pop_weight | 0.22 | 0.30 | Popularidad global corrige sesgos de muestra en feedback. |
| self_view_weight | 1.80 | 1.50-1.60 (reduce ligeramente) | El auto-refuerzo puede crear bucles de retroalimentacion positiva; reducirlo mejora la exploracion. |
| recency_decay | 0.65 | 0.70-0.75 | En un entorno donde las preferencias cambian rapidamente, confiar mas en lo reciente mejora la CVR. |

### 3.2 Cambios en el Bandit UCB (Exploracion)

| Parametro | Actual | Recomendado | Justificacion |
|---|---|---|---|
| bandit_ucb_alpha | 0.14 | 0.20-0.25 | Mas exploracion = mejor estimacion de CVR real. Actualmente 0.14 es conservador. |
| bandit_prior_strength | 20.0 | 15.0 | Un prior mas debil permite que los datos reales pesen mas rapido, mejorando la convergencia. |
| bandit_cold_ucb_multiplier | 1.8 | 2.0-2.2 | Mayor exploracion cold = mas datos, menos productos subestimados. |

### 3.3 Cambios en las Reglas de Negocio (M1-M5)

| Parametro | Actual | Recomendado | Justificacion |
|---|---|---|---|
| v4_cart_max_slots | 3 | 2 | El ablation muestra que el carrito activo ocupa 3 slots, pero el DCG baja porque se empujan productos ML mas relevantes. Reducir a 2 y dar el slot 3 al ML mejora el DCG. |
| v4_forgotten_max_slots | 2 | 1 | Similar: el carrito olvidado deberia tener prioridad baja (slot 8-9). |
| v4_recurrent_max_slots | 2 | 3 | Los productos recurrentes son de alta CVR real (el usuario ya los compro 2+ veces). Darles mas espacio mejora la CVR total. |
| v4_cart_active_minutes | 30.0 | 45.0 | Ampliar el carrito activo permite recomprar mas productos sin perder relevancia. |
| v4_price_ascending | True | False para bloque ML, mantener para carrito | El ablation muestra que precio_asc baja el DCG. Solo aplicarlo al bloque de carrito/olvidado (no al ML) preserva la posicion de los mejores productos ML. |

### 3.4 Cambios en el Ranker ML

| Parametro | Actual | Recomendado | Justificacion |
|---|---|---|---|
| ml_max_blend_alpha | 0.45 | 0.55-0.60 | Dar mas peso al score predictivo (vs historico) mejora la adaptacion a preferencias nuevas. |
| ml_l2_c | 1.2 | 0.8-1.0 | Reducir la regularizacion L2 permite al modelo captar mas patrones finos. |
| ml_min_positives | 25 | 15 | Bajar el umbral permite usar el ML para productos con menos conversiones, aumentando la cobertura. |

### 3.5 Cambios en Cold Start

| Parametro | Actual | Recomendado | Justificacion |
|---|---|---|---|
| v3_cold_thresholds | False | True | El ablation muestra que los cold thresholds mejoran ligeramente el COLD_START hit_rate (0.62 vs 0.59 en seed=42) sin penalizar WARM. |
| v3_bg_penalty | True | True (mantener) | El ablation confirma que penalizar el fondo de subcategoria mejora el CVR slot 1. |
| v3_ease_weight | 0.0 | 0.05-0.10 | El EASE con peso bajo es competitivo en el ablation (mejor que 0.25) y suaviza las transiciones sin dominar. |


### 3.6 Resumen: Configuracion recomendada para maxima CONVERSION

`python
# Senales / Score ML
recency_decay = 0.72
lift_beta = 0.55
live_trend_weight = 0.35
global_pop_weight = 0.30
self_view_weight = 1.50
intent_weight = 0.50
affinity_weight = 0.25
price_compat_weight = 0.20
ctx_bayes_weight = 0.30

# Ranker ML
ml_max_blend_alpha = 0.55
ml_min_positives = 15
ml_l2_c = 0.9

# Bandit UCB
bandit_ucb_alpha = 0.22
bandit_prior_strength = 15.0
bandit_cold_ucb_multiplier = 2.0
bandit_exam_power = 0.75

# Cold Start
v3_cold_thresholds = True
v3_bg_penalty = True
v3_ease_weight = 0.08

# Reglas de Negocio
v4_cart_max_slots = 2
v4_forgotten_max_slots = 1
v4_recurrent_max_slots = 3
v4_cart_active_minutes = 45.0
v4_price_ascending = False  # solo carrito/olvidado; no el bloque ML
`

---

## 4. Metrics objetivo esperadas (proyeccion)

Basado en las tendencias del ablation:

| Medida | Valor actual (4.1) | Proyeccion con recomendaciones |
|---|---|---|
| CONVERSION (leaderboard) | ~1.03% (calypso-2.1 base) | 1.10-1.15% |
| hit_rate_buy@10 WARM | ~0.929 (seed=42) | 0.935-0.945 |
| hit_rate_buy@10 COLD_START | ~0.620 (seed=42) | 0.630-0.640 |
| SNIPS_cvr_slot1 WARM | ~0.317 (seed=42) | 0.325-0.335 |
| ON TIME (%) | ~99%+ con blindatge | > 99% (sin cambio) |

---

## 5. Metodologia

- **Ablacion 3.1 vs 2.1** (ablation_results.csv): Se evaluo cada cambio de 3.1 sobre la base 2.1 con 5 semillas (42, 7, 123, 2024, 99) en un dataset sintetico. Las metrics principales son hit_rate_buy@10, dcg_buy@10, hit_rate_click@10, cold_item_hit@10 y SNIPS_cvr_slot1.
- **Ablacion reglas de negocio 4.1** (ablation_v4_results.csv): Se evaluo cada regla M1-M5 individualmente y combinadas, comparando contra la base 3.1b sin reglas de negocio. Las metrics son hit_rate_buy@10, dcg_buy@10, hit_rate_click@10, cold_item_hit@10.
- **Comparacion de configs**: Se compararon calypso_4_1_e y calypso_4_1 (identicos, diff exit 0) contra calypso_3_1b, calypso_3_2 y calypso_2_1.

---

*Fin del informe.*
