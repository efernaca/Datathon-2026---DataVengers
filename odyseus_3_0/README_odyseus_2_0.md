# Odyseus 2.0 (`odyseus-2.0`) · Calypso 5.1 + residuo aprendido

Odyseus 2.0 es **Calypso 5.1 sin tocar** más una red pequeña que **corrige** su puntuación. No sustituye
nada: si la red no existe, falla o se apaga, el sistema responde exactamente lo mismo que Calypso 5.1.

```
score(producto) = puntuación de Calypso 5.1  +  alpha · f(x) / s
                  └────── «prior» ──────┘      └── residuo aprendido ──┘
```

## 1. Por qué existe

Hasta `odyseus`, los ~65 parámetros de `weights_profile.json` se ajustaban por búsqueda iterativa contra
386 compras reales. Ese camino estaba agotado: en `offline_eval_cache/tuning_trials_results.csv`, 22 de 23
pruebas dan el mismo `hit_rate_buy`. Además, la evaluación «real» de esa búsqueda usa un snapshot construido
con los 60 días completos, incluidos los 7 días de test, así que sus métricas están contaminadas.

Odyseus 2.0 cambia la fuente de aprendizaje: en vez de 386 compras usa las de `shop.history_sessions`
(13.129 compras de entrenamiento), y en vez de buscar pesos a ciegas entrena un modelo con una función
de pérdida. Los pesos a mano siguen ahí, legibles y editables en caliente.

## 2. Cómo responde una petición

1. **Candidatos.** `_gather_candidates` de Calypso 5.1 reúne hasta 140 productos (vistos, carrito,
   recurrentes, top ventas garantizado, contexto, transiciones, productos nuevos). Sin cambios.
2. **Puntuación de Calypso (prior).** Suma ponderada de las 14 señales con multiplicadores por segmento,
   penalización de fondo y bonus de carrito, recurrentes, descuento y categoría. Sin cambios.
3. **Residuo (solo rama B).** Para cada candidato se calculan 34 variables, la red devuelve una corrección
   y se suma a la puntuación. Solo se corrigen candidatos **de la categoría del último producto visto**;
   si no hay sesión (home) se corrigen todos.
4. **Capa de política.** Bonus de slot 1, exploración UCB de productos nuevos, MMR, desempates, reglas de
   carrito, exclusión de comprados y guardrails. Sin cambios, y siempre después del residuo.
5. **Envío.** 10 IDs a `team_NN.submissions` con `model_version` = `odyseus-2.0-A` o `odyseus-2.0-B`.

Si el modelo lanza cualquier error o devuelve un valor no finito, la petición se responde con el prior.

## 3. Reparto A/B

`serve.py --ab-share X` manda una fracción `X` de las peticiones a la rama B, elegida por
`crc32(request_id)` (estable: la misma petición cae siempre en la misma rama).

| Rama | `model_version` | Qué sirve |
|---|---|---|
| A | `odyseus-2.0-A` | Calypso 5.1 |
| B | `odyseus-2.0-B` | Calypso 5.1 + residuo, si existe `ml_model.json`; si no, idéntica a A |

Sin `--ab-share` (o con 0) no hay ramas y `model_version` es `odyseus-2.0`.
`ml_model.json` se recarga en caliente cada 5 segundos: se puede publicar, cambiar o retirar un modelo sin
reiniciar el servicio.

## 4. El modelo

**Entrada: 34 variables por (petición, producto)**, ninguna es una identidad (`user_id`, `product_id`,
marca o región), porque la ronda final trae compradores y productos nuevos.

| Grupo | Variables |
|---|---|
| 14 señales de Calypso | `seq_lift_score`, `self_cvr_view`, `intent_score`, `user_affinity`, `price_compat`, `ctx_bayes_score`, `global_pop_norm`, `live_trend_norm`, `is_new_product`, `reverse_graft_score`, `forward_cold_score`, `log_price_norm`, `same_top_subcat`, `same_top_brand` |
| Contexto | `page_home`, `page_product`, `page_category`, `device_web`, `device_ios`, `device_android` |
| Usuario y sesión | `has_profile`, `has_views`, `n_recent_views`, `last_view_gap` |
| Producto × sesión | `in_recent_views`, `is_last_viewed`, `view_recency`, `dlogprice_last`, `abs_dlogprice_last` |
| Producto / usuario × producto | `self_cvr_raw`, `in_cart`, `is_recurrent`, `already_bought` |
| Posición | `slot` (solo al entrenar; al servir vale 0 = «como si fuera al slot 1») |

**Forma.** `logit(compra) = b + s · prior + f(x)`. El prior entra como offset con una escala `s` aprendida,
así que `f` solo aprende lo que Calypso no explica. Hay una segunda salida auxiliar para clic con peso 0,1.

**Escalones que se comparan** (todos con el mismo offset):

| Escalón | Forma | Papel |
|---|---|---|
| `prior` | solo `b`, `s` y posición | Referencia: Calypso recalibrado |
| `lineal` | `f` lineal | Suelo de comparación, el más seguro |
| `mlp16` | 34 → 16 → 1 | **Candidato**: cabe en el techo de 875 parámetros (15 compras por parámetro) |
| `mlp32x16` | 34 → 32 → 16 → 1 | Solo comparación: supera el techo |

Entrenamiento en NumPy puro (sin PyTorch): Adam, weight decay 1e-4, parada temprana sobre validación,
5 semillas promediadas al servir.

**Variables que la red no usa** (siguen contando dentro del prior):

- `is_new_product`, `reverse_graft_score`, `forward_cold_score`: en el histórico no hay productos nuevos y
  valen siempre 0. **La red no aprende nada de cold start**; de eso se ocupa Calypso.
- `live_trend_norm`: en vivo son los últimos 60 minutos y eso no se puede reconstruir en el histórico.

## 5. De dónde salen los datos (sin fuga de etiqueta)

`ml_build_dataset.py` parte el histórico en bloques de 7 días. Para cada bloque construye un snapshot con
`build_snapshot_v4_from_records` usando **solo datos anteriores al inicio del bloque** (popularidad,
transiciones, perfiles, prior de contexto, carritos, recurrentes) y puntúa las filas del bloque con él.
Ninguna variable ha visto la compra que intenta predecir.

| Bloque | Fechas | Uso | Peticiones | Compras |
|---|---|---|---|---|
| 0 | 7–14 nov | test | 33.064 | 3.530 |
| 1 | 31 oct–6 nov | validación (parada temprana, elección de escalón) | 31.247 | 3.238 |
| 2–5 | 3–30 oct | entrenamiento | 125.624 | 13.129 |

Se descargan todas las peticiones con clic o compra y 1 de cada 10 del resto (por hash); el peso
`g_weight` deshace ese muestreo. Cada petición aporta 10 filas, una por producto mostrado.
`official_test_data.parquet` cae entero dentro del bloque de test.

## 6. Cómo se decide si el modelo vale

**Puerta offline (`ml_train.py`).** Métrica: entre los 10 productos mostrados en cada petición con compra,
¿en qué puesto coloca el modelo el comprado? (MRR, acierto@1, acierto@3), con la posición fijada a 0 para
que no pueda copiar el orden del log. El escalón elegido debe:

- superar al `prior` en MRR de validación por más de un error típico,
- superarlo en MRR de test,
- mejorar el logloss de validación,
- no perder en peticiones de usuarios sin perfil.

Si no pasa, `--export` no escribe `ml_model.json`.

**Puerta en vivo (`ml_ab_report.py`).** Compras por 100 peticiones por rama, total y en usuarios nuevos,
con test z. Con una conversión del 0,76 % y reparto 50/50, cuatro horas solo distinguen mejoras de
alrededor del 25 % o más: la decisión se apoya sobre todo en la puerta offline, y el A/B sirve para
descartar que la rama B empeore o pierda fiabilidad.

## 7. Uso

```powershell
cd odyseus_2_0

# 1. Datos (reanudable; lo ya descargado en ml_data/ no se repite)
python ml_build_dataset.py --questions-only
python ml_build_dataset.py                    # -> ml_data/dataset.npz, dataset_meta.json

# 2. Entrenar y comparar
python ml_train.py                            # -> ml_data/train_report.md
python ml_train.py --export                   # escribe ml_model.json solo si pasa la puerta
python ml_train.py --export --export-model mlp16   # fuerza un escalón concreto

# 3. Servir
python serve.py --self-check
python serve.py --watchdog --ab-share 0.2     # 20 % a la rama B; subir a 0.5 si va estable

# 4. Leer el A/B
python ml_ab_report.py --hours 4
```

**Vuelta atrás:** renombrar `ml_model.json` (efecto en 5 segundos) o relanzar sin `--ab-share`.
**Dejar una sola rama para la final:** `--ab-share 1` (todo B) o sin flag (todo Calypso 5.1).
**Atenuar la red:** `--alpha 0.5` al exportar aplica la mitad de la corrección.

## 8. Piezas

| Fichero | Qué hace |
|---|---|
| `etapa_4/src/ml_features.py` | Las 34 variables y la réplica de la puntuación de `_recommend_core` (el offset). Mismo código al entrenar y al servir. |
| `etapa_4/src/ml_dataset.py` | Preguntas abiertas, descarga por días y snapshots a fecha de. |
| `etapa_4/src/ml_train.py` | Entrenamiento, puerta offline, importancia por permutación y pesos equivalentes. |
| `etapa_4/src/ml_model.py` | Inferencia NumPy y recarga en caliente. |
| `etapa_4/src/ml_ab_report.py` | Resultado del A/B por rama. |
| `etapa_4/src/recommender.py` | `recommend(..., ml_model=)`: único cambio respecto a 5.1, el bloque que suma el residuo. |
| `etapa_4/src/serve.py` | `--ab-share`, `request_arm`, etiquetado de `model_version` y carga del modelo. |
| `ml_data/` | Datos descargados, dataset e informes (fuera de git). |

El resto de la carpeta es Calypso 5.1 (`README_5_1.md`). `optimize.py`, `odyseus_optimizer.py` y
`README_odyseus.md` son restos del Odyseus anterior y no forman parte de esta versión.

## 9. Cómo leer lo que aprendió la red

`ml_data/train_report.md` incluye dos tablas pensadas para explicar el modelo sin abrir la red:

- **Importancia por permutación:** cuánto cae el MRR al barajar cada variable.
- **Pesos equivalentes:** aproximación lineal del residuo, en unidades de la puntuación de Calypso. Dice en
  qué dirección corrige la red cada señal. Si una corrección convence, se puede escribir a mano en
  `weights_profile.json` y prescindir de la red.

## 10. Límites conocidos

- **No mejora el cold start** (10 de los 70 puntos automáticos): solo se vigila que no lo empeore.
- **Solo misma categoría.** El carrusel antiguo mostraba siempre 10 productos de la categoría del último
  visto; no hay evidencia para corregir recomendaciones de otra categoría.
- **La métrica offline solo compara productos que el sistema antiguo ya mostró**, y hereda su sesgo hacia
  los más vendidos. No es una estimación de compras por 100 peticiones.
- **El prior del entrenamiento usa los pesos del perfil sin la mezcla en vivo** del re-ranker logístico;
  al servir esa mezcla sí está activa, así que hay una pequeña diferencia entre ambos.
- **Si se cambian pesos de `weights_profile.json`**, el prior cambia y la red queda entrenada contra otro
  prior: conviene regenerar el dataset y reentrenar (unos minutos, sin descargas).
- **Test heredado en rojo:** `test_slot_gated_calypso_protects_warm_and_beats_previous_stages` falla igual
  en `calypso_5_1` (0,8247 frente al umbral 0,83); no lo introduce esta versión.
