# Calypso 4.2 · Medidas de conversión aplicadas

Calypso 4.2 = copia de `calypso_4_1_e` con las medidas de `calypso_4_1/MEDIDAS_CONVERSION.md`.
`calypso_4_1_e` queda intacta.
Todo es conmutable desde `weights_profile.json` (hot-reload). Los presets anteriores
(`conversion_first`, `cheap_boost`, …) dan **exactamente** las mismas recomendaciones que antes
(comprobado sobre 2.400 peticiones sintéticas por preset). El nuevo preset es `medidas_conversion`,
y es el que queda activo en `weights_profile.json`.

## Qué se ha cambiado

| Medida | Parámetro(s) del perfil | Valor en `medidas_conversion` | Código |
|---|---|---|---|
| M1 Sin orden por precio | `v4_price_ascending` | `false` | ya existía |
| M3 Tasa base × relación | `scoring_mode`, `cal_w_*` | `calibrated`; rank 0,69 · visto 2,30 · relación 0,61 | `recommender._recommend_core` |
| M3/D4 Sin penalización de fondo | `v3_bg_penalty` | `false` (el modo calibrado no la usa) | |
| M3/D6 Sin ×0,65 al último visto | `self_view_page_factor` | `1.0` | `_prepare_request_signals` |
| M3/D3 Superventas siempre candidatos | `candidate_pool_size`, `candidate_top_sellers` | 140 / 30 | `_gather_candidates` |
| M4 Ranking de ventas en vivo | `live_window_minutes`, `live_kappa`, `live_cart_weight` | 60 min, κ = 40, 0 | `serve.sql_live_trends`, `recommender._build_sales_ranking` |
| M4 Prior de productos nuevos | `new_product_prior_factor` | 1,0 (mediana de sus vecinos por contenido) | `_build_sales_ranking` |
| M5 Exploración solo en slots 8–10 | `explore_min_slot` | 7 (también en sesiones frías) | bucle de selección |
| M5 Curva de atención medida | `attention_curve` | 1,0 · 0,78 · 0,72 · 0,60 · 0,58 · 0,55 · 0,52 · 0,50 · 0,50 · 0,50 | `bandits.py`, `reranker.py` |
| M6 Carrito activo como señal | `v4_cart_as_signal`, `cal_w_cart_active`, `v4_cart_priority` | `true`, +2,6, `false` | `recommend` |
| M6 Sin olvidado ni recurrentes | `v4_forgotten_cart`, `v4_recurrent` | `false`, `false` | |
| M7 `home` sin MMR | `mmr_subcat_lambda_home`, `mmr_brand_lambda` | 0, 0 | |
| M2 A/B por hash de `request_id` | `ab_split_ratio`, `ab_preset` | 0 (apagado) | `serve.CalypsoFastLoopV4._ab_branch` |

**Puntuación calibrada (M3):**

```
score(i) = −0,69·ln(puesto_ventas_i + 1)
         + 2,30·[visto en la sesión]   (o +2,6 si está en el carrito activo)
         + 0,61·relación_i              (lift de co-visita normalizado, solo si no visto)
         + 0,40·intención + 0,30·afinidad + 0,15·encaje de precio + 0,25·contexto
         + 0,45·señal cold-start + 0,10·misma subcategoría + 0,05·misma marca
```

Los pesos de la segunda línea de señales no están validados (la muestra no trae catálogo ni
historial de usuario); son pequeños a propósito. Ajustarlos es M8.

**Ranking vivo (M4):** `puesto` sale de
`share_mix(i) = (compras_60min(i) + 40·share_hist(i)) / (compras_60min_total + 40)`,
recalculado en cada snapshot (cada 15 s). La ventana de 60 min se mide desde el último evento de
`shop.events`, no desde `now()`, para no depender del reloj del simulador.

## Resultados

**Log real** (`python calypso_4_2/offline_eval_medidas.py`, 25.000 peticiones, 386 compras):

| Política | hit@10 clic | hit@10 compra | SNIPS ctr slot-1 | Compras esp./100 |
|---|---|---|---|---|
| Baseline top-10 ventas | 1,00 | 1,00 | 1,00 | 1,00 |
| 4.1e (`conversion_first`) | 0,07 | 0,06 | 0,01 | 0,09 |
| 4.2 (`medidas_conversion`) | **1,09** | **1,08** | 0,98 | **1,04** |

(relativo al baseline). Limitación: el snapshot del script solo tiene ranking de ventas,
co-visitas y vistas de sesión; sin afinidad, precio ni contexto. La 4.1e real tendrá algo más de
señal, pero el problema D2–D5 sigue ahí.

**Benchmark sintético** (`eval.py`): `medidas_conversion` cae de hit@10 0,80 a 0,24. El generador
sintético premia las co-visitas (en él el baseline de ventas solo saca 0,20), lo contrario de lo que
dice el log real (allí el baseline gana a las co-visitas ×5). **No uséis `tuner.py --optimize` ni
`eval.py` para decidir entre perfiles:** optimizan contra un mundo que no es el del simulador.

## Cómo usarlo

```powershell
cd calypso_4_2
python -m unittest discover -s etapa_4/tests            # 32 tests
python offline_eval_medidas.py                            # comparación sobre el log real
python serve.py --self-check
python tuner.py --preset medidas_conversion               # ya activo
python tuner.py --preset conversion_first                 # mismo comportamiento que 4.1e

# M2: repartir el 50 % del tráfico con 4.1e antigua (model_version calypso-4.2-ab-conversion_first)
python tuner.py --set ab_split_ratio=0.5 ab_preset=conversion_first
python tuner.py --set ab_split_ratio=0                    # apagar el A/B
```

## Pendiente

- **M8** (re-ranker con 250.000 sesiones): requiere descargar datos de RawTree.
- La alternativa "mejor todavía" de M3 (conversión del propio carrusel por producto) no está hecha.
- Probar en vivo con el A/B de M2 antes de la final.
