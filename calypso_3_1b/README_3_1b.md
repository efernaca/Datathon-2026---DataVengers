# Calypso 3.1b · los cambios de 3.1 convertidos en interruptores

3.1b es una copia de `calypso_3_1` en la que cada uno de los 4 cambios respecto a 2.1 se puede activar o
desactivar por variable de entorno (sin tocar código). Por defecto queda la combinación que mejor sale en la
ablación offline.

| Variable de entorno | Qué controla | 2.1 | 3.1 | **3.1b (defecto)** |
|---|---|---|---|---|
| `CALYPSO_V3_NO_SUBCAT_FLOOR` | Quita los suelos `max(w12,0.28)` / `max(w13,0.12)` del bonus de subcategoría/marca | 0 | 1 | **0** |
| `CALYPSO_V3_BG_PENALTY` | Fondo sin señal x0.18 (otra subcat) / x0.45 (misma subcat), en vez de x0.25 / x1.0 | 0 | 1 | **1** |
| `CALYPSO_V3_EASE_WEIGHT` | Peso del "EASE" mezclado con las transiciones lift | 0 | 0.25 | **0** |
| `CALYPSO_V3_DIRECTIONAL` | SQL de transiciones solo hacia delante (`b.t >= a.t`, LIMIT 60) | 0 | 1 | **0** |
| `CALYPSO_V3_COLD_THRESHOLDS` | Cold start con `min_similarity=0.08` y normalización por `max(max,0.25)` | 0 | 1 | **0** |

Con todo a 0 el modelo reproduce exactamente 2.1; con todo a 1 (y EASE 0.25), exactamente 3.1.

## Ablación offline (benchmark sintético de `etapa_4/src/eval.py`, sin RawTree)

```
python etapa_4/ablation_v3.py --seeds 42 7 123 2024 99 --out-csv ablacion.csv
```

Media de 5 semillas, segmento ALL (2.400 peticiones por semilla). Resultados por semilla en `etapa_4/ablation_results.csv`.

| Variante | hit_rate_buy@10 | Δ vs 2.1 | dcg_buy@10 |
|---|---|---|---|
| 2.1 | 0.7725 | — | 0.4749 |
| 3.1 completo | 0.7454 | **−0.027** (5/5 semillas peor) | 0.4603 |
| 2.1 + sin suelo subcat | 0.7529 | **−0.020** (5/5 peor) | 0.4640 |
| 2.1 + EASE 0.25 | 0.7642 | −0.008 (4/5 peor) | 0.4711 |
| 2.1 + EASE 0.10 | 0.7724 | 0.000 | 0.4746 |
| 2.1 + umbrales cold | 0.7725 | 0.000 | 0.4749 |
| **2.1 + penalización de fondo (= 3.1b)** | **0.7904** | **+0.018** (5/5 mejor) | **0.4835** |

El cambio 3 (SQL direccional) no se puede medir aquí porque el mundo sintético ya trae las transiciones; solo se ve con datos reales.

Nota: los resultados dependen ligeramente del orden de los `set()` de Python; el script fija `PYTHONHASHSEED=0`
para que sean reproducibles. Diferencias < 0.003 son ruido.
