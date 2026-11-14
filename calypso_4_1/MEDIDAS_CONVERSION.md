# Calypso 4.1 · Medidas para maximizar la conversión

**Fecha:** sábado 14-nov-2026, 20:00 CET (quedan 14 h hasta la ronda final)
**Base:** código de `calypso_4_1/etapa_4/` y análisis del log real que ya está en el repo (`offline_eval_cache/`: 25.000 peticiones de `shop.history_sessions` de los últimos 7 días, 386 compras).
**Reproducir las cifras:** `python calypso_4_1/analisis_conversion.py` (no accede a RawTree).

> **Qué es dato y qué no.** Las tablas de la sección 1 son recuentos del log real. Los coeficientes de la sección 1.4 salen de un modelo sencillo ajustado sobre ese log. La comparación de políticas (1.5) es una estimación bajo ese modelo, no una medida en vivo. La muestra no incluye catálogo ni historial de usuario, así que no he podido medir categoría, marca, precio, carrito ni afinidad personal: la sección 4 dice cómo medirlos. El SQL de la sección 4 no está ejecutado.

---

## 1. Qué dice el log real

### 1.1 Dónde están las compras

| Página | Peticiones | Vistas recientes (media) | CVR de la lista | % de todas las compras |
|---|---|---|---|---|
| `product` | 56 % | 2,5 | 1,80 % | 65 % |
| `category` | 24 % | 2,5 | 1,99 % | 31 % |
| `home` | 20 % | **0 siempre** | 0,28 % | 4 % |

Toda compra pasa por un clic en ese mismo producto (0 excepciones), y el 42 % de los clics acaba en compra. `home` nunca trae `recent_views` y aporta el 4 % de las compras: **el 96 % de la conversión se juega en `product` y `category`.**

### 1.2 Qué producto se compra: manda el puesto en el ranking de ventas

Conversión **por impresión** en páginas con sesión, según la relación del producto con la sesión y su puesto en el ranking de ventas (`best_sellers.json`):

| Puesto de ventas | Visto en la sesión | Co-visitado | Sin relación |
|---|---|---|---|
| 1–10 | **6,11 %** (37 / 606) | 1,10 % (47 / 4.288) | 0,70 % (120 / 17.093) |
| 11–30 | 2,24 % (12 / 536) | 0,50 % (30 / 6.013) | 0,21 % (41 / 19.538) |
| 31–100 | 0,90 % (6 / 666) | 0,19 % (16 / 8.500) | 0,08 % (29 / 36.154) |
| 101–300 | 0,34 % (2 / 597) | 0,09 % (7 / 8.236) | 0,04 % (17 / 43.927) |
| 301–1000 | 0 % (0 / 344) | 0 % (0 / 5.355) | 0,02 % (8 / 38.102) |

"Co-visitado" = está entre los 20 vecinos de co-visita de algún producto visto.

Tres lecturas:

1. **El puesto de ventas es el factor dominante.** La conversión cae aproximadamente como `puesto^-0,87`. Un top-10 **sin relación** con la sesión (0,70 %) convierte casi cuatro veces más que un co-visitado del puesto 31–100 (0,19 %).
2. **Haber visto el producto multiplica por ~10**, y el efecto es multiplicativo: ×8,7 en el top-10, ×10,7 en 11–30, ×11 en 31–100.
3. **La co-visita multiplica por ~2.** Es una señal real, pero cinco veces más débil que "visto" y mucho más débil que el puesto de ventas.

Por debajo del puesto 300 casi no hay compras, ni siquiera entre productos vistos.

### 1.3 Posición

El slot 1 concentra el 34 % de las compras y los slots 1–3 el 63 %. Parte de eso es que el sistema antiguo ponía arriba los más vendidos. Descontando el producto, la atención relativa por slot es aproximadamente:

| Slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Atención | 1,00 | 0,78 | 0,75 | 0,51 | 0,69 | 0,49 | 0,58 | 0,43 | 0,56 | 0,61 |

Con 386 compras la curva tiene ruido; la forma fiable es **1 → ~0,75 (slots 2–3) → ~0,55 (slots 4–10, plana)**. Es más plana que la que asume `bandits.py`, `(k+1)^-0,75`, que da 0,18 al slot 10.

### 1.4 Modelo: una fórmula de puntuación calibrada

Regresión logística por impresión sobre compra:

```
logit P(compra) = −3,84 − 0,69·ln(puesto_ventas + 1) + 2,30·[visto] + 0,61·[co-visitado] − 1,65·[home] + efecto_slot
```

Traducido a equivalencias de ranking:

- **"Visto" equivale a dividir el puesto de ventas entre ~28.** Un producto visto en el puesto 250 vale lo que el más vendido nº 9 sin relación.
- **"Co-visitado" equivale a dividirlo entre ~2,4.** Un co-visitado en el puesto 24 vale lo que el nº 10 sin relación.

A nivel de lista las probabilidades se suman sin saturar en el rango observado: en el decil más alto el modelo predice 5,1 % y se observa 6,1 %. Llenar la lista de productos fuertes suma.

### 1.5 Políticas simples bajo ese modelo

Compras esperadas por 100 peticiones, relativas al baseline (estimación):

| Política | Con sesión | Total |
|---|---|---|
| Baseline: top-10 ventas | 1,00 | 1,00 |
| Co-visitas primero, relleno con ventas (starter) | **0,20** | 0,23 |
| Vistos primero, luego top ventas | 0,85 | 0,86 |
| Puntuación calibrada de 1.4 | 1,04 | 1,04 |
| La misma, en orden aleatorio | 0,92 | 0,92 |

- Una lista ordenada por relación con la sesión, sin el puesto de ventas, queda **muy por debajo del baseline**.
- Poner los vistos arriba sin mirar su puesto también pierde: un visto del puesto 500 desplaza a un superventas.
- Con solo estas tres señales el margen sobre el baseline es pequeño (+4 %). El resto de la distancia al oráculo tiene que venir de lo que esta muestra no contiene: afinidad del usuario por categoría, marca y precio, y popularidad en vivo.
- Desordenar una buena lista cuesta ~8 %.

---

## 2. Dónde choca Calypso 4.1 con estos datos

| # | Qué hace el código | Qué dicen los datos |
|---|---|---|
| D1 | `v4_price_ascending = True` por defecto: cada bloque se ordena por precio. | El orden importa. La ablación propia ya daba −25 % de DCG; con la curva real, desordenar cuesta del orden del 8 % o más. |
| D2 | La popularidad entra como `0,22 × log1p(pop)/max`: comprimida con logaritmo y con el peso más bajo de las señales principales. | Es el factor dominante. Entre el puesto 5 y el 200 la conversión cambia ~20 veces; en la puntuación actual cambian unas centésimas. |
| D3 | `_gather_candidates` llena los 64 huecos con las transiciones de cada producto visto (hasta 35 por vista). Con 2 vistas o más, los más vendidos no llegan a ser candidatos. | El 59 % de las compras del log son de productos ni vistos ni co-visitados. |
| D4 | Penalización de fondo: productos sin señal de sesión ×0,18 (o ×0,25 con el interruptor apagado). | Un top-10 sin relación convierte más que casi cualquier co-visitado. Se está hundiendo justo lo que más vende. |
| D5 | El lift divide las transiciones por `pop(destino)^0,45` y no hay ningún término que devuelva la tasa base. | El lift es correcto como medida de *relación*, pero la probabilidad es `tasa base × relación`. Falta el primer factor. |
| D6 | Producto visto: `self_cvr × 5` (tope 1,2) × 1,85, con ×0,65 si es el último visto en página de producto. Una transición fuerte (1,05) puede superarlo. | "Visto" vale ~5 veces más que "co-visitado". En página de producto, el último visto convierte igual o mejor que los anteriores (18/945 frente a 17/983): no hay base para el ×0,65. |
| D7 | Sesión fría (`is_new_user`): `cold_gate = 0`, exploración UCB desde el slot 1. | Los slots 1–3 concentran el 63 % de las compras. |
| D8 | `SQL_LIVE_TRENDS` agrega **todo** `shop.events` sin ventana temporal. | El domingo a las 10:00 habrá 24 h de eventos de práctica acumulados: las tendencias nuevas de la final quedarán diluidas. Con la popularidad como factor principal, esto es crítico. |
| D9 | El re-ranker se entrena con 3.000 sesiones (~46 compras), sus pesos se recortan a [0,7×, 2,2×] del prior manual y se mezclan al 45 % como máximo. | En la práctica los pesos son los manuales. |
| D10 | Carrito olvidado fijado en los primeros slots; recurrentes garantizados. | Sin medir. La ablación propia sale negativa para el carrito olvidado, y las bases dicen que quien ya compró un producto tiende a no quererlo otra vez. |
| D11 | MMR en `home` con λ = 0,15 por subcategoría repetida. | En `home` solo convierte el top de ventas (0,18 % por impresión en el top-10, ~0 a partir del puesto 100). Diversificar ahí cambia superventas por productos que no se compran. |

---

## 3. Medidas, por orden de impacto esperado

### M1 · Quitar el orden por precio — 1 minuto

`CALYPSO_V4_PRICE_ASC=0`. La lista debe salir ordenada por probabilidad de compra y nada más.

### M2 · Comprobar que estáis por encima del baseline — 30 minutos

La sección 1.5 predice que una lista dominada por co-visitas pierde contra el top-10 de ventas. Mirad la componente de conversión en el leaderboard. Si está en 0 o cerca, la causa probable es D2–D5.

Prueba directa: repartir el tráfico al 50 % por hash de `request_id` entre la versión actual y una variante mínima (M3 reducida: top ventas + vistos con la regla del ÷28), con `model_version` distinto. Con ~2.700 peticiones por rama en 30 minutos, una diferencia de CTR de 2–3 puntos se ve con claridad.

### M3 · Puntuar como tasa base × relación — 2–3 horas, la medida principal

Sustituir la suma de pesos por una puntuación en escala logarítmica con la tasa base explícita:

```
score(i) = −0,69·ln(puesto_i + 1)          # tasa base
         + 2,30 · visto(i)                  # está en recent_views y no comprado
         + 0,61 · relación(i)               # s_seq normalizado a [0, 1]
         + (resto de señales, ver M8)
```

Cambios concretos en `etapa_4/src/recommender.py`:

1. **Tasa base.** Añadir a `product_static` el término `−0,69·ln(puesto+1)`, con el puesto calculado por **compras** en `history_events` (como `best_sellers.json`), y usarlo en lugar de `w6 * gpop`. Mejor todavía: la conversión del propio carrusel por producto, `compras / Σ atención_slot` sobre `history_sessions`, suavizada hacia la curva `puesto^-0,87` para los productos poco mostrados.
2. **Sin penalización de fondo** (D4). Con la tasa base explícita sobra. Hay que quitar también el camino `bg_score *= 0.25`, no solo poner `CALYPSO_V3_BG_PENALTY=0`.
3. **Visto** (D6): +2,30 fijo en vez de `self_cvr × 5 × 1,85`; quitar el `page_factor` de 0,65.
4. **Candidatos** (D3): reservar siempre los 30 primeros de ventas y subir `candidate_pool_size` a 120–150. El cálculo sigue muy por debajo de 1 ms.
5. **Mantener el lift** como `relación(i)`: ahora sí cumple su papel, porque la popularidad la aporta el primer término.

Resultado esperado con sesión: arriba los vistos que además venden, luego los superventas y los co-visitados bien situados, y fuera los productos de cola.

### M4 · Popularidad en vivo para la final — 1–2 horas

La final trae "new shoppers, new products, new trends", y la tasa base es el factor que más pesa: si el ranking de ventas se queda en el histórico, lo principal del modelo estará desfasado.

- `SQL_LIVE_TRENDS` con ventana: `WHERE ts::DateTime64(3) > now() - INTERVAL 60 MINUTE` (D8), o contadores con decaimiento.
- Puesto de ventas mezclado: `compras_mix(i) = compras_60min(i) · escala + κ · compras_hist(i)`, y el puesto se recalcula sobre esa mezcla cada 15 s. Con pocos eventos manda el histórico; cuando hay señal, manda lo vivo.
- **Productos nuevos:** no tienen puesto. Darles como prior el puesto mediano de sus vecinos por contenido y dejar que las compras de `shop.events` lo corrijan. Es también la palanca para el segmento de cold start, donde el baseline usa un ranking fijo calculado al inicio.
- Ensayarlo esta noche: comparar el ranking mezclado con el histórico y ver si mejora el CTR.

### M5 · Orden estricto y exploración solo al final — 30 minutos

- Lista ordenada por `score` de mayor a menor, sin reordenaciones posteriores.
- Exploración (UCB, productos nuevos sin confirmar) solo en los slots 8–10, también en sesiones frías: `cold_gate` nunca por debajo de 7 (D7).
- Sustituir la curva de atención de `bandits.py` por la medida: `[1,0, 0,78, 0,72, 0,60, 0,58, 0,55, 0,52, 0,50, 0,50, 0,50]`. La actual infravalora la exposición de los slots bajos y por tanto sobrestima la conversión de lo que se muestra ahí.

### M6 · Carrito y recurrentes como señales, no como posiciones fijas — 30 minutos

Hasta medirlos (sección 4):

- **Carrito activo (≤ 30 min):** tratarlo como "visto" reforzado (+2,3 a +3 en la puntuación) y dejar que compita. Si además vende, saldrá primero por sí solo.
- **Carrito olvidado:** `CALYPSO_V4_FORGOTTEN_CART=0`. Si la medición lo respalda, que vuelva como bonus con decaimiento por antigüedad.
- **Recurrentes:** `CALYPSO_V4_RECURRENT=0`. Va contra la pista explícita de las bases y no hay dato que lo apoye.

Mantener la regla de 4.1 que cuenta las compras orgánicas en vivo como "ya comprado" (`CALYPSO_V4_LIVE_EVENT_PURCHASES=1`): esa sí está alineada con las bases.

### M7 · `home`: superventas en vivo y nada más — 15 minutos

`mmr_subcat_lambda_home = 0` (D11) y lista por tasa base de M4. `home` es el 4 % de las compras: no merece más tiempo, salvo que la medición de afinidad de usuario (sección 4) muestre que el historial personal cambia el cuadro.

### M8 · Re-ranker entrenado de verdad — 2–3 horas, después de M3

- Descargar una muestra de 250.000 sesiones (unas 3.800 compras) en vez de 3.000.
- Variables: `ln(puesto+1)`, visto, fuerza de la relación, misma subcategoría, misma marca, afinidad del usuario, encaje de precio, `home`, y el slot como variable de control (fijado a 1 al puntuar).
- Sin recorte a los priors ni mezcla al 45 %: si en validación (últimos días) mejora el log-loss del modelo de 1.4, se usan sus coeficientes tal cual; si no, se quedan los de 1.4.
- Objetivo: compra. Los clics sirven de señal auxiliar si faltan positivos.

### Resumen

| Medida | Qué ataca | Esfuerzo | Evidencia |
|---|---|---|---|
| M1 Sin orden por precio | Orden de la lista | 1 min | Log real + ablación propia |
| M2 Prueba contra baseline | Diagnóstico | 30 min | Estimación 1.5 |
| **M3 Tasa base × relación** | Qué productos salen | 2–3 h | Log real (1.2, 1.4) |
| **M4 Popularidad en vivo** | Final con tendencias nuevas | 1–2 h | Bases + peso de la tasa base |
| M5 Exploración a la cola | Slots 1–5 | 30 min | Log real (1.3) |
| M6 Carrito/recurrentes sin fijar | Slots 1–3 | 30 min | Falta de evidencia; bases |
| M7 `home` sin MMR | 4 % de las compras | 15 min | Log real (1.1) |
| M8 Re-ranker real | Afinar pesos | 2–3 h | — |

Si solo da tiempo a tres: **M1, M3 y M4**.

---

## 4. Lo que falta por medir

Estas señales pueden valer tanto o más que las de la sección 1, y Calypso ya las calcula (afinidad, marca, precio, carrito). Falta saber cuánto pesan en el simulador.

**Más datos, mismo script.** Ampliar la descarga de `run_offline_eval_official.py` a 250.000 filas, guardar también `shop.catalog` como parquet y repetir `analisis_conversion.py`. Con el catálogo se añaden al modelo "misma subcategoría / marca que la última vista" y el precio relativo.

**S6 · Subcategoría y marca respecto a la última vista** (sin ejecutar):

```sql
WITH cat AS (
    SELECT product_id::String AS pid, subcategory::String AS sub, brand::String AS br
    FROM shop.catalog
)
SELECT ci.sub = cl.sub AS misma_subcategoria,
       ci.br = cl.br   AS misma_marca,
       count() AS impresiones,
       sum(comprado) AS compras,
       round(100 * sum(comprado) / count(), 3) AS cvr_pct
FROM (
    SELECT arrayJoin(arrayEnumerate(shown)) AS k,
           shown[k]::String AS item,
           ifNull(purchased::String, '') = item AS comprado,
           recent_views[length(recent_views)].product_id::String AS ultima_vista
    FROM shop.history_sessions
    WHERE length(recent_views) > 0
) AS s
INNER JOIN cat AS ci ON ci.pid = s.item
INNER JOIN cat AS cl ON cl.pid = s.ultima_vista
WHERE s.item != s.ultima_vista
GROUP BY misma_subcategoria, misma_marca
ORDER BY cvr_pct DESC
```

**S7 · Carrito y compras previas del usuario** (sin ejecutar):

```sql
WITH hist AS (
    SELECT user_id::String AS u, product_id::String AS p,
           minIf(ts::DateTime64(3), event_type::String = 'add_to_cart') AS t_carrito,
           minIf(ts::DateTime64(3), event_type::String = 'purchase')    AS t_compra1,
           countIf(event_type::String = 'purchase')                     AS n_compras
    FROM shop.history_events
    WHERE event_type::String IN ('add_to_cart', 'purchase')
    GROUP BY u, p
)
SELECT multiIf(
           h.n_compras >= 2 AND h.t_compra1 < s.t, 'comprado 2+ veces',
           h.n_compras = 1 AND h.t_compra1 < s.t, 'comprado 1 vez antes',
           h.t_carrito > '2000-01-01' AND h.t_carrito < s.t
               AND dateDiff('minute', h.t_carrito, s.t) <= 30, 'carrito <= 30 min',
           h.t_carrito > '2000-01-01' AND h.t_carrito < s.t, 'carrito antiguo',
           'sin historial') AS estado,
       count() AS impresiones,
       sum(comprado) AS compras,
       round(100 * sum(comprado) / count(), 3) AS cvr_pct
FROM (
    SELECT user_id::String AS u, ts::DateTime64(3) AS t,
           arrayJoin(arrayEnumerate(shown)) AS k,
           shown[k]::String AS item,
           ifNull(purchased::String, '') = item AS comprado
    FROM shop.history_sessions
) AS s
LEFT JOIN hist AS h ON h.u = s.u AND h.p = s.item
GROUP BY estado
ORDER BY cvr_pct DESC
```

`n_compras` cuenta todo el histórico, no solo lo anterior a la impresión: sirve como primera aproximación. Lectura: si "carrito antiguo" no convierte claramente más que "sin historial" a igual puesto de ventas, la regla de carrito olvidado se queda apagada; si "comprado 2+ veces" convierte menos que "sin historial", los recurrentes también.

**Afinidad de usuario.** Misma idea que S7, cruzando cada impresión con las subcategorías en las que el usuario compró o vio antes. Es probablemente la mayor fuente de mejora que queda sobre el baseline, y hoy pesa 0,25 sin validar.

---

## 5. Limitaciones

- 25.000 peticiones y 386 compras: las celdas pequeñas de 1.2 tienen pocos casos (se indican los recuentos) y la curva de posición es ruidosa.
- El log es del sistema antiguo, que mostraba sobre todo superventas: de los productos de cola hay pocas impresiones como "vistos".
- El modelo de 1.4 supone efectos multiplicativos e independencia entre productos de la lista. Lo primero se cumple en 1.2; lo segundo, en el rango observado (hasta ~9 % por lista).
- El ranking de ventas usado es el histórico. En la final cambiará: por eso M4.
- Nada de esto sustituye a la prueba en vivo de M2. El sábado no puntúa: cualquier duda se resuelve antes repartiendo tráfico que discutiéndola.
