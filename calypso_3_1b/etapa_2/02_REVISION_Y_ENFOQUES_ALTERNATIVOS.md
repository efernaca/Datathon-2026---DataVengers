# Revisión de la Etapa 2 y enfoques alternativos

**Proyecto:** Datathon FME 2026 · RawTree Real-Time Recommendation
**Fecha de la revisión:** sábado 14-nov-2026, 14:15 CET (quedan ~20 h hasta la ronda final)
**Alcance:** lectura completa de `etapa_2/` (código, SQL, tests y documentación), comparada con `etapa_1/` y con `etapa_1/04_REVISION_Y_ENFOQUES_ALTERNATIVOS.md` (en adelante, "la revisión 04").

> **Límite de esta revisión.** Igual que la anterior: está hecha leyendo el código, sin consultar RawTree. Lo que depende de los datos va marcado como *hipótesis* con la forma de comprobarlo, y el SQL nuevo **no está ejecutado**.

---

## 1. Resumen

La Etapa 2 añade bastante lógica de recomendación: lift en las transiciones, perfiles de usuario, cold start en los dos sentidos, MMR y desempate por ingresos. Varias cosas de la revisión 04 quedan resueltas. El balance, sin embargo, tiene tres problemas:

1. **Se ha añadido modelo sin añadir medida.** Hay ~22 hiperparámetros puestos a mano (antes 12) y ninguno se ha contrastado con datos reales. `eval.py` sigue usando solo un generador sintético, ampliado ahora con marca favorita, nivel de presupuesto y carritos abandonados: justo los mecanismos nuevos del recomendador. El test `test_etapa2_beats_baseline_and_etapa1…` convierte esa circularidad en un requisito. En el repo no hay resultados de EDA, cifras del leaderboard ni registros de ejecución en vivo.
2. **La parte operativa no se ha tocado y ahora pesa más.** El bucle rápido es idéntico al de la Etapa 1. La recarga completa pasa de 5 a 8 consultas pesadas cada 3 minutos, y tres de ellas devuelven filas por usuario. Los productos nuevos siguen tardando hasta 3 minutos o más en existir para el sistema.
3. **Casi toda la inversión es en usuarios conocidos** (perfil, intención, marca, precio). La final trae "new shoppers, new products, new trends": hay que medir hoy cuánto de eso sobrevive.

La propuesta central de este documento: **el sábado no puntúa, así que explorar hoy es gratis.** En vez de seguir deduciendo el simulador a partir de un log sesgado o de un generador propio, se puede experimentar directamente contra él y dejar que el feedback decida qué componentes se quedan (sección 4).

---

## 2. Qué ha pasado con las observaciones de la revisión 04

| Ref. | Tema | Estado en Etapa 2 |
|---|---|---|
| U2 | `serve.py` no encontraba `group_API.env` | **Resuelto**: `config.py` lo carga. |
| U3 | Sin `.gitignore` en la raíz | **Parcial**: existe, pero `group_API.env` y 10 ficheros de `__pycache__` siguen en el índice de git (`git rm --cached`). |
| V1 | Historial de usuario sin usar | **Resuelto**: `UserProfile`. |
| V2 | Compras en vivo sin excluir | **Parcial**: se leen las de nuestro `feedback`; las compras orgánicas de `shop.events` siguen fuera. Ver N3. |
| V3 | Transiciones sin lift ni orden temporal | **Parcial**: hay lift, pero aplicado después de truncar (N5), y el `JOIN` sigue sin comparar `ts`. |
| F3 | La lectura no trae `ts` | **Parcial**: se selecciona `ts`, pero no se filtra por tiempo ni se calcula la edad de la petición. |
| F1, F2, F8 | Peor caso 4,05 s; marcar antes del push; descartar en silencio | **Sin cambios.** |
| F4, F5 | Stock bloqueado durante la recarga; histórico recalculado cada 180 s | **Peor** (N1). |
| F6, F7 | Supervisor y alarma | Sin evidencia en el repo. |
| C1 | Productos nuevos solo en la recarga completa | **Sin cambios.** `with_fast_updates` acepta `products`, `cold_neighbors` y `warm_to_cold`, pero nadie se los pasa. |
| V4, V5, V6 | Contexto sesgado; tendencia ruidosa; descuento contra precio inicial | **Sin cambios.** |
| Sec. 3.4 | Validación sintética; SNIPS en slot 1 | **Sin cambios.** |
| B1–B8 | EDA del simulador | Sin resultados en el repo. |

---

## 3. Hallazgos nuevos

### 3.1 Operación

| # | Problema | Consecuencia | Arreglo |
|---|---|---|---|
| N1 | `refresh_full()` lanza 8 consultas sobre el histórico cada 180 s. `SQL_USER_PROFILES`, `SQL_USER_INTENT` (hasta 10 filas por usuario) y `SQL_USER_AFFINITY` (hasta 8) devuelven resultados proporcionales al número de usuarios. | *Hipótesis: con cientos de miles de usuarios son millones de filas JSON cada 3 minutos.* Tres efectos: (a) más probabilidad de timeout, y un solo fallo descarta toda la recarga; (b) `resp.json()` decodifica en una única llamada en C que **no suelta el GIL**, así que el hilo que responde peticiones se queda parado mientras dura; (c) reconstruir millones de objetos Python dispara recolecciones de basura largas, que también paran todos los hilos. | Calcular el histórico una vez y guardarlo en disco. Mejor aún: separar en dos procesos (sección 4.5). Como parche inmediato, `history_refresh_sec = 86400` y `gc.freeze()` tras la primera carga. |
| N2 | El stock sigue dependiendo del mismo hilo. Con N1, la ventana sin refresco de stock es más larga. | Un producto agotado en la lista invalida la respuesta entera. | Hilo propio para stock y catálogo incremental (consulta S2 de la revisión 04). |
| N3 | `_sql_live_bought` une **todo** `feedback` con **todo** `shop.requests` cada 10 s, con timeout de 3 s. | Las dos tablas crecen ~260.000 filas al día. La consulta irá más lenta cada hora y acabará agotando el timeout justo el domingo; entonces `query_safe` devuelve `None` y las compras dejan de actualizarse sin ningún aviso. | El `user_id` de cada petición ya lo conocemos al responderla: guardar `request_id → user_id` en memoria y leer `feedback` de forma incremental (`WHERE ts > último_ts AND purchased != ''`), sin `JOIN`. Añadir las compras de `shop.events` igual. |
| N4 | Cada recarga escribe hasta 200 filas de vecinos de cold start en `scratch`, las mismas cada 3 minutos, y nadie las lee. | Unas 4.000 filas duplicadas por hora. Para el criterio "Use of RawTree" cuenta cómo el pipeline *lee y escribe*; un registro de solo escritura aporta poco. | Escribir una vez por producto nuevo, y usar `scratch` para algo que el sistema lea de vuelta (sección 4.5). |
| N5 | `etapa_2/` es una copia de `etapa_1/` (`guardrails`, `rawtree_client`, `state`, bucles de `serve`). | Cada arreglo de fiabilidad hay que aplicarlo dos veces, y la Etapa 3 será una tercera copia. | Un solo paquete con la infraestructura común; cada etapa aporta solo su clase `Recommender` y sus consultas. |

### 3.2 Algoritmo

| # | Problema | Consecuencia | Arreglo |
|---|---|---|---|
| N6 | El lift se aplica **después** de truncar: el SQL ordena por puntuación bruta y hace `LIMIT 40 BY src_product`; Python reordena esos 40. | Los 40 que sobreviven son los destinos más populares. Las asociaciones de nicho que el lift debía hacer aflorar ya se cortaron en la consulta. | Calcular el lift en SQL antes del `LIMIT` (S4), con un soporte mínimo para no promocionar coincidencias de un solo evento. |
| N7 | En el cold start inverso, la similitud se normaliza por el máximo de cada producto: `norm_s = val / max_s`. El mejor producto nuevo de cada producto con historial obtiene siempre 1,0 y pasa el filtro `val ≥ 0,9 · max_s` aunque sea de otra subcategoría. | En cuanto existe un solo producto nuevo, **todos** los productos con historial quedan enlazados a él con similitud 1,0, se parezcan o no. Lo mismo ocurre en el sentido directo: el primer vecino vale siempre 1,0. | Conservar la similitud absoluta y exigir un umbral fijo más misma subcategoría (o categoría). |
| N8 | El injerto inverso suma `0,28 · w · sim`; una transición fuerte suma `1,35 · w · t`. | *Hipótesis:* en sesiones con transiciones, un producto nuevo no llega casi nunca al top 10; en sesiones pobres entra el que marque N7. | Medirlo: porcentaje de listas con al menos un producto nuevo, y su posición media. Decidir el peso con el experimento de 4.1. |
| N9 | El "desempate" por ingresos multiplica la puntuación por hasta 1,06. | No es un desempate: reordena cualquier par con menos de un 6 % de diferencia a favor del más caro. Si el simulador es sensible al precio, resta conversión (50 puntos) a cambio de un criterio que solo cuenta en caso de empate exacto en el ranking. Además tira en sentido contrario a `PriceFit`, que acerca al precio habitual del usuario. | `revenue_tie_alpha = 0` hasta medir B4 de la revisión 04. Si se mantiene, que sea lexicográfico: solo entre candidatos con diferencia menor que un ε pequeño. |
| N10 | MMR resta penalizaciones absolutas (0,15 y 0,04 por repetición) a puntuaciones sin escala definida. | Con sesión vacía las puntuaciones rondan 0,1–0,6, así que 0,15 es una penalización enorme; con sesión rica es despreciable. El efecto real depende de una escala que nadie ha fijado. Y no sabemos si la composición de la lista importa al simulador (B6). | Penalización relativa (multiplicar por `1 − λ·n`) y decidir si se queda con el experimento de 4.1. |
| N11 | La recuperación de intención usa carritos de hasta 60 días atrás sin decaimiento, y se suma con independencia de la sesión actual. | Un carrito abandonado hace 8 semanas pesa igual que uno de ayer, y puede desplazar a lo que el usuario mira ahora. | Decaimiento por antigüedad y medir su efecto (4.1). En la final probablemente aplique a pocos usuarios (4.4). |
| N12 | El contexto "Empirical Bayes" suma `3,0 ×` una puntuación normalizada (0–1) a recuentos brutos. | Mezcla dos escalas: con recuentos de cientos el prior no pinta nada; con recuentos de 1–5 solo importa para el primer producto. No es un suavizado bayesiano. Además parte de listas ya truncadas a 35 y hereda el sesgo V4. | `(n + κ·p_padre) / (N + κ)` con tasas, no puntuaciones normalizadas. Prioridad baja. |
| N13 | `self_conversion` es la tasa global compras/vistas del producto, no la probabilidad de comprar un producto *que ya está en `recent_views`*. `time_decay_half_life_sec` es una constante de tiempo, no una semivida (la semivida real es 624 s). | Nombres que no describen lo que se calcula; en la presentación penaliza lo que no se sabe explicar. | Renombrar, o calcular la magnitud que dice el nombre con la consulta S3 de la revisión 04. |

**S4 · Lift calculado antes de truncar, con orden temporal** (sin ejecutar):

```sql
WITH ev AS (
    SELECT session_id::String AS sid, product_id::String AS pid,
           event_type::String AS et, ts::DateTime64(3) AS t
    FROM shop.history_events
),
first_view AS (SELECT sid, pid, min(t) AS t0 FROM ev WHERE et = 'view' GROUP BY sid, pid),
outcome AS (
    SELECT sid, pid, max(et = 'purchase') AS bought,
           max(et = 'add_to_cart') AS carted, max(t) AS t1
    FROM ev GROUP BY sid, pid
),
pop AS (
    SELECT pid, countIf(et = 'purchase') * 5.0 + countIf(et = 'add_to_cart') * 2.0
              + countIf(et = 'view') * 0.2 AS wp
    FROM ev GROUP BY pid
)
SELECT a.pid AS src_product, b.pid AS dst_product,
       count() AS sessions,
       sum(b.bought) * 6.0 + sum(b.carted) * 2.5 + count() AS raw_score,
       raw_score / pow(any(p.wp) + 30.0, 0.45) AS lift_score
FROM first_view AS a
INNER JOIN outcome AS b ON a.sid = b.sid
INNER JOIN pop AS p ON p.pid = b.pid
WHERE a.pid != b.pid AND b.t1 > a.t0
GROUP BY src_product, dst_product
HAVING sessions >= 3
ORDER BY lift_score DESC
LIMIT 40 BY src_product
```

Cuenta cada par una vez por sesión (corrige el doble conteo), exige que B ocurra después de ver A y aplica el lift antes del corte.

### 3.3 Validación

- El generador sintético de la Etapa 2 decide los clics con reglas como "el producto está entre las intenciones no compradas del usuario" o "coincide con su marca favorita". El recomendador busca exactamente eso. Que la Etapa 2 gane a la Etapa 1 ahí está garantizado por construcción.
- Recomendación: mantener el sintético solo como test de integración y latencia, quitar la aserción "Etapa 2 > Etapa 1", y sustituir la validación de calidad por la de la sección 4.

---

## 4. Enfoques alternativos

### 4.1 Experimentación activa contra el simulador (diseño factorial)

**Idea.** La ronda de práctica no cuenta para la nota y dura hasta el domingo a las 10:00. Cada petición de hoy es un experimento gratuito contra el modelo oculto real. En lugar de discutir si MMR, el injerto inverso o el desempate por ingresos ayudan, se encienden y apagan al azar y se mira el feedback.

**Diseño.** En cada petición, cada componente se activa con probabilidad 0,5 de forma independiente (por hash de `request_id` con una sal distinta por componente). Los interruptores se codifican en `model_version`, por ejemplo `e2|mmr1|rev0|rg1|int0|pf1`; la consulta 3 de `monitoring.sql` ya cruza `submissions` con `feedback` por esa columna.

| Interruptor | Qué responde | Hallazgo |
|---|---|---|
| `mmr` | ¿Diversificar ayuda? ¿Solo con sesión vacía? | N10, B6 |
| `rev` | ¿El factor de ingresos cuesta conversión? | N9 |
| `rg` | ¿El injerto inverso de productos nuevos aporta algo? | N7, N8 |
| `int` | ¿Vale la recuperación de carritos antiguos? | N11 |
| `pf` | ¿Ayuda el ajuste al precio habitual? | — |
| `self` | ¿Conviene mostrar lo ya visto, y en qué slot? | B2 |

**Por qué factorial.** Los seis interruptores usan el 100 % del tráfico cada uno. Con A/B secuenciales, cada prueba recibiría una sexta parte.

**Potencia.** Hay ~10.800 peticiones por hora, 5.400 por rama de cada interruptor. *Si el CTR por lista ronda el 10 % (a medir)*, el error estándar de la diferencia es ~0,6 puntos por hora y ~0,18 tras 10 horas: en una noche se detectan diferencias relativas del orden del 4–5 %. Usar clics como métrica principal; las compras son demasiado escasas para efectos pequeños. Analizar por segmento (sesión vacía / con vistas, usuario nuevo / conocido), con la advertencia de que un segmento con el 20 % del tráfico necesita cinco veces más tiempo.

**Aleatorizar el orden.** En un 10 % de las peticiones, barajar el top 10 antes de enviarlo. El CTR por slot bajo orden aleatorio da la curva de atención directamente, sin depender de `propensity`. Es B1, medido en vez de inferido, y hace falta para 4.2.

**Análisis.** Una regresión logística `clic ~ interruptores + segmento` da el efecto de cada componente con su intervalo. Lo que no muestre efecto positivo se apaga para la final: menos piezas que explicar y que puedan fallar.

**Encaje con el jurado.** Es evidencia real para "Method" (cómo se validó) y para "Learning from the stream".

### 4.2 Aprender los pesos con el feedback propio

**Idea.** El recomendador ya calcula una docena de señales por candidato (transición, autoconversión, vecino frío, subcategoría, intención, contexto, popularidad, tendencia, descuento, marca, precio). Hoy se suman con pesos a mano. Cada lista servida produce 10 filas *(señales, posición) → clic / compra*: con eso los pesos se pueden ajustar.

**Cambios necesarios.**

1. `recommend()` devuelve, además de la lista, el vector de señales de cada producto mostrado (hoy solo existe la suma).
2. Un hilo aparte lo escribe en un JSONL local (y, por lotes, en `scratch`). Nunca desde el hilo que responde.
3. Cada hora: unir con `feedback` y ajustar una regresión logística con un término por posición. Los coeficientes son los pesos nuevos.
4. El sistema lee los pesos desde `scratch` y los aplica solo si mejoran en una partición de validación; si no, conserva los anteriores.

**Requisitos.** Solo aprende de lo que se mostró, así que necesita la aleatorización de 4.1 para no confirmar sus propias decisiones. Unas 45.000 peticiones cada 4 horas dan 450.000 filas, suficiente para ~15 coeficientes por segmento.

**Relación con el enfoque A de la revisión 04.** Es la versión barata: reutiliza las señales que ya existen y entrena con la respuesta del simulador *actual*, no con el log del sistema antiguo. El pipeline es el mismo que luego admitiría un modelo más rico.

### 4.3 Una política por segmento

**Idea.** La misma fórmula atiende situaciones muy distintas, y dos de ellas se puntúan aparte (cold start, 10 puntos).

| Segmento | Qué información hay | Qué debería mandar |
|---|---|---|
| Usuario conocido con sesión | Vistas recientes e historial | Señales de sesión; el historial como ajuste |
| Usuario conocido sin sesión | Solo historial | Perfil, intención reciente, tendencia |
| Usuario nuevo con sesión | Solo vistas recientes | Sesión y contenido |
| Usuario nuevo sin sesión | Contexto (`page`, `device`, `region`) | Tendencia en vivo y diversificación |
| Sesión con producto nuevo | Contenido del producto | Vecinos por contenido, validados |

Un vector de pesos por segmento (aprendido con 4.2) en lugar de uno global. El coste es pequeño: el segmento se decide con tres condiciones al entrar en `recommend()`. Además da la estructura de la presentación: "qué hacemos en cada caso y por qué".

### 4.4 Ensayo de la final esta noche

**Idea.** No sabemos cuánta de la personalización sobrevivirá el domingo. Se puede estimar hoy.

1. **Medir la composición del tráfico** (S5): proporción de `is_new_user`, de sesiones vacías y de usuarios que aparecen en `history_events`. Y si ya llegan `new_product` el sábado: en ese caso el segmento frío de producto se puede medir hoy mismo.
2. **Interruptor "modo final"**: en un 20 % de las peticiones, tratar al usuario como nuevo (sin perfil) aunque sea conocido. La diferencia de CTR con el resto es lo que se pierde si el domingo todos los compradores son nuevos. Si la caída es grande, la prioridad pasa a las señales de sesión, contenido y tendencia.
3. **Arranque en frío del propio sistema**: reiniciar una vez con los contadores en vivo a cero y medir cuánto tarda en recuperar el CTR.

**S5 · Composición del tráfico en vivo** (sin ejecutar):

```sql
SELECT is_new_user::Bool AS nuevo,
       length(recent_views) = 0 AS sesion_vacia,
       context.page::String AS page,
       count() AS peticiones
FROM shop.requests
GROUP BY nuevo, sesion_vacia, page
ORDER BY peticiones DESC;

SELECT change::String AS cambio, count() AS n, min(ts) AS primero, max(ts) AS ultimo
FROM shop.catalog_updates
GROUP BY cambio;
```

### 4.5 Dos procesos y agregados materializados en RawTree

**Idea.** Resolver N1–N4 con un cambio de estructura en lugar de parches.

- **Proceso servidor.** Solo lee peticiones, puntúa y envía. Carga el estado desde un fichero local y lo recarga cuando cambia. No decodifica respuestas grandes ni construye estructuras pesadas, así que nada le para el GIL.
- **Proceso constructor.** Hace las consultas pesadas, construye el snapshot y lo escribe en disco de forma atómica (fichero temporal + renombrado). Si se cae o tarda, el servidor sigue con el último válido.
- **Agregados en `scratch`.** Los resultados de las consultas caras (transiciones con lift, popularidad, perfiles) se guardan una vez en `team_NN.scratch` con un campo `kind` y una versión. El arranque lee tablas pequeñas ya agregadas en vez de recalcular sobre 12 M de eventos. Esto sí es "features computed in SQL, how the pipeline reads and writes". Comprobar en la documentación de RawTree si admite `INSERT … SELECT`; si no, el constructor consulta y reinserta.
- **Lo incremental, al servidor ligero.** Stock, catálogo, compras y contadores de tendencia se leen con `WHERE ts > último_ts`: respuestas de pocas filas.

Coste estimado: 2–3 horas. A cambio, reiniciar deja de costar decenas de segundos de fallback y las Etapas 3 en adelante heredan la fiabilidad sin copiar código.

### Comparativa

| Enfoque | Qué resuelve | Esfuerzo | Riesgo | Cuándo |
|---|---|---|---|---|
| **4.1** Experimento factorial | Saber qué componentes funcionan | 1–2 h + una noche de datos | Bajo: hoy no puntúa | Ya |
| **4.4** Ensayo de la final | Saber qué sobrevive el domingo | 1 h | Bajo | Ya, junto a 4.1 |
| **4.5** Dos procesos | Fiabilidad (N1–N4, F4, F5) | 2–3 h | Medio: tocar el arranque | Esta tarde |
| **4.3** Segmentos | Cold start; claridad | 1 h | Bajo | Tras los primeros datos de 4.1 |
| **4.2** Pesos aprendidos | Sustituir los 22 pesos manuales | 3–4 h | Medio: necesita aleatorización | Noche, con datos de 4.1 |

Siguen pendientes de la revisión 04 y encajan aquí: EASE como sustituto de la matriz de transiciones (una señal más para 4.2) y los contadores con decaimiento sobre `shop.events` para la tendencia.

---

## 5. Orden de trabajo sugerido

1. **Ahora (30 min).** Ejecutar S5 y las consultas 1b y 4 de `eda_queries.sql`; guardar los resultados en el repo. Poner `revenue_tie_alpha = 0` y `history_refresh_sec` alto hasta tener el punto 3.
2. **Tarde (2 h).** Interruptores de 4.1, barajado del 10 % y modo final de 4.4, codificados en `model_version`. Arreglar N7 (similitud absoluta) y N3 (compras incrementales sin `JOIN`).
3. **Tarde (2–3 h, en paralelo).** 4.5 junto con F1–F3 y C1 de la revisión 04: es lo que protege los 10 puntos de fiabilidad y los 10 de cold start.
4. **Noche.** El sistema corre con los experimentos activos. Dejar preparado el registro de señales de 4.2.
5. **Domingo 07:00–08:00.** Leer los efectos, apagar lo que no aporta, fijar pesos por segmento, desactivar el barajado y los interruptores, y congelar.
6. **Domingo 08:00–10:00.** Vigilancia y presentación, con las tablas de efectos como evidencia.

Si solo da tiempo a una cosa: el punto 3. Un sistema sencillo que responde siempre a tiempo y conoce los productos nuevos en 2 segundos puntúa más que uno sofisticado que se arriesga a caer en fallback en cada recarga.
