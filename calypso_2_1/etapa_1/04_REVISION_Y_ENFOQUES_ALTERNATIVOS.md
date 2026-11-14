# Revisión de la Etapa 1 y enfoques alternativos

**Proyecto:** Datathon FME 2026 · RawTree Real-Time Recommendation
**Fecha de la revisión:** sábado 14-nov-2026, 13:30 CET (quedan ~20 h hasta la ronda final)
**Alcance:** lectura completa de `etapa_1/` (código, SQL, tests y los tres documentos) contrastada con el `README.md` oficial.

> **Límite de esta revisión.** Está hecha leyendo el código, sin ejecutar consultas contra RawTree. Todo lo que depende de los datos reales aparece marcado como *hipótesis* y va acompañado de la consulta o el experimento que lo confirma. Las consultas SQL nuevas de este documento **no están ejecutadas**: revisad los casts antes de fiaros.

---

## 1. Resumen

La arquitectura de la Etapa 1 es buena: bucle rápido en memoria, snapshot inmutable, guardrails deterministas y el mismo `recommend()` en offline y en vivo. No hay que rehacerla. Los problemas están en otro sitio:

1. **`serve.py` no encuentra las credenciales tal como están guardadas**: busca un `.env` y el fichero se llama `group_API.env`.
2. **El sistema no se ha validado nunca con datos reales.** `eval.py` solo corre contra un simulador sintético escrito por nosotros, que genera los datos con las mismas hipótesis que el recomendador explota. Que "gane al baseline" ahí no dice nada sobre el simulador de los organizadores.
3. **Hay cuatro fallos que cuestan puntos directamente**: los productos nuevos tardan hasta 3 minutos en existir para el sistema (cold start), el stock deja de refrescarse durante cada recarga completa (fiabilidad), el peor caso del ciclo supera los 3 s (fiabilidad) y el historial del usuario solo se usa para excluir compras (conversión).
4. **El enfoque de fondo es mejorable.** Ahora mismo es una suma de 12 pesos puestos a mano. `shop.history_sessions` son 2 M de listas mostradas con su resultado, generadas por *el mismo modelo oculto que nos puntúa*: lo más rentable es aprender ese modelo directamente (sección 4, enfoque A).
5. **Sobra presupuesto de cómputo.** Usamos ~0,3 ms de los ~1.500 ms disponibles por petición. Podemos puntuar los 5.000 productos en cada petición en lugar de ~100 candidatos.

Orden recomendado: sección 2 (hoy, 30 min) → sección 3 fiabilidad y cold start (2–3 h) → EDA real (1–2 h) → enfoque A con el C como apoyo → adaptación en vivo (D). El plan con horas está en la sección 6.

---

## 2. Puesta en marcha y limpieza del repositorio

| # | Qué pasa | Qué hacer |
|---|---|---|
| U1 | `group_API.env` (con `TEAM_ID` y `RAWTREE_API_KEY`) está versionado. El repositorio es privado, así que hoy no hay exposición. Queda en el historial de git, y las bases dicen que los mejores equipos comparten el repositorio con el jurado. | Sin urgencia. Antes de abrir o compartir el repo: `git rm --cached group_API.env` y añadirlo al `.gitignore`. Si se llega a hacer público con el historial intacto, pedir entonces una clave nueva. |
| U2 | `config.py` busca un fichero `.env`; `group_API.env` no está entre los candidatos, así que `serve.py` arranca sin credenciales y sale con error. | Renombrar a `.env` (ya ignorado por el `.gitignore` del starter, pero no por uno propio en la raíz: crearlo). |
| U3 | No hay `.gitignore` en la raíz y `__pycache__/` está versionado. | `.gitignore` con `.env`, `*.env`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`. |
| U4 | `03_FUNCIONAMENT_ACTUAL.md` enlaza con `file:///c:/Users/Èric.../OneDrive...`: enlaces rotos en GitHub y con una ruta personal. | Sustituir por rutas relativas (`src/serve.py`). El jurado lee el repositorio. |
| U5 | No hay nada que reinicie el proceso si muere, ni aviso si empieza a caer en fallback. "Shoppers keep arriving while you sleep." | Ver F6 y F7 más abajo. |

---

## 3. Hallazgos en el código

Ordenados por el componente de la nota al que afectan. "Coste" es tiempo estimado de arreglo.

### 3.1 Fiabilidad (10 pts, y además cada fallback puntúa como baseline)

| # | Problema | Consecuencia | Arreglo | Coste |
|---|---|---|---|---|
| F1 | El bucle rápido es secuencial: lectura (timeout 1,2 s) + push (1,4 s) + reintento (0,05 s + 1,4 s). | Peor caso 4,05 s > 3 s. Mientras un push lento reintenta, tampoco se leen las peticiones nuevas, así que un solo tropiezo de red retrasa varias tandas. | Separar lectura y escritura: el hilo principal lee y puntúa; un pool de 2–3 hilos hace los push. Reintentar solo si la petición más antigua del lote tiene < 2,2 s. | 1 h |
| F2 | `_mark_unseen` marca las peticiones como vistas **antes** de que el push tenga éxito. | Si el push falla dos veces, esas peticiones no se reintentan nunca aunque quede tiempo. | Marcar como vistas al confirmar el push, o reencolar las que aún tengan edad < 2,2 s. | 20 min |
| F3 | La consulta de lectura no trae `ts` ni filtra por tiempo (`ORDER BY ts DESC LIMIT 50` sobre toda la tabla). | No sabemos la edad de cada petición, no podemos descartar las que ya van tarde, y la consulta ordena una tabla que crece ~260.000 filas al día. | Ver consulta S1. Registrar `age_ms` al leer y al enviar. | 20 min |
| F4 | El stock se refresca en el mismo hilo que `refresh_full()`, y junto a la consulta de tendencias (que escanea 195 min de eventos). | Cada 180 s se relanzan las cinco consultas pesadas sobre el histórico (timeout 30 s cada una). Mientras duran, el stock no se actualiza: un solo producto agotado invalida la respuesta entera. Fuera de esa ventana, el retardo real es ~3 s o más, no 2 s. | Un hilo solo para stock y catálogo, cada 1 s, sin nada pesado al lado. | 45 min |
| F5 | El histórico (60 días, no cambia) se recalcula entero cada 3 minutos. Si una de las cinco consultas falla, se descarta toda la recarga. La primera `refresh_full()` en `main()` no tiene `try`: si falla, el proceso no arranca. | Carga inútil sobre RawTree, arranque frágil, y reiniciar cuesta decenas de segundos de fallback. | Calcular el histórico **una vez**, guardarlo en disco (pickle/parquet) y cargarlo al arrancar. En vivo, solo lo incremental (`WHERE ts > último_ts`). Refresco por componente conservando el último válido. | 1,5 h |
| F6 | Sin supervisor. | Una excepción no capturada o un corte a las 4:00 = fallback hasta que alguien mire. | Bucle `while true; do python …; sleep 1; done` dentro de `tmux`/`systemd`, en una máquina que no se suspenda. | 15 min |
| F7 | Sin alarma. La monitorización es SQL manual. | No nos enteramos de una degradación. | Un script aparte que cada minuto lea `feedback` y avise (sonido, Telegram) si `served_by='team'` < 98 % o p95 > 2 s. | 30 min |
| F8 | Si `validate_submission_record` falla, la respuesta se descarta en silencio. | Fallback `missing` en vez de enviar algo válido. | En ese caso enviar `global_ranked[:10]` filtrado y registrar el incidente. | 10 min |

**S1 · Lectura con edad de la petición** (sin ejecutar):

```sql
SELECT request_id, ts, user_id, is_new_user, context, recent_views,
       dateDiff('millisecond', ts::DateTime64(3), now64(3)) AS age_ms
FROM shop.requests
WHERE ts::DateTime64(3) > now64(3) - INTERVAL 4 SECOND
ORDER BY ts ASC
```

Dos comprobaciones antes de confiar en `age_ms`: que el reloj de RawTree y el `ts` de la petición estén alineados (comparar con `latency_ms` de `feedback`), y si conviene `ASC` (primero las que caducan antes) o descartar directamente las de más de ~2,5 s.

Sobre redundancia: las bases dicen que con varias respuestas a una petición "only one counts", sin decir cuál. Antes de lanzar dos instancias en paralelo, **preguntarlo en el stand**. La alternativa segura es una instancia de reserva que solo responde si el latido de la principal lleva más de 2 s parado.

### 3.2 Cold start (10 pts)

Una petición es fría si el usuario es nuevo o si `recent_views` contiene un producto dado de alta durante el evento. El código actual cubre mal los dos casos.

| # | Problema | Consecuencia | Arreglo | Coste |
|---|---|---|---|---|
| C1 | Los `new_product` solo entran al snapshot en `refresh_full()` (cada 180 s más lo que tarden las consultas). `catalog_refresh_sec = 30` y `trends_refresh_sec = 20` existen en `config.py` pero no se usan en ningún sitio. `01_PLANIFICACIO` describe 2 s / 30 s / 120 s; el código hace otra cosa. | Durante hasta ~3 min, un producto nuevo en `recent_views` no aporta nada: `products.get()` devuelve `None`, no hay transiciones ni vecinos. Es justo la petición que define el segmento frío. | Consulta incremental a `catalog_updates` cada 1–2 s (S2). Para cada producto nuevo, calcular sus vecinos en el momento: un `transform` de TF-IDF para un texto son milisegundos. | 1 h |
| C2 | Usuario nuevo sin `recent_views`: la lista es popularidad por contexto más popularidad global. | Prácticamente la lista del baseline, que por definición puntúa 0 en este segmento. | Es el caso donde más pesan las tendencias en vivo y la diversificación (enfoques D y E). | — |
| C3 | Un producto nuevo solo puede recomendarse si el usuario ya lo ha visto. | Si el modelo oculto hace atractivos los productos nuevos (probable: es lo que quieren medir), nunca los mostramos. | Prior heredado de los vecinos más exploración acotada (enfoque D). | 1–2 h |
| C4 | Los bonus de similitud (0,35 subcategoría, 0,15 categoría, 0,10 marca, 0,15 precio) no están validados. | No sabemos si los vecinos elegidos son buenos. | Test offline: tratar como "nuevos" un 5 % de productos con historial (ocultar sus transiciones) y medir cuánto de su comportamiento real recupera el método. | 1 h |

**S2 · Catálogo incremental** (sin ejecutar):

```sql
SELECT ts, change, product_id, category, subcategory, brand,
       price_eur, description, is_active
FROM shop.catalog_updates
WHERE ts::DateTime64(3) > '{ultimo_ts_visto}'
ORDER BY ts ASC
```

Se aplica sobre el estado en memoria con `with_fast_updates(products=…, cold_neighbors=…)`, que ya admite esos parámetros.

### 3.3 Conversión (50 pts)

| # | Problema | Consecuencia | Arreglo |
|---|---|---|---|
| V1 | `user_subcat_rows` nunca se pasa a `build_snapshot_from_records`, así que `user_subcat_affinity` está siempre vacío. | Del historial de 60 días de cada usuario solo usamos la lista de compras para excluir. Un usuario conocido que llega a `home` sin vistas recibe lo mismo que un desconocido. | Añadir la consulta de afinidad (subcategoría, marca, rango de precio por usuario) y pasarla. Ojo con la nota de la sección 5 sobre cuánto vale esto en la final. |
| V2 | `user_bought` sale solo de `history_events`. | Las compras hechas durante el evento no se excluyen. Las bases dicen expresamente que quien ya compró un producto tiende a no quererlo otra vez. | Sumar de forma incremental las compras de `shop.events` y las de nuestro `feedback`. |
| V3 | Las "transiciones dirigidas" no lo son: el `JOIN` por sesión no compara `ts`, cuenta B aunque ocurra antes que A, y cuenta doble si A se vio varias veces. No hay normalización por popularidad del destino. | Los destinos más populares dominan casi todas las filas, y la lista tiende a parecerse a los más vendidos. | Añadir `a.ts < b.ts`, deduplicar por sesión y dividir por `pop(B)^β` (el lift que ya propone `02_PROPOSTES`). O sustituir por EASE (enfoque C). |
| V4 | `context_ranked` cuenta compras de `history_sessions`. | Solo puede contener productos que el carrusel antiguo mostraba: reproduce su sesgo hacia los populares. | Calcularlo con `history_events` (orgánico) o ponderar cada compra por `1/propensity`. |
| V5 | Tendencia = cociente con denominador mínimo 1, normalizado por el máximo. | Un producto con 3 eventos y base casi nula da un cociente enorme; al normalizar por ese máximo, el resto queda cerca de 0. La señal es casi ruido. | Contadores con decaimiento exponencial (semivida 30–60 min) y un z-score de Poisson o `log((n_reciente+κ)/(n_esperado+κ))`. |
| V6 | El descuento se mide siempre contra el precio inicial. | Una bajada de hace horas pesa igual que una reciente; una subida se ignora. | Usar el cambio respecto al precio anterior con decaimiento, y medir en el histórico si la conversión depende del precio (ver B4). |
| V7 | Doce pesos manuales que suman señales en escalas distintas. | Imposible saber si la combinación es buena; cada señal nueva obliga a reajustar a ojo. | Enfoque A: que los pesos los aprenda un modelo. |
| V8 | Reglas como "atenuar 0,6 el último visto en página de producto" o el peso 0,75 al producto ya visto. | Son hipótesis razonables sin comprobar. | La consulta 4 de `eda_queries.sql` ya lo mide: ejecutarla y anotar el resultado en el repo. |

### 3.4 Evaluación offline (10 pts del jurado: "How was it validated offline?")

- **Validación circular.** `generate_synthetic_dataset` crea usuarios con una subcategoría favorita y pares de co-compra dentro de la misma subcategoría y marca; el recomendador busca exactamente eso. Sirve como test de integración y de latencia, no como evidencia de calidad. Falta un cargador que alimente `evaluate_stratified` con `history_sessions` reales.
- **SNIPS en el slot 1 no discrimina políticas personalizadas.** Solo cuentan las filas donde nuestro primer producto coincide con el que mostró el sistema antiguo. Como el sistema antiguo muestra casi siempre populares, una política que personaliza coincide muy pocas veces: `slot1_matches` será pequeño y el error estándar, enorme. Bien que se reporte, pero no servirá para ordenar modelos.
- **HitRate sobre clics registrados favorece imitar al sistema antiguo.** Solo se puede clicar lo que se mostró, y se mostraban populares. Una política mejor que recomiende productos que el log nunca enseñó sale penalizada.
- **Alternativa**: evaluación a nivel de ítem y posición, que aprovecha las diez posiciones de cada fila (sección 4, A3).

---

## 4. Enfoques alternativos

Cinco enfoques, no excluyentes. La tabla final los compara.

### A. Aprender el simulador (modelo de respuesta)

**Idea.** El oráculo (100 puntos) es "los 10 productos que el propio modelo oculto elegiría". `history_sessions` contiene 2 M de peticiones × 10 posiciones = 20 M de ejemplos *(contexto, producto, posición) → clic, compra* producidos por ese mismo modelo. La aleatoriedad del sistema antiguo y la columna `propensity` hacen que se pueda aprender de ellos sin heredar su sesgo. En vez de inventar una fórmula de puntuación, se entrena un modelo que imite al simulador y se ordena por su predicción.

**A1 · Datos de entrenamiento.** Una fila por (petición, posición): todas las positivas (clic o compra) y una muestra del 5–10 % de las negativas, con peso para corregir el muestreo. La explosión con `arrayJoin(arrayEnumerate(shown))` se hace en RawTree (ya está en la consulta 1b del EDA) y se descarga solo la muestra.

**A2 · Variables.**

| Grupo | Variables |
|---|---|
| Posición | `slot` (en entrenamiento; fijo a 1 al predecir, para que el modelo separe "se vio" de "gustó") |
| Sesión | ¿el producto está en `recent_views`? ¿hace cuánto? ¿misma subcategoría / marca que la última vista? nº de vistas; similitud EASE o de contenido con las vistas |
| Producto | categoría, subcategoría, precio, precio relativo dentro de la subcategoría, popularidad orgánica, CVR suavizada, antigüedad, descuento reciente |
| Usuario | compras y vistas previas en esa subcategoría / marca, precio medio de compra, ¿ya lo compró?, `is_new_user` |
| Contexto | `page`, `device`, `region`, hora |

**A3 · Modelo y validación.** LightGBM (≤ 200 árboles, profundidad ≤ 6) o regresión logística con cruces, con peso `1/propensity` recortado. Corte temporal: entrenar con los primeros 50 días, validar con los últimos 10. Métricas: log-loss y AUC por segmento (`ALL` / `WARM` / `COLD`), **curva de calibración** (el score oficial usa probabilidades, así que importa que estén bien calibradas) y, para comparar políticas, el valor estimado de una lista como `Σ_k e_k · p̂(ítem_k)`, con `e_k` la curva de atención por posición medida en B1.

**A4 · Servir.** Si la atención decrece con la posición, ordenar por `p̂` descendente es óptimo para compras esperadas. Con ~30 variables, puntuar los 5.000 productos de una petición con un modelo lineal es un producto de matrices (microsegundos); con LightGBM, 5.000 filas son unos pocos ms. A 3 peticiones/s cabe de sobra, y elimina el riesgo de que el producto bueno no esté entre los candidatos.

**Riesgos.** (1) El log cubre mal los productos poco populares: para ellos el modelo extrapola por contenido y por señales orgánicas, que entran como *variables*, no como etiquetas. (2) Si el mundo de la final difiere mucho del histórico, las variables de identidad (producto, usuario) valen poco y las estructurales (coincidencia con la sesión, precio relativo, categoría) valen mucho: conviene entrenar una variante **sin** variables de identidad y compararla. (3) Todo lo que no sepamos explicar penaliza: guardar la importancia de variables y dos o tres ejemplos para la presentación.

### B. Ingeniería inversa del simulador (EDA como experimentos)

**Idea.** El simulador es código sintético con reglas probablemente sencillas. Cada regla que identifiquemos se convierte en una decisión de diseño o en una variable del enfoque A. Coste: 1–2 h de consultas. Es lo primero que hay que hacer con la clave en la mano.

| # | Pregunta | Cómo medirla | Decisión que cambia |
|---|---|---|---|
| B1 | ¿Cómo cae la atención con la posición? | CTR por slot ponderado por `1/propensity` (para que no se mezcle con que los populares van arriba). | Cuánto vale acertar el slot 1 frente al 5; los `γ_k` del bandit. |
| B2 | ¿Se compra lo ya visto? | CTR y CVR de un ítem mostrado según esté o no en `recent_views` (S3). | El peso de "re-engagement", hoy fijado a mano en 0,75. |
| B3 | ¿Importa la afinidad de subcategoría / marca con la sesión? | Igual que B2, agrupando por "misma subcategoría que la última vista". | Si el carrusel debe ser de sustitutos o de complementos. |
| B4 | ¿Hay sensibilidad al precio? | CVR dado clic por decil de precio relativo dentro de la subcategoría. | Señal de descuento; desempate por ingresos. |
| B5 | ¿Compra implica clic? ¿Como mucho uno por petición? | `countIf(purchased != '' AND purchased != clicked)`. | Si se modela compra directamente o clic × compra-dado-clic. |
| B6 | ¿El resultado de un ítem depende de sus vecinos en la lista? | CTR del slot k según la popularidad de los ítems de arriba. | Si hace falta diversificar (enfoque E) o basta con ordenar. |
| B7 | ¿Cuánta personalización real hay por usuario? | ¿Predice la subcategoría favorita histórica del usuario su próximo clic, a igualdad de sesión? | Cuánto invertir en V1. |
| B8 | ¿Se parece el stream del sábado al histórico? | CTR/CVR de un mismo modelo en validación histórica frente a `feedback` en vivo. | Cuánto fiarse del histórico en la final (enfoque D). |

**S3 · Respuesta por ítem según relación con la sesión** (sin ejecutar):

```sql
SELECT in_recent, slot, count() AS imps,
       sum(w) AS imps_w,
       sum(w * (clicked_id = item)) / sum(w)   AS ctr_ips,
       sum(w * (purchased_id = item)) / sum(w) AS cvr_ips
FROM (
    SELECT ifNull(clicked::String, '')   AS clicked_id,
           ifNull(purchased::String, '') AS purchased_id,
           arrayJoin(arrayEnumerate(shown)) AS slot,
           shown[slot]::String AS item,
           1.0 / greatest(propensity[slot]::Float64, 0.01) AS w,
           has(arrayMap(x -> x.product_id::String, recent_views), item) AS in_recent
    FROM shop.history_sessions
)
GROUP BY in_recent, slot
ORDER BY in_recent, slot
```

Guardar los resultados en `etapa_1/eda/` como tablas pequeñas: son la base de la parte de "Method" de la presentación.

### C. Factorización ítem-ítem sobre todo el catálogo (EASE / embeddings)

**Idea.** Sustituir la matriz de transiciones con `LIMIT 30` y pesos 6 / 2,5 / 1 por un modelo ítem-ítem aprendido. Con 5.000 productos, **EASE** encaja muy bien: es una regresión ridge de cada producto sobre los demás, con solución cerrada.

- Matriz X sesión × producto a partir de `history_events` (binaria, o con peso por tipo de evento). `G = XᵀX` es la matriz de co-ocurrencias, que se puede agregar en RawTree.
- `P = (G + λI)⁻¹`, `B = −P / diag(P)` con diagonal a 0. Invertir 5.000 × 5.000 en NumPy tarda segundos.
- En vivo: `scores = Σ_{v ∈ recent_views} w_v · B[v, :]`, una suma de filas sobre todo el catálogo.
- Un solo hiperparámetro (λ), se explica en una frase y descuenta por construcción a los productos populares (resuelve V3).

**Productos nuevos.** En vez de los bonus ad hoc de `cold_start.py`, aprender una regresión del contenido (TF-IDF + subcategoría + marca + log-precio) a la fila de `B` de cada producto con historial, y aplicarla al producto nuevo. El test C4 dice si mejora a los vecinos actuales.

**Encaje.** Sirve solo como recomendador de sesión y, mejor, como variable de entrada del enfoque A. Los embeddings y la matriz se pueden guardar en `team_NN.scratch` (criterio "Use of RawTree").

### D. Adaptación en vivo dimensionada a los datos que habrá

**Idea.** En la final cambian compradores, productos y tendencias. La pregunta útil no es "¿qué bandit?" sino "¿cuántos parámetros se pueden aprender con los datos de dos horas?".

- **Feedback propio:** ~21.600 peticiones en 2 h. Si la conversión ronda el 2 % (*a medir*), son ~400 compras. Con eso se aprenden unas decenas de parámetros, no una tasa por producto para 5.000 productos.
- **`shop.events`:** ~50 eventos/s ≈ 360.000 eventos en 2 h, compartidos y orgánicos. Ahí sí hay señal por producto.

De ahí sale un reparto de papeles:

1. **Nivel de producto ← `shop.events`.** Popularidad en vivo con decaimiento y suavizado hacia un prior: `pop(i) = (n_vivo(i) + κ · prior(i)) / (N_vivo + κ)`. El prior es la popularidad histórica para productos conocidos y la media de sus vecinos para los nuevos. Al empezar la final manda el prior; a medida que llegan eventos, mandan los datos. Resuelve V5 y C3 con una sola fórmula.
2. **Nivel de estrategia ← `feedback`.** Thompson sampling sobre pocos brazos: 3–5 variantes de mezcla (por ejemplo, cuánto pesa lo vivo frente a lo histórico) por segmento (frío / templado × `page`). Con ~12 segmentos × 4 brazos sí hay datos. Usar los **clics** como señal principal, que son más frecuentes que las compras.
3. **Exploración acotada.** El score usa la probabilidad esperada de compra, así que cada producto de relleno cuesta. Explorar solo en los slots 7–10 y solo con productos nuevos o en tendencia.

**Ensayo general.** El sábado es la única ocasión de probar el cambio de régimen: medir B8 y, si se puede, arrancar el sistema "en frío" (sin contadores en vivo) para ver cuánto tarda en converger.

### E. Optimizar la lista, no los productos

**Idea.** El objetivo es que haya compra en la lista, no la suma de relevancias. Cuando la intención es clara (hay `recent_views`), concentrar la lista es correcto. Cuando no se sabe nada (usuario nuevo en `home`), diez productos de la misma categoría fallan todos a la vez si no era esa.

- Modelar la intención desconocida como mezcla de categorías con probabilidad `π_c` (por contexto y tendencia en vivo).
- Selección voraz: en cada slot, elegir el producto que más aumenta `Σ_c π_c · [1 − Π_k (1 − e_k · p(ítem_k | c))]`. Es la versión con fundamento del MMR que propone `02_PROPOSTES`, y se reduce a ordenar por `p` cuando `π` está concentrada.
- **Desempate por ingresos:** entre candidatos con probabilidad casi igual, el más caro. No cambiar conversión por precio; es solo desempate.

Solo merece la pena si B6 indica que la composición de la lista importa. Si los ítems se evalúan de forma independiente, ordenar por `p̂` basta.

### Comparativa

| Enfoque | Qué ataca | Esfuerzo | Riesgo | Criterio del jurado que alimenta |
|---|---|---|---|---|
| **A** Modelo de respuesta | Conversión (y cold, vía contenido) | 4–6 h | Medio: cobertura del log, cambio de régimen | Method |
| **B** EDA experimental | Todo (dirige el resto) | 1–2 h | Bajo | Method, Use of RawTree |
| **C** EASE / embeddings | Conversión con sesión; cold de producto | 2–3 h | Bajo | Method, Use of RawTree |
| **D** Adaptación en vivo | Final con datos nuevos; cold | 3–4 h | Medio: ruido, coste de explorar | Learning from the stream |
| **E** Lista diversificada | Cold de usuario sin sesión | 1–2 h | Bajo, si B6 lo respalda | Method |
| Actual (pesos manuales) | — | hecho | Alto: sin validar | — |

**Recomendación.** B primero, porque decide lo demás. Después C como señal de sesión (rápido y seguro), A como modelo que combina todo, D para la final y E solo si el EDA lo justifica. Si el tiempo aprieta, C + D sobre la infraestructura actual es una versión defendible sin A.

No doy ganancias en puntos por enfoque porque no hay datos para estimarlas. Las cifras de `02_PROPOSTES_MILLORA_FUTURES.md` ("+15–25 pts") tampoco están medidas; conviene sustituirlas por resultados reales del leaderboard a medida que se desplieguen.

---

## 5. Decisiones estratégicas que conviene tomar ya

**Optimizar para la final, no para el sábado.** Solo puntúan las dos horas del domingo, con "new shoppers, new products, new trends". Medir hoy qué fracción de peticiones trae `is_new_user = true` y `recent_views` vacío. Si en la final casi todos los compradores son nuevos, la personalización por historial (V1) vale poco y la calidad de las señales de sesión, de contenido y en vivo lo es casi todo. Mantener en `eval.py` un escenario "todos los usuarios como nuevos" para no optimizar algo que desaparece el domingo.

**Cómo comparar versiones en vivo.** Con una conversión del 2 %, detectar una mejora relativa del 10 % con el feedback propio requiere ~78.000 peticiones por variante (unas 14 h con el tráfico repartido en dos); una del 20 %, ~20.000 (unas 3,5 h). Los A/B por compras solo ven diferencias grandes. Tres recursos:

- El leaderboard usa la probabilidad esperada del modelo oculto ("No luck"), no la compra realizada: es una medida con mucho menos ruido que nuestro feedback. Desplegar una variante cada vez durante ventanas fijas y leer el leaderboard. Comprobar antes si muestra el acumulado o la última ventana.
- Comparar por clics, que son más frecuentes.
- Repartir el tráfico por `hash(request_id)`, no por tiempo, y etiquetar con `model_version` (la consulta 3 de `monitoring.sql` ya lo cruza).

**Congelar a tiempo.** Las bases piden tener la mejor versión funcionando a las 10:00. Último despliegue de modelo a las 08:00; de 08:00 a 10:00 solo vigilancia. Durante la final, tocar algo solo si la luz se pone roja.

---

## 6. Plan propuesto (sábado 13:30 → domingo 12:00)

| Franja (CET) | Trabajo | Resultado |
|---|---|---|
| Sáb 13:30–14:30 | U2–U3. Arrancar `serve.py` tal cual contra RawTree con supervisor (F6). | En el leaderboard, con un proceso que se reinicia solo. |
| Sáb 14:30–17:00 | F1–F5, C1, V2. En paralelo otra persona: EDA B1–B7 con resultados guardados. | ~100 % de respuestas a tiempo; productos nuevos visibles en < 2 s; reglas del simulador documentadas. |
| Sáb 17:00–19:00 | Cargador de datos reales para `eval.py`. Enfoque C (EASE) y test C4. Alarma F7. | Primera comparación offline con datos reales; v2 desplegada. |
| Sáb 19:00–23:00 | Enfoque A: dataset, entrenamiento, calibración, variante sin identidad. | v3 desplegada en una parte del tráfico. |
| Sáb 23:00–02:00 | Enfoque D: contadores con decaimiento sobre `shop.events`, priors para nuevos, Thompson por segmento. E si B6 lo pide. | v4. Dejar la mejor versión estable para la noche. |
| Noche | Sistema solo, con alarma. | Datos de una noche entera por versión. |
| Dom 07:00–08:00 | Revisar la noche, elegir versión, último despliegue. | Versión final congelada. |
| Dom 08:00–10:00 | Vigilancia. Preparar la presentación con los resultados de B y las comparaciones entre versiones. | — |
| Dom 10:00–12:00 | No tocar salvo luz roja. | — |

### Comprobaciones antes de las 10:00 del domingo

- [ ] Si el repo se va a compartir con el jurado: `group_API.env` fuera del índice de git.
- [ ] Máquina enchufada, sin suspensión, con red estable; proceso bajo supervisor.
- [ ] Histórico en caché de disco: reinicio completo en menos de 10 s.
- [ ] Últimas 6 h: `served_by = 'team'` ≥ 99,5 % y p95 de `latency_ms` < 2.000.
- [ ] Probado a mano: matar el proceso, cortar la red 10 s, que una consulta lenta agote su timeout.
- [ ] Un `new_product` aparece en el snapshot en menos de 2 s y tiene vecinos.
- [ ] Un producto que pasa a `out_of_stock` deja de recomendarse en menos de 2 s.
- [ ] Las compras en vivo del usuario se excluyen.
- [ ] Alarma activa y alguien asignado a cada franja.
- [ ] `01`, `02` y `03` actualizados para que describan lo que el código hace de verdad.
