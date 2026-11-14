# Etapa 1 · versión especial (Etapa 1 + todas las mejoras de la revisión 04)

**Versión:** `etapa1-especial` · **Base:** el modelo de `etapa_1/` · **Guía:** [`etapa_1/04_REVISION_Y_ENFOQUES_ALTERNATIVOS.md`](../etapa_1/04_REVISION_Y_ENFOQUES_ALTERNATIVOS.md)

La arquitectura de la Etapa 1 se mantiene (bucle rápido en memoria, snapshot inmutable, guardrails, mismo código offline y en vivo). Lo que cambia es lo que la revisión señalaba: fiabilidad, cold start, señales mal construidas y, sobre todo, que los 12 pesos a mano se sustituyen por un **modelo de respuesta aprendido del propio log del simulador** (enfoque A), con EASE (C), adaptación en vivo (D) y lista diversificada (E) alrededor.

> **Estado de validación.** Todo el código está probado con 17 tests y con un ensayo de punta a punta contra un RawTree local (ClickHouse embebido, `chdb`, con columnas `Dynamic` como RawTree). **No se ha ejecutado aún contra RawTree real**: la primera ejecución real es `python -m src.eda` (sección 3) y hay que revisar los logs. Las cifras de la sección 5 salen de un mundo sintético y **no predicen el leaderboard**.

---

## 1. Cómo funciona, en una frase por pieza

| Pieza | Qué hace | Fichero |
|---|---|---|
| Histórico | 7 componentes calculados en RawTree **una vez** y guardados en `cache/` (arranque en segundos). Si uno falla se conserva el anterior. | `src/history.py` |
| Catálogo vivo | Lee `catalog_updates` cada 0,5 s: stock, precios y productos nuevos **con vecinos en el momento**. | `src/catalog.py`, `src/cold_start.py` |
| Señales vivas | Contadores con decaimiento sobre `shop.events`: popularidad suavizada hacia el prior y tendencia con z-score suave. Compras en vivo para excluir. | `src/live.py` |
| Variables | 30 variables (sesión, producto, usuario, contexto, posición), vectorizadas: se puntúan **los ~5.000 productos** en ~1,5 ms. | `src/features.py` |
| Modelo | Logística o GBM entrenados con `history_sessions` (corte temporal, IPS, sin fuga). Sin modelo, pesos a mano equivalentes a la Etapa 1. | `src/model.py`, `src/train.py` |
| EASE | Matriz ítem-ítem con solución cerrada sobre co-ocurrencias calculadas en SQL. | `src/ease.py` |
| Política | p̂ del modelo + tendencia y bajada de precio en vivo + Thompson por segmento + exploración en slots 8/10 + lista diversificada sin sesión + guardrails. | `src/policy.py`, `src/bandit.py` |
| Servicio | Hilos separados: lectura/puntuación, pool de push, catálogo, eventos, feedback. | `src/serve.py` |
| Operación | Supervisor, alarma, instancia de reserva por latido. | `run_forever.ps1/.sh`, `src/monitor.py` |
| Medida | EDA como experimentos, evaluación offline con datos reales, test C4, ensayo local. | `src/eda.py`, `src/eval.py`, `src/simulate.py` |

## 2. Qué se ha hecho con cada punto de la revisión

✅ implementado y probado (tests o simulación) · 🟡 implementado, falta comprobarlo con datos reales · ✋ acción manual

### Puesta en marcha (sección 2)

| # | Estado | Qué se ha hecho |
|---|---|---|
| U1 | ✋ | `group_API.env` ya está en `.gitignore`, pero sigue en el índice. Antes de compartir el repo: `git rm --cached group_API.env` (sección 6). |
| U2 | ✅ | `config.py` lee `.env` y `group_API.env`, también con finales de línea Windows (test `test_loads_group_api_env_with_crlf`). |
| U3 | ✋ | `.gitignore` añade `etapa_1_version_especial/cache/`, `etapa_1_version_especial/logs/`. Quitar del índice los `__pycache__` ya versionados (sección 6). |
| U4 | ✅ | Este README usa rutas relativas. En `etapa_1/03_FUNCIONAMENT_ACTUAL.md` se han cambiado los enlaces `file:///…` por rutas relativas. |
| U5 | ✅ | Ver F6 y F7. |

### Fiabilidad (3.1)

| # | Estado | Qué se ha hecho |
|---|---|---|
| F1 | ✅ | Lectura y push desacoplados: el hilo principal lee y puntúa; un pool de 3 hilos hace los push. Reintento solo si la petición más antigua tiene < 2,2 s; timeout del push ajustado al tiempo que queda. En el ensayo con un 25 % de push fallidos (0,8 s cada uno): 96 % a tiempo, latencia máx. 2,7 s. |
| F2 | ✅ | Una petición se marca como hecha **solo al confirmar el push**. Si falla y aún está a tiempo, se relee y se reenvía **la misma respuesta** (caché por `request_id`). |
| F3 | ✅ | Consulta S1: solo los últimos 4 s, `ORDER BY ts ASC`, `age_ms` calculada por el servidor. Se registran p50/p95 de edad al leer y al confirmar, y la diferencia con `latency_ms` del feedback (alineación de relojes). Si S1 falla 5 veces seguidas, cae a la lectura antigua. |
| F4 | ✅ | Hilo propio para stock y catálogo cada 0,5 s, sin nada pesado al lado. |
| F5 | ✅ | Histórico calculado una vez → `cache/history_full.pkl`. Refresco por componente conservando el último válido. El arranque reintenta en bucle en vez de morir. |
| F6 | ✅ | `run_forever.ps1` (Windows) y `run_forever.sh` (tmux). Instrucciones para desactivar la suspensión. |
| F7 | ✅ | `python -m src.monitor`: cada minuto, alarma si `team` < 98 %, p95 > 2 s, sin feedback > 3 min o latido parado. Pitido y Telegram opcional (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`). |
| F8 | ✅ | Si una respuesta no pasa la validación se envían populares válidos y se cuenta (`guard_fallback`). |
| Redundancia | ✅ | `--standby`: segunda instancia que solo responde si el latido de la principal lleva > 2 s parado (misma máquina; para otra máquina, ver `--ease-from-scratch`). No se lanzan dos respuestas a la vez: **preguntar en el stand** antes de cambiar esto. |

### Cold start (3.2)

| # | Estado | Qué se ha hecho |
|---|---|---|
| C1 | ✅ | Un `new_product` entra en el catálogo en ≤ 0,5 s con vecinos de contenido calculados en ese momento (test `test_new_product_gets_neighbors_immediately`). La similitud es **absoluta** y exige misma categoría: un producto sin relación no recibe vecinos (test `test_unrelated_product…`). |
| C2 | ✅ | Usuario nuevo sin sesión: popularidad viva suavizada, prior por contexto con IPS, lista diversificada (E) y brazo del bandit propio del segmento frío. |
| C3 | ✅ | Un producto nuevo hereda de sus vecinos el prior de popularidad, el engagement y su fila/columna EASE (encogido al 60 %), así que puede recomendarse **aunque el usuario no lo haya visto**. Además, exploración acotada (D3). |
| C4 | 🟡 | `eval.cold_start_test`: oculta un 5 % de productos con historial y mide cuánto de su fila EASE real recuperan los vecinos. En sintético: 0,69 recall@20 frente a 0,58 de "populares de la subcategoría". Se ejecuta en `train.py` con datos reales. |

### Conversión (3.3)

| # | Estado | Qué se ha hecho |
|---|---|---|
| V1 | ✅ | Perfil de usuario: afinidad por subcategoría y marca, rango de precio (μ, σ de log-precio) y compras. Entra como variables del modelo. |
| V2 | ✅ | Se excluyen las compras del histórico, las de `shop.events` en vivo y las de nuestro `feedback` (sin JOIN: el `user_id` de cada petición se guarda en memoria). |
| V3 | ✅ | Transiciones con `a.t < b.t`, deduplicadas por sesión, lift `/ (sesiones(B)+30)^β` **antes** del `LIMIT`, soporte mínimo 3. Y EASE como señal principal. |
| V4 | ✅ | Prior por contexto ponderado por 1/propensity del slot donde ocurrió la compra/clic. |
| V5 | ✅ | Tendencia = `log((corto+κ)/(esperado+κ))` con contadores de semivida 15 min y 3 h. |
| V6 | ✅ | Cambio respecto al precio **anterior**, con decaimiento (τ = 2 h); las subidas restan. Peso `weight_price_drop`, a ajustar con B4. |
| V7 | ✅ | Los pesos los aprende el modelo (A). |
| V8 | 🟡 | La consulta `V8_ultimo_visto_por_pagina` del EDA lo mide; el modelo lo aprende con `is_last_viewed` y `x_product_in_recent`. |

### Evaluación offline (3.4)

| Punto | Estado | Qué se ha hecho |
|---|---|---|
| Validación circular | ✅ | `eval.load_sessions` carga `history_sessions` **real** (muestra por hash). El mundo sintético solo se usa como test de integración, y su modelo oculto usa variables latentes que el recomendador no ve. |
| SNIPS slot 1 / HitRate | ✅ | Se siguen reportando, pero la política se compara con **IPS ítem-posición** (las 10 posiciones) usando la exposición marginal empírica Q_i como propensity, más un método directo con un modelo distinto. |
| ¿Funciona el estimador? | ✅ | Comprobado en sintético contra el valor verdadero: IPS ítem-posición da 8,2 / 13,2 / 14,8 compras por 100 (baseline / heurística / lineal) frente a 8,5 / 16,4 / 16,4 reales, y ordena bien. HitRate y SNIPS-slot1 prefieren el baseline (el sesgo que anticipaba la revisión). Usar la `propensity` condicional como si fuera marginal también favorecía al baseline: por eso se usa Q_i. |

### Enfoques (sección 4)

| Enfoque | Estado | Qué se ha hecho |
|---|---|---|
| **A** modelo de respuesta | 🟡 | `train.py`: corte temporal; variables "a fecha de" (las filas de entrenamiento solo ven el histórico anterior a su ventana, para que la CVR por ítem y el prior por contexto no contengan la etiqueta); negativas al 10 %; IPS; logística y GBM; variante **sin identidad**; log-loss, AUC y calibración por segmento; importancia de variables en el informe. Al servir: slot = 1, 5.000 productos, recorte de variables al rango de entrenamiento (riesgo 1: no extrapolar a productos que el log nunca mostró). |
| **B** EDA experimental | 🟡 | `eda.py`: B1 (dos versiones), B2/S3, B3, B4, B5, B6, B7, B8, V8 y fracción de nuevos/sesiones vacías en el stream. Escribe `eda/*.csv`, `EDA_RESULTS.md` con la decisión que cambia cada una y `eda/attention.json` (curva por posición que pasa a usar el sistema). En sintético recupera la curva de atención verdadera. |
| **C** EASE | ✅ | Co-ocurrencias en SQL, inversión en NumPy (5.000 productos en segundos), productos nuevos en ambos sentidos vía vecinos. Vecinos publicados en `team.scratch` y **leídos de vuelta** por una instancia sin caché (`--ease-from-scratch`). |
| **D** adaptación en vivo | ✅ | D1: popularidad `(n_vivo + κ·prior)/(N + κ)`. D2: Thompson con 4 brazos (`hist`, `mix`, `live`, `explore`: κ, peso de tendencia, tasa de exploración) × 6 segmentos (frío/templado × página), sobre **clics**, estado guardado en disco. D3: exploración solo en slots 8 y 10, solo con productos nuevos (< 300 impresiones nuestras) o con la tendencia duplicada. Ensayo del cambio de régimen: `--cold-boot`. |
| **E** lista diversificada | ✅ | Selección voraz que maximiza Σ_c π_c [1 − Π(1 − e_k p_c)]. Por defecto solo sin sesión (`--diversify`); activarla siempre si B6 lo justifica. Desempate por ingresos con peso pequeño. |

### Decisiones de la sección 5

* **Optimizar para la final:** la evaluación incluye el escenario `ALL_AS_NEW` (todos los usuarios como nuevos) y el modelo sin identidad; el ensayo local usa un 50 % de usuarios nuevos y productos nuevos.
* **Comparar versiones:** reparto por `hash(request_id)` (`--ab-share`) y `model_version = etapa1-especial-<modelo|heur>-<brazo>`; consulta 2 de `sql/monitoring_v3.sql`.
* **Congelar:** último despliegue a las 08:00 (sección 6).

## 3. Cómo ejecutarlo (en orden)

```powershell
cd etapa_1_version_especial
pip install -r requirements.txt

# 0. Tests (incluye un ensayo de 20 s contra RawTree local con fallos de red inyectados)
python -m pytest tests -q

# 1. EDA real (B1–B8). Revisar eda/EDA_RESULTS.md y rellenar las conclusiones
python -m src.eda

# 2. Entrenar (calcula y cachea el histórico, ~10 % de las sesiones). Revisar artifacts/train_report.md
python -m src.train
#    opciones útiles: --sample-pages 8 (20 %), --target click, --no-ips

# 3. Servir bajo supervisor, en una máquina que no se suspenda
powershell -ExecutionPolicy Bypass -File run_forever.ps1
#    la primera vez el histórico completo se calcula y se guarda en cache/ (minutos);
#    después arranca en segundos. Forzar recálculo: añadir --rebuild-history

# 4. Alarma, en otra terminal
python -m src.monitor

# Opcional
python -m src.serve --heuristic        # sin modelo entrenado (pesos a mano)
python -m src.serve --ab-share 0.5     # 50 % modelo / 50 % heurística por hash(request_id)
python -m src.serve --standby          # instancia de reserva
python -m src.simulate --seconds 60    # ensayo general local
python -m src.train --local-sim        # pipeline de entrenamiento sobre el mundo sintético
```

Sin `artifacts/model.json` el servicio usa la heurística, así que el paso 3 se puede lanzar antes que el 2.

**Memoria:** la matriz EASE de 5.000 productos ocupa ~100 MB; el proceso completo, unos 0,5–1 GB.

## 4. Lo primero que hay que mirar en la primera ejecución real

1. Que todos los componentes del histórico salgan `OK` en el log. Si alguno falla, el error de SQL aparece entero; el resto sigue funcionando.
2. Que la lectura S1 no caiga a "MODO LEGACY" (eso indicaría que RawTree no acepta `now64()`/`INTERVAL` así).
3. Línea `stats` cada 30 s: edad al confirmar el push (p95 debería quedar < 1,5 s) y `latency_ms(feedback) − edad_push` cerca de 0 (relojes alineados).
4. `python -m src.monitor` en verde.

Las consultas acceden a `context.page` y castean los arrays (`shown::Array(String)`, `recent_views::Array(JSON)`). Así funcionan en ClickHouse 26 con columnas `Dynamic`; si RawTree se comporta distinto, los errores saldrán en el paso 1 del EDA.

## 5. Resultados en el mundo sintético (solo integración)

Mundo de 500 productos con un modelo oculto con variables latentes (calidad, gusto del usuario, subcategorías complementarias). Compras esperadas por 100 peticiones según ese modelo oculto, y score 0–100 entre baseline y oráculo, como en el leaderboard:

| Política | Compras/100 | Score | Score COLD |
|---|---|---|---|
| Popularidad (baseline) | 8,4 | 0 | 0 |
| Heurística (pesos a mano sobre las variables nuevas) | 15,2 | 76,7 | 78,2 |
| Logística entrenada | 15,3 | 77,6 | 78,3 |
| GBM entrenado | 13,2 | 53,7 | 45,6 |
| Oráculo | 17,3 | 100 | 100 |

Ensayo general (`simulate.py`, 3 peticiones/s, 50 % usuarios nuevos, productos nuevos, agotados y cambios de precio): p50 de latencia ~0,1 s, p95 ~0,2 s sin fallos inyectados; 96 % a tiempo con un 25 % de push fallidos.

Qué se puede concluir: el pipeline funciona y el entrenamiento recupera estructura que nadie le dio a mano. Qué **no**: cuánto ganará en el simulador real. El GBM rinde peor aquí con pocos positivos (449); con datos reales habrá unos 20 veces más. El script elige entre lineal y GBM por log-loss de validación.

## 6. Pendiente (manual)

- [ ] Si el repo se va a compartir: `git rm --cached group_API.env` y `git rm -r --cached etapa_1/src/__pycache__ etapa_1/tests/__pycache__`, y commit.
- [ ] Ejecutar EDA, entrenar y revisar `artifacts/train_report.md` antes de desplegar el modelo.
- [ ] Preguntar en el stand qué respuesta cuenta si llegan dos para la misma petición.
- [ ] Desactivar la suspensión del portátil; lanzar supervisor y alarma; asignar a alguien a cada franja.
- [ ] Probar a mano: matar el proceso, cortar la red 10 s, comprobar que un `new_product` aparece en < 2 s y que un agotado desaparece en < 2 s (los logs lo muestran).
- [ ] Último despliegue de modelo a las 08:00 del domingo; de 08:00 a 10:00 solo vigilancia.
