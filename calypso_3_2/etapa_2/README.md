# Etapa 2 · versión especial (Etapa 2 + todas las mejoras de su revisión)

**Versión:** `e2esp` · **Base:** el modelo de [`etapa_2/`](../etapa_2/) · **Guía:** [`etapa_2/02_REVISION_Y_ENFOQUES_ALTERNATIVOS.md`](../etapa_2/02_REVISION_Y_ENFOQUES_ALTERNATIVOS.md)

Conserva lo que define a la Etapa 2: una docena de señales por producto (lift, autoconversión, intención, marca, precio, cold start en los dos sentidos, contexto) combinadas con pesos, más MMR y desempate por ingresos. Sobre eso aplica todos los puntos de la revisión:

1. **Medir antes que añadir.** El sábado no puntúa, así que cada componente discutible (MMR, ingresos, injerto inverso, intención, ajuste de precio, mostrar lo ya visto) se enciende o apaga al azar en cada petición (diseño factorial, 4.1). Además, un 10 % de listas se baraja para medir la curva de atención y un 20 % trata al usuario conocido como nuevo (ensayo de la final, 4.4). El feedback decide qué se queda.
2. **Pesos aprendidos y por segmento** (4.2, 4.3) a partir de ese feedback, en lugar de los 22 pesos a mano.
3. **Dos procesos** (4.5): un constructor hace todo lo pesado; el servidor solo responde.
4. Arreglos N1–N13 y la infraestructura de fiabilidad de la [Etapa 1 · versión especial](../etapa_1_version_especial/README.md).

> **Estado de validación.** 15 tests pasan, incluido un ensayo de punta a punta contra un RawTree local (ClickHouse embebido con columnas `Dynamic`). **No se ha ejecutado aún contra RawTree real.** Las cifras de la sección 5 salen de un mundo sintético y no predicen el leaderboard.

---

## 1. Cómo funciona

```
 ┌──────────────── constructor (src/builder.py) ────────────────┐
 │ histórico 60 días (una vez) → cache/history_full.pkl          │
 │                              → state/history_version.txt      │──► team.scratch (agregados, kind+version)
 │ cada hora: aprender pesos (4.2) → state/weights.json          │
 │            analizar experimentos (4.1/4.4) → experiments/     │
 └───────────────────────────────────────────────────────────────┘
                 │ ficheros atómicos (tmp + rename)
 ┌──────────────── servidor (src/serve.py) ──────────────────────┐
 │ lee peticiones (S1) → señales de 5.000 productos → lista      │──► submissions (model_version = interruptores)
 │ hilos ligeros: catálogo 0,5 s · eventos 2 s · feedback 5 s     │
 │ recarga pesos/histórico cuando cambian · registra señales     │──► logs/signals_*.jsonl (+ scratch opcional)
 └───────────────────────────────────────────────────────────────┘
```

| Fichero | Qué hace |
|---|---|
| `src/signals.py` | Las 15 señales de la Etapa 2 (nombres corregidos), vectorizadas sobre todo el catálogo y en escalas fijas para poder aprender sus pesos. |
| `src/policy.py` | `score = Σ peso_segmento · interruptor · señal`; MMR relativo; desempate lexicográfico; asignación de interruptores, barajado y modo final por hash de `request_id`. |
| `src/learner.py` | 4.2: une el registro de señales con `feedback` y ajusta una logística por segmento; solo acepta pesos que mejoren la validación. |
| `src/analyze.py` | 4.1/4.4: efecto de cada interruptor con IC95 %, logística ajustada, curva de atención medida, caída en modo final, N8 → `experiments/EFFECTS.md` y `state/switches.json`. |
| `src/builder.py` | 4.5: proceso constructor. |
| `src/serve.py` | 4.5: proceso servidor (hilos de la Etapa 1 especial + recarga de estado + registro de señales). |
| `src/history.py`, `catalog.py`, `cold_start.py`, `live.py`, `ease.py` | Histórico, catálogo vivo, cold start, señales vivas y EASE. |
| `src/eda.py`, `eval.py`, `simulate.py`, `monitor.py` | EDA (con S5), evaluación offline con datos reales, ensayo local, alarma. |

## 2. Qué se ha hecho con cada punto de la revisión

✅ implementado y probado (tests o simulación) · 🟡 implementado, falta comprobarlo con datos reales · ✋ acción manual

### Puntos heredados de la revisión 04 (tabla de la sección 2)

| Ref. | Estado | Qué se ha hecho |
|---|---|---|
| U2 | ✅ | Lee `.env` y `group_API.env` con CRLF. |
| U3 | ✋ | `.gitignore` añade `cache/` y `logs/` (`state/` se puede versionar: pesos e interruptores). Quitar del índice `group_API.env` y los `__pycache__` de `etapa_1`/`etapa_2` (sección 6). |
| V1, V2, V3, F3 | ✅ | Perfil completo; compras en vivo de `shop.events` **y** del feedback; transiciones S4; lectura con edad y ventana de 4 s. |
| F1, F2, F8 | ✅ | Push en pool aparte, reintento solo si queda tiempo, marcado al confirmar, nunca se descarta en silencio. |
| F4, F5 | ✅ | Stock y catálogo cada 0,5 s en su hilo; histórico calculado una vez en el constructor. |
| F6, F7 | ✅ | `run_forever.ps1 serve|builder`; `python -m src.monitor`. |
| C1 | ✅ | Productos nuevos con vecinos en ≤ 0,5 s; el injerto inverso se recalcula en ese momento. |
| V4, V5, V6 | ✅ | Contexto con IPS y bayesiano jerárquico; tendencia con decaimiento; bajada respecto al precio anterior. |
| 3.4 / B1–B8 | 🟡 | `eval.py` con datos reales e IPS ítem-posición; `eda.py` con B1–B8, V8 y S5. |

### Hallazgos nuevos (sección 3)

| # | Estado | Qué se ha hecho |
|---|---|---|
| N1 | ✅ | Las 8 consultas pesadas ya no corren en el proceso que responde: las hace el constructor, una vez. Tras cargar, `gc.freeze()` para que el recolector no vuelva a recorrer el histórico. |
| N2 | ✅ | Hilo propio de stock y catálogo incremental (S2). |
| N3 | ✅ | Feedback incremental (`WHERE ts > último`) **sin JOIN**: el `user_id` de cada petición ya se conoce al responderla. Compras de `shop.events` también, incrementales. |
| N4 | ✅ | Los vecinos de cada producto nuevo se escriben **una vez** en `scratch` (`kind = cold_nb`). Los agregados también se publican **y se leen de vuelta** (`serve.py --from-scratch`). |
| N5 | 🟡 | La infraestructura es la misma que la de la Etapa 1 especial (cliente, guardrails, catálogo, señales vivas, bucle rápido); lo propio de esta etapa está en `signals.py`, `policy.py`, `learner.py`, `analyze.py` y `builder.py`. Sigue siendo una copia: cuando se elija versión para el domingo, conviene borrar las demás. |
| N6 | ✅ | Consulta S4: B después de ver A, una vez por sesión, lift `raw / (pop(B) + 30)^0,45` **antes** del `LIMIT 40 BY`, soporte ≥ 3. |
| N7 | ✅ | Similitud **absoluta** en los dos sentidos; injerto inverso solo si la similitud es ≥ 0,45 **y** es de la misma subcategoría. Test: un producto sin relación no se enlaza con nada (`test_reverse_graft_absolute_and_same_subcat`). |
| N8 | 🟡 | Medido en vivo: `% de listas con producto nuevo` y posición media en el log cada 30 s y en `EFFECTS.md`. El peso se decide con el interruptor `rg` y con el aprendizaje de pesos. En el ensayo local: 29 % de listas, posición media 6,6. |
| N9 | ✅ | `rev` apagado por defecto. Si se enciende, es lexicográfico: solo intercambia vecinos con \|Δscore\| < 1 % (test `test_revenue_tiebreak_only_within_eps`). Su efecto lo mide el interruptor `rev`. |
| N10 | ✅ | Penalización relativa: valor·(1 − λ·n), con valor = exp(score). Se mantiene o se quita según el interruptor `mmr`. |
| N11 | ✅ | La intención guarda la antigüedad y decae con semivida de 7 días (más el tiempo desde el final del histórico). Test `test_intent_decays_with_age`. Interruptor `int`. |
| N12 | ✅ | `(n + κ·tasa_padre) / (N + κ)` con tasas en tres niveles: (región, dispositivo, página) → (región, página) → página → popularidad. Las compras/clics están ponderadas por 1/propensity. |
| N13 | ✅ | `self_conversion` → `visto_conv`, que ahora sí es P(compra \| el producto ya estaba en `recent_views`), medida con IPS. `time_decay_half_life_sec` → `recent_halflife_sec = 624` (la semivida real). |
| 3.3 | ✅ | El test "Etapa 2 > Etapa 1" ha desaparecido. El mundo sintético solo se usa para integración, y su modelo oculto usa variables latentes que el recomendador no ve. |

### Enfoques (sección 4)

| Enfoque | Estado | Qué se ha hecho |
|---|---|---|
| **4.1** experimento factorial | ✅ | 6 interruptores (`mmr`, `rev`, `rg`, `int`, `pf`, `self`) al 50 % e independientes, por hash de `request_id` con una sal distinta por interruptor. Reintentos reproducibles. Barajado del top-10 en el 10 %. Todo va en `model_version`. Análisis: diferencia de medias con IC95 % (también por segmento) y logística `clic ~ interruptores + segmento + modo final`. La regla de decisión escribe `state/switches.json`, que el domingo usa `serve.py --freeze`. |
| **4.2** pesos con feedback propio | ✅ | El servidor registra las señales de los 10 mostrados (JSONL + scratch opcional) desde un hilo aparte. El constructor reentrena cada hora con un término por posición y acepta solo si mejora la log-loss de validación frente a los pesos actuales recalibrados. El servidor recarga los pesos solo. En sintético, con 30.000 listas, pasa de 53 a 73 sobre 100 (sección 5). |
| **4.3** una política por segmento | ✅ | 5 segmentos: conocido con sesión, conocido sin sesión, nuevo con sesión, nuevo sin sesión, sesión con producto nuevo. Un vector de pesos por segmento, y el global cuando hay pocos datos. |
| **4.4** ensayo de la final | ✅ | Modo final en el 20 % (usuario tratado como nuevo; las compras se siguen excluyendo); su caída de CTR sale en `EFFECTS.md`. `eda.py` incluye la S5 (composición del tráfico y cambios de catálogo). Arranque en frío con `--cold-boot`. |
| **4.5** dos procesos + scratch | ✅ | `builder.py` y `serve.py`; estado en disco con escritura atómica; agregados en `scratch` con `kind` y `version`, con lectura de vuelta. RawTree no documenta `INSERT … SELECT`, así que el constructor consulta y reinserta. |
| EASE y contadores vivos (revisión 04) | ✅ | EASE es una señal más que se aprende con 4.2; la tendencia sale de contadores con decaimiento sobre `shop.events`. |

## 3. Cómo ejecutarlo

```powershell
cd etapa_2_version_especial
pip install -r requirements.txt
python -m pytest tests -q                 # 15 tests (incluye ensayo local de 20 s)

python -m src.eda                         # EDA + S5 → eda/EDA_RESULTS.md

# Sábado: dos terminales, cada una con supervisor
powershell -ExecutionPolicy Bypass -File run_forever.ps1 builder    # histórico (primera vez: minutos)
powershell -ExecutionPolicy Bypass -File run_forever.ps1 serve      # experimentos activos
python -m src.monitor                                               # tercera terminal: alarma

# Durante la noche el constructor reentrena y analiza cada hora. A mano:
python -m src.learner          # → state/weights.json + logs/learn_report.md
python -m src.analyze          # → experiments/EFFECTS.md + state/switches.json

# Domingo 07:00–08:00: revisar EFFECTS.md y learn_report.md, ajustar switches.json si hace falta
powershell -ExecutionPolicy Bypass -File run_forever.ps1 serve --freeze   # sin experimentos
```

Otras opciones: `serve --no-experiments`, `serve --standby`, `serve --from-scratch`, `serve --signals-to-scratch`, `serve --cold-boot`, `builder --once`, `builder --rebuild`, `python -m src.simulate --seconds 60`.

## 4. Lo primero que hay que mirar en real

1. Constructor: que los 9 componentes del histórico salgan `OK` y que publique los agregados en `scratch`.
2. Servidor: línea `stats` cada 30 s (edad del push p95 < 1,5 s, sin "MODO LEGACY") y línea `N8`.
3. Tras 1–2 h: `python -m src.analyze`. Con ~10.000 listas por hora, cada interruptor tiene ~5.000 por rama: los efectos del orden del 4–5 % se ven tras una noche, no en una hora.
4. Que `logs/signals_*.jsonl` crezca y que `learner` acepte pesos en algún segmento.

## 5. Resultados en el mundo sintético (solo integración)

Compras esperadas por 100 peticiones según el modelo oculto del mundo sintético y score 0–100 entre baseline y oráculo:

| Política | Compras/100 | Score | Score COLD |
|---|---|---|---|
| Popularidad (baseline) | 8,2 | 0 | 0 |
| Etapa 2 especial con los pesos de la Etapa 2 | 12,5 | 53 | 53 |
| Etapa 2 especial con pesos aprendidos (4.2 + 4.3), 30.000 listas con experimentos | 14,1 | **73** | **78** |
| Oráculo | 16,3 | 100 | 100 |

- **4.1 en sintético:** con 30.000 listas, el interruptor `self` (mostrar lo ya visto) sale positivo con IC95 % por encima de 0 y `mmr` queda al borde. El resto no es concluyente: es lo esperable con efectos pequeños.
- **Barajado:** la curva de atención medida sale de forma automática. Es más empinada que la atención pura porque, una vez que alguien clica, ya no mira los slots de abajo.
- **Ensayo de 60 s a 10 peticiones/s con experimentos activos:** 99 % a tiempo, latencia p95 0,16 s.
- **Test C4:** 0,66 recall@20 frente a 0,45 de "populares de la subcategoría".

Esto demuestra que el circuito "experimento → feedback → pesos" funciona y mejora la política sin intervención manual. No dice cuánto ganará contra el simulador real.

## 6. Pendiente (manual)

- [ ] `git rm --cached group_API.env` y `git rm -r --cached etapa_1/src/__pycache__ etapa_1/tests/__pycache__` (y los de `etapa_2` si existen) antes de compartir el repo.
- [ ] Elegir qué versión corre el domingo (Etapa 1 especial o Etapa 2 especial) **con datos**. Las dos escriben `model_version` distinto (`etapa1-especial…` / `e2esp|…`), así que se pueden alternar en ventanas fijas y comparar en el leaderboard. No las corráis a la vez con el mismo equipo.
- [ ] Revisar a mano `experiments/EFFECTS.md` antes de `--freeze`: con menos de una noche de datos, la regla deja casi todo por defecto.
- [ ] Desactivar la suspensión, supervisores y alarma activos; último cambio a las 08:00 del domingo.
