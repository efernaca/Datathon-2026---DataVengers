# Medidas implementadas · de la Etapa 1 a Calypso 4.1

Recopilación de todo lo que hace el recomendador, en el orden en que se fue añadiendo, con un comentario corto
de cada medida. El modelo que se sirve en vivo es la Etapa 4 de cada carpeta `calypso_*` (`serve.py` → `etapa_4/src/serve.py`);
las etapas 1–3 se conservan para compararlas en el benchmark offline.

Detalle de cada etapa: `calypso_*/etapa_N/01_*.md`, `calypso_3_1b/README_3_1b.md` y `calypso_4_1/README_4_1.md`.

---

## Etapa 1 · Recomendador híbrido (`etapa1-hybrid-v1`)

| Medida | Comentario |
|---|---|
| Bucle pull → puntuar → push | Lee `shop.requests`, decide 10 productos y los escribe por lotes en `submissions`. |
| Snapshot inmutable en memoria | El estado se recalcula aparte y se sustituye de golpe; responder no espera a SQL. |
| Transiciones en sesión | "Quien vio A después miró/compró B", sacado de `history_events`. Lo más reciente de `recent_views` pesa más. |
| Re-engagement | Volver a enseñar lo que el usuario acaba de mirar (atenuado en el slot 1 de una ficha de producto). |
| Afinidad de subcategoría | Las subcategorías dominantes de la sesión y del historial del usuario empujan a sus productos. |
| Popularidad por contexto | Ranking por (región, dispositivo, página) para cuando no hay sesión. |
| Popularidad global, tendencia y descuentos | Señales de fondo: más vendidos, lo que sube en `shop.events` y bajadas de precio. |
| Cold start por contenido (TF-IDF) | Un producto nuevo hereda los vecinos de productos parecidos por texto, categoría y marca. |
| **No repetir compras** | Lo que el usuario ya compró no se recomienda. |
| Guardrails | Siempre 10 IDs distintos, activos y válidos; si faltan, se rellena con los más vendidos. |

## Etapa 2 · Secuencial + personalización + cold start bidireccional (`etapa2-seq-mmr-v1`)

| Medida | Comentario |
|---|---|
| Lift en transiciones | Se divide por la popularidad del destino para que los superventas no salgan en todas partes. |
| Decaimiento doble | El peso de cada vista depende de su posición y del tiempo transcurrido. |
| Perfil de usuario | Intención no comprada (`add_to_cart` o varias vistas), afinidad por subcategoría + marca y rango de precio habitual. |
| Encaje de precio | Bonus gaussiano para productos en el rango de precio en que suele comprar el usuario. |
| Contexto jerárquico | (región, dispositivo, página) → (región, dispositivo) → (página), suavizado bayesiano. |
| Cold start bidireccional | Nuevo → conocido (si miras un nuevo) y conocido → nuevo (si miras uno parecido a un nuevo). TF-IDF + SVD. |
| Desempate por ingresos | Ligero empuje a productos más caros dentro de su subcategoría (criterio de desempate del reto). |
| MMR (diversidad) | Slots 1–5 intactos con sesión; a partir de ahí se evita repetir marca/subcategoría. |
| Compras en vivo excluidas | Lo comprado en nuestro carrusel (`feedback`) tampoco se vuelve a recomendar. |
| Uso de `team.scratch` | Se guardan vecinos de cold start y telemetría en RawTree. |

## Etapa 3 · `calypso-2.1`: ML + bandit + agente

| Medida | Comentario |
|---|---|
| Re-ranker logístico con IPS | 14 variables por candidato; corrige el sesgo de posición del log antiguo con pesos (k+1)^0.75. |
| Fusión con priors monótonos | El peso del ML crece con los datos (`ml_blend_alpha`); con pocos datos mandan los pesos razonables a mano. |
| Bandit UCB por posición | Conversión empírica por producto corregida por el slot en que se mostró, más bonus de exploración (mayor para nuevos). |
| Agente autónomo | Cada ciclo diagnostica CTR/CVR, oleadas de productos nuevos, etc., y ajusta hiperparámetros; deja la traza en `scratch`. |
| Cambio sin parada | Las versiones conviven por `model_version`; se puede pasar de una a otra sin perder peticiones. |

## Etapa 4 · Blindaje operativo (base de todas las `calypso_*`)

| Medida | Comentario |
|---|---|
| Caché en disco + arranque rápido | El snapshot se guarda en disco; un reinicio responde en < 0,4 s. |
| Solo peticiones frescas | Se leen las de los últimos 2,4 s (máx. 25), sin perder tiempo con peticiones ya caducadas. |
| `_seen` tras confirmar el push | Si un envío falla, la petición se reintenta en el siguiente ciclo. |
| Watchdog | Reinicia el servicio en 1 s si se cae. |
| Histórico estático separado | Las consultas de 60 días se hacen una vez; en vivo solo stock (1,5 s) y events/feedback (15 s). |
| Slot-gating | En sesiones normales los primeros slots son precisión pura; la exploración (UCB, productos nuevos) va a la cola. |
| Monitor y gestor de modelos | `monitor.py` vigila on-time y conversión; `gestor_models` arranca/cambia de modelo y lee el latido. |

## Calypso 3.1 · cuatro cambios sobre 2.1

| Medida | Comentario |
|---|---|
| Sin suelos del bonus subcat/marca | Quitar `max(w12,0.28)` / `max(w13,0.12)`. En la ablación empeora (−0,020). |
| Penalización de fondo | Productos sin ninguna señal de sesión ×0,18 (otra subcat) / ×0,45 (misma subcat). Mejora (+0,018). |
| EASE mezclado con lift | Modelo ítem-ítem de solución cerrada, peso 0,25. Empeora ligeramente (−0,008). |
| Transiciones direccionales | SQL solo hacia delante en el tiempo. Solo medible con datos reales. |
| Umbrales de cold start | Similitud mínima 0,08 y normalización. Neutro. |

## Calypso 3.1b · los cambios de 3.1 como interruptores

| Medida | Comentario |
|---|---|
| Interruptores `CALYPSO_V3_*` | Cada cambio de 3.1 se activa por variable de entorno, sin tocar código. |
| Ablación offline (5 semillas) | Por defecto solo queda la penalización de fondo: mejor variante (0,7904 vs 0,7725 de 2.1 en hit_rate_buy@10). |

## Calypso 4.1 · reglas de negocio (nuevas)

Todas conmutables con `CALYPSO_V4_*`; con todas a 0 se reproduce 3.1b exactamente.

| Medida | Comentario |
|---|---|
| **Carrito → alta prioridad** | Producto añadido al carrito hace ≤ 30 min y no comprado: primeros slots (máx. 3 entre carrito activo y olvidado). |
| **Carrito olvidado → volver a recomendar** | Añadido hace más de 30 min (hasta 30 días) y no comprado: justo detrás del carrito activo (máx. 2). Busca cerrar la compra pendiente. |
| **Orden de más barato a más caro** | Cada bloque (carrito activo, olvidado, recomendados) se ordena por precio. No cambia qué productos salen, solo el orden. En el sintético cuesta −25 % de DCG: se puede apagar con `CALYPSO_V4_PRICE_ASC=0`. |
| **No repetir comprados** | Comprado una vez → no aparece en recomendados (puede seguir en el historial de vistas). Ya existía; ahora también cuenta las compras orgánicas en vivo de `shop.events`. |
| **Recurrentes sí se recomiendan** | Comprado ≥ 2 veces → producto recurrente: no se excluye y se garantiza en recomendados (máx. 2). Para no contar dos veces la misma compra (events + feedback) se usa el máximo, no la suma. |

Ablación de 4.1 y detalle de consultas: `calypso_4_1/README_4_1.md`.
