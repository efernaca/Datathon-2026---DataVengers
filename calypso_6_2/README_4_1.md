# Calypso 4.1 · 3.1b + reglas de negocio de carrito, compras y precio

4.1 es una copia de `calypso_3_1b` (mismo modelo, mismos valores por defecto de los interruptores `CALYPSO_V3_*`)
a la que se añaden cinco reglas que actúan **después** del modelo, al construir la lista de 10.
Cada una se activa o desactiva por variable de entorno, sin tocar código. **Con todas a 0, 4.1 devuelve
exactamente las mismas listas que 3.1b** (comprobado petición a petición sobre las 2.400 del benchmark).

## Las reglas

| # | Regla | Variable (defecto) | Cómo funciona |
|---|---|---|---|
| M1 | Producto en el carrito → alta prioridad | `CALYPSO_V4_CART_PRIORITY=1` | "En carrito" = último `add_to_cart` posterior a la última compra de ese producto. Si se añadió hace ≤ `CALYPSO_V4_CART_ACTIVE_MIN` (30 min) va a los **primeros slots**. |
| M2 | Carrito olvidado → volver a recomendar con prioridad | `CALYPSO_V4_FORGOTTEN_CART=1` | Igual, pero añadido hace más de 30 min y menos de `CALYPSO_V4_FORGOTTEN_MAX_DAYS` (30 días). Va **justo detrás** del carrito activo, máx. `CALYPSO_V4_FORGOTTEN_MAX_SLOTS` (2). Entre M1 y M2, como mucho `CALYPSO_V4_CART_MAX_SLOTS` (3) slots. |
| M3 | De más barato a más caro | `CALYPSO_V4_PRICE_ASC=1` | Se ordena **cada bloque** por precio ascendente: [carrito activo] [carrito olvidado] [recomendados]. Los 10 productos elegidos no cambian, solo su orden. |
| M4 | No repetir comprados, salvo recurrentes | `CALYPSO_V4_RECURRENT=1` | Comprado 1 vez → excluido (ya existía desde la Etapa 1). Comprado ≥ `CALYPSO_V4_RECURRENT_MIN` (2) veces → **no se excluye y se garantiza** en el bloque de recomendados (máx. `CALYPSO_V4_RECURRENT_MAX_SLOTS`, 2), sustituyendo las últimas posiciones si el modelo no lo había elegido. |
| M5 | Compras orgánicas en vivo cuentan como "ya comprado" | `CALYPSO_V4_LIVE_EVENT_PURCHASES=1` | Antes solo se excluían las compras del histórico y las de nuestro carrusel (`feedback`); ahora también las de `shop.events`. |

Un producto agotado/inactivo nunca se fija, aunque esté en el carrito. Si un producto se compró y después se
vuelve a añadir al carrito, manda el carrito (no se excluye).

## De dónde salen los datos

| Fuente | Consulta (`etapa_4/src/serve.py`) | Cadencia |
|---|---|---|
| Carritos pendientes del histórico (máx. 6 por usuario) | `SQL_CART_HIST` sobre `shop.history_events` | una vez, se cachea en disco |
| Productos comprados ≥ 2 veces en el histórico | `SQL_RECURRENT_HIST` | una vez, se cachea en disco |
| Carrito y compras orgánicas en vivo (stream entero) | `SQL_LIVE_CART_EVENTS` sobre `shop.events` | 15 s |
| Lo mismo, últimos 35 min (para que un carrito recién añadido salga ya) | `SQL_LIVE_CART_EVENTS_RECENT` | 1,5 s |
| Compras en nuestro carrusel con contador | `_sql_feedback_purchases` sobre `team_NN.feedback` | 15 s |

La fusión la hace `etapa_4/src/business_rules.py` (`CartPurchaseIndex`): la parte histórica se precalcula una vez
y en cada refresco solo se recalculan los usuarios con actividad en vivo.

**Doble conteo.** Si el simulador escribiera las compras del carrusel también en `shop.events`, sumar ambas
fuentes convertiría una compra única en "recurrente". Por eso, en vivo se usa `max(compras en events, compras en feedback)`
y no la suma. Conviene comprobarlo en el log de la primera ejecución real (línea `carts_users=… recurrent_users=…`).

## Ablación offline (benchmark sintético, sin RawTree)

```
python etapa_4/ablation_v4.py --seeds 42 7 123 2024 99 --out-csv etapa_4/ablation_v4_results.csv
```

Media de 5 semillas, segmento ALL (2.400 peticiones por semilla). `hit_rate_buy@10` mira si el producto comprado
está entre los 10; `dcg_buy@10` además premia que esté arriba (que es lo que pasa en el simulador real).

| Variante | hit_rate_buy@10 | dcg_buy@10 | Δ dcg vs 3.1b |
|---|---|---|---|
| 3.1b | 0.7906 | 0.4851 | — |
| 3.1b + carrito activo (M1) | 0.7906 | 0.4845 | −0.001 |
| 3.1b + carrito olvidado (M2) | 0.7892 | 0.4776 | −0.008 |
| 3.1b + carrito activo + olvidado | 0.7872 | 0.4799 | −0.005 |
| 3.1b + precio ascendente (M3) | 0.7906 | **0.3651** | **−0.120 (−25 %)** |
| 3.1b + recurrentes (M4/M5) | 0.7906 | 0.4851 | 0 (no medible: en este mundo nadie recompra) |
| **4.1 completo (defecto)** | 0.7872 | **0.3941** | **−0.091** |
| 4.1 sin precio ascendente | 0.7872 | 0.4799 | −0.005 |

Lectura:

* **El orden por precio es caro.** No cambia qué 10 productos se enseñan, pero aparta del slot 1 al que el modelo
  considera más probable, y los primeros slots se miran mucho más. En el sintético pierde ~25 % de DCG.
  Para quitarlo sin tocar código: `CALYPSO_V4_PRICE_ASC=0`.
* **Carrito**: en usuarios WARM sube un poco (dcg 0.598 → 0.601), pero en COLD_START baja (0.326 → 0.310) porque los
  slots fijados desplazan productos nuevos. En el simulador real el efecto depende de cuánto pese el carrito en el modelo
  oculto; solo el feedback en vivo lo dirá (comparar `model_version`).
* El mundo sintético lo diseñamos nosotros: sirve para detectar regresiones, no para predecir el leaderboard.

## Cómo ejecutar

```powershell
cd calypso_4_1
python -m pytest etapa_4/tests -q          # 13 tests (4 de 3.1b + 9 de las reglas 4.1)
python serve.py --self-check
python serve.py --dry-run
python serve.py --train-only               # descarga el histórico (incluye carritos/recurrentes) y lo cachea
python serve.py --watchdog                 # servicio en vivo
```

La primera vez hay que dejar que descargue el histórico de nuevo (la caché de 3.1b no tiene carritos ni recurrentes,
y además vive en otra carpeta).
