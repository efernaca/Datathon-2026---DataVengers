-- Etapa 1 · versión especial · consultas del histórico (src/history.py). Se ejecutan UNA vez y se guardan en cache/ (F5).
-- Placeholders {where_ev}/{and_ev}: corte temporal opcional. Probadas en ClickHouse embebido con columnas Dynamic.

-- ======================================================================
-- Catálogo base
-- ======================================================================
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       price_eur::Float64 AS price_eur, description::String AS description,
       is_active::Bool AS is_active, toString(created_at) AS created_at
FROM shop.catalog
;

-- ======================================================================
-- Popularidad orgánica
-- ======================================================================
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'view') AS views,
       countIf(event_type::String = 'add_to_cart') AS carts,
       countIf(event_type::String = 'purchase') AS purchases
FROM shop.history_events
{where_ev}
GROUP BY product_id
;

-- ======================================================================
-- Estadísticas IPS por ítem (identidad / prior de CVR)
-- ======================================================================
SELECT item AS product_id, count() AS imps, sum(w) AS imps_w,
       sum(w * (clicked_id = item)) AS clicks_w,
       sum(w * (purchased_id = item)) AS buys_w
FROM (
    SELECT ifNull(clicked::String, '') AS clicked_id,
           ifNull(purchased::String, '') AS purchased_id,
           arrayJoin(arrayEnumerate(shown::Array(String))) AS slot,
           (shown::Array(String))[slot] AS item,
           1.0 / greatest((propensity::Array(Float64))[slot], 0.01) AS w
    FROM shop.history_sessions
    {where_ev}
)
GROUP BY item
;

-- ======================================================================
-- V4 · Popularidad por contexto ponderada por 1/propensity ({col} = purchased | clicked)
-- ======================================================================
SELECT context.region::String AS region, context.device::String AS device,
       context.page::String AS page, {col}::String AS product_id,
       sum(1.0 / greatest((propensity::Array(Float64))[
           greatest(indexOf(shown::Array(String), {col}::String), 1)], 0.01)) AS score_w,
       count() AS n
FROM shop.history_sessions
WHERE {col} IS NOT NULL AND {col}::String != '' {and_ev}
GROUP BY region, device, page, product_id
ORDER BY score_w DESC
LIMIT 80 BY region, device, page
;

-- ======================================================================
-- V3 · Transiciones dirigidas (a.t < b.t), deduplicadas por sesión, lift antes del LIMIT
-- ======================================================================
WITH ev AS (
    SELECT session_id::String AS s, product_id::String AS p,
           event_type::String AS e, ts::DateTime64(3) AS t
    FROM shop.history_events
    {where_ev}
),
ok AS (SELECT s FROM ev GROUP BY s HAVING uniqExact(p) BETWEEN 2 AND {max_items}),
fv AS (SELECT s, p, min(t) AS t FROM ev WHERE e = 'view' AND s IN ok GROUP BY s, p),
lt AS (SELECT s, p, e, min(t) AS t FROM ev WHERE s IN ok GROUP BY s, p, e),
pairs AS (
    SELECT a.p AS src, b.p AS dst,
           max(b.e = 'purchase') AS bp, max(b.e = 'add_to_cart') AS bc
    FROM fv AS a INNER JOIN lt AS b ON a.s = b.s
    WHERE a.p != b.p AND a.t < b.t
    GROUP BY a.s, src, dst
),
agg AS (
    SELECT src, dst, count() AS support,
           sum(bp) * 6.0 + sum(bc) * 2.5 + count() * 1.0 AS raw
    FROM pairs GROUP BY src, dst HAVING support >= {min_support}
),
popd AS (SELECT p AS dst, uniqExact(s) AS n_sessions FROM ev GROUP BY p)
SELECT agg.src AS src_product, agg.dst AS dst_product, agg.support AS support,
       agg.raw / pow(popd.n_sessions + 30.0, {beta}) AS lift
FROM agg INNER JOIN popd ON agg.dst = popd.dst
ORDER BY lift DESC
LIMIT {topk} BY src_product
;

-- ======================================================================
-- C · EASE: diagonal de G = XᵀX
-- ======================================================================
WITH sp AS (
    SELECT session_id::String AS s, product_id::String AS p
    FROM shop.history_events {where_ev} GROUP BY s, p
),
ok AS (SELECT s FROM sp GROUP BY s HAVING count() BETWEEN 2 AND {max_items})
SELECT p AS product_id, count() AS n FROM sp WHERE s IN ok GROUP BY p
;

-- ======================================================================
-- C · EASE: pares de G (paginado por hash)
-- ======================================================================
WITH sp AS (
    SELECT session_id::String AS s, product_id::String AS p
    FROM shop.history_events {where_ev} GROUP BY s, p
),
ok AS (SELECT s FROM sp GROUP BY s HAVING count() BETWEEN 2 AND {max_items}),
spf AS (SELECT s, p FROM sp WHERE s IN ok)
SELECT a.p AS i, b.p AS j, count() AS c
FROM spf AS a INNER JOIN spf AS b ON a.s = b.s
WHERE a.p < b.p AND cityHash64(a.p) % {{n_pages}} = {{page}}
GROUP BY i, j
HAVING c >= {min_c}
;

-- ======================================================================
-- V1 · Afinidad de usuario por subcategoría / marca (paginado)
-- ======================================================================
WITH e AS (
    SELECT user_id::String AS user_id, product_id::String AS pid,
           multiIf(event_type::String = 'purchase', 5.0,
                   event_type::String = 'add_to_cart', 2.0, 0.3) AS w
    FROM shop.history_events
    WHERE cityHash64(user_id::String) % {{n_pages}} = {{page}} {and_ev}
),
c AS (SELECT product_id::String AS pid, {field}::String AS val FROM shop.catalog)
SELECT e.user_id AS user_id, c.val AS val, sum(e.w) AS w
FROM e INNER JOIN c ON e.pid = c.pid
GROUP BY user_id, val
ORDER BY w DESC
LIMIT {limit} BY user_id
;

-- ======================================================================
-- V1 · Compras y rango de precio por usuario (paginado)
-- ======================================================================
SELECT user_id::String AS user_id,
       groupUniqArray(product_id::String) AS bought,
       avg(log1p(price_eur::Float64)) AS mu,
       stddevPop(log1p(price_eur::Float64)) AS sigma,
       count() AS n
FROM shop.history_events
WHERE event_type::String = 'purchase'
  AND cityHash64(user_id::String) % {{n_pages}} = {{page}} {and_ev}
GROUP BY user_id
;

