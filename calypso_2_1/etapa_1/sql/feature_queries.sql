-- =============================================================================
-- Etapa 1 · Consultes SQL de producció i generació de features (Push-Down)
-- Totes les columnes Dynamic es castegen explícitament per rendiment i seguretat.
-- =============================================================================

-- 1. Catàleg base (segur abans que arribin updates a shop.catalog_updates)
SELECT
    product_id::String     AS product_id,
    category::String       AS category,
    subcategory::String    AS subcategory,
    brand::String          AS brand,
    price_eur::Float64     AS price_eur,
    description::String    AS description,
    created_at::String     AS created_at,
    is_active::Bool        AS is_active
FROM shop.catalog;


-- 2. Catàleg consolidat en viu (shop.catalog + última actualització de shop.catalog_updates)
SELECT
    product_id,
    argMax(category, v)     AS category,
    argMax(subcategory, v)  AS subcategory,
    argMax(brand, v)        AS brand,
    argMax(price_eur, v)    AS price_eur,
    argMax(description, v)  AS description,
    argMax(is_active, v)    AS is_active,
    max(is_new)             AS is_new_product
FROM (
    SELECT
        product_id::String  AS product_id,
        category::String    AS category,
        subcategory::String AS subcategory,
        brand::String       AS brand,
        price_eur::Float64  AS price_eur,
        description::String AS description,
        is_active::Bool     AS is_active,
        toUInt8(0)          AS is_new,
        ''                  AS v
    FROM shop.catalog
    UNION ALL
    SELECT
        product_id::String,
        category::String,
        subcategory::String,
        brand::String,
        price_eur::Float64,
        description::String,
        is_active::Bool,
        toUInt8(if(change::String = 'new_product', 1, 0)) AS is_new,
        toString(ts::DateTime64(3))                       AS v
    FROM shop.catalog_updates
)
GROUP BY product_id;


-- 3. Productes fora d'estoc ara mateix (Slow Loop cada ~2s)
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock';


-- 4. Popularitat i conversió suavitzada (Empirical Bayes) per producte (amb tall opcional {until_filter})
-- {until_filter}: p. ex. "AND ts < '2026-11-08'" en offline eval, o "" en producció
SELECT
    product_id::String                              AS product_id,
    countIf(event_type::String = 'view')            AS views,
    countIf(event_type::String = 'add_to_cart')     AS carts,
    countIf(event_type::String = 'purchase')        AS purchases,
    (purchases + 0.5) / (views + 20.0)              AS smoothed_cvr,
    (purchases * 5.0 + carts * 2.0 + views * 0.2)   AS weighted_pop
FROM shop.history_events
WHERE 1 = 1 {until_filter}
GROUP BY product_id
ORDER BY weighted_pop DESC;


-- 5. Popularitat contextual per (region, device, page) des de shop.history_sessions
SELECT
    context.region::String                          AS region,
    context.device::String                          AS device,
    context.page::String                            AS page,
    purchased::String                               AS product_id,
    count()                                         AS buys
FROM shop.history_sessions
WHERE purchased IS NOT NULL
  AND purchased::String != ''
  {until_filter}
GROUP BY region, device, page, product_id
ORDER BY buys DESC
LIMIT 30 BY region, device, page;


-- 6. Matriu dirigida de co-ocorrència en sessió (view -> purchase i view -> view)
-- Captura quins productes B s'acaben comprant o veient quan s'ha vist A a la mateixa sessió
SELECT
    a.product_id                                    AS src_product,
    b.product_id                                    AS dst_product,
    countIf(b.event_type = 'purchase')              AS co_purchases,
    countIf(b.event_type = 'add_to_cart')           AS co_carts,
    countIf(b.event_type = 'view')                  AS co_views,
    (co_purchases * 6.0 + co_carts * 2.5 + co_views * 1.0) AS transition_score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type::String = 'view' {until_filter}
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS event_type
    FROM shop.history_events
    WHERE 1 = 1 {until_filter}
) AS b
    ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY src_product, dst_product
ORDER BY transition_score DESC
LIMIT 30 BY src_product;


-- 7. Compres històriques per usuari (per suprimir productes ja comprats)
SELECT
    user_id::String                                 AS user_id,
    groupUniqArray(product_id::String)              AS bought_products
FROM shop.history_events
WHERE event_type::String = 'purchase' {until_filter}
GROUP BY user_id;


-- 8. Tendències en viu (últims 15 min vs 3 hores prèvies a shop.events)
SELECT
    product_id::String                                                    AS product_id,
    countIf(event_type::String = 'purchase' AND ts::DateTime64(3) > now() - INTERVAL 15 MINUTE) AS buys_15m,
    countIf(event_type::String = 'view'     AND ts::DateTime64(3) > now() - INTERVAL 15 MINUTE) AS views_15m,
    countIf(ts::DateTime64(3) <= now() - INTERVAL 15 MINUTE) / 12.0       AS avg_events_15m_prev,
    (buys_15m * 5.0 + views_15m) / greatest(avg_events_15m_prev, 1.0)     AS live_momentum
FROM shop.events
WHERE ts::DateTime64(3) > now() - INTERVAL 195 MINUTE
GROUP BY product_id
HAVING (buys_15m + views_15m) >= 3
ORDER BY live_momentum DESC
LIMIT 100;
