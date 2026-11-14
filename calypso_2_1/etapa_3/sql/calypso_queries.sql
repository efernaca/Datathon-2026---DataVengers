-- =============================================================================
-- Etapa 3 · Calypso-2.1 · Consultes SQL Push-Down per RawTree (ClickHouse)
-- Nota: substituir team_NN pel TEAM_ID real (ex. team_32) en execució.
-- =============================================================================

-- 1. Catàleg base (actius, preus i descripció per Cold Start)
SELECT
    product_id::String     AS product_id,
    category::String       AS category,
    subcategory::String    AS subcategory,
    brand::String          AS brand,
    price_eur::Float64     AS price_eur,
    description::String    AS description,
    is_active::Bool        AS is_active
FROM shop.catalog;

-- 2. Actualitzacions incrementals de catàleg (nous productes i canvis d'estoc/preu)
SELECT
    product_id::String                             AS product_id,
    argMax(category::String, ts::DateTime64(3))    AS category,
    argMax(subcategory::String, ts::DateTime64(3)) AS subcategory,
    argMax(brand::String, ts::DateTime64(3))       AS brand,
    argMax(price_eur::Float64, ts::DateTime64(3))  AS price_eur,
    argMax(description::String, ts::DateTime64(3)) AS description,
    argMax(is_active::Bool, ts::DateTime64(3))     AS is_active,
    max(if(change::String = 'new_product', 1, 0))  AS is_new_product
FROM shop.catalog_updates
GROUP BY product_id;

-- 3. Auto-conversió per producte i popularitat ponderada històrica
SELECT
    product_id::String                          AS product_id,
    countIf(event_type::String = 'view')        AS views,
    countIf(event_type::String = 'add_to_cart') AS carts,
    countIf(event_type::String = 'purchase')    AS purchases,
    (purchases + 0.5) / (views + 25.0)          AS self_cvr,
    (purchases * 5.0 + carts * 2.0 + views * 0.2) AS weighted_pop
FROM shop.history_events
GROUP BY product_id
ORDER BY weighted_pop DESC;

-- 4. Transicions dirigides de sessió (per calcular Lift en Python)
SELECT
    a.product_id                                AS src_product,
    b.product_id                                AS dst_product,
    (countIf(b.event_type = 'purchase') * 6.0 +
     countIf(b.event_type = 'add_to_cart') * 2.5 +
     countIf(b.event_type = 'view') * 1.0)      AS raw_transition_score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type::String = 'view'
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS event_type
    FROM shop.history_events
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY src_product, dst_product
ORDER BY raw_transition_score DESC
LIMIT 40 BY src_product;

-- 5. Perfil profund d'usuari (compres prèvies i estadístiques de preu logarítmic)
SELECT
    user_id::String                                                              AS user_id,
    groupUniqArrayIf(product_id::String, event_type::String = 'purchase')        AS bought_products,
    avgIf(log1p(greatest(price_eur::Float64, 0.0)), event_type::String = 'purchase') AS log_price_mean,
    stddevPopIf(log1p(greatest(price_eur::Float64, 0.0)), event_type::String = 'purchase') AS log_price_std
FROM shop.history_events
GROUP BY user_id;

-- 6. Ítems d'alta intenció no comprats (cistelles abandonades i múltiples vistes)
SELECT
    user_id::String                                                     AS user_id,
    product_id::String                                                  AS product_id,
    (countIf(event_type::String = 'add_to_cart') * 3.0 +
     countIf(event_type::String = 'view') * 0.5)                        AS intent_score
FROM shop.history_events
GROUP BY user_id, product_id
HAVING countIf(event_type::String = 'purchase') = 0
   AND (countIf(event_type::String = 'add_to_cart') >= 1 OR countIf(event_type::String = 'view') >= 2)
ORDER BY intent_score DESC
LIMIT 10 BY user_id;

-- 7. Afinitat històrica d'usuari per subcategoria i marca
SELECT
    e.user_id::String                                                   AS user_id,
    c.subcategory::String                                               AS subcategory,
    c.brand::String                                                     AS brand,
    (countIf(e.event_type::String = 'purchase') * 5.0 +
     countIf(e.event_type::String = 'add_to_cart') * 2.0 +
     countIf(e.event_type::String = 'view') * 0.3)                      AS affinity
FROM shop.history_events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
GROUP BY user_id, subcategory, brand
ORDER BY affinity DESC
LIMIT 8 BY user_id;

-- 8. Popularitat contextual jeràrquica (regió x dispositiu x pàgina)
SELECT
    context.region::String AS region,
    context.device::String AS device,
    context.page::String   AS page,
    purchased::String      AS product_id,
    count()                AS buys
FROM shop.history_sessions
WHERE purchased IS NOT NULL AND purchased::String != ''
GROUP BY region, device, page, product_id
ORDER BY buys DESC
LIMIT 35 BY region, device, page;

-- 9. Esdeveniments en viu recents (tendència de demanda i senyals de sessió recent)
SELECT
    product_id::String     AS product_id,
    countIf(event_type::String = 'purchase') * 5.0 +
    countIf(event_type::String = 'add_to_cart') * 2.0 +
    countIf(event_type::String = 'view') * 0.2 AS live_score,
    countIf(event_type::String = 'purchase')   AS live_purchases,
    count()                                    AS live_events
FROM shop.events
GROUP BY product_id
ORDER BY live_score DESC
LIMIT 200;

-- 10. Compres recents en viu (evitar recomanar productes ja comprats en directe)
SELECT
    r.user_id::String                   AS user_id,
    groupUniqArray(f.purchased::String) AS live_bought_products
FROM team_NN.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND f.purchased IS NOT NULL
  AND f.purchased::String != ''
GROUP BY user_id;

-- 11. Retroalimentació en viu per al PositionAwareBandit i PositionDebiasedRanker (últimes 4.000 peticions)
SELECT
    f.request_id::String                   AS request_id,
    r.user_id::String                      AS user_id,
    r.is_new_user::Bool                    AS is_new_user,
    r.context.region::String               AS region,
    r.context.device::String               AS device,
    r.context.page::String                 AS page,
    r.recent_views                         AS recent_views,
    f.shown                                AS shown,
    f.clicked::String                      AS clicked,
    f.purchased::String                    AS purchased,
    f.revenue_eur::Float64                 AS revenue_eur
FROM team_NN.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
ORDER BY f.ts DESC
LIMIT 4000;

-- 12. Sessions històriques d'entrenament inicial (per arrencar el Re-Ranker ML abans de tenir feedback)
SELECT
    session_id::String                     AS request_id,
    user_id::String                        AS user_id,
    is_new_user::Bool                      AS is_new_user,
    context.region::String                 AS region,
    context.device::String                 AS device,
    context.page::String                   AS page,
    recent_views                           AS recent_views,
    shown                                  AS shown,
    clicked::String                        AS clicked,
    purchased::String                      AS purchased
FROM shop.history_sessions
WHERE length(shown) >= 10
LIMIT 5000;

-- 13. Inspecció de memòria persistent a team_NN.scratch (bandits, decisions CalypsoAgent i cold start)
SELECT
    kind::String          AS kind,
    model_version::String AS model_version,
    ts::String            AS ts,
    count()               AS num_records
FROM team_NN.scratch
GROUP BY kind, model_version, ts
ORDER BY ts DESC
LIMIT 50;
