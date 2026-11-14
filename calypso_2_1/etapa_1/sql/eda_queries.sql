-- =============================================================================
-- Etapa 1 · Consultes EDA per entendre el simulador ocult de RawTree
-- Executar contra RawTree amb:
--   rtree query --api-key $RAWTREE_API_KEY --database $TEAM_ID "<sql>"
-- =============================================================================

-- 1. Distribució d'en quin slot s'ha produït cada clic i cada compra (Position Bias)
SELECT
    indexOf(shown, clicked::String)                                      AS click_slot,
    count()                                                              AS clicks,
    countIf(ifNull(purchased::String, '') = clicked::String)             AS purchases_from_click,
    round(100.0 * count() / sum(count()) OVER (), 2)                     AS pct_of_all_clicks
FROM shop.history_sessions
WHERE clicked IS NOT NULL AND clicked::String != ''
GROUP BY click_slot
ORDER BY click_slot;


-- 1b. Propensió mitjana i CTR observat per posició (slot 1..10)
SELECT
    slot,
    count()                                                              AS impressions,
    countIf(clicked_id = slot_product)                                   AS clicks,
    countIf(purchased_id = slot_product)                                 AS purchases,
    round(100.0 * clicks / impressions, 4)                               AS slot_ctr_pct,
    round(100.0 * purchases / impressions, 4)                            AS slot_cvr_pct,
    round(avg(slot_propensity), 5)                                       AS avg_propensity
FROM (
    SELECT
        ifNull(clicked::String, '')                                      AS clicked_id,
        ifNull(purchased::String, '')                                    AS purchased_id,
        arrayJoin(arrayEnumerate(shown))                                 AS slot,
        shown[slot]::String                                              AS slot_product,
        propensity[slot]::Float64                                        AS slot_propensity
    FROM shop.history_sessions
)
GROUP BY slot
ORDER BY slot;


-- 2. Conversió i ingressos segons context (page, device) i longitud de recent_views
SELECT
    context.page::String                                                 AS page,
    context.device::String                                               AS device,
    count()                                                              AS sessions,
    round(100.0 * countIf(ifNull(clicked::String, '') != '') / count(), 3)   AS ctr_pct,
    round(100.0 * countIf(ifNull(purchased::String, '') != '') / count(), 3) AS cvr_pct,
    round(avg(ifNull(revenue_eur::Float64, 0.0)), 4)                     AS avg_revenue_eur,
    round(avg(length(recent_views)), 2)                                  AS avg_recent_views
FROM shop.history_sessions
GROUP BY page, device
ORDER BY sessions DESC;


-- 2b. Efecte de la regió sobre preu mitjà de compra i taxa de conversió
SELECT
    context.region::String                                               AS region,
    count()                                                              AS sessions,
    round(100.0 * countIf(ifNull(purchased::String, '') != '') / count(), 3) AS cvr_pct,
    round(avgIf(revenue_eur::Float64, ifNull(purchased::String, '') != ''), 2) AS avg_order_value_eur
FROM shop.history_sessions
GROUP BY region
ORDER BY sessions DESC;


-- 3. Saturació per compra prèvia: probabilitat que un usuari compri un producte ja comprat abans
WITH user_first_buy AS (
    SELECT
        user_id::String                                                  AS user_id,
        product_id::String                                               AS product_id,
        min(ts::DateTime64(3))                                           AS first_buy_ts
    FROM shop.history_events
    WHERE event_type::String = 'purchase'
    GROUP BY user_id, product_id
)
SELECT
    countIf(ufb.first_buy_ts IS NOT NULL AND ufb.first_buy_ts < hs.ts::DateTime64(3)) AS repeat_product_purchases,
    count()                                                                           AS total_session_purchases,
    round(100.0 * repeat_product_purchases / greatest(total_session_purchases, 1), 3) AS repeat_purchase_pct
FROM shop.history_sessions AS hs
LEFT JOIN user_first_buy AS ufb
    ON hs.user_id::String = ufb.user_id
   AND hs.purchased::String = ufb.product_id
WHERE hs.purchased IS NOT NULL AND hs.purchased::String != '';


-- 4. Relació entre recent_views i el producte comprat (exact match vs empty recent_views)
SELECT
    count()                                                              AS total_purchases,
    countIf(has(arrayMap(x -> x.product_id::String, hs.recent_views), hs.purchased::String)) AS bought_exact_recent_view,
    round(100.0 * bought_exact_recent_view / count(), 2)                 AS pct_exact_recent_view,
    countIf(length(hs.recent_views) = 0)                                 AS bought_with_empty_recent_views,
    round(100.0 * bought_with_empty_recent_views / count(), 2)           AS pct_empty_recent_views
FROM shop.history_sessions AS hs
WHERE hs.purchased IS NOT NULL AND hs.purchased::String != '';


-- 5. Transicions de sessió a shop.history_events (taxa view -> purchase mateix ítem vs cross-item)
SELECT
    event_type::String                                                   AS event_type,
    count()                                                              AS n_events,
    count(DISTINCT user_id::String)                                      AS unique_users,
    count(DISTINCT session_id::String)                                   AS unique_sessions,
    count(DISTINCT product_id::String)                                   AS unique_products
FROM shop.history_events
GROUP BY event_type
ORDER BY n_events DESC;


-- 6. Resum del catàleg per categoria
SELECT
    category::String                                                     AS category,
    count(DISTINCT product_id::String)                                   AS n_products,
    round(avg(price_eur::Float64), 2)                                    AS avg_price_eur,
    round(min(price_eur::Float64), 2)                                    AS min_price_eur,
    round(max(price_eur::Float64), 2)                                    AS max_price_eur
FROM shop.catalog
WHERE is_active::Bool = true
GROUP BY category
ORDER BY n_products DESC;
