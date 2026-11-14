-- =============================================================================
-- Etapa 4 · Consultes SQL Separades: Històric Estàtic vs. Fast/Live & Monitor
-- =============================================================================

-- [FAST-POLL] Petites finestres SLA-strictes (2400 ms respecte al cap de stream, tolerant a clock skew)
SELECT
    request_id,
    user_id,
    is_new_user,
    context,
    recent_views
FROM shop.requests
WHERE ts::DateTime64(3) >= (SELECT max(ts::DateTime64(3)) FROM shop.requests) - INTERVAL 2400 MILLISECOND
ORDER BY ts ASC
LIMIT 25;

-- [STOCK-FAST · cada 1.5s] Productes fora d'estoc
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock';

-- [LIVE-CATALOG · cada 15s] Actualitzacions incrementals i nous productes
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

-- [MONITOR] Estat en viu per model_version a team_NN.feedback
SELECT
    coalesce(nullIf(s.model_version::String, ''), 'fallback_or_unknown') AS model_version,
    count()                                                             AS total_reqs,
    countIf(f.served_by::String = 'team')                               AS team_on_time,
    round(100.0 * team_on_time / greatest(total_reqs, 1), 2)            AS on_time_pct,
    round(quantile(0.50)(f.latency_ms::Float64), 1)                     AS p50_ms,
    round(quantile(0.95)(f.latency_ms::Float64), 1)                     AS p95_ms,
    countIf(f.clicked IS NOT NULL AND f.clicked::String != '')          AS clicks,
    countIf(f.purchased IS NOT NULL AND f.purchased::String != '')      AS purchases,
    round(sum(coalesce(f.revenue_eur::Float64, 0.0)), 2)                AS revenue_eur
FROM team_NN.feedback AS f
LEFT JOIN team_NN.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.request_id::String != '__init__'
  AND f.ts::DateTime64(3) >= now64(3) - INTERVAL 30 MINUTE
GROUP BY model_version
ORDER BY total_reqs DESC;
