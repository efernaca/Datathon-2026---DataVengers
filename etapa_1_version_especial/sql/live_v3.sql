-- Etapa 1 · versión especial · consultas en vivo (src/serve.py). Todas incrementales por marca de agua `ts` con solape y deduplicación.

-- ======================================================================
-- S1 · Lectura de peticiones con edad (bucle rápido)
-- ======================================================================
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       user_id::String AS user_id, is_new_user, context, recent_views,
       dateDiff('millisecond', ts::DateTime64(3), now64(3)) AS age_ms
FROM shop.requests
WHERE ts::DateTime64(3) > now64(3) - INTERVAL {window} SECOND
ORDER BY ts::DateTime64(3) ASC
;

-- ======================================================================
-- S1 legacy (si S1 falla 5 veces seguidas)
-- ======================================================================
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       user_id::String AS user_id, is_new_user, context, recent_views
FROM shop.requests ORDER BY ts::DateTime64(3) DESC LIMIT 60
;

-- ======================================================================
-- S2 · Catálogo incremental: stock, precios y productos nuevos (cada 0,5 s)
-- ======================================================================
SELECT toString(ts::DateTime64(3)) AS ts, change::String AS change,
       product_id::String AS product_id, ifNull(category::String, '') AS category,
       ifNull(subcategory::String, '') AS subcategory, ifNull(brand::String, '') AS brand,
       price_eur::Nullable(Float64) AS price_eur, ifNull(description::String, '') AS description,
       is_active::Nullable(Bool) AS is_active
FROM shop.catalog_updates
WHERE ts::DateTime64(3) >= '{since}'
ORDER BY ts::DateTime64(3) ASC
LIMIT 20000
;

-- ======================================================================
-- D1/V5/V2 · Eventos incrementales (cada 2 s)
-- ======================================================================
SELECT toString(ts::DateTime64(3)) AS ts, event_type::String AS event_type,
       user_id::String AS user_id, product_id::String AS product_id
FROM shop.events
WHERE ts::DateTime64(3) >= '{since}'
ORDER BY ts::DateTime64(3) ASC
LIMIT 50000
;

-- ======================================================================
-- D2/V2 · Feedback incremental (cada 5 s)
-- ======================================================================
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       served_by::String AS served_by, ifNull(fallback_reason::String, '') AS fallback_reason,
       latency_ms::Float64 AS latency_ms, ifNull(clicked::String, '') AS clicked,
       ifNull(purchased::String, '') AS purchased, revenue_eur::Float64 AS revenue_eur
FROM {team}.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '{since}'
ORDER BY ts::DateTime64(3) ASC
LIMIT 20000
;

