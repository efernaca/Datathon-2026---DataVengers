-- Etapa 2 · versión especial · consultas del proceso servidor: todas incrementales y pequeñas.

-- ======================================================================
-- S1 · Peticiones con edad
-- ======================================================================
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       user_id::String AS user_id, is_new_user, context, recent_views,
       dateDiff('millisecond', ts::DateTime64(3), now64(3)) AS age_ms
FROM shop.requests
WHERE ts::DateTime64(3) > now64(3) - INTERVAL {window} SECOND
ORDER BY ts::DateTime64(3) ASC
;

-- ======================================================================
-- S1 legacy
-- ======================================================================
SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
       user_id::String AS user_id, is_new_user, context, recent_views
FROM shop.requests ORDER BY ts::DateTime64(3) DESC LIMIT 60
;

-- ======================================================================
-- S2 · Catálogo incremental
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
-- Eventos incrementales
-- ======================================================================
SELECT toString(ts::DateTime64(3)) AS ts, event_type::String AS event_type,
       user_id::String AS user_id, product_id::String AS product_id
FROM shop.events
WHERE ts::DateTime64(3) >= '{since}'
ORDER BY ts::DateTime64(3) ASC
LIMIT 50000
;

-- ======================================================================
-- N3 · Feedback incremental SIN JOIN (el user_id de cada petición se guarda en memoria)
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

-- ======================================================================
-- 4.2 · Feedback para aprender pesos y analizar experimentos
-- ======================================================================
SELECT request_id::String AS request_id, served_by::String AS served_by,
       ifNull(clicked::String, '') AS clicked, ifNull(purchased::String, '') AS purchased,
       ifNull(revenue_eur::Float64, 0) AS revenue_eur
FROM {team}.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '{since}'
;

