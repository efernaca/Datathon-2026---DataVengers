-- =============================================================================
-- Etapa 1 · Consultes de monitorització en temps real (substituir team_NN)
-- =============================================================================

-- 1. Salut global i latència (últims 30 minuts)
SELECT
    served_by::String                              AS served_by,
    fallback_reason::String                        AS fallback_reason,
    count()                                        AS requests,
    round(100.0 * count() / sum(count()) OVER (), 2) AS share_pct,
    round(quantile(0.50)(latency_ms::Float64), 1)  AS p50_ms,
    round(quantile(0.95)(latency_ms::Float64), 1)  AS p95_ms,
    round(max(latency_ms::Float64), 1)             AS max_ms
FROM team_NN.feedback
WHERE ts::DateTime64(3) > now() - INTERVAL 30 MINUTE
  AND request_id::String != '__init__'
GROUP BY served_by, fallback_reason
ORDER BY requests DESC;


-- 2. Rendiment de negoci per hora (CTR, CVR i Revenue per petició)
SELECT
    toStartOfHour(ts::DateTime64(3))               AS hour_utc,
    served_by::String                              AS served_by,
    count()                                        AS requests,
    round(100.0 * countIf(clicked::String != '') / count(), 3)   AS ctr_pct,
    round(100.0 * countIf(purchased::String != '') / count(), 3) AS cvr_pct,
    round(sum(revenue_eur::Float64) / count(), 4)  AS revenue_per_request
FROM team_NN.feedback
WHERE request_id::String != '__init__'
GROUP BY hour_utc, served_by
ORDER BY hour_utc DESC, served_by;


-- 3. Comparació A/B per model_version unint team_NN.submissions i team_NN.feedback
SELECT
    s.model_version::String                        AS model_version,
    count()                                        AS requests,
    round(100.0 * countIf(f.served_by::String = 'team') / count(), 2)     AS on_time_pct,
    round(100.0 * countIf(f.clicked::String != '') / count(), 3)          AS ctr_pct,
    round(100.0 * countIf(f.purchased::String != '') / count(), 3)        AS cvr_pct,
    round(sum(f.revenue_eur::Float64) / count(), 4)                       AS revenue_per_request,
    round(quantile(0.95)(f.latency_ms::Float64), 1)                       AS p95_ms
FROM team_NN.feedback AS f
INNER JOIN team_NN.submissions AS s
    ON f.request_id::String = s.request_id::String
WHERE f.request_id::String != '__init__'
GROUP BY model_version
ORDER BY requests DESC;
