-- Etapa 1 · versión especial · monitorización manual (complementa src/monitor.py). Sustituir team_NN.

-- 1) Fiabilidad y latencia de los últimos 15 minutos, por minuto
SELECT toStartOfMinute(ts::DateTime64(3)) AS minute, count() AS n,
       countIf(served_by::String = 'team') / count() AS team_share,
       quantile(0.95)(latency_ms::Float64) AS p95_latency_ms,
       countIf(fallback_reason::String = 'late') AS late,
       countIf(fallback_reason::String = 'missing') AS missing,
       countIf(fallback_reason::String = 'invalid') AS invalid
FROM team_NN.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) > now64(3) - INTERVAL 15 MINUTE
GROUP BY minute ORDER BY minute;

-- 2) Comparar variantes (A/B por hash(request_id) y brazos del bandit) por model_version
--    model_version = etapa1-especial-<modelo|heur>-<brazo>
SELECT s.model_version::String AS model_version, count() AS n,
       avg(ifNull(f.clicked::String, '') != '') AS ctr,
       avg(ifNull(f.purchased::String, '') != '') AS cvr,
       avg(ifNull(f.revenue_eur::Float64, 0)) AS revenue_per_req
FROM team_NN.submissions AS s
INNER JOIN team_NN.feedback AS f ON s.request_id::String = f.request_id::String
WHERE f.served_by::String = 'team' AND f.ts::DateTime64(3) > now64(3) - INTERVAL 6 HOUR
GROUP BY model_version ORDER BY n DESC;

-- 3) Alineación de relojes: latency_ms del feedback frente a la edad al insertar
SELECT quantiles(0.5, 0.95, 0.99)(latency_ms::Float64) AS lat_q
FROM team_NN.feedback
WHERE served_by::String = 'team' AND ts::DateTime64(3) > now64(3) - INTERVAL 1 HOUR;

-- 4) Conversión en peticiones frías (usuario nuevo) frente a templadas
SELECT r.is_new_user::Bool AS new_user, count() AS n,
       avg(ifNull(f.clicked::String, '') != '') AS ctr,
       avg(ifNull(f.purchased::String, '') != '') AS cvr
FROM team_NN.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.ts::DateTime64(3) > now64(3) - INTERVAL 2 HOUR
GROUP BY new_user;
