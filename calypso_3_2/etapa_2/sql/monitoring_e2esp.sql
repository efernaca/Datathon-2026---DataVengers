-- Etapa 2 · versión especial · monitorización y experimentos (sustituir team_NN)

-- 1) Fiabilidad por minuto (últimos 15 min)
SELECT toStartOfMinute(ts::DateTime64(3)) AS minute, count() AS n,
       countIf(served_by::String = 'team') / count() AS team_share,
       quantile(0.95)(latency_ms::Float64) AS p95_latency_ms
FROM team_NN.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) > now64(3) - INTERVAL 15 MINUTE
GROUP BY minute ORDER BY minute;

-- 2) Experimento factorial (4.1) directamente en RawTree: CTR por interruptor.
--    model_version = e2esp|<pesos>|mmr1|rev0|rg1|int0|pf1|self1|shuf0|fin0
SELECT arrayJoin(['mmr', 'rev', 'rg', 'int', 'pf', 'self']) AS interruptor,
       position(s.model_version::String, concat('|', interruptor, '1')) > 0 AS encendido,
       count() AS n, avg(ifNull(f.clicked::String, '') != '') AS ctr,
       avg(ifNull(f.purchased::String, '') != '') AS cvr
FROM team_NN.submissions AS s
INNER JOIN team_NN.feedback AS f ON s.request_id::String = f.request_id::String
WHERE f.served_by::String = 'team' AND position(s.model_version::String, 'shuf0') > 0
GROUP BY interruptor, encendido ORDER BY interruptor, encendido;

-- 3) Curva de atención medida (listas barajadas, 4.1)
SELECT indexOf(f.shown::Array(String), f.clicked::String) AS slot, count() AS clicks
FROM team_NN.submissions AS s
INNER JOIN team_NN.feedback AS f ON s.request_id::String = f.request_id::String
WHERE position(s.model_version::String, 'shuf1') > 0 AND ifNull(f.clicked::String, '') != ''
GROUP BY slot ORDER BY slot;

-- 4) Modo final (4.4): CTR con y sin perfil
SELECT position(s.model_version::String, 'fin1') > 0 AS modo_final, count() AS n,
       avg(ifNull(f.clicked::String, '') != '') AS ctr
FROM team_NN.submissions AS s
INNER JOIN team_NN.feedback AS f ON s.request_id::String = f.request_id::String
WHERE f.served_by::String = 'team'
GROUP BY modo_final;
