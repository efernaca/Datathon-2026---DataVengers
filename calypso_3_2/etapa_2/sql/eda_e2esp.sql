-- Etapa 2 · versión especial · EDA (src/eda.py), incluida la S5 de la revisión (4.4).

-- ======================================================================
-- B1_atencion_ips · ¿Cómo cae la atención con la posición? (CTR por slot ponderado por 1/propensity)
-- ======================================================================
SELECT slot, count() AS imps, sum(w * (c = item)) / sum(w) AS ctr_ips,
       sum(w * (b = item)) / sum(w) AS cvr_ips, avg(c = item) AS ctr_raw
FROM ({explode}) GROUP BY slot ORDER BY slot
;

-- ======================================================================
-- B1b_atencion_mismo_item · Atención por slot con los mismos productos (top-100 más mostrados)
-- ======================================================================
WITH x AS ({explode}),
top AS (SELECT item FROM x GROUP BY item ORDER BY count() DESC LIMIT 100),
base AS (SELECT item, avg(c = item) AS item_ctr FROM x WHERE item IN top GROUP BY item)
SELECT x.slot AS slot, count() AS imps, avg(x.c = x.item) AS ctr,
       avg(x.c = x.item) / avg(base.item_ctr) AS rel_to_item_mean
FROM x INNER JOIN base ON x.item = base.item
GROUP BY slot ORDER BY slot
;

-- ======================================================================
-- B2_S3_visto_en_sesion · ¿Se clica/compra lo ya visto? (S3: por slot y si está en recent_views)
-- ======================================================================
SELECT has(rv, item) AS in_recent, slot, count() AS imps, sum(w) AS imps_w,
       sum(w * (c = item)) / sum(w) AS ctr_ips, sum(w * (b = item)) / sum(w) AS cvr_ips
FROM ({explode}) GROUP BY in_recent, slot ORDER BY in_recent, slot
;

-- ======================================================================
-- V8_ultimo_visto_por_pagina · ¿Atenuar el último visto en página de producto? (consulta V8)
-- ======================================================================
SELECT page, has(rv, item) AS in_recent, (length(rv) > 0 AND rv[-1] = item) AS is_last,
       count() AS imps, avg(c = item) AS ctr, avg(b = item) AS cvr,
       sum(w * (b = item)) / sum(w) AS cvr_ips
FROM ({explode}) GROUP BY page, in_recent, is_last ORDER BY page, in_recent, is_last
;

-- ======================================================================
-- B3_afinidad_sesion · ¿Importa la subcategoría/marca de la última vista? (sustitutos vs complementos)
-- ======================================================================
WITH x AS (SELECT * FROM ({explode}) WHERE length(rv) > 0),
cat AS (SELECT product_id::String AS pid, subcategory::String AS sub, category::String AS cat,
               brand::String AS brand FROM shop.catalog)
SELECT (ci.sub = cl.sub) AS same_subcat, (ci.cat = cl.cat) AS same_cat, (ci.brand = cl.brand) AS same_brand,
       count() AS imps, sum(x.w * (x.c = x.item)) / sum(x.w) AS ctr_ips,
       sum(x.w * (x.b = x.item)) / sum(x.w) AS cvr_ips
FROM x INNER JOIN cat AS ci ON x.item = ci.pid INNER JOIN cat AS cl ON rv[-1] = cl.pid
GROUP BY same_subcat, same_cat, same_brand ORDER BY same_subcat, same_cat, same_brand
;

-- ======================================================================
-- B4_precio · ¿Sensibilidad al precio? (compra dado clic por precio relativo en la subcategoría)
-- ======================================================================
WITH cat AS (SELECT product_id::String AS pid,
                    price_eur::Float64 / avg(price_eur::Float64) OVER (PARTITION BY subcategory::String) AS rel
             FROM shop.catalog),
x AS (SELECT ifNull(clicked::String, '') AS c, ifNull(purchased::String, '') AS b
      FROM shop.history_sessions {sample} )
SELECT least(floor(cat.rel / 0.25) * 0.25, 3.0) AS rel_price_bucket, count() AS clicks,
       avg(x.b = x.c) AS cvr_given_click
FROM x INNER JOIN cat ON x.c = cat.pid
WHERE x.c != ''
GROUP BY rel_price_bucket ORDER BY rel_price_bucket
;

-- ======================================================================
-- B5_compra_y_clic · ¿Compra implica clic? ¿Como mucho una por petición?
-- ======================================================================
SELECT count() AS n, countIf(ifNull(clicked::String, '') != '') AS clicks,
       countIf(ifNull(purchased::String, '') != '') AS purchases,
       countIf(ifNull(purchased::String, '') != '' AND purchased::String != ifNull(clicked::String, '')) AS buy_without_same_click,
       clicks / n AS ctr_request, purchases / n AS cvr_request
FROM shop.history_sessions {sample}
;

-- ======================================================================
-- S5_peticiones_en_vivo · Sección 5: ¿cuántos usuarios nuevos y sesiones vacías trae el stream?
-- ======================================================================
SELECT toStartOfHour(ts::DateTime64(3)) AS hour, count() AS n,
       avg(is_new_user::Bool) AS share_new_user,
       avg(length(recent_views::Array(JSON)) = 0) AS share_empty_session
FROM shop.requests GROUP BY hour ORDER BY hour
;

-- ======================================================================
-- S5_productos_nuevos_en_sesion · Sección 5: peticiones con un producto nuevo en recent_views
-- ======================================================================
WITH newp AS (SELECT DISTINCT product_id::String AS pid FROM shop.catalog_updates
              WHERE change::String = 'new_product')
SELECT count() AS n,
       countIf(hasAny(arrayMap(x -> x.product_id::String, recent_views::Array(JSON)),
                      (SELECT groupArray(pid) FROM newp))) AS with_new_product
FROM shop.requests
;

-- ======================================================================
-- S5_composicion_trafico · 4.4: composición del tráfico (nuevo × sesión vacía × página)
-- ======================================================================
SELECT is_new_user::Bool AS nuevo, length(recent_views::Array(JSON)) = 0 AS sesion_vacia,
       context.page::String AS page, count() AS peticiones
FROM shop.requests GROUP BY nuevo, sesion_vacia, page ORDER BY peticiones DESC
;

-- ======================================================================
-- S5_cambios_catalogo · 4.4: ¿llegan ya productos nuevos, agotados, cambios de precio?
-- ======================================================================
SELECT change::String AS cambio, count() AS n, toString(min(ts::DateTime64(3))) AS primero,
       toString(max(ts::DateTime64(3))) AS ultimo
FROM shop.catalog_updates GROUP BY cambio
;

-- ======================================================================
-- B8_historico_vs_vivo · B8: ¿se parece el stream al histórico? (tasas por petición)
-- ======================================================================
SELECT 'historico' AS source, count() AS n,
       avg(ifNull(clicked::String, '') != '') AS ctr, avg(ifNull(purchased::String, '') != '') AS cvr
FROM shop.history_sessions {sample}
UNION ALL
SELECT 'feedback_team', count(), avg(ifNull(clicked::String, '') != ''), avg(ifNull(purchased::String, '') != '')
FROM {team}.feedback WHERE served_by::String = 'team' AND request_id::String != '__init__'
UNION ALL
SELECT 'feedback_fallback', count(), avg(ifNull(clicked::String, '') != ''), avg(ifNull(purchased::String, '') != '')
FROM {team}.feedback WHERE served_by::String = 'fallback' AND request_id::String != '__init__'
;

-- ======================================================================
-- {explode}
-- ======================================================================
SELECT ifNull(clicked::String, '') AS c, ifNull(purchased::String, '') AS b,
           context.page::String AS page,
           arrayJoin(arrayEnumerate(shown::Array(String))) AS slot,
           (shown::Array(String))[slot] AS item,
           1.0 / greatest((propensity::Array(Float64))[slot], 0.01) AS w,
           arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS rv
    FROM shop.history_sessions {sample}
;

