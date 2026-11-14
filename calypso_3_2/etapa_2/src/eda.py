"""EDA como experimentos sobre el simulador (enfoque B, B1–B8, S3, V8 y sección 5).

    python -m src.eda                 # contra RawTree real → etapa_2_version_especial/eda/*.csv + EDA_RESULTS.md
    python -m src.eda --local-sim     # mismo código contra el mundo sintético (prueba)

Cada pregunta guarda una tabla pequeña en `eda/` y una línea en `EDA_RESULTS.md` con la
decisión que cambia. B1 escribe además `eda/attention.json`, que `Config.from_env` usa como
curva de atención por posición (γ_k / e_k) en la política y en la evaluación.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ETAPA_DIR, Config

logger = logging.getLogger("e2esp.eda")

SAMPLE = "WHERE cityHash64(request_id::String) % {mod} = 0"

EXPLODE = """
    SELECT ifNull(clicked::String, '') AS c, ifNull(purchased::String, '') AS b,
           context.page::String AS page,
           arrayJoin(arrayEnumerate(shown::Array(String))) AS slot,
           (shown::Array(String))[slot] AS item,
           1.0 / greatest((propensity::Array(Float64))[slot], 0.01) AS w,
           arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS rv
    FROM shop.history_sessions {sample}
"""

QUERIES: dict[str, tuple[str, str]] = {
    "B1_atencion_ips": ("¿Cómo cae la atención con la posición? (CTR por slot ponderado por 1/propensity)", """
SELECT slot, count() AS imps, sum(w * (c = item)) / sum(w) AS ctr_ips,
       sum(w * (b = item)) / sum(w) AS cvr_ips, avg(c = item) AS ctr_raw
FROM ({explode}) GROUP BY slot ORDER BY slot
"""),
    # Los populares salen en muchos slots gracias a la aleatoriedad: comparar el mismo
    # producto en slots distintos aísla el efecto de la posición ("intervention harvesting").
    "B1b_atencion_mismo_item": ("Atención por slot con los mismos productos (top-100 más mostrados)", """
WITH x AS ({explode}),
top AS (SELECT item FROM x GROUP BY item ORDER BY count() DESC LIMIT 100),
base AS (SELECT item, avg(c = item) AS item_ctr FROM x WHERE item IN top GROUP BY item)
SELECT x.slot AS slot, count() AS imps, avg(x.c = x.item) AS ctr,
       avg(x.c = x.item) / avg(base.item_ctr) AS rel_to_item_mean
FROM x INNER JOIN base ON x.item = base.item
GROUP BY slot ORDER BY slot
"""),
    "B2_S3_visto_en_sesion": ("¿Se clica/compra lo ya visto? (S3: por slot y si está en recent_views)", """
SELECT has(rv, item) AS in_recent, slot, count() AS imps, sum(w) AS imps_w,
       sum(w * (c = item)) / sum(w) AS ctr_ips, sum(w * (b = item)) / sum(w) AS cvr_ips
FROM ({explode}) GROUP BY in_recent, slot ORDER BY in_recent, slot
"""),
    "V8_ultimo_visto_por_pagina": ("¿Atenuar el último visto en página de producto? (consulta V8)", """
SELECT page, has(rv, item) AS in_recent, (length(rv) > 0 AND rv[-1] = item) AS is_last,
       count() AS imps, avg(c = item) AS ctr, avg(b = item) AS cvr,
       sum(w * (b = item)) / sum(w) AS cvr_ips
FROM ({explode}) GROUP BY page, in_recent, is_last ORDER BY page, in_recent, is_last
"""),
    "B3_afinidad_sesion": ("¿Importa la subcategoría/marca de la última vista? (sustitutos vs complementos)", """
WITH x AS (SELECT * FROM ({explode}) WHERE length(rv) > 0),
cat AS (SELECT product_id::String AS pid, subcategory::String AS sub, category::String AS cat,
               brand::String AS brand FROM shop.catalog)
SELECT (ci.sub = cl.sub) AS same_subcat, (ci.cat = cl.cat) AS same_cat, (ci.brand = cl.brand) AS same_brand,
       count() AS imps, sum(x.w * (x.c = x.item)) / sum(x.w) AS ctr_ips,
       sum(x.w * (x.b = x.item)) / sum(x.w) AS cvr_ips
FROM x INNER JOIN cat AS ci ON x.item = ci.pid INNER JOIN cat AS cl ON rv[-1] = cl.pid
GROUP BY same_subcat, same_cat, same_brand ORDER BY same_subcat, same_cat, same_brand
"""),
    "B4_precio": ("¿Sensibilidad al precio? (compra dado clic por precio relativo en la subcategoría)", """
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
"""),
    "B5_compra_y_clic": ("¿Compra implica clic? ¿Como mucho una por petición?", """
SELECT count() AS n, countIf(ifNull(clicked::String, '') != '') AS clicks,
       countIf(ifNull(purchased::String, '') != '') AS purchases,
       countIf(ifNull(purchased::String, '') != '' AND purchased::String != ifNull(clicked::String, '')) AS buy_without_same_click,
       clicks / n AS ctr_request, purchases / n AS cvr_request
FROM shop.history_sessions {sample}
"""),
    "S5_peticiones_en_vivo": ("Sección 5: ¿cuántos usuarios nuevos y sesiones vacías trae el stream?", """
SELECT toStartOfHour(ts::DateTime64(3)) AS hour, count() AS n,
       avg(is_new_user::Bool) AS share_new_user,
       avg(length(recent_views::Array(JSON)) = 0) AS share_empty_session
FROM shop.requests GROUP BY hour ORDER BY hour
"""),
    "S5_productos_nuevos_en_sesion": ("Sección 5: peticiones con un producto nuevo en recent_views", """
WITH newp AS (SELECT DISTINCT product_id::String AS pid FROM shop.catalog_updates
              WHERE change::String = 'new_product')
SELECT count() AS n,
       countIf(hasAny(arrayMap(x -> x.product_id::String, recent_views::Array(JSON)),
                      (SELECT groupArray(pid) FROM newp))) AS with_new_product
FROM shop.requests
"""),
    "S5_composicion_trafico": ("4.4: composición del tráfico (nuevo × sesión vacía × página)", """
SELECT is_new_user::Bool AS nuevo, length(recent_views::Array(JSON)) = 0 AS sesion_vacia,
       context.page::String AS page, count() AS peticiones
FROM shop.requests GROUP BY nuevo, sesion_vacia, page ORDER BY peticiones DESC
"""),
    "S5_cambios_catalogo": ("4.4: ¿llegan ya productos nuevos, agotados, cambios de precio?", """
SELECT change::String AS cambio, count() AS n, toString(min(ts::DateTime64(3))) AS primero,
       toString(max(ts::DateTime64(3))) AS ultimo
FROM shop.catalog_updates GROUP BY cambio
"""),
    "B8_historico_vs_vivo": ("B8: ¿se parece el stream al histórico? (tasas por petición)", """
SELECT 'historico' AS source, count() AS n,
       avg(ifNull(clicked::String, '') != '') AS ctr, avg(ifNull(purchased::String, '') != '') AS cvr
FROM shop.history_sessions {sample}
UNION ALL
SELECT 'feedback_team', count(), avg(ifNull(clicked::String, '') != ''), avg(ifNull(purchased::String, '') != '')
FROM {team}.feedback WHERE served_by::String = 'team' AND request_id::String != '__init__'
UNION ALL
SELECT 'feedback_fallback', count(), avg(ifNull(clicked::String, '') != ''), avg(ifNull(purchased::String, '') != '')
FROM {team}.feedback WHERE served_by::String = 'fallback' AND request_id::String != '__init__'
"""),
}


def run_sql_part(client: Any, team: str, mod: int, out: Path) -> dict[str, pd.DataFrame | str]:
    sample = SAMPLE.format(mod=mod)
    explode = EXPLODE.format(sample=sample)
    res: dict[str, pd.DataFrame | str] = {}
    for name, (_, sql) in QUERIES.items():
        q = sql.format(explode=explode, sample=sample, team=team)
        try:
            df = pd.DataFrame(client.query(q, timeout=600))
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
            df.to_csv(out / f"{name}.csv", index=False)
            res[name] = df
            logger.info("%s: %d filas", name, len(df))
        except Exception as exc:  # noqa: BLE001
            res[name] = f"ERROR: {str(exc)[:300]}"
            logger.warning("%s falló: %s", name, str(exc)[:200])
    return res


def python_part(rows: list[dict[str, Any]], h, cfg: Config, out: Path) -> dict[str, pd.DataFrame]:
    """B6 y B7 sobre una muestra descargada (más cómodo en Python que en SQL)."""
    from .eval import as_logged

    cat = {str(r["product_id"]): r for r in h.catalog_rows}
    res: dict[str, pd.DataFrame] = {}
    # B6: ¿el CTR de un slot depende de cuántos productos de su categoría hay encima?
    agg: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        shown = list(r["shown"])[:10]
        c = r.get("clicked") or ""
        cats = [cat.get(p, {}).get("category", "") for p in shown]
        for s in range(1, len(shown)):
            same_above = min(sum(1 for x in cats[:s] if x == cats[s]), 3)
            a = agg[(s + 1, same_above)]
            a[0] += 1
            a[1] += shown[s] == c
    b6 = pd.DataFrame([{"slot": k[0], "same_cat_above": k[1], "imps": v[0], "ctr": v[1] / v[0]}
                       for k, v in sorted(agg.items()) if v[0] >= 30])
    if len(b6):
        b6s = b6.groupby("same_cat_above").apply(
            lambda d: pd.Series({"imps": d["imps"].sum(), "ctr_mean_over_slots": d["ctr"].mean()}))
        b6s.to_csv(out / "B6_saturacion_categoria.csv")
        res["B6_saturacion_categoria"] = b6s.reset_index()
    # B7: ¿la subcategoría favorita histórica predice el clic más allá de lo que se mostró?
    hits = shown_share = n = 0.0
    for r in rows:
        c = r.get("clicked") or ""
        if not c:
            continue
        q = as_logged(r, h)
        prof = h.users.get(q.user_id)
        if prof is None or not prof.subcat:
            continue
        fav = max(prof.subcat, key=prof.subcat.get)
        shown = list(r["shown"])[:10]
        hits += cat.get(c, {}).get("subcategory") == fav
        shown_share += np.mean([cat.get(p, {}).get("subcategory") == fav for p in shown])
        n += 1
    if n:
        b7 = pd.DataFrame([{"clicks_known_users": n, "share_clicks_in_fav_subcat": hits / n,
                            "share_shown_in_fav_subcat": shown_share / n,
                            "lift": (hits / n) / max(shown_share / n, 1e-9)}])
        b7.to_csv(out / "B7_personalizacion.csv", index=False)
        res["B7_personalizacion"] = b7
    return res


def derive_attention(res: dict[str, Any], out: Path) -> list[float] | None:
    df = res.get("B1b_atencion_mismo_item")
    if not isinstance(df, pd.DataFrame) or len(df) < 10:
        df = res.get("B1_atencion_ips")
        col = "ctr_ips"
    else:
        col = "rel_to_item_mean"
    if not isinstance(df, pd.DataFrame) or len(df) < 10:
        return None
    v = df.sort_values("slot")[col].astype(float).to_numpy()[:10]
    if v[0] <= 0:
        return None
    e = np.maximum.accumulate((v / v[0])[::-1])[::-1]   # monótona decreciente
    e = [round(float(x), 4) for x in e]
    (out / "attention.json").write_text(json.dumps({"attention": e, "source": col}))
    return e


DECISIONS = {
    "B1_atencion_ips": "Curva e_k → `eda/attention.json` (política, lista diversificada y evaluación IPS).",
    "B1b_atencion_mismo_item": "Más fiable que B1 para e_k: compara el mismo producto en slots distintos.",
    "B2_S3_visto_en_sesion": "Si in_recent=1 multiplica el CTR, el modelo ya lo aprende (`in_recent`, `recent_recency`).",
    "V8_ultimo_visto_por_pagina": "Si is_last en página `product` tiene CTR bajo, mantener la atenuación (lo aprende `is_last_viewed`).",
    "B3_afinidad_sesion": "same_subcat alto → sustitutos; same_cat sin same_subcat alto → complementos (pesa EASE/transiciones).",
    "B4_precio": "Si cvr_given_click cae con el precio relativo, subir `weight_price_drop`; si no, dejarlo bajo.",
    "B5_compra_y_clic": "buy_without_same_click≈0 → compra ⊂ clic: modelar compra directamente (objetivo `buy`).",
    "B6_saturacion_categoria": "Si el CTR baja con same_cat_above, activar la lista diversificada (`--diversify always`).",
    "B7_personalizacion": "lift ≫ 1 → la personalización por historial (V1) aporta; ≈1 → apenas.",
    "S5_peticiones_en_vivo": "Si casi todos son nuevos/sin sesión, priorizar señales de sesión, contenido y vivo.",
    "S5_productos_nuevos_en_sesion": "Tamaño del segmento frío por producto.",
    "S5_composicion_trafico": "Peso de cada segmento de 4.3; si casi todo es nuevo/sin sesión, mandan contexto, tendencia y diversificación.",
    "S5_cambios_catalogo": "Si el sábado ya llegan new_product, el segmento frío de producto se puede medir hoy (N8).",
    "S5_composicion_trafico": ("4.4: composición del tráfico (nuevo × sesión vacía × página)", """
SELECT is_new_user::Bool AS nuevo, length(recent_views::Array(JSON)) = 0 AS sesion_vacia,
       context.page::String AS page, count() AS peticiones
FROM shop.requests GROUP BY nuevo, sesion_vacia, page ORDER BY peticiones DESC
"""),
    "S5_cambios_catalogo": ("4.4: ¿llegan ya productos nuevos, agotados, cambios de precio?", """
SELECT change::String AS cambio, count() AS n, toString(min(ts::DateTime64(3))) AS primero,
       toString(max(ts::DateTime64(3))) AS ultimo
FROM shop.catalog_updates GROUP BY cambio
"""),
    "B8_historico_vs_vivo": "Si el vivo difiere mucho del histórico, dar más peso al brazo `live` del bandit.",
}


def write_summary(res: dict[str, Any], att: list[float] | None, out: Path) -> None:
    lines = ["# Resultados del EDA (enfoque B)", "",
             "Generado por `python -m src.eda`. Cada tabla está en `eda/<nombre>.csv`.", ""]
    for name, val in res.items():
        title = QUERIES.get(name, (name,))[0]
        lines += [f"## {name}", "", f"**Pregunta:** {title}", "",
                  f"**Decisión que cambia:** {DECISIONS.get(name, '')}", ""]
        if isinstance(val, pd.DataFrame):
            lines += ["```", val.head(30).to_string(index=False), "```", ""]
        else:
            lines += [f"`{val}`", ""]
    if att:
        lines += ["## Curva de atención usada", "", f"`{att}`", ""]
    lines += ["## Conclusiones (rellenar a mano)", "", "- ", ""]
    (out / "EDA_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-sim", action="store_true")
    ap.add_argument("--mod", type=int, default=10, help="muestra 1/mod de history_sessions")
    ap.add_argument("--out", default=str(ETAPA_DIR / "eda"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.local_sim:
        from .localdb import LocalRawTree
        from .sim import build_world

        W = build_world(seed=5, n_products=500, n_sessions_events=15000, n_sessions_log=20000)
        client = LocalRawTree()
        client.insert("shop", "catalog", W["catalog"])
        client.insert("shop", "history_events", W["events"])
        client.insert("shop", "history_sessions", W["sessions"])
        cfg = Config()
        args.mod = 1
    else:
        from .rawtree_client import RawTreeClient

        cfg = Config.from_env()
        client = RawTreeClient(cfg.api_url, cfg.api_key, cfg.team_id)
    res = run_sql_part(client, client.team_id, args.mod, out)
    try:
        from .eval import load_sessions
        from .history import get_or_build_history

        h = get_or_build_history(client, cfg) if not args.local_sim else __import__(
            "src.history", fromlist=["build_history"]).build_history(client, cfg)
        rows = load_sessions(client, n_pages=1, mod=max(args.mod * 4, 1))
        res.update(python_part(rows, h, cfg, out))
    except Exception as exc:  # noqa: BLE001
        logger.warning("parte Python del EDA falló: %s", exc)
    att = derive_attention(res, out)
    write_summary(res, att, out)
    print(f"EDA guardado en {out}")


if __name__ == "__main__":
    main()
