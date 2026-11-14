"""Cálculo del histórico (una sola vez), caché en disco y publicación en `team.scratch`.

F5: el histórico de 60 días no cambia, así que se calcula una vez, se guarda en
`cache/history_<cutoff>.pkl` y se carga al arrancar en segundos. Cada componente
se calcula por separado: si uno falla se conserva el último válido.

Todas las consultas agregan en RawTree y solo descargan resultados pequeños.
Los arrays se castean explícitamente (`::Array(String)`, `::Array(JSON)`) porque
en RawTree cada columna es `Dynamic`. Probadas con ClickHouse embebido (chdb) sobre
tablas `Dynamic` sintéticas (ver `tests/`); revisad contra RawTree real la primera vez.
"""
from __future__ import annotations

import logging
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .ease import ease_from_neighbors, ease_top_neighbors, fit_ease
from .state import HistoryBundle, UserProfile

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- SQL

SQL_CATALOG = """
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       price_eur::Float64 AS price_eur, description::String AS description,
       is_active::Bool AS is_active, toString(created_at) AS created_at
FROM shop.catalog
"""

SQL_POPULARITY = """
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'view') AS views,
       countIf(event_type::String = 'add_to_cart') AS carts,
       countIf(event_type::String = 'purchase') AS purchases
FROM shop.history_events
{where_ev}
GROUP BY product_id
"""

# Estadísticas IPS por ítem (variable de identidad del enfoque A y prior de CVR).
SQL_ITEM_IPS = """
SELECT item AS product_id, count() AS imps, sum(w) AS imps_w,
       sum(w * (clicked_id = item)) AS clicks_w,
       sum(w * (purchased_id = item)) AS buys_w
FROM (
    SELECT ifNull(clicked::String, '') AS clicked_id,
           ifNull(purchased::String, '') AS purchased_id,
           arrayJoin(arrayEnumerate(shown::Array(String))) AS slot,
           (shown::Array(String))[slot] AS item,
           1.0 / greatest((propensity::Array(Float64))[slot], 0.01) AS w
    FROM shop.history_sessions
    {where_ev}
)
GROUP BY item
"""

# V4: popularidad por contexto ponderada por 1/propensity del slot donde ocurrió,
# para no reproducir el sesgo del carrusel antiguo hacia los populares.
SQL_CONTEXT_IPS = """
SELECT context.region::String AS region, context.device::String AS device,
       context.page::String AS page, {col}::String AS product_id,
       sum(1.0 / greatest((propensity::Array(Float64))[
           greatest(indexOf(shown::Array(String), {col}::String), 1)], 0.01)) AS score_w,
       count() AS n
FROM shop.history_sessions
WHERE {col} IS NOT NULL AND {col}::String != '' {and_ev}
GROUP BY region, device, page, product_id
ORDER BY score_w DESC
LIMIT 300 BY region, device, page
"""

# S4 de la revisión de la Etapa 2 (N6) / V3: transiciones dirigidas (B después de ver A),
# deduplicadas por sesión, lift raw / (pop(B) + 30)^β aplicado ANTES del LIMIT, soporte ≥ 3.
SQL_TRANSITIONS = """
WITH ev AS (
    SELECT session_id::String AS s, product_id::String AS p,
           event_type::String AS e, ts::DateTime64(3) AS t
    FROM shop.history_events
    {where_ev}
),
ok AS (SELECT s FROM ev GROUP BY s HAVING uniqExact(p) BETWEEN 2 AND {max_items}),
fv AS (SELECT s, p, min(t) AS t FROM ev WHERE e = 'view' AND s IN ok GROUP BY s, p),
lt AS (SELECT s, p, e, min(t) AS t FROM ev WHERE s IN ok GROUP BY s, p, e),
pairs AS (
    SELECT a.p AS src, b.p AS dst,
           max(b.e = 'purchase') AS bp, max(b.e = 'add_to_cart') AS bc
    FROM fv AS a INNER JOIN lt AS b ON a.s = b.s
    WHERE a.p != b.p AND a.t < b.t
    GROUP BY a.s, src, dst
),
agg AS (
    SELECT src, dst, count() AS support,
           sum(bp) * 6.0 + sum(bc) * 2.5 + count() * 1.0 AS raw
    FROM pairs GROUP BY src, dst HAVING support >= {min_support}
),
popd AS (SELECT p AS dst, countIf(e = 'purchase') * 5.0 + countIf(e = 'add_to_cart') * 2.0
                 + countIf(e = 'view') * 0.2 AS wp FROM ev GROUP BY p)
SELECT agg.src AS src_product, agg.dst AS dst_product, agg.support AS support,
       agg.raw / pow(popd.wp + {prior}, {beta}) AS lift
FROM agg INNER JOIN popd ON agg.dst = popd.dst
ORDER BY lift DESC
LIMIT {topk} BY src_product
"""

# C: matriz de co-ocurrencias G = XᵀX para EASE (triángulo superior, paginado por hash).
SQL_EASE_DIAG = """
WITH sp AS (
    SELECT session_id::String AS s, product_id::String AS p
    FROM shop.history_events {where_ev} GROUP BY s, p
),
ok AS (SELECT s FROM sp GROUP BY s HAVING count() BETWEEN 2 AND {max_items})
SELECT p AS product_id, count() AS n FROM sp WHERE s IN ok GROUP BY p
"""

SQL_EASE_PAIRS = """
WITH sp AS (
    SELECT session_id::String AS s, product_id::String AS p
    FROM shop.history_events {where_ev} GROUP BY s, p
),
ok AS (SELECT s FROM sp GROUP BY s HAVING count() BETWEEN 2 AND {max_items}),
spf AS (SELECT s, p FROM sp WHERE s IN ok)
SELECT a.p AS i, b.p AS j, count() AS c
FROM spf AS a INNER JOIN spf AS b ON a.s = b.s
WHERE a.p < b.p AND cityHash64(a.p) % {{n_pages}} = {{page}}
GROUP BY i, j
HAVING c >= {min_c}
"""

# N13: lo que la Etapa 2 llamaba "self_conversion" era compras/vistas globales del producto.
# Aquí se mide lo que el nombre promete: P(compra del producto | ya estaba en recent_views), IPS.
SQL_VISTO_CONV = """
SELECT item AS product_id, count() AS imps, sum(w) AS imps_w, sum(w * (b = item)) AS buys_w
FROM (
    SELECT ifNull(purchased::String, '') AS b,
           arrayJoin(arrayEnumerate(shown::Array(String))) AS slot,
           (shown::Array(String))[slot] AS item,
           1.0 / greatest((propensity::Array(Float64))[slot], 0.01) AS w,
           arrayMap(x -> x.product_id::String, recent_views::Array(JSON)) AS rv
    FROM shop.history_sessions
    {where_ev}
)
WHERE has(rv, item)
GROUP BY item
"""

# N11: intención (carrito o ≥ 2 vistas sin compra) con su antigüedad, para decaerla.
SQL_USER_INTENT = """
SELECT user_id, pid, strength, age_days FROM (
    SELECT user_id::String AS user_id, product_id::String AS pid,
           countIf(event_type::String = 'add_to_cart') * 3.0 + countIf(event_type::String = 'view') * 0.5 AS strength,
           countIf(event_type::String = 'purchase') AS buys,
           countIf(event_type::String = 'add_to_cart') AS carts,
           countIf(event_type::String = 'view') AS views,
           dateDiff('second', max(ts::DateTime64(3)), {ref}) / 86400.0 AS age_days
    FROM shop.history_events
    WHERE cityHash64(user_id::String) % {{n_pages}} = {{page}} {and_ev}
    GROUP BY user_id, pid
)
WHERE buys = 0 AND (carts >= 1 OR views >= 2)
ORDER BY strength * exp(-age_days * 0.693 / {hl}) DESC
LIMIT {lim} BY user_id
"""

SQL_HISTORY_END = "SELECT toString(max(ts::DateTime64(3))) AS t FROM shop.history_events {where_ev}"

# V1: afinidad de usuario por subcategoría y marca (paginado por hash de usuario).
SQL_USER_AFFINITY = """
WITH e AS (
    SELECT user_id::String AS user_id, product_id::String AS pid,
           multiIf(event_type::String = 'purchase', 5.0,
                   event_type::String = 'add_to_cart', 2.0, 0.3) AS w
    FROM shop.history_events
    WHERE cityHash64(user_id::String) % {{n_pages}} = {{page}} {and_ev}
),
c AS (SELECT product_id::String AS pid, {field}::String AS val FROM shop.catalog)
SELECT e.user_id AS user_id, c.val AS val, sum(e.w) AS w
FROM e INNER JOIN c ON e.pid = c.pid
GROUP BY user_id, val
ORDER BY w DESC
LIMIT {limit} BY user_id
"""

SQL_USER_PURCHASES = """
SELECT user_id::String AS user_id,
       groupUniqArray(product_id::String) AS bought,
       avg(log1p(price_eur::Float64)) AS mu,
       stddevPop(log1p(price_eur::Float64)) AS sigma,
       count() AS n
FROM shop.history_events
WHERE event_type::String = 'purchase'
  AND cityHash64(user_id::String) % {{n_pages}} = {{page}} {and_ev}
GROUP BY user_id
"""


def _where(cutoff: str | None) -> tuple[str, str]:
    if not cutoff:
        return "", ""
    cond = f"ts::DateTime64(3) < '{cutoff}'"
    return f"WHERE {cond}", f"AND {cond}"


# ----------------------------------------------------------------------------- build


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def build_history(
    client: Any,
    cfg: Config,
    cutoff: str | None = None,
    previous: HistoryBundle | None = None,
    components: set[str] | None = None,
) -> HistoryBundle:
    """Calcula el histórico componente a componente. `client` necesita `.query(sql, timeout)`."""
    where_ev, and_ev = _where(cutoff)
    T = cfg.query_timeout_slow_sec
    ok: dict[str, bool] = {}
    want = components or {"catalog", "pop", "item_stats", "ctx", "transitions", "ease", "users",
                          "visto", "intent"}

    def run(name: str, fn: Callable[[], Any], fallback: Any) -> Any:
        if name not in want:
            return fallback
        t0 = time.perf_counter()
        try:
            val = fn()
            ok[name] = True
            logger.info("histórico[%s] OK en %.1fs", name, time.perf_counter() - t0)
            return val
        except Exception as exc:  # noqa: BLE001
            ok[name] = False
            logger.error("histórico[%s] FALLA (%s); se conserva el anterior", name, str(exc)[:300])
            return fallback

    prev = previous
    catalog_rows = run("catalog", lambda: client.query(SQL_CATALOG, timeout=T),
                       prev.catalog_rows if prev else [])
    if not catalog_rows:
        raise RuntimeError("Sin catálogo no se puede construir el histórico.")

    def _pop() -> dict[str, float]:
        out = {}
        for r in client.query(SQL_POPULARITY.format(where_ev=where_ev), timeout=T):
            out[str(r["product_id"])] = (cfg.event_weight("purchase") * _f(r["purchases"])
                                         + cfg.event_weight("add_to_cart") * _f(r["carts"])
                                         + cfg.event_weight("view") * _f(r["views"]))
        return out

    pop = run("pop", _pop, prev.pop if prev else {})

    def _items() -> dict[str, tuple[float, float, float, float]]:
        rows = client.query(SQL_ITEM_IPS.format(where_ev=where_ev), timeout=T)
        return {str(r["product_id"]): (_f(r["imps"]), _f(r["imps_w"]), _f(r["clicks_w"]), _f(r["buys_w"]))
                for r in rows}

    item_stats = run("item_stats", _items, prev.item_stats if prev else {})

    def _ctx() -> dict[tuple[str, str, str], dict[str, float]]:
        out: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for col, w in (("purchased", 1.0), ("clicked", 0.15)):
            rows = client.query(SQL_CONTEXT_IPS.format(col=col, and_ev=and_ev), timeout=T)
            for r in rows:
                key = (str(r["region"]), str(r["device"]), str(r["page"]))
                s = w * _f(r["score_w"])
                out[key][str(r["product_id"])] += s
        return {k: dict(v) for k, v in out.items()}

    ctx_prior = run("ctx", _ctx, prev.ctx_prior if prev else {})

    def _trans() -> dict[str, list[tuple[str, float]]]:
        sql = SQL_TRANSITIONS.format(where_ev=where_ev, max_items=cfg.ease_max_items_per_session,
                                     min_support=cfg.transition_min_support, prior=cfg.transition_prior,
                                     beta=cfg.transition_beta, topk=cfg.transition_topk)
        out: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for r in client.query(sql, timeout=T):
            out[str(r["src_product"])].append((str(r["dst_product"]), _f(r["lift"])))
        for k in out:
            out[k].sort(key=lambda x: -x[1])
        return dict(out)

    transitions = run("transitions", _trans, prev.transitions if prev else {})

    def _ease():
        diag_rows = client.query(SQL_EASE_DIAG.format(where_ev=where_ev,
                                                      max_items=cfg.ease_max_items_per_session), timeout=T)
        counts = {str(r["product_id"]): _f(r["n"]) for r in diag_rows}
        tmpl = SQL_EASE_PAIRS.format(where_ev=where_ev, max_items=cfg.ease_max_items_per_session,
                                     min_c=cfg.ease_min_pair_count)
        pairs = client.query_paged(tmpl, n_pages=8, timeout=T) if hasattr(client, "query_paged") else \
            [r for p in range(8) for r in client.query(tmpl.format(page=p, n_pages=8), timeout=T)]
        return fit_ease(counts, pairs, lam=cfg.ease_lambda)

    ease = run("ease", _ease, prev.ease if prev else None)

    def _users() -> dict[str, UserProfile]:
        users: dict[str, UserProfile] = {}
        n_pages = 4

        def paged(tmpl: str) -> list[dict[str, Any]]:
            if hasattr(client, "query_paged"):
                return client.query_paged(tmpl, n_pages=n_pages, timeout=T)
            return [r for p in range(n_pages) for r in client.query(tmpl.format(page=p, n_pages=n_pages), timeout=T)]

        for field_name, attr, limit in (("subcategory", "subcat", 5), ("brand", "brand", 3)):
            rows = paged(SQL_USER_AFFINITY.format(field=field_name, limit=limit, and_ev=and_ev))
            for r in rows:
                u = users.setdefault(str(r["user_id"]), UserProfile())
                getattr(u, attr)[str(r["val"])] = _f(r["w"])
        for u in users.values():
            for attr in ("subcat", "brand"):
                d = getattr(u, attr)
                m = max(d.values(), default=0.0)
                if m > 0:
                    setattr(u, attr, {k: v / m for k, v in d.items()})
        for r in paged(SQL_USER_PURCHASES.format(and_ev=and_ev)):
            u = users.setdefault(str(r["user_id"]), UserProfile())
            u.bought = frozenset(str(p) for p in (r.get("bought") or ()))
            u.n_buys = int(_f(r["n"]))
            u.price_mu = _f(r["mu"], float("nan"))
            u.price_sigma = _f(r["sigma"], float("nan"))
        return users

    users = run("users", _users, prev.users if prev else {})

    def _visto() -> dict[str, tuple[float, float]]:
        rows = client.query(SQL_VISTO_CONV.format(where_ev=where_ev), timeout=T)
        return {str(r["product_id"]): (_f(r["imps_w"]), _f(r["buys_w"])) for r in rows}

    visto = run("visto", _visto, prev.visto_stats if prev else {})

    end_rows = client.query_safe(SQL_HISTORY_END.format(where_ev=where_ev), timeout=T) or []
    history_end = end_rows[0]["t"] if end_rows else None

    def _intent() -> dict[str, tuple[tuple[str, float, float], ...]]:
        ref = f"toDateTime64('{history_end}', 3)" if history_end else "now64(3)"
        tmpl = SQL_USER_INTENT.format(ref=ref, and_ev=and_ev, hl=cfg.intent_halflife_days,
                                      lim=cfg.intent_limit_per_user)
        rows = client.query_paged(tmpl, n_pages=4, timeout=T) if hasattr(client, "query_paged") else \
            [r for p in range(4) for r in client.query(tmpl.format(page=p, n_pages=4), timeout=T)]
        out: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for r in rows:
            out[str(r["user_id"])].append((str(r["pid"]), _f(r["strength"]), max(_f(r["age_days"]), 0.0)))
        return {u: tuple(v) for u, v in out.items()}

    intent = run("intent", _intent, prev.intent if prev else {})

    built = time.time()
    return HistoryBundle(
        catalog_rows=[dict(r) for r in catalog_rows], pop=pop, item_stats=item_stats,
        ctx_prior=ctx_prior, transitions=transitions, ease=ease, users=users,
        visto_stats=visto, intent=intent, version=f"h{int(built)}",
        meta={"cutoff": cutoff, "built_at": built, "n_users": len(users), "history_end": history_end},
        ok_components=ok,
    )


# ----------------------------------------------------------------------------- caché (F5)


def cache_path(cfg: Config, cutoff: str | None) -> Path:
    tag = (cutoff or "full").replace(":", "").replace(" ", "_").replace("-", "")
    return Path(cfg.cache_dir) / f"history_{tag}.pkl"


def save_history(bundle: HistoryBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_history(path: Path) -> HistoryBundle | None:
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def get_or_build_history(client: Any, cfg: Config, cutoff: str | None = None,
                         rebuild: bool = False) -> HistoryBundle:
    path = cache_path(cfg, cutoff)
    prev = load_history(path)
    if prev is not None and not rebuild and all(prev.ok_components.values()):
        logger.info("Histórico cargado de caché: %s", path)
        return prev
    bundle = build_history(client, cfg, cutoff=cutoff, previous=prev)
    save_history(bundle, path)
    logger.info("Histórico guardado en %s (componentes: %s)", path, bundle.ok_components)
    return bundle


# ----------------------------------------------------------------------------- team.scratch (4.5)
# Los agregados caros se guardan una vez en `team.scratch` con `kind` y `version`, y se
# pueden leer de vuelta para arrancar sin recalcular sobre 12 M de eventos (p. ej. una
# instancia de reserva en otra máquina). Perfiles e intenciones por usuario no se suben
# (son proporcionales al nº de usuarios): esa instancia arranca sin personalización.

def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def publish_aggregates(client: Any, bundle: HistoryBundle, ease_k: int = 20) -> int:
    v = bundle.version or f"h{int(time.time())}"
    rows: list[dict[str, Any]] = [{"kind": "agg_meta", "version": v, "built_at": bundle.meta.get("built_at"),
                                   "components": [k for k, ok in bundle.ok_components.items() if ok]}]
    for part in _chunks(sorted(bundle.pop.items()), 1000):
        rows.append({"kind": "agg_pop", "version": v, "pid": [p for p, _ in part], "val": [x for _, x in part]})
    for part in _chunks(sorted(bundle.item_stats.items()), 1000):
        rows.append({"kind": "agg_items", "version": v, "pid": [p for p, _ in part],
                     "val": [list(x) for _, x in part]})
    for part in _chunks(sorted(bundle.visto_stats.items()), 1000):
        rows.append({"kind": "agg_visto", "version": v, "pid": [p for p, _ in part],
                     "val": [list(x) for _, x in part]})
    for key, d in bundle.ctx_prior.items():
        rows.append({"kind": "agg_ctx", "version": v, "ctx": list(key), "pid": list(d), "val": list(d.values())})
    for src, lst in bundle.transitions.items():
        rows.append({"kind": "agg_trans", "version": v, "src": src, "pid": [d for d, _ in lst],
                     "val": [w for _, w in lst]})
    if bundle.ease is not None:
        for src, lst in ease_top_neighbors(bundle.ease, k=ease_k).items():
            rows.append({"kind": "agg_ease", "version": v, "src": src, "pid": [d for d, _ in lst],
                         "val": [w for _, w in lst]})
    for part in _chunks(rows, 1000):
        client.push_scratch(part)
    return len(rows)


def load_aggregates(client: Any, version: str | None = None, timeout: float = 120.0) -> HistoryBundle | None:
    t = f"{client.team_id}.scratch"
    if version is None:
        r = client.query(f"SELECT version::String AS v FROM {t} WHERE kind::String = 'agg_meta' "
                         f"ORDER BY built_at::Float64 DESC LIMIT 1", timeout=timeout)
        if not r:
            return None
        version = r[0]["v"]
    rows = client.query(f"SELECT kind::String AS kind, src, ctx, pid, val FROM {t} "
                        f"WHERE version::String = '{version}' AND kind::String != 'agg_meta'", timeout=timeout)

    pop: dict[str, float] = {}
    items: dict[str, tuple] = {}
    visto: dict[str, tuple] = {}
    ctx: dict[tuple[str, str, str], dict[str, float]] = {}
    trans: dict[str, list[tuple[str, float]]] = {}
    ease_nb: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        kind, src, cx, pids, vals = (r.get(k) for k in ("kind", "src", "ctx", "pid", "val"))
        if kind == "agg_pop":
            pop.update(zip(pids, map(float, vals)))
        elif kind == "agg_items":
            items.update((p, tuple(map(float, v))) for p, v in zip(pids, vals))
        elif kind == "agg_visto":
            visto.update((p, tuple(map(float, v))) for p, v in zip(pids, vals))
        elif kind == "agg_ctx":
            ctx[tuple(cx)] = dict(zip(pids, map(float, vals)))
        elif kind == "agg_trans":
            trans[src] = list(zip(pids, map(float, vals)))
        elif kind == "agg_ease":
            ease_nb[src] = list(zip(pids, map(float, vals)))
    catalog_rows = client.query(SQL_CATALOG, timeout=timeout)
    return HistoryBundle(catalog_rows=catalog_rows, pop=pop, item_stats=items, ctx_prior=ctx, transitions=trans,
                         ease=ease_from_neighbors(ease_nb) if ease_nb else None, users={}, visto_stats=visto,
                         intent={}, version=version, meta={"source": "scratch", "built_at": time.time()},
                         ok_components={"scratch": True})
