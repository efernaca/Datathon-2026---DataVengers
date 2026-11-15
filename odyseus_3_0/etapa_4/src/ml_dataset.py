"""Odyseus-ML · paso 1: preguntas abiertas + extracción de history_sessions + variables «a fecha de».

Uso (desde la carpeta odyseus/):

    python ml_build_dataset.py --questions-only     # solo cuenta filas/compras (1 minuto)
    python ml_build_dataset.py                      # dataset completo -> ml_data/dataset.npz
    python ml_build_dataset.py --local-test         # prueba de humo sin RawTree (usa cache/ y el parquet)

Cómo se evita la fuga de etiqueta
---------------------------------
El histórico se parte en bloques de `--block-days` días contados hacia atrás desde CUT_TEST
(= último día − 7, el mismo corte que `run_offline_eval_official.py`). Para cada bloque se
construye un snapshot con `build_snapshot_v4_from_records` usando SOLO datos con ts < inicio del
bloque (popularidad, transiciones, perfiles, prior de contexto, carritos…). Las filas del bloque
se puntúan con ese snapshot, así ninguna variable ha visto la compra que intenta predecir.

    bloque 0  (ts >= CUT_TEST)            -> split 2 = test
    bloque 1  (los 7 días anteriores)     -> split 1 = validación / parada temprana
    bloques 2+                            -> split 0 = entrenamiento

Muestreo: se bajan TODAS las peticiones con clic o compra y 1 de cada `--bg-mod` del resto
(por hash del request_id). `w_req` guarda el peso que deshace ese muestreo.
Todo lo descargado se guarda en ml_data/raw y ml_data/asof: relanzar el script no repite consultas.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConfigV4
from .disk_cache import DiskCache
from .ml_features import (
    ML_FEATURE_NAMES,
    NUM_BASE,
    build_ml_features,
    parse_ts_ms,
    prior_scores,
)
from .rawtree_client import RawTreeClient
from .recommender import (
    CalypsoRecommenderV4,
    _extract_feature_matrix,
    _parse_request_fields,
    build_snapshot_v4_from_records,
)

logger = logging.getLogger("odyseus.ml_dataset")
MODEL_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "calypso-5.1"  # las variables y el prior dependen del recomendador base: no mezclar datasets
SQL_TIMEOUT = 300.0

# ── SQL «a fecha de»: mismas consultas que serve.py con un filtro ts < {cut} ──────────────
_EV = "ts::DateTime64(3) < toDateTime64('{cut}', 3)"

SQL_CATALOG_BASE = """
SELECT product_id::String AS product_id, category::String AS category,
       subcategory::String AS subcategory, brand::String AS brand,
       price_eur::Float64 AS price_eur, description::String AS description,
       is_active::Bool AS is_active
FROM shop.catalog
"""

SQL_POPULARITY = f"""
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'view') AS views,
       countIf(event_type::String = 'add_to_cart') AS carts,
       countIf(event_type::String = 'purchase') AS purchases,
       (purchases + 0.5) / (views + 25.0) AS self_cvr,
       (purchases * 5.0 + carts * 2.0 + views * 0.2) AS weighted_pop
FROM shop.history_events
WHERE {_EV}
GROUP BY product_id
ORDER BY weighted_pop DESC
"""

SQL_TRANSITIONS_DIRECTIONAL = f"""
SELECT a.product_id AS src_product,
       b.product_id AS dst_product,
       (countIf(b.event_type = 'purchase') * 6.0 +
        countIf(b.event_type = 'add_to_cart') * 2.5 +
        countIf(b.event_type = 'view') * 1.0) AS raw_transition_score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id, ts::DateTime64(3) AS t
    FROM shop.history_events
    WHERE event_type::String = 'view' AND {_EV}
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id,
           event_type::String AS event_type, ts::DateTime64(3) AS t
    FROM shop.history_events
    WHERE {_EV}
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id AND b.t >= a.t
GROUP BY src_product, dst_product
ORDER BY raw_transition_score DESC
LIMIT 60 BY src_product
"""

SQL_TRANSITIONS_SYMMETRIC = f"""
SELECT a.product_id AS src_product,
       b.product_id AS dst_product,
       (countIf(b.event_type = 'purchase') * 6.0 +
        countIf(b.event_type = 'add_to_cart') * 2.5 +
        countIf(b.event_type = 'view') * 1.0) AS raw_transition_score
FROM (
    SELECT session_id::String AS session_id, product_id::String AS product_id
    FROM shop.history_events
    WHERE event_type::String = 'view' AND {_EV}
) AS a
INNER JOIN (
    SELECT session_id::String AS session_id, product_id::String AS product_id, event_type::String AS event_type
    FROM shop.history_events
    WHERE {_EV}
) AS b ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY src_product, dst_product
ORDER BY raw_transition_score DESC
LIMIT 40 BY src_product
"""

SQL_USER_PROFILE = f"""
SELECT user_id::String AS user_id,
       groupUniqArrayIf(product_id::String, event_type::String = 'purchase') AS bought_products,
       avgIf(log1p(greatest(price_eur::Float64, 0.0)), event_type::String = 'purchase') AS log_price_mean,
       stddevPopIf(log1p(greatest(price_eur::Float64, 0.0)), event_type::String = 'purchase') AS log_price_std
FROM shop.history_events
WHERE {_EV}
GROUP BY user_id
"""

SQL_USER_INTENT = f"""
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       (countIf(event_type::String = 'add_to_cart') * 3.0 +
        countIf(event_type::String = 'view') * 0.5) AS intent_score
FROM shop.history_events
WHERE {_EV}
GROUP BY user_id, product_id
HAVING countIf(event_type::String = 'purchase') = 0
   AND (countIf(event_type::String = 'add_to_cart') >= 1 OR countIf(event_type::String = 'view') >= 2)
ORDER BY intent_score DESC
LIMIT 10 BY user_id
"""

SQL_USER_AFFINITY = """
SELECT e.user_id::String AS user_id,
       c.subcategory::String AS subcategory,
       c.brand::String AS brand,
       (countIf(e.event_type::String = 'purchase') * 5.0 +
        countIf(e.event_type::String = 'add_to_cart') * 2.0 +
        countIf(e.event_type::String = 'view') * 0.3) AS affinity
FROM shop.history_events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
WHERE e.ts::DateTime64(3) < toDateTime64('{cut}', 3)
GROUP BY user_id, subcategory, brand
ORDER BY affinity DESC
LIMIT 8 BY user_id
"""

SQL_CONTEXT_POP = f"""
SELECT context.region::String AS region,
       context.device::String AS device,
       context.page::String AS page,
       purchased::String AS product_id,
       count() AS buys
FROM shop.history_sessions
WHERE purchased IS NOT NULL AND purchased::String != '' AND {_EV}
GROUP BY region, device, page, product_id
ORDER BY buys DESC
LIMIT 35 BY region, device, page
"""

# En vivo la tendencia sale de shop.events (lo ocurrido desde que abrió el stream). Aquí se imita
# con las últimas 24 h de eventos orgánicos anteriores al corte.
SQL_LIVE_TRENDS = """
SELECT product_id::String AS product_id,
       countIf(event_type::String = 'purchase') AS live_purchases,
       countIf(event_type::String = 'add_to_cart') AS live_carts,
       (countIf(event_type::String = 'purchase') * 5.0 +
        countIf(event_type::String = 'add_to_cart') * 2.0 +
        countIf(event_type::String = 'view') * 0.2) AS live_score
FROM shop.history_events
WHERE ts::DateTime64(3) < toDateTime64('{cut}', 3)
  AND ts::DateTime64(3) >= toDateTime64('{cut_prev}', 3)
GROUP BY product_id
ORDER BY live_score DESC
LIMIT 400
"""

SQL_CART_HIST = f"""
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       countIf(event_type::String = 'purchase') AS n_purchases,
       toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'add_to_cart')) AS last_cart_ms,
       toUnixTimestamp64Milli(maxIf(ts::DateTime64(3), event_type::String = 'purchase')) AS last_purchase_ms
FROM shop.history_events
WHERE event_type::String IN ('add_to_cart', 'purchase') AND {_EV}
GROUP BY user_id, product_id
HAVING last_cart_ms > last_purchase_ms
ORDER BY last_cart_ms DESC
LIMIT 6 BY user_id
"""

SQL_RECURRENT_HIST = f"""
SELECT user_id::String AS user_id,
       product_id::String AS product_id,
       count() AS n_purchases,
       toUnixTimestamp64Milli(max(ts::DateTime64(3))) AS last_purchase_ms
FROM shop.history_events
WHERE event_type::String = 'purchase' AND {_EV}
GROUP BY user_id, product_id
HAVING n_purchases >= 2
ORDER BY n_purchases DESC
LIMIT 10 BY user_id
"""

_HAS_BUY = "(purchased IS NOT NULL AND purchased::String != '')"
_HAS_CLICK = "(clicked IS NOT NULL AND clicked::String != '')"

SQL_Q_SESSIONS = f"""
SELECT count() AS n,
       countIf({_HAS_BUY}) AS buys,
       countIf({_HAS_CLICK}) AS clicks,
       toString(min(ts::DateTime64(3))) AS t0,
       toString(max(ts::DateTime64(3))) AS t1
FROM shop.history_sessions
"""

SQL_Q_SESSIONS_BY_DAY = f"""
SELECT toString(toDate(ts::DateTime64(3))) AS day,
       count() AS n,
       countIf({_HAS_BUY}) AS buys,
       countIf({_HAS_CLICK}) AS clicks
FROM shop.history_sessions
GROUP BY day
ORDER BY day
"""

SQL_Q_FEEDBACK = """
SELECT count() AS n,
       countIf(served_by::String = 'team') AS served_team,
       countIf(purchased IS NOT NULL AND purchased::String != '') AS buys,
       countIf(clicked IS NOT NULL AND clicked::String != '') AS clicks,
       toString(min(ts::DateTime64(3))) AS t0,
       toString(max(ts::DateTime64(3))) AS t1
FROM {team}.feedback
WHERE request_id::String != '__init__'
"""

SQL_SESSIONS_RANGE = """
SELECT request_id::String AS request_id,
       toString(ts::DateTime64(3)) AS ts,
       user_id::String AS user_id,
       context.region::String AS region,
       context.device::String AS device,
       context.page::String AS page,
       recent_views, shown, propensity, clicked, purchased
FROM shop.history_sessions
WHERE ts::DateTime64(3) >= toDateTime64('{lo}', 3) AND ts::DateTime64(3) < toDateTime64('{hi}', 3)
  AND ({has_buy} OR {has_click} OR {sample})
"""


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _to_dt(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _clean_id(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    s = str(v).strip()
    return "" if s in ("None", "nan", "null") else s


# ── 1 · preguntas abiertas ───────────────────────────────────────────────────────────────
def answer_questions(client: RawTreeClient, team_id: str, out_dir: Path) -> dict[str, Any]:
    tot = client.query(SQL_Q_SESSIONS, timeout=SQL_TIMEOUT)[0]
    by_day = client.query(SQL_Q_SESSIONS_BY_DAY, timeout=SQL_TIMEOUT)
    fb: dict[str, Any] = {}
    if team_id:
        rows = client.query_safe(SQL_Q_FEEDBACK.format(team=team_id), timeout=60.0)
        fb = rows[0] if rows else {}
    res = {"history_sessions": tot, "history_sessions_by_day": by_day, "feedback": fb}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preguntas.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    n, buys, clicks = int(tot["n"]), int(tot["buys"]), int(tot["clicks"])
    print("\n=== PREGUNTAS ABIERTAS ===")
    print(f"history_sessions: {n:,} peticiones | {buys:,} compras ({100.0 * buys / max(n, 1):.2f} %) "
          f"| {clicks:,} clics | {tot['t0']} -> {tot['t1']}")
    if fb:
        fn, fbuy = int(fb.get("n") or 0), int(fb.get("buys") or 0)
        print(f"{team_id}.feedback: {fn:,} filas | {fbuy:,} compras ({100.0 * fbuy / max(fn, 1):.2f} %) "
              f"| servidas por el equipo: {int(fb.get('served_team') or 0):,} | {fb.get('t0')} -> {fb.get('t1')}")
    else:
        print("feedback: sin datos (o TEAM_ID vacío)")
    print("official_test_data.parquet = 25.000 peticiones al azar con ts >= CUT_TEST (último día − 7): "
          "es el bloque de test de este dataset; nunca se usa para entrenar.")
    print("(detalle por día en ml_data/preguntas.json)\n")
    return res


# ── 2 · snapshots «a fecha de» ───────────────────────────────────────────────────────────
def fetch_asof_bundle(client: RawTreeClient, cfg: ConfigV4, cut: datetime, cache_dir: Path) -> dict[str, list]:
    path = cache_dir / f"bundle_{cut.strftime('%Y%m%d')}.pkl"
    if path.is_file():
        with path.open("rb") as fh:
            return pickle.load(fh)
    c = _fmt(cut)
    cprev = _fmt(cut - timedelta(days=1))
    trans_sql = SQL_TRANSITIONS_DIRECTIONAL if cfg.v3_directional_transitions else SQL_TRANSITIONS_SYMMETRIC
    plan = [
        ("popularity_rows", SQL_POPULARITY.format(cut=c)),
        ("transition_rows", trans_sql.format(cut=c)),
        ("user_profile_rows", SQL_USER_PROFILE.format(cut=c)),
        ("user_intent_rows", SQL_USER_INTENT.format(cut=c)),
        ("user_affinity_rows", SQL_USER_AFFINITY.format(cut=c)),
        ("context_rows", SQL_CONTEXT_POP.format(cut=c)),
        ("live_trend_rows", SQL_LIVE_TRENDS.format(cut=c, cut_prev=cprev)),
        ("cart_hist_rows", SQL_CART_HIST.format(cut=c)),
        ("recurrent_hist_rows", SQL_RECURRENT_HIST.format(cut=c)),
    ]
    bundle: dict[str, list] = {}
    for name, sql in plan:
        t0 = time.perf_counter()
        bundle[name] = client.query(sql, timeout=SQL_TIMEOUT)
        logger.info("  [asof %s] %-20s %8d filas en %.1fs", cut.date(), name, len(bundle[name]), time.perf_counter() - t0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return bundle


def snapshot_from_bundle(catalog_rows: list, bundle: dict[str, list], cfg: ConfigV4, wp: Any):
    snap, _ = build_snapshot_v4_from_records(
        catalog_rows=catalog_rows,
        popularity_rows=bundle.get("popularity_rows", ()),
        context_rows=bundle.get("context_rows", ()),
        transition_rows=bundle.get("transition_rows", ()),
        user_profile_rows=bundle.get("user_profile_rows", ()),
        user_intent_rows=bundle.get("user_intent_rows", ()),
        user_affinity_rows=bundle.get("user_affinity_rows", ()),
        live_trend_rows=bundle.get("live_trend_rows", ()),
        cart_hist_rows=bundle.get("cart_hist_rows", ()),
        recurrent_hist_rows=bundle.get("recurrent_hist_rows", ()),
        config=cfg,
        version=1,
        weights_profile=wp,
    )
    snap.config_override = cfg
    return snap


# ── 3 · descarga de sesiones ─────────────────────────────────────────────────────────────
def fetch_sessions_day(client: RawTreeClient, lo: datetime, hi: datetime, bg_mod: int, raw_dir: Path) -> list[dict]:
    path = raw_dir / f"sessions_{lo.strftime('%Y%m%d')}_mod{bg_mod}.json.gz"
    if path.is_file():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    samples = [f"cityHash64(request_id::String) % {bg_mod} = 0", f"rand() % {bg_mod} = 0"]
    if bg_mod <= 1:
        samples = ["1 = 1"]
    rows: list[dict] | None = None
    last_exc: Exception | None = None
    for sample in samples:
        sql = SQL_SESSIONS_RANGE.format(lo=_fmt(lo), hi=_fmt(hi), has_buy=_HAS_BUY, has_click=_HAS_CLICK, sample=sample)
        for attempt in range(3):
            try:
                rows = client.query(sql, timeout=SQL_TIMEOUT)
                break
            except Exception as exc:  # red o SQL
                last_exc = exc
                logger.warning("  sesiones %s intento %d: %s", lo.date(), attempt + 1, str(exc)[:200])
                time.sleep(2.0 * (attempt + 1))
        if rows is not None:
            break
    if rows is None:
        raise RuntimeError(f"No se pudieron descargar las sesiones de {lo.date()}: {last_exc}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(rows, fh, default=str)
    tmp.replace(path)
    return rows


# ── 4 · filas de entrenamiento ───────────────────────────────────────────────────────────
class _Acc:
    def __init__(self) -> None:
        self.X: list[np.ndarray] = []
        self.prior: list[np.ndarray] = []
        self.prop: list[np.ndarray] = []
        self.y_buy: list[np.ndarray] = []
        self.y_click: list[np.ndarray] = []
        self.g_n = 0
        self.g_w: list[float] = []
        self.g_split: list[int] = []
        self.g_block: list[int] = []
        self.g_ts: list[float] = []
        self.g_page: list[int] = []
        self.g_has_views: list[int] = []
        self.g_has_profile: list[int] = []
        self.g_request_id: list[str] = []
        self.skipped = 0


_PAGES = {"home": 0, "product": 1, "category": 2}


def featurize_rows(rows: list[dict], snap: Any, cfg: ConfigV4, wp: Any, *, bg_mod: int, split: int, block: int, acc: _Acc) -> None:
    c = NUM_BASE
    slots = list(range(10))
    for row in rows:
        shown_raw = row.get("shown")
        prop_raw = row.get("propensity")
        try:
            shown = [str(x) for x in shown_raw][:10]
            prop = np.asarray([float(x) for x in prop_raw][:10], dtype=np.float32)
        except (TypeError, ValueError):
            acc.skipped += 1
            continue
        if len(shown) != 10 or len(prop) != 10:
            acc.skipped += 1
            continue
        clicked = _clean_id(row.get("clicked"))
        purchased = _clean_id(row.get("purchased"))
        ts_ms = parse_ts_ms(row.get("ts"))
        req = {
            "user_id": _clean_id(row.get("user_id")),
            "is_new_user": False,  # en el histórico no hay flag: decide la existencia de perfil a fecha de
            "region": _clean_id(row.get("region")),
            "device": _clean_id(row.get("device")),
            "page": _clean_id(row.get("page")),
            "recent_views": row.get("recent_views"),
        }
        if not req["page"] and isinstance(row.get("context"), dict):  # formato del parquet oficial
            ctx = row["context"]
            req["region"], req["device"], req["page"] = (
                _clean_id(ctx.get("region")), _clean_id(ctx.get("device")), _clean_id(ctx.get("page")),
            )
        parsed = _parse_request_fields(req)
        base = _extract_feature_matrix(snap, req, shown, parsed, weights_profile=wp)
        X = build_ml_features(
            snap, req, shown, slots=slots, request_ts_ms=ts_ms, parsed=parsed,
            weights_profile=wp, base_matrix=base,
        )
        acc.X.append(X)
        acc.prior.append(prior_scores(snap, parsed, shown, base, weights_profile=wp, config=cfg, now_ms=ts_ms).astype(np.float32))
        acc.prop.append(prop)
        acc.y_buy.append(np.asarray([1 if (purchased and p == purchased) else 0 for p in shown], dtype=np.int8))
        acc.y_click.append(np.asarray([1 if (clicked and p == clicked) else 0 for p in shown], dtype=np.int8))
        acc.g_w.append(1.0 if (clicked or purchased or bg_mod <= 1) else float(bg_mod))
        acc.g_split.append(split)
        acc.g_block.append(block)
        acc.g_ts.append(ts_ms or 0.0)
        acc.g_page.append(_PAGES.get(req["page"], 3))
        acc.g_has_views.append(int(X[0, c + 7] > 0.5))
        acc.g_has_profile.append(int(X[0, c + 6] > 0.5))
        acc.g_request_id.append(_clean_id(row.get("request_id")))
        acc.g_n += 1


def save_dataset(acc: _Acc, out_dir: Path, meta: dict[str, Any], name: str = "dataset") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if acc.g_n == 0:
        raise RuntimeError("Dataset vacío: ninguna petición válida.")
    X = np.concatenate(acc.X).astype(np.float32)
    y_buy = np.concatenate(acc.y_buy)
    y_click = np.concatenate(acc.y_click)
    g_split = np.asarray(acc.g_split, dtype=np.int8)
    g_w = np.asarray(acc.g_w, dtype=np.float32)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(
        path,
        X=X,                                                     # (n_peticiones*10, n_variables)
        prior=np.concatenate(acc.prior),                         # score_p de Calypso (offset)
        propensity=np.concatenate(acc.prop),                     # del log antiguo, por slot
        y_buy=y_buy,
        y_click=y_click,
        group=np.repeat(np.arange(acc.g_n, dtype=np.int32), 10),  # índice de petición por fila
        g_weight=g_w,                                            # peso que deshace el muestreo de fondo
        g_split=g_split,                                         # 0 train · 1 val · 2 test
        g_block=np.asarray(acc.g_block, dtype=np.int8),
        g_ts_ms=np.asarray(acc.g_ts, dtype=np.float64),
        g_page=np.asarray(acc.g_page, dtype=np.int8),            # 0 home · 1 product · 2 category
        g_has_views=np.asarray(acc.g_has_views, dtype=np.int8),
        g_has_profile=np.asarray(acc.g_has_profile, dtype=np.int8),
        g_request_id=np.asarray(acc.g_request_id),
        feature_names=np.asarray(ML_FEATURE_NAMES),
        base_model=np.asarray(BASE_MODEL),                       # recomendador con el que se calcularon las variables
    )
    yb = y_buy.reshape(-1, 10).max(axis=1)
    yc = y_click.reshape(-1, 10).max(axis=1)
    splits = {}
    for s, label in ((0, "train"), (1, "val"), (2, "test")):
        m = g_split == s
        splits[label] = {
            "peticiones": int(m.sum()),
            "peticiones_equivalentes": float(g_w[m].sum()),
            "compras": int(yb[m].sum()),
            "clics": int(yc[m].sum()),
        }
    feat_stats = {
        n: {"mean": float(X[:, i].mean()), "std": float(X[:, i].std()), "nonzero": float((X[:, i] != 0).mean())}
        for i, n in enumerate(ML_FEATURE_NAMES)
    }
    meta = dict(meta)
    meta.update({
        "n_features": int(X.shape[1]),
        "feature_names": list(ML_FEATURE_NAMES),
        "n_rows": int(X.shape[0]),
        "n_requests": int(acc.g_n),
        "skipped_requests": int(acc.skipped),
        "splits": splits,
        "feature_stats": feat_stats,
        "dataset_file": path.name,
    })
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\n=== DATASET {path} ===")
    print(f"{X.shape[0]:,} filas x {X.shape[1]} variables | {acc.g_n:,} peticiones | descartadas: {acc.skipped}")
    for label, d in splits.items():
        print(f"  {label:5s}: {d['peticiones']:>8,} peticiones ({d['peticiones_equivalentes']:>11,.0f} equiv.) "
              f"| {d['compras']:>6,} compras | {d['clics']:>6,} clics")
    n_pos = splits["train"]["compras"]
    print(f"\nCompras de entrenamiento: {n_pos:,} -> techo de parámetros (15 compras/parámetro): {n_pos // 15:,}")
    dead = [n for n, st in feat_stats.items() if st["nonzero"] == 0.0]
    if dead:
        print(f"Variables siempre a cero en este dataset: {', '.join(dead)}")
    return path


# ── orquestación ─────────────────────────────────────────────────────────────────────────
def build_blocks(t0_ms: float, t1_ms: float, block_days: int, min_history_days: int, max_blocks: int) -> tuple[datetime, list[tuple[datetime, datetime]]]:
    t0, t1 = _to_dt(t0_ms), _to_dt(t1_ms)
    last_day = t1.replace(hour=0, minute=0, second=0, microsecond=0)
    cut_test = last_day - timedelta(days=7)
    end = last_day + timedelta(days=1)
    blocks = [(cut_test, end)]
    lo = cut_test
    earliest = t0 + timedelta(days=min_history_days)
    while len(blocks) < max_blocks:
        new_lo = lo - timedelta(days=block_days)
        if new_lo < earliest:
            break
        blocks.append((new_lo, lo))
        lo = new_lo
    return cut_test, blocks


def run_build(args: argparse.Namespace) -> None:
    cfg = ConfigV4.from_env()
    wp = cfg.load_weights_profile()
    out_dir = (MODEL_ROOT / args.out).resolve()
    if not cfg.api_key:
        raise SystemExit("Falta RAWTREE_API_KEY (group_API.env en la raíz del repo).")
    client = RawTreeClient(api_url=cfg.api_url, api_key=cfg.api_key, team_id=cfg.team_id)

    try:
        res = answer_questions(client, cfg.team_id, out_dir)
    except Exception as exc:  # sin red: se reutiliza lo ya descargado
        q_path = out_dir / "preguntas.json"
        if args.questions_only or not q_path.is_file():
            raise
        logger.warning("RawTree no responde (%s): se usa %s y lo ya descargado en ml_data/", str(exc)[:120], q_path.name)
        res = json.loads(q_path.read_text(encoding="utf-8"))
    if args.questions_only:
        return
    tot = res["history_sessions"]
    t0_ms, t1_ms = parse_ts_ms(tot["t0"]), parse_ts_ms(tot["t1"])
    if t0_ms is None or t1_ms is None:
        raise SystemExit(f"No se pudo leer el rango de fechas: {tot}")
    cut_test, blocks = build_blocks(t0_ms, t1_ms, args.block_days, args.min_history_days, args.max_blocks)
    print(f"CUT_TEST = {cut_test.date()} | {len(blocks)} bloques:")
    for b, (lo, hi) in enumerate(blocks):
        print(f"  bloque {b}: [{lo.date()}, {hi.date()})  -> {'test' if b == 0 else 'val' if b == 1 else 'train'}")
    if len(blocks) < 3:
        raise SystemExit("Histórico demasiado corto para train/val/test: baja --min-history-days o --block-days.")

    cat_path = out_dir / "asof" / "catalog.pkl"
    try:
        catalog_rows = client.query(SQL_CATALOG_BASE, timeout=SQL_TIMEOUT)
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        with cat_path.open("wb") as fh:
            pickle.dump(catalog_rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        if cat_path.is_file():
            with cat_path.open("rb") as fh:
                catalog_rows = pickle.load(fh)
        else:
            hist = DiskCache(MODEL_ROOT / "cache").load_static_history()
            if hist is None or not hist.catalog_rows:
                raise
            catalog_rows = hist.catalog_rows
        logger.warning("Catálogo desde copia local (%s)", str(exc)[:120])
    acc = _Acc()
    t_start = time.perf_counter()
    for b, (lo, hi) in enumerate(blocks):
        split = 2 if b == 0 else (1 if b == 1 else 0)
        logger.info("Bloque %d [%s, %s): snapshot a fecha de %s", b, lo.date(), hi.date(), lo.date())
        bundle = fetch_asof_bundle(client, cfg, lo, out_dir / "asof")
        t0 = time.perf_counter()
        snap = snapshot_from_bundle(catalog_rows, bundle, cfg, wp)
        logger.info("  snapshot listo en %.1fs (%d perfiles, %d productos con transiciones)",
                    time.perf_counter() - t0, len(snap.user_profiles), len(snap.transitions))
        day = lo
        while day < hi:
            nxt = min(day + timedelta(days=1), hi)
            rows = fetch_sessions_day(client, day, nxt, args.bg_mod, out_dir / "raw")
            before = acc.g_n
            featurize_rows(rows, snap, cfg, wp, bg_mod=args.bg_mod, split=split, block=b, acc=acc)
            logger.info("  %s: %6d peticiones descargadas, %6d válidas (total %d, %.0fs)",
                        day.date(), len(rows), acc.g_n - before, acc.g_n, time.perf_counter() - t_start)
            day = nxt
        del snap, bundle
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights_profile": getattr(wp, "profile_name", ""),
        "cut_test": _fmt(cut_test),
        "blocks": [{"block": b, "lo": _fmt(lo), "hi": _fmt(hi)} for b, (lo, hi) in enumerate(blocks)],
        "bg_mod": args.bg_mod,
        "block_days": args.block_days,
        "min_history_days": args.min_history_days,
        "asof": True,
        "base_model": BASE_MODEL,
    }
    save_dataset(acc, out_dir, meta)


def run_local_test(args: argparse.Namespace) -> None:
    """Prueba de humo sin RawTree: snapshot del cache (NO es «a fecha de») + parquet oficial."""
    import pandas as pd

    cfg = ConfigV4.from_env()
    wp = cfg.load_weights_profile()
    hist = DiskCache(MODEL_ROOT / "cache").load_static_history()
    if hist is None:
        raise SystemExit("No hay cache/static_history_v4.pkl")
    bundle = {
        "popularity_rows": hist.popularity_rows,
        "context_rows": hist.context_rows,
        "transition_rows": hist.transition_rows,
        "user_profile_rows": hist.user_profile_rows,
        "user_intent_rows": hist.user_intent_rows,
        "user_affinity_rows": hist.user_affinity_rows,
        "cart_hist_rows": getattr(hist, "cart_hist_rows", None) or [],
        "recurrent_hist_rows": getattr(hist, "recurrent_hist_rows", None) or [],
    }
    t0 = time.perf_counter()
    snap = snapshot_from_bundle(hist.catalog_rows, bundle, cfg, wp)
    print(f"snapshot (cache, CON fuga: solo para probar el código) en {time.perf_counter() - t0:.1f}s")
    df = pd.read_parquet(MODEL_ROOT.parent / "offline_eval_cache" / "official_test_data.parquet")
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    acc = _Acc()
    t0 = time.perf_counter()
    featurize_rows(rows, snap, cfg, wp, bg_mod=1, split=2, block=0, acc=acc)
    dt = time.perf_counter() - t0
    print(f"{acc.g_n} peticiones en {dt:.1f}s ({1000.0 * dt / max(acc.g_n, 1):.2f} ms/petición)")

    # Comprobación de la réplica del prior: el top-1 de recommend() debe ser casi siempre
    # el candidato con mayor prior (difiere por carrito, UCB y exclusión de comprados).
    rec = CalypsoRecommenderV4(cfg)
    agree = total = 0
    for row in rows[:400]:
        ctx = row.get("context") or {}
        req = {"user_id": _clean_id(row.get("user_id")), "is_new_user": False,
               "region": _clean_id(ctx.get("region")), "device": _clean_id(ctx.get("device")),
               "page": _clean_id(ctx.get("page")), "recent_views": row.get("recent_views")}
        parsed = _parse_request_fields(req)
        top = rec._recommend_core(parsed, snap, 10, frozenset(), _cfg=cfg)
        cands = rec._gather_candidates(parsed, snap, frozenset())
        if not cands or not top:
            continue
        base = _extract_feature_matrix(snap, req, cands, parsed, weights_profile=wp)
        pr = prior_scores(snap, parsed, cands, base, weights_profile=wp, config=cfg)
        if parsed[5] and wp.slot1_category_match_bonus > 0.0:  # bonus de slot 1 de 5.1 (capa de política, fuera del prior)
            a_info = snap.products.get(parsed[5][-1])
            for j, pid in enumerate(cands):
                info = snap.products.get(pid)
                if pid in parsed[5] or (a_info is not None and info is not None and info.category == a_info.category):
                    pr[j] += wp.slot1_category_match_bonus
        total += 1
        agree += int(cands[int(np.argmax(pr))] == top[0])
    print(f"réplica del prior: top-1 coincide con _recommend_core en {agree}/{total} peticiones")
    save_dataset(acc, (MODEL_ROOT / args.out / "smoke").resolve(), {"asof": False, "note": "prueba de humo con fuga"}, name="smoke")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Odyseus-ML · dataset de entrenamiento a fecha de")
    ap.add_argument("--questions-only", action="store_true", help="Solo responde las preguntas abiertas y sale")
    ap.add_argument("--local-test", action="store_true", help="Prueba de humo sin RawTree")
    ap.add_argument("--limit", type=int, default=0, help="(local-test) nº de peticiones")
    ap.add_argument("--bg-mod", type=int, default=10, help="Se conserva 1 de cada N peticiones sin clic ni compra")
    ap.add_argument("--block-days", type=int, default=7, help="Días por bloque de snapshot")
    ap.add_argument("--min-history-days", type=int, default=14, help="Histórico mínimo antes del primer bloque")
    ap.add_argument("--max-blocks", type=int, default=6, help="Nº máximo de bloques (incluye test y val)")
    ap.add_argument("--out", default="ml_data", help="Carpeta de salida dentro de odyseus/")
    args = ap.parse_args()
    if args.local_test:
        run_local_test(args)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
