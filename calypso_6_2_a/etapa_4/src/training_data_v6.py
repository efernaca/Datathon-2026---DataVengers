"""Multi-source zero-leakage historical training dataset & pre-CUT covisit loader for Calypso 6.2."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from pathlib import Path
from typing import Any

import numpy as np

from .reranker import NUM_FEATURES, PositionDebiasedRanker, build_ips_training_dataset
from .state import RecommenderSnapshotV4
from .weights_profile import WeightsProfile


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        res = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(res) else res


def resolve_offline_covisit_path(start_path: Path | None = None) -> Path | None:
    """Locate pre-CUT offline_eval_cache/covisit.parquet without touching any test files."""
    base = Path(start_path) if start_path is not None else Path(__file__).resolve()
    for parent in (base, *base.parents):
        cand = parent / "offline_eval_cache" / "covisit.parquet"
        if cand.is_file():
            return cand
    return None


def load_covisit_edges(
    parquet_path: Path | None = None,
    *,
    active_ids: frozenset[str] | None = None,
    max_edges: int = 120_000,
    top_per_item: int = 24,
) -> tuple[list[tuple[str, str, float]], dict[str, tuple[tuple[str, float], ...]]]:
    """Load pre-CUT co-visitation pairs (ts < CUT) cleanly into edge list & normalized lookup."""
    path = parquet_path or resolve_offline_covisit_path()
    if path is None or not path.is_file():
        return [], {}

    try:
        import pandas as pd

        df = pd.read_parquet(path)
    except Exception:
        return [], {}

    if df.empty or not {"p", "q", "n"}.issubset(df.columns):
        return [], {}

    edges: list[tuple[str, str, float]] = []
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)

    p_col = df["p"].astype(str).tolist()
    q_col = df["q"].astype(str).tolist()
    n_col = df["n"].to_numpy(dtype=np.float64, copy=False)

    limit = min(len(p_col), max_edges)
    for idx in range(limit):
        p = p_col[idx].strip()
        q = q_col[idx].strip()
        val = float(n_col[idx])
        if not p or not q or p == q or val <= 0.0:
            continue
        if active_ids is not None and (p not in active_ids or q not in active_ids):
            continue
        edges.append((p, q, val))
        grouped[p].append((q, math.log1p(val)))

    cov_map: dict[str, tuple[tuple[str, float], ...]] = {}
    for src, nbs in grouped.items():
        nbs.sort(key=lambda kv: kv[1], reverse=True)
        top_nbs = nbs[:top_per_item]
        max_v = top_nbs[0][1] if top_nbs else 0.0
        if max_v > 0.0:
            cov_map[src] = tuple((dst, float(sc / max_v)) for dst, sc in top_nbs)

    return edges, cov_map


def build_multisource_v6_training_dataset(
    snapshot: RecommenderSnapshotV4,
    feature_extractor: Any,
    *,
    combined_logs: Iterable[Mapping[str, Any]] = (),
    cart_hist_rows: Iterable[Mapping[str, Any]] = (),
    recurrent_hist_rows: Iterable[Mapping[str, Any]] = (),
    user_intent_rows: Iterable[Mapping[str, Any]] = (),
    transition_rows: Iterable[Mapping[str, Any]] = (),
    weights_profile: WeightsProfile | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble multi-source zero-leakage training matrix (X, y, sample_weights) for Calypso 6.2."""
    wp = weights_profile if weights_profile is not None else snapshot.weights_profile
    rng = np.random.default_rng(seed)

    parts_X: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_w: list[np.ndarray] = []

    gold_w = wp.train_gold_weight if wp is not None else 1.0
    cart_w = wp.train_cart_weight if wp is not None else 0.55
    intent_w = wp.train_intent_weight if wp is not None else 0.40
    seq_w = wp.train_seq_weight if wp is not None else 0.30

    # ── Source 1: Gold IPS Impression Logs ────────────────────────────────────────
    logs_list = list(combined_logs)
    if logs_list and gold_w > 0.0:
        if len(logs_list) > 12000:
            # Keep all positive-outcome sessions while filling remaining budget without ordering bias
            pos_logs = [
                r for r in logs_list
                if str(r.get("purchased") or "").strip() or str(r.get("clicked") or "").strip()
            ]
            neg_logs = [
                r for r in logs_list
                if not (str(r.get("purchased") or "").strip() or str(r.get("clicked") or "").strip())
            ]
            take_neg = max(0, 12000 - len(pos_logs))
            logs_ordered = pos_logs[:12000] + neg_logs[:take_neg]
            rng_ord = np.random.default_rng(seed)
            perm = rng_ord.permutation(len(logs_ordered))
            logs_ordered = [logs_ordered[int(i)] for i in perm]
        else:
            logs_ordered = logs_list
        Xg, yg, wg = build_ips_training_dataset(
            logs_ordered,
            lambda req, cands: feature_extractor(snapshot, req, cands, weights_profile=wp),
            exam_power=wp.bandit_exam_power if wp is not None else 0.70,
            max_sessions=12000,
            purchase_weight=wp.ips_purchase_weight if wp is not None else 2.5,
            click_weight=wp.ips_click_weight if wp is not None else 0.12,
            use_empirical_exam_curve=wp.use_empirical_exam_curve if wp is not None else True,
        )
        if Xg.shape[0] > 0:
            parts_X.append(Xg)
            parts_y.append(yg)
            parts_w.append(wg * gold_w)

    # Determine Gold IPS positive count first so auxiliary sources boost sparse/cold regimes
    # without diluting calibrated real IPS serving distributions.
    n_gold_pos = 0
    gold_w_sum = 0.0
    if parts_y:
        n_gold_pos = int(np.sum(parts_y[0] > 0))
        gold_w_sum = float(np.sum(parts_w[0]))

    if n_gold_pos >= 60 and parts_X:
        return np.vstack(parts_X), np.concatenate(parts_y), np.concatenate(parts_w)

    active_arr = list(snapshot.fallback_ranked) if snapshot.fallback_ranked else sorted(snapshot.active_ids)
    n_active = len(active_arr)
    if n_active < 10:
        if not parts_X:
            return (
                np.empty((0, NUM_FEATURES), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        return np.vstack(parts_X), np.concatenate(parts_y), np.concatenate(parts_w)

    # Pre-index subcategories for hard negative mining
    subcat_items: dict[str, list[str]] = defaultdict(list)
    for pid in active_arr:
        p = snapshot.products.get(pid)
        if p is not None and p.subcategory:
            subcat_items[p.subcategory].append(pid)

    top_pop_pool = active_arr[: min(250, n_active)]
    pages = ("product", "category", "home")
    page_probs = (0.56, 0.28, 0.16)
    max_synth_per_source = 1200 if n_active < 400 else 2400

    def _pick_hard_negatives(target: str, uid: str, n_neg: int = 5) -> list[str]:
        negs: list[str] = []
        used = {target}
        info = snapshot.products.get(target)
        sc_pool = subcat_items.get(info.subcategory, ()) if info is not None else ()
        # Serving-aligned hard distractors: sequential neighbors + subcat peers
        trans_nbs = snapshot.transitions.get(target, ())
        if trans_nbs:
            lim = min(6, len(trans_nbs))
            for _ in range(2):
                cand = trans_nbs[int(rng.integers(0, lim))][0]
                if cand not in used and cand in snapshot.active_ids:
                    negs.append(cand)
                    used.add(cand)
                    break
        if len(sc_pool) > 1:
            for _ in range(3):
                cand = sc_pool[int(rng.integers(0, len(sc_pool)))]
                if cand not in used:
                    negs.append(cand)
                    used.add(cand)
                    if len(negs) >= 3:
                        break
        # Popularity distractors
        while len(negs) < n_neg:
            cand = (
                top_pop_pool[int(rng.integers(0, len(top_pop_pool)))]
                if rng.random() < 0.72
                else active_arr[int(rng.integers(0, n_active))]
            )
            if cand not in used:
                negs.append(cand)
                used.add(cand)
        return negs

    extra_rows_X: list[np.ndarray] = []
    extra_rows_y: list[int] = []
    extra_rows_w: list[float] = []

    # ── Source 2: Historical Cart Conversion vs Abandonment ─────────────────────────
    if cart_w > 0.0:
        cart_list = list(cart_hist_rows)
        if len(cart_list) > max_synth_per_source:
            idx_sample = rng.choice(len(cart_list), size=max_synth_per_source, replace=False)
            cart_sample = [cart_list[int(i)] for i in idx_sample]
        else:
            cart_sample = cart_list

        for row in cart_sample:
            uid = str(row.get("user_id") or "").strip()
            pid = str(row.get("product_id") or "").strip()
            if not uid or pid not in snapshot.active_ids:
                continue
            n_buys = _safe_float(row.get("n_purchases"), 0.0)
            last_cart = _safe_float(row.get("last_cart_ms"), 0.0)
            last_buy = _safe_float(row.get("last_purchase_ms"), 0.0)
            is_pos = (n_buys >= 1.0 and last_buy >= last_cart) or (n_buys >= 2.0)

            pg = str(rng.choice(pages, p=page_probs))
            rv: list[str] = []
            prof = snapshot.user_profiles.get(uid)
            if pg != "home":
                if prof and prof.intent_items:
                    rv = [ip for ip, _ in prof.intent_items[:3] if ip != pid][:2]
                if not rv:
                    nbs = snapshot.transitions.get(pid, ())
                    if nbs:
                        rv = [nbs[0][0]]
            req = {"user_id": uid, "context": {"page": pg, "region": "", "device": ""}, "recent_views": rv}
            cands = [pid] + _pick_hard_negatives(pid, uid, n_neg=4)
            req_now_ms = (last_cart + 600_000.0) if last_cart > 0.0 else None
            mat = feature_extractor(snapshot, req, cands, weights_profile=wp, now_ms=req_now_ms)
            if mat.shape[0] != len(cands):
                continue
            extra_rows_X.append(mat[0])
            extra_rows_y.append(1 if is_pos else 0)
            extra_rows_w.append(cart_w * (1.2 if is_pos else 0.75))
            for j in range(1, len(cands)):
                extra_rows_X.append(mat[j])
                extra_rows_y.append(0)
                extra_rows_w.append(cart_w * 0.35)

    # ── Source 3: Recurrent & User Intent Conversion Pairs ─────────────────────────
    if intent_w > 0.0:
        int_pool: list[tuple[str, str, float]] = []
        for row in recurrent_hist_rows:
            uid = str(row.get("user_id") or "").strip()
            pid = str(row.get("product_id") or "").strip()
            if uid and pid in snapshot.active_ids:
                int_pool.append((uid, pid, 1.35))
        for row in user_intent_rows:
            uid = str(row.get("user_id") or "").strip()
            pid = str(row.get("product_id") or "").strip()
            sc = _safe_float(row.get("intent_score"), 0.0)
            if uid and pid in snapshot.active_ids and sc > 0.5:
                int_pool.append((uid, pid, 0.85))
                if len(int_pool) >= max_synth_per_source * 3:
                    break

        if len(int_pool) > max_synth_per_source:
            idx_sample = rng.choice(len(int_pool), size=max_synth_per_source, replace=False)
            int_sample = [int_pool[int(i)] for i in idx_sample]
        else:
            int_sample = int_pool

        for uid, pid, rel_w in int_sample:
            pg = str(rng.choice(pages, p=page_probs))
            rv: list[str] = []
            if pg == "product":
                nbs = snapshot.transitions.get(pid, ())
                if nbs:
                    rv = [nbs[0][0]]
            elif pg == "category":
                prof = snapshot.user_profiles.get(uid)
                if prof and prof.intent_items:
                    rv = [ip for ip, _ in prof.intent_items[:3] if ip != pid][:2]
            req = {"user_id": uid, "context": {"page": pg, "region": "", "device": ""}, "recent_views": rv}
            cands = [pid] + _pick_hard_negatives(pid, uid, n_neg=4)
            mat = feature_extractor(snapshot, req, cands, weights_profile=wp)
            if mat.shape[0] != len(cands):
                continue
            extra_rows_X.append(mat[0])
            extra_rows_y.append(1)
            extra_rows_w.append(intent_w * rel_w)
            for j in range(1, len(cands)):
                extra_rows_X.append(mat[j])
                extra_rows_y.append(0)
                extra_rows_w.append(intent_w * 0.30)

    # ── Source 4: Sequential Transition & Co-visit Supervision ─────────────────────
    if seq_w > 0.0:
        seq_edges: list[tuple[str, str]] = []
        for row in transition_rows:
            src = str(row.get("src_product") or "").strip()
            dst = str(row.get("dst_product") or "").strip()
            if src in snapshot.active_ids and dst in snapshot.active_ids and src != dst:
                seq_edges.append((src, dst))
                if len(seq_edges) >= max_synth_per_source * 4:
                    break
        if len(seq_edges) > max_synth_per_source:
            idx_sample = rng.choice(len(seq_edges), size=max_synth_per_source, replace=False)
            seq_sample = [seq_edges[int(i)] for i in idx_sample]
        else:
            seq_sample = seq_edges

        for src, dst in seq_sample:
            pg = "product" if rng.random() < 0.72 else "category"
            req = {"user_id": "", "is_new_user": True, "context": {"page": pg, "region": "", "device": ""}, "recent_views": [src]}
            cands = [dst] + _pick_hard_negatives(dst, "", n_neg=4)
            mat = feature_extractor(snapshot, req, cands, weights_profile=wp)
            if mat.shape[0] != len(cands):
                continue
            extra_rows_X.append(mat[0])
            extra_rows_y.append(1)
            extra_rows_w.append(seq_w)
            for j in range(1, len(cands)):
                extra_rows_X.append(mat[j])
                extra_rows_y.append(0)
                extra_rows_w.append(seq_w * 0.30)

    if extra_rows_X:
        aux_X = np.vstack(extra_rows_X)
        aux_y = np.asarray(extra_rows_y, dtype=np.int64)
        aux_w = np.asarray(extra_rows_w, dtype=np.float64)
        if n_gold_pos >= 60 and gold_w_sum > 0.0:
            aux_cap = 0.08 * gold_w_sum
            cur_aux_sum = float(np.sum(aux_w))
            if cur_aux_sum > aux_cap and cur_aux_sum > 0.0:
                aux_w = aux_w * (aux_cap / cur_aux_sum)
        parts_X.append(aux_X)
        parts_y.append(aux_y)
        parts_w.append(aux_w)

    if not parts_X:
        return (
            np.empty((0, NUM_FEATURES), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    return np.vstack(parts_X), np.concatenate(parts_y), np.concatenate(parts_w)


def compute_ranker_holdout_auc(
    ranker: PositionDebiasedRanker,
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_fraction: float = 0.20,
) -> float:
    """Deterministic stratified-like AUC diagnostic across all multisource blocks."""
    n = int(X.shape[0])
    if n < 100:
        return 0.0
    rng = np.random.default_rng(1337)
    take = max(100, int(n * val_fraction))
    idx = rng.choice(n, size=min(take, n), replace=False)
    Xv, yv = X[idx], y[idx]
    if len(np.unique(yv)) < 2:
        return 0.0
    try:
        from sklearn.metrics import roc_auc_score

        scores = ranker.score_matrix(Xv)
        return float(roc_auc_score(yv, scores))
    except Exception:
        return 0.0
