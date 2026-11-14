"""Política de recomendación (A4 + D + E).

1. Variables para los ~5.000 productos del catálogo (no ~100 candidatos).
2. p̂ = modelo de respuesta (slot fijo a 1).
3. Ajustes en vivo que el histórico no puede enseñar al modelo: tendencia (V5) y
   cambio de precio reciente (V6), con el peso del brazo elegido por Thompson (D2).
4. Lista: ordenar por p̂, o selección voraz con intención incierta (E) cuando no hay sesión.
5. Exploración acotada a los slots 8 y 10 con productos nuevos o en tendencia (D3).
6. Guardrails: 10 productos distintos, activos, con stock y no comprados ya (V2).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bandit import ThompsonBandit, segment_of
from .config import Config
from .features import Req, compute_features, parse_request
from .guardrails import NUM_SLOTS, ensure_valid_submission
from .model import LinearModel, sigmoid
from .state import Snapshot


@dataclass
class Decision:
    request_id: str
    items: list[str]
    arm: str = "mix"
    segment: str = ""
    variant: str = "model"
    explored: list[str] = field(default_factory=list)
    is_cold: bool = False
    user_id: str = ""
    scores: np.ndarray | None = None

    def tag(self, base: str) -> str:
        return f"{base}-{self.variant}-{self.arm}"


def stable_unit(s: str) -> float:
    return (zlib.crc32(s.encode()) % 10_000) / 10_000.0


class Recommender:
    def __init__(self, cfg: Config, model: Any = None, control_model: Any = None,
                 bandit: ThompsonBandit | None = None, seed: int | None = None) -> None:
        self.cfg = cfg
        self.model = model or LinearModel.default()
        self.control = control_model or LinearModel.default()
        self.arms = {a[0]: a for a in cfg.bandit_arms}
        self.bandit = bandit if cfg.bandit_enabled else None
        self.rng = np.random.default_rng(seed)
        self.attention = np.array(cfg.attention, dtype=float)

    # ------------------------------------------------------------------ scoring

    def score(self, req: Req, snap: Snapshot, arm: str, variant: str = "model",
              use_profile: bool = True) -> tuple[np.ndarray, Any]:
        cfg = self.cfg
        cat = snap.catalog
        live = snap.live.get(arm) or snap.live.get("mix")
        X, info = compute_features(req, snap, live=live, use_profile=use_profile)
        model = self.model if variant == "model" else self.control
        logit = np.asarray(model.logit(X), dtype=float)
        _, trend_w, _ = self._arm_params(arm)
        if live is not None and live.n_items == cat.n and trend_w:
            logit = logit + trend_w * live.trend
        now = live.now_ts if (live is not None and live.now_ts) else (req.ts or 0.0)
        if cfg.weight_price_drop and now:
            age = np.maximum(now - cat.price_change_ts, 0.0)
            decay = np.where(cat.price_change_ts > 0, np.exp(-age / (cfg.price_change_tau_min * 60)), 0.0)
            logit = logit + cfg.weight_price_drop * cat.price_change_rel * decay
        logit = logit + cfg.revenue_tiebreak * np.clip(cat.price_rel_z, -2, 2)
        return logit, info

    def _arm_params(self, arm: str) -> tuple[float, float, float]:
        a = self.arms.get(arm)
        if a is None:
            return 1.0, self.cfg.weight_trend, 0.0
        return a[1], a[2], a[3]

    def excluded(self, req: Req, snap: Snapshot, info: Any) -> set[str]:
        out: set[str] = set()
        prof = snap.history.users.get(req.user_id) if req.user_id else None
        if prof is not None:
            out |= prof.bought
        if req.user_id:
            out |= snap.live_bought.get(req.user_id, frozenset())
        return out

    # ------------------------------------------------------------------ list building

    def _greedy_diverse(self, cand: np.ndarray, p: np.ndarray, cats: np.ndarray, k: int) -> list[int]:
        """E: maximiza Σ_c π_c [1 − Π_k (1 − e_k p_c(i_k))] de forma voraz.
        Con p_c(i) = p_i / π_c, la ganancia marginal es e_k · p_i · Π(1 − e_j p_c(i_j))."""
        p = np.clip(p[cand], 1e-9, None)
        cc = cats[cand]
        pi = np.bincount(cc, weights=p)
        pi = pi / pi.sum()
        none = np.ones_like(pi)
        chosen: list[int] = []
        avail = np.ones(len(cand), dtype=bool)
        for slot in range(min(k, len(cand))):
            e = self.attention[min(slot, len(self.attention) - 1)]
            gain = np.where(avail, e * p * none[cc], -1.0)
            j = int(np.argmax(gain))
            chosen.append(int(cand[j]))
            avail[j] = False
            c = cc[j]
            none[c] *= 1.0 - min(1.0, e * p[j] / max(pi[c], 1e-9))
        return chosen

    def recommend(self, raw: Any, snap: Snapshot, k: int = NUM_SLOTS, arm: str | None = None,
                  variant: str | None = None, use_profile: bool = True) -> Decision:
        cfg = self.cfg
        req = raw if isinstance(raw, Req) else parse_request(raw)
        cat = snap.catalog
        is_cold_hint = req.is_new_user or any(
            (p in cat.idx and cat.is_new[cat.idx[p]]) for p in req.recent)
        segment = segment_of(is_cold_hint, req.page)
        if variant is None:
            variant = "model" if stable_unit(req.request_id) < cfg.ab_treatment_share else "control"
        if arm is None:
            arm = self.bandit.choose(segment) if self.bandit is not None else "mix"

        logit, info = self.score(req, snap, arm, variant, use_profile=use_profile)
        excl = self.excluded(req, snap, info)
        valid = cat.active.copy()
        if excl:
            for pid in excl:
                j = cat.idx.get(pid)
                if j is not None:
                    valid[j] = False
        logit = np.where(valid, logit, -np.inf)
        n_valid = int(valid.sum())
        C = min(200, n_valid)
        if C > 0:
            cand = np.argpartition(-logit, C - 1)[:C]
            cand = cand[np.argsort(-logit[cand], kind="stable")]
            cand = cand[np.isfinite(logit[cand])]
        else:
            cand = np.zeros(0, dtype=np.int64)

        diversify = cfg.diversify == "always" or (cfg.diversify == "no_session" and len(info.recent_idx) == 0)
        if diversify and len(cand) > k:
            order = self._greedy_diverse(cand, sigmoid(logit), cat.category, k)
        else:
            order = [int(i) for i in cand[:k]]
        items = [cat.ids[i] for i in order]

        explored: list[str] = []
        _, _, explore_rate = self._arm_params(arm)
        if explore_rate > 0 and self.rng.random() < explore_rate and len(cand):
            inserted = []
            for pos, pid in zip(cfg.explore_slots, self._explore(cand, logit, info, snap, arm, items)):
                if pid in items:
                    continue
                items.insert(min(pos - 1, len(items)), pid)
                inserted.append(pid)
            items = items[:k]
            explored = [p for p in inserted if p in items]

        final = ensure_valid_submission(items, cat.active_ids, cat.global_ranked, excl, k=k)
        return Decision(request_id=req.request_id, items=final, arm=arm, segment=segment,
                        variant=variant, explored=explored, is_cold=info.is_cold, user_id=req.user_id,
                        scores=None)

    def _explore(self, cand: np.ndarray, logit: np.ndarray, info: Any, snap: Snapshot,
                 arm: str, current: list[str]) -> list[str]:
        """D3: productos nuevos con pocas impresiones nuestras o en tendencia clara."""
        cat = snap.catalog
        live = snap.live.get(arm) or snap.live.get("mix")
        imps = snap.new_impressions
        pool_mask = np.isfinite(logit) & cat.is_new
        if imps:
            for j in np.where(pool_mask)[0]:
                if imps.get(cat.ids[j], 0) >= self.cfg.explore_max_impressions:
                    pool_mask[j] = False
        if live is not None and live.n_items == cat.n:
            pool_mask |= np.isfinite(logit) & (live.trend > math.log(2.0))
        if info.cat_weights.sum() > 0:
            top_cats = np.where(info.cat_weights >= 0.3 * info.cat_weights.max())[0]
            restricted = pool_mask & np.isin(cat.category, top_cats)
            if restricted.any():
                pool_mask = restricted
        pool = np.where(pool_mask)[0]
        pool = np.array([j for j in pool if cat.ids[j] not in current[:6]], dtype=np.int64)
        if not len(pool):
            return []
        top = pool[np.argsort(-logit[pool])][:8]
        w = np.exp(logit[top] - logit[top].max())
        n_pick = min(len(self.cfg.explore_slots), len(top))
        picks = self.rng.choice(top, size=n_pick, replace=False, p=w / w.sum())
        return [cat.ids[int(j)] for j in picks]
