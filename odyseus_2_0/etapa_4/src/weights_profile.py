"""Centralized modular hyperparameter and preset profile for Calypso 5.1."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

USER_SEGMENTS: tuple[str, ...] = (
    "new_with_views",
    "new_no_views",
    "known_with_views",
    "known_with_history",
    "known_cold",
)

SEGMENT_SIGNAL_NAMES: tuple[str, ...] = (
    "seq",
    "intent",
    "aff",
    "ctx",
    "pop",
    "trend",
    "cold",
)

_SEGMENT_FIELD_MAP: dict[str, str] = {
    "new_with_views": "seg_new_with_views",
    "new_no_views": "seg_new_no_views",
    "known_with_views": "seg_known_with_views",
    "known_with_history": "seg_known_with_history",
    "known_cold": "seg_known_cold",
}


@dataclass(frozen=True, slots=True)
class WeightsProfile:
    """Hot-reloadable modular optimization profile for Calypso 5.1."""

    # Identification
    profile_name: str = "conversion_first"
    description: str = "Calypso 5.1: Prioritat absoluta a la conversió efectiva a compra (taxa base M3, tendències 60m M4, pool 140 amb Top-35 vendes D3, cold-start X10, carret/recurrents per puntuació M6)."

    # 14 Prior Weights matching FEATURE_NAMES order
    w_seq_lift: float = 1.15
    w_self_cvr_view: float = 2.15
    w_intent_score: float = 0.65
    w_user_affinity: float = 0.34
    w_price_compat: float = 0.22
    w_ctx_bayes: float = 0.36
    w_global_pop: float = 0.56
    w_live_trend: float = 0.36
    w_is_new_product: float = 0.03
    w_reverse_graft: float = 0.35
    w_forward_cold: float = 0.50
    w_log_price_norm: float = 0.00
    w_same_top_subcat: float = 0.22
    w_same_top_brand: float = 0.08

    # ML anchoring ratios
    min_prior_ratios: tuple[float, ...] = (
        0.72,  # seq_lift_score
        0.75,  # self_cvr_view
        0.65,  # intent_score
        0.55,  # user_affinity
        0.45,  # price_compat
        0.50,  # ctx_bayes_score
        0.25,  # global_pop_norm
        0.30,  # live_trend_norm
        0.00,  # is_new_product
        0.55,  # reverse_graft_score
        0.55,  # forward_cold_score
        0.00,  # log_price_norm
        0.45,  # same_top_subcat
        0.00,  # same_top_brand
    )
    max_prior_multiplier: float = 2.2

    # 5 User Segment Multipliers: (seq, intent, aff, ctx, pop, trend, cold)
    seg_new_with_views: tuple[float, ...] = (1.12, 0.0, 0.0, 1.15, 1.05, 1.20, 1.15)
    seg_new_no_views: tuple[float, ...] = (0.0, 0.0, 0.0, 1.45, 1.35, 1.40, 1.20)
    seg_known_with_views: tuple[float, ...] = (1.15, 1.15, 1.10, 0.90, 0.85, 1.05, 1.00)
    seg_known_with_history: tuple[float, ...] = (0.80, 1.40, 1.35, 1.10, 1.00, 1.15, 1.05)
    seg_known_cold: tuple[float, ...] = (0.0, 1.10, 1.10, 1.35, 1.25, 1.30, 1.15)

    # Price compatibility & tie-breaking
    price_std_floor: float = 0.45
    price_z2_clip: float = 6.0
    price_cheap_asymmetry: float = 0.0
    revenue_tie_weight: float = 0.0
    tie_break_mode: str = "conversion"  # "conversion" | "cheap" | "revenue"
    tie_break_eps: float = 0.015

    # Product profile & Cold-Start levers
    cold_sparse_cos_weight: float = 0.65
    cold_dense_svd_weight: float = 0.35
    cold_subcat_match_bonus: float = 0.35
    cold_cat_match_bonus: float = 0.15
    cold_brand_match_bonus: float = 0.12
    cold_price_prox_bonus: float = 0.15
    cold_min_similarity: float = 0.08
    cold_norm_floor: float = 0.25
    cold_explore_start_slot_warm: int = 7
    cold_explore_start_slot_cold: int = 0
    max_cold_explore_in_warm: int = 2
    cold_neighbor_pop_prior_weight: float = 0.80

    # Stage 5.1 Base-Rate, Candidate Pool & Conversion Signals
    candidate_pool_size: int = 140
    guaranteed_top_sellers_in_pool: int = 35
    use_sales_rank_base_rate: bool = True
    sales_rank_blend_weight: float = 0.68
    live_pop_mix_weight: float = 0.45
    transition_dest_pop_gamma: float = 0.50
    view_last_product_page_factor: float = 1.0
    view_base_bonus: float = 0.55
    recency_decay: float = 0.72
    slot1_category_match_bonus: float = 0.20
    same_cat_bonus_weight: float = 0.08
    cart_active_score_bonus: float = 0.95
    cart_forgotten_score_bonus: float = 0.35
    recurrent_score_bonus: float = 0.25
    allow_recurrent_rebuy: bool = True
    discount_boost_weight: float = 0.15

    # Conversion-oriented IPS & Bandit learning
    ips_purchase_weight: float = 2.5
    ips_click_weight: float = 0.10
    bandit_purchase_reward: float = 1.0
    bandit_click_reward: float = 0.05
    bandit_ucb_alpha: float = 0.12
    bandit_cold_ucb_multiplier: float = 1.8
    bandit_warm_ucb_multiplier: float = 0.20
    bandit_exam_power: float = 0.75
    use_empirical_exam_curve: bool = True
    bandit_prior_strength: float = 20.0
    ml_max_blend_alpha: float = 0.45
    ml_min_positives: int = 25
    ml_l2_c: float = 1.2

    # MMR & Background penalties
    mmr_start_slot_session: int = 5
    mmr_start_slot_home: int = 3
    mmr_subcat_lambda_session: float = 0.0
    mmr_subcat_lambda_home: float = 0.0
    mmr_brand_lambda: float = 0.02
    mmr_relative_penalty: bool = True
    subcat_floor_min: float = 0.28
    brand_floor_min: float = 0.12
    v3_no_subcat_floor: bool = False
    v3_bg_penalty: bool = True
    bg_penalty_diff_subcat: float = 0.48
    bg_penalty_same_subcat: float = 0.85
    v3_ease_weight: float = 0.0

    # Stage 4.1 / 5.1 Business rules
    v4_cart_priority: bool = True
    v4_cart_active_minutes: float = 30.0
    v4_cart_max_slots: int = 2
    v4_forgotten_cart: bool = False
    v4_forgotten_max_slots: int = 2
    v4_forgotten_max_days: float = 30.0
    v4_price_ascending: bool = False
    v4_recurrent: bool = False
    v4_recurrent_min_purchases: int = 2
    v4_recurrent_max_slots: int = 2
    v4_live_event_purchases: bool = True

    def prior_weights_array(self) -> np.ndarray:
        """Return 14 prior weights ordered by reranker.FEATURE_NAMES."""
        return np.array(
            [
                self.w_seq_lift,
                self.w_self_cvr_view,
                self.w_intent_score,
                self.w_user_affinity,
                self.w_price_compat,
                self.w_ctx_bayes,
                self.w_global_pop,
                self.w_live_trend,
                self.w_is_new_product,
                self.w_reverse_graft,
                self.w_forward_cold,
                self.w_log_price_norm,
                self.w_same_top_subcat,
                self.w_same_top_brand,
            ],
            dtype=np.float64,
        )

    def min_prior_ratios_array(self) -> np.ndarray:
        """Return 14 minimum prior anchoring ratios aligned with FEATURE_NAMES."""
        vals = list(self.min_prior_ratios)
        if len(vals) < 14:
            vals.extend([0.0] * (14 - len(vals)))
        return np.asarray(vals[:14], dtype=np.float64)

    def segment_multipliers(self, segment: str) -> tuple[float, float, float, float, float, float, float]:
        """Return (seq, intent, aff, ctx, pop, trend, cold) multipliers for a user segment."""
        attr = _SEGMENT_FIELD_MAP.get(segment)
        raw = getattr(self, attr, None) if attr else None
        if raw is None or len(raw) < 7:
            return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        return tuple(float(x) for x in raw[:7])  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to JSON-friendly dict."""
        data = asdict(self)
        for key in ("min_prior_ratios", *_SEGMENT_FIELD_MAP.values()):
            data[key] = [float(x) for x in data[key]]
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base: WeightsProfile | None = None) -> WeightsProfile:
        """Construct WeightsProfile coercing types safely from a raw mapping."""
        template = base or cls()
        kwargs: dict[str, Any] = {}
        tuple_fields = {"min_prior_ratios", *_SEGMENT_FIELD_MAP.values()}
        valid_fields = {f.name: f for f in fields(cls)}
        for name, fobj in valid_fields.items():
            default_val = getattr(template, name)
            if name not in raw or raw[name] is None:
                kwargs[name] = default_val
                continue
            val = raw[name]
            if name in tuple_fields:
                if isinstance(val, str):
                    items = [float(x.strip()) for x in val.split(",") if x.strip()]
                else:
                    items = [float(x) for x in val]
                expected_len = len(default_val)
                if len(items) < expected_len:
                    items.extend(list(default_val)[len(items) :])
                kwargs[name] = tuple(items[:expected_len])
            elif isinstance(default_val, bool):
                if isinstance(val, str):
                    kwargs[name] = val.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    kwargs[name] = bool(val)
            elif isinstance(default_val, int):
                kwargs[name] = int(float(val))
            elif isinstance(default_val, float):
                kwargs[name] = float(val)
            else:
                kwargs[name] = str(val).strip()
        if kwargs.get("tie_break_mode") not in {"conversion", "cheap", "revenue"}:
            kwargs["tie_break_mode"] = template.tie_break_mode
        return cls(**kwargs)

    def save_json(self, path: str | Path) -> Path:
        """Atomically persist WeightsProfile to disk so live updater reloads safely."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> WeightsProfile:
        """Load WeightsProfile from JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        base = PRESETS.get(str(data.get("profile_name") or ""), cls())
        return cls.from_dict(data, base=base)


_BASE = WeightsProfile()

PRESETS: dict[str, WeightsProfile] = {
    "conversion_first": _BASE,
    "cheap_boost": replace(
        _BASE,
        profile_name="cheap_boost",
        description="Prioritza productes més barats mitjançant pesos suaus (negatiu en preu i asimetria barata) sense trencar el rànquing top-1.",
        w_log_price_norm=-0.18,
        w_price_compat=0.38,
        price_cheap_asymmetry=0.45,
        revenue_tie_weight=-0.08,
        tie_break_mode="cheap",
        v4_price_ascending=False,
        min_prior_ratios=(
            0.72, 0.75, 0.65, 0.55, 0.45, 0.50, 0.25, 0.30, 0.00, 0.55, 0.55, 0.40, 0.45, 0.00
        ),
    ),
    "cheap_hard_sort": replace(
        _BASE,
        profile_name="cheap_hard_sort",
        description="Combina cheap_boost amb ordenació estricta de més barat a més car dins de cada bloc (M3 activada).",
        w_log_price_norm=-0.20,
        w_price_compat=0.38,
        price_cheap_asymmetry=0.50,
        revenue_tie_weight=-0.10,
        tie_break_mode="cheap",
        v4_price_ascending=True,
        min_prior_ratios=(
            0.72, 0.75, 0.65, 0.55, 0.45, 0.50, 0.25, 0.30, 0.00, 0.55, 0.55, 0.40, 0.45, 0.00
        ),
    ),
    "user_trend_focus": replace(
        _BASE,
        profile_name="user_trend_focus",
        description="Potencia tendències contextuals, senyals d'afinitat d'usuari i tendències en viu adaptades per segment.",
        w_intent_score=0.75,
        w_user_affinity=0.45,
        w_ctx_bayes=0.42,
        w_live_trend=0.42,
        w_same_top_subcat=0.22,
        seg_new_with_views=(1.10, 0.0, 0.0, 1.35, 1.15, 1.45, 1.10),
        seg_new_no_views=(0.0, 0.0, 0.0, 1.65, 1.40, 1.65, 1.15),
        seg_known_with_views=(1.15, 1.30, 1.28, 1.10, 0.90, 1.30, 1.00),
        seg_known_with_history=(0.85, 1.55, 1.50, 1.25, 1.05, 1.35, 1.05),
        seg_known_cold=(0.0, 1.25, 1.30, 1.55, 1.30, 1.55, 1.10),
    ),
    "product_profile_focus": replace(
        _BASE,
        profile_name="product_profile_focus",
        description="Potencia la similitud semàntica i d'atributs de producte (subcategoria, marca, preu) i l'exploració cold-start.",
        w_reverse_graft=0.45,
        w_forward_cold=0.62,
        w_same_top_subcat=0.26,
        w_same_top_brand=0.14,
        cold_sparse_cos_weight=0.55,
        cold_dense_svd_weight=0.45,
        cold_subcat_match_bonus=0.48,
        cold_cat_match_bonus=0.20,
        cold_brand_match_bonus=0.22,
        cold_price_prox_bonus=0.22,
        cold_explore_start_slot_warm=5,
        max_cold_explore_in_warm=3,
        bandit_ucb_alpha=0.16,
    ),
    "calibrated_m3": replace(
        _BASE,
        profile_name="calibrated_m3",
        description="Variant M3 pura: fins i tot el carretó actiu competeix al rànquing per puntuació sense bloqueig de posicions ni penalització de fons.",
        v4_cart_priority=False,
        cart_active_score_bonus=1.60,
        v3_bg_penalty=False,
        w_global_pop=0.72,
        w_self_cvr_view=2.30,
        sales_rank_blend_weight=0.80,
    ),
}


def list_presets() -> list[str]:
    """Return available built-in profile preset names."""
    return list(PRESETS.keys())


def get_preset(name: str) -> WeightsProfile:
    """Return a built-in WeightsProfile preset by name."""
    key = str(name or "").strip()
    if key not in PRESETS:
        raise KeyError(f"Unknown WeightsProfile preset {name!r}. Available: {list_presets()}")
    return PRESETS[key]


def load_active_profile(path: str | Path | None = None) -> WeightsProfile:
    """Load profile from JSON path if present, else fall back to CALYPSO_PRESET / default."""
    env_preset = os.environ.get("CALYPSO_PRESET", "").strip()
    if path is not None:
        p = Path(path)
        if p.is_file():
            try:
                loaded = WeightsProfile.load_json(p)
                if env_preset and env_preset in PRESETS and env_preset != loaded.profile_name:
                    return PRESETS[env_preset]
                return loaded
            except Exception:
                pass
    if env_preset and env_preset in PRESETS:
        return PRESETS[env_preset]
    return PRESETS["conversion_first"]
