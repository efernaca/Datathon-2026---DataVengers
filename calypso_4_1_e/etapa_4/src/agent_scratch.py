"""Autonomous Slow-Loop agent diagnostics and team_NN.scratch persistence."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .bandits import PositionAwareBandit
from .cold_start import build_cold_start_scratch_records
from .config import ConfigV4
from .reranker import PositionDebiasedRanker
from .state import AgentOverrides, BanditState, ProductInfo


@dataclass(frozen=True, slots=True)
class LiveTelemetry:
    """Compact summary of live serving & catalog conditions."""

    feedback_count: int
    ctr: float
    cvr: float
    new_product_share: float
    new_product_count: int
    new_product_purchases: int
    total_revenue_eur: float


class CalypsoAgent:
    """Inspects live feedback/catalog telemetry and adapts runtime overrides safely."""

    def diagnose_and_adapt(
        self,
        config: ConfigV4,
        products: Mapping[str, ProductInfo],
        active_ids: frozenset[str],
        new_product_ids: frozenset[str],
        feedback_rows: Iterable[Mapping[str, Any]],
        bandit_state: BanditState,
        ranker: PositionDebiasedRanker,
        *,
        weights_profile: Any = None,
    ) -> tuple[AgentOverrides, LiveTelemetry, list[dict[str, Any]]]:
        """Analyze live conditions, produce AgentOverrides, and format scratch logs."""
        fb_list = list(feedback_rows)
        n_fb = len(fb_list)
        clicks = 0
        purchases = 0
        new_buys = 0
        revenue_eur = 0.0

        for row in fb_list:
            clicked = str(row.get("clicked") or "").strip()
            purchased = str(row.get("purchased") or "").strip()
            if clicked:
                clicks += 1
            if purchased:
                purchases += 1
                if purchased in new_product_ids:
                    new_buys += 1
            rev_val = row.get("revenue_eur")
            if rev_val is not None:
                try:
                    revenue_eur += float(rev_val)
                except (TypeError, ValueError):
                    pass

        ctr = clicks / n_fb if n_fb > 0 else 0.0
        cvr = purchases / n_fb if n_fb > 0 else 0.0
        n_active = max(len(active_ids), 1)
        n_new = len(new_product_ids)
        new_share = n_new / n_active

        telemetry = LiveTelemetry(
            feedback_count=n_fb,
            ctr=ctr,
            cvr=cvr,
            new_product_share=new_share,
            new_product_count=n_new,
            new_product_purchases=new_buys,
            total_revenue_eur=revenue_eur,
        )

        base_ucb = weights_profile.bandit_ucb_alpha if weights_profile is not None else config.bandit_ucb_alpha
        base_rev = weights_profile.w_reverse_graft if weights_profile is not None else config.weight_reverse_cold_graft
        base_fwd = weights_profile.w_forward_cold if weights_profile is not None else config.weight_forward_cold
        base_tie = weights_profile.revenue_tie_weight if weights_profile is not None else config.revenue_tie_weight
        base_brand = weights_profile.mmr_brand_lambda if weights_profile is not None else config.mmr_brand_lambda

        ml_scale = 1.0
        ucb_alpha = base_ucb
        rev_graft_w = base_rev
        fwd_cold_w = base_fwd
        rev_tie_w = base_tie
        brand_lam = base_brand
        regime = "nominal"
        reason = "Steady-state Modular Slot-Gated ML + Bandit policy"

        if n_fb >= 80 and cvr < 0.015 and ranker.ml_blend_alpha > 0.15:
            ml_scale = 0.45
            ucb_alpha = min(0.24, base_ucb + 0.04)
            regime = "conservative_fallback"
            reason = f"Low live CVR ({cvr:.4f} on {n_fb} reqs): pulling ML blend toward domain priors"
        elif new_share >= 0.06 or n_new >= 25:
            ucb_alpha = min(0.22, base_ucb + 0.06)
            rev_graft_w = min(max(0.42, base_rev + 0.10), base_rev + 0.12)
            fwd_cold_w = min(max(0.58, base_fwd + 0.08), base_fwd + 0.10)
            regime = "cold_surge"
            reason = (
                f"Cold Start surge ({n_new} new items / {new_share:.1%}): "
                "Slot-Gated UCB exploration in slots 5-9"
            )
        elif n_fb >= 250 and cvr >= 0.045:
            ml_scale = 1.15
            ucb_alpha = max(0.08, base_ucb - 0.04)
            regime = "high_velocity_ml"
            reason = f"Strong live CVR ({cvr:.4f} on {n_fb} reqs): increasing ML exploit weight"

        ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        overrides = AgentOverrides(
            ml_blend_alpha_scale=ml_scale,
            bandit_ucb_alpha=ucb_alpha,
            weight_reverse_cold_graft=rev_graft_w,
            weight_forward_cold=fwd_cold_w,
            revenue_tie_weight=rev_tie_w,
            mmr_brand_lambda=brand_lam,
            regime=regime,
            reason=reason,
            diagnosed_at=ts_iso,
        )

        decision_record: dict[str, Any] = {
            "kind": "agent_decision",
            "model_version": config.model_version,
            "weights_profile": (
                weights_profile.profile_name if weights_profile is not None else "config_default"
            ),
            "ts": ts_iso,
            "regime": regime,
            "reason": reason,
            "telemetry": {
                "feedback_count": n_fb,
                "ctr": round(ctr, 5),
                "cvr": round(cvr, 5),
                "new_product_share": round(new_share, 5),
                "new_product_count": n_new,
                "new_product_purchases": new_buys,
                "revenue_eur": round(revenue_eur, 2),
                "bandit_eff_exposures": round(bandit_state.total_effective_exposures, 1),
            },
            "overrides": {
                "ml_blend_alpha_scale": ml_scale,
                "bandit_ucb_alpha": round(ucb_alpha, 4),
                "weight_reverse_cold_graft": round(rev_graft_w, 4),
                "weight_forward_cold": round(fwd_cold_w, 4),
            },
        }

        scratch_records: list[dict[str, Any]] = [
            decision_record,
            ranker.export_scratch_record(config.model_version, ts_iso),
        ]
        return overrides, telemetry, scratch_records

    def build_full_scratch_payload(
        self,
        config: ConfigV4,
        bandit: PositionAwareBandit,
        bandit_state: BanditState,
        cold_neighbors: Mapping[str, tuple[tuple[str, float], ...]],
        products: Mapping[str, ProductInfo],
        overrides: AgentOverrides,
        base_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Assemble bounded telemetry batch for team_NN.scratch."""
        ts_iso = overrides.diagnosed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = list(base_records)
        payload.extend(
            bandit.build_scratch_records(
                bandit_state,
                products,
                config.model_version,
                ts_iso,
                overrides.bandit_ucb_alpha,
                limit=60,
            )
        )
        payload.extend(
            build_cold_start_scratch_records(
                cold_neighbors,
                products,
                config.model_version,
                ts_iso,
                limit=80,
            )
        )
        return payload
