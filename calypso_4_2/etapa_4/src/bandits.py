"""Position-aware Empirical Bayes & UCB multi-armed bandit for Stage 4."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .state import BanditState, ProductInfo


def slot_examination_weights(
    k_slots: int = 10,
    exam_power: float = 0.75,
    attention_curve: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Examination curve per slot.

    Si `attention_curve` trae valores (M5 de MEDIDAS_CONVERSION: curva medida en el log real),
    se usa tal cual (rellenando con el último valor). Si no, curva paramétrica (k + 1)^(-exam_power).
    """
    if attention_curve:
        vals = [max(float(x), 1e-4) for x in attention_curve]
        return tuple(vals[idx] if idx < len(vals) else vals[-1] for idx in range(k_slots))
    return tuple((idx + 1.0) ** (-exam_power) for idx in range(k_slots))


class PositionAwareBandit:
    """Estimates position-debiased conversion lift and UCB exploration bonuses."""

    def __init__(
        self,
        *,
        k_slots: int = 10,
        exam_power: float = 0.75,
        prior_strength: float = 20.0,
        cold_ucb_multiplier: float = 1.8,
        warm_ucb_multiplier: float = 0.20,
        click_reward: float = 0.25,
        purchase_reward: float = 1.0,
        attention_curve: tuple[float, ...] | None = None,
    ) -> None:
        self.k_slots = k_slots
        self.exam_weights = slot_examination_weights(
            k_slots=k_slots, exam_power=exam_power, attention_curve=attention_curve
        )
        self.prior_strength = prior_strength
        self.cold_ucb_multiplier = cold_ucb_multiplier
        self.warm_ucb_multiplier = warm_ucb_multiplier
        self.click_reward = click_reward
        self.purchase_reward = purchase_reward

    def fit_from_feedback(
        self,
        feedback_rows: Iterable[Mapping[str, Any]],
        fallback_cvr: Mapping[str, float] | None = None,
    ) -> BanditState:
        """Aggregate effective slot exposures and clicks/purchases from feedback/sessions."""
        eff_exp: dict[str, float] = defaultdict(float)
        weighted_rew: dict[str, float] = defaultdict(float)
        raw_imp: dict[str, int] = defaultdict(int)
        raw_buy: dict[str, int] = defaultdict(int)
        raw_clk: dict[str, int] = defaultdict(int)

        total_eff = 0.0
        total_rew = 0.0
        rows_count = 0

        for row in feedback_rows:
            items = row.get("items")
            if items is None or len(items) == 0:
                items = row.get("shown")
            if items is None or len(items) == 0:
                items = row.get("recommendations")
            if items is None or len(items) == 0:
                continue

            rows_count += 1
            purchased_id = str(row.get("purchased") or "").strip()
            clicked_id = str(row.get("clicked") or "").strip()

            for slot_idx, raw_pid in enumerate(items[: self.k_slots]):
                pid = str(raw_pid).strip()
                if not pid:
                    continue
                ew = self.exam_weights[slot_idx]
                eff_exp[pid] += ew
                raw_imp[pid] += 1
                total_eff += ew

                if pid == purchased_id:
                    weighted_rew[pid] += self.purchase_reward
                    raw_buy[pid] += 1
                    total_rew += self.purchase_reward
                elif pid == clicked_id:
                    weighted_rew[pid] += self.click_reward
                    raw_clk[pid] += 1
                    total_rew += self.click_reward

        if total_eff > 0.0:
            global_rate = min(max(total_rew / total_eff, 0.005), 0.35)
        elif fallback_cvr:
            vals = list(fallback_cvr.values())
            global_rate = min(max(sum(vals) / max(len(vals), 1), 0.005), 0.35)
        else:
            global_rate = 0.04

        arms: dict[str, tuple[float, float, int, int, int]] = {}
        for pid, e_val in eff_exp.items():
            arms[pid] = (
                e_val,
                weighted_rew.get(pid, 0.0),
                raw_imp[pid],
                raw_buy.get(pid, 0),
                raw_clk.get(pid, 0),
            )

        return BanditState(
            arms=arms,
            total_effective_exposures=total_eff,
            global_reward_rate=global_rate,
            exam_weights=self.exam_weights,
            feedback_rows_processed=rows_count,
        )

    def score_components(
        self,
        product_id: str,
        is_new: bool,
        state: BanditState,
        ucb_alpha: float,
    ) -> tuple[float, float]:
        """Return separate (exploit_delta, ucb_bonus) for slot-gated ranking."""
        if state.total_effective_exposures <= 0.0 and not is_new:
            return 0.0, 0.0

        arm = state.arms.get(product_id)
        eff_i = arm[0] if arm is not None else 0.0
        rew_i = arm[1] if arm is not None else 0.0

        mu_0 = state.global_reward_rate
        mu_hat = (rew_i + self.prior_strength * mu_0) / (eff_i + self.prior_strength)
        exploit_delta = max(-0.18, min(0.35, (mu_hat - mu_0) * 2.5))

        log_total = math.log(max(state.total_effective_exposures, 12.0))
        mult = self.cold_ucb_multiplier if is_new else self.warm_ucb_multiplier
        ucb_bonus = min(ucb_alpha * mult * math.sqrt(log_total / (eff_i + 4.0)), 0.38)
        return exploit_delta, ucb_bonus

    def score_adjustment(
        self,
        product_id: str,
        is_new: bool,
        state: BanditState,
        ucb_alpha: float,
    ) -> float:
        """Combined exploit + exploration adjustment."""
        exploit_delta, ucb_bonus = self.score_components(product_id, is_new, state, ucb_alpha)
        return exploit_delta + ucb_bonus

    def build_scratch_records(
        self,
        state: BanditState,
        products: Mapping[str, ProductInfo],
        model_version: str,
        ts_iso: str,
        ucb_alpha: float,
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        """Serialize top-active and cold-start bandit arms for team_NN.scratch."""
        if not state.arms and not products:
            return []

        ranked_pids: list[tuple[float, str]] = []
        for pid, (eff_i, rew_i, _, _, _) in state.arms.items():
            adj = self.score_adjustment(
                pid,
                bool(products.get(pid) and products[pid].is_new),
                state,
                ucb_alpha,
            )
            ranked_pids.append((eff_i + rew_i * 5.0 + adj * 10.0, pid))

        ranked_pids.sort(reverse=True)
        selected_ids: list[str] = [pid for _, pid in ranked_pids[: limit // 2]]
        selected_set = set(selected_ids)
        for pid, info in products.items():
            if len(selected_ids) >= limit:
                break
            if info.is_active and info.is_new and pid not in selected_set:
                selected_ids.append(pid)
                selected_set.add(pid)

        records: list[dict[str, Any]] = []
        for pid in selected_ids[:limit]:
            arm = state.arms.get(pid, (0.0, 0.0, 0, 0, 0))
            eff_i, rew_i, imp, buys, clks = arm
            mu_hat = (rew_i + self.prior_strength * state.global_reward_rate) / (
                eff_i + self.prior_strength
            )
            info = products.get(pid)
            records.append(
                {
                    "kind": "bandit_arm",
                    "model_version": model_version,
                    "ts": ts_iso,
                    "product_id": pid,
                    "is_new": bool(info.is_new) if info else False,
                    "eff_exposures": round(eff_i, 3),
                    "weighted_reward": round(rew_i, 3),
                    "raw_impressions": imp,
                    "raw_purchases": buys,
                    "raw_clicks": clks,
                    "eb_rate": round(mu_hat, 5),
                }
            )
        return records
