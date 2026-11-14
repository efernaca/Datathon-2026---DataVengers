"""Modular profile manager, offline preset evaluator, and iterative conversion optimizer for Calypso 6.1."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import ConfigV4
from .eval import evaluate_policy, generate_synthetic_dataset_v4
from .recommender import CalypsoRecommenderV4, build_snapshot_v4_from_records
from .reranker import PositionDebiasedRanker
from .weights_profile import (
    PRESETS,
    WeightsProfile,
    get_preset,
    list_presets,
    load_active_profile,
)

# Bounded search ranges for iterative conversion optimization: (min, max, step_scale)
OPTIMIZABLE_PARAMS: dict[str, tuple[float, float, float]] = {
    "w_seq_lift": (0.40, 2.20, 0.20),
    "w_self_cvr_view": (0.80, 2.80, 0.25),
    "w_intent_score": (0.15, 1.40, 0.15),
    "w_user_affinity": (0.05, 0.85, 0.10),
    "w_price_compat": (0.00, 0.70, 0.10),
    "w_ctx_bayes": (0.05, 0.80, 0.10),
    "w_global_pop": (0.15, 0.90, 0.10),
    "w_live_trend": (0.02, 0.75, 0.10),
    "w_reverse_graft": (0.05, 0.75, 0.08),
    "w_forward_cold": (0.10, 0.90, 0.10),
    "w_log_price_norm": (-0.35, 0.15, 0.06),
    "w_same_top_subcat": (0.00, 0.45, 0.06),
    "w_same_top_brand": (0.00, 0.25, 0.04),
    "w_ease_score": (0.05, 1.20, 0.12),
    "w_hop2_score": (0.00, 0.70, 0.08),
    "w_covisit_score": (0.00, 0.75, 0.08),
    "w_svd_session_sim": (0.00, 0.75, 0.08),
    "w_svd_user_sim": (0.00, 0.65, 0.08),
    "w_is_last_viewed": (-0.40, 0.40, 0.08),
    "w_cart_active_decay": (0.10, 1.20, 0.12),
    "w_recurrent_log": (0.05, 0.85, 0.10),
    "w_discount_ratio": (0.00, 0.45, 0.06),
    "w_same_category": (0.00, 0.45, 0.06),
    "v3_ease_weight": (0.10, 0.90, 0.10),
    "hop2_weight": (0.05, 0.55, 0.06),
    "covisit_blend_weight": (0.05, 0.65, 0.08),
    "ml_nonlin_blend_alpha": (0.00, 0.85, 0.10),
    "price_cheap_asymmetry": (0.00, 0.90, 0.15),
    "bandit_ucb_alpha": (0.04, 0.28, 0.04),
    "mmr_brand_lambda": (0.00, 0.12, 0.02),
    "sales_rank_blend_weight": (0.20, 0.95, 0.12),
    "transition_dest_pop_gamma": (0.00, 0.85, 0.12),
    "view_base_bonus": (0.00, 0.90, 0.12),
    "slot1_category_match_bonus": (0.00, 0.60, 0.08),
    "cart_active_score_bonus": (0.20, 2.20, 0.25),
    "cold_explore_start_slot_cold": (4.0, 9.0, 1.0),
    "cold_boost_on_cold_session": (0.60, 1.80, 0.15),
}


def conversion_objective(metrics_all: dict[str, float | int]) -> float:
    """Composite purchase-conversion objective prioritizing buy hit-rate, DCG, and slot-1 SNIPS CVR."""
    hr = float(metrics_all.get("hit_rate_buy@10", 0.0))
    dcg = float(metrics_all.get("dcg_buy@10", 0.0))
    snips = float(metrics_all.get("SNIPS_cvr_slot1", 0.0))
    return 0.55 * hr + 0.35 * dcg + 0.10 * snips


def parse_set_assignments(assignments: list[str], base: WeightsProfile) -> WeightsProfile:
    """Apply KEY=VALUE CLI overrides cleanly onto a WeightsProfile."""
    raw = base.to_dict()
    valid_keys = set(raw.keys())
    updates: dict[str, Any] = {}
    for expr in assignments:
        if "=" not in expr:
            raise ValueError(f"Expected KEY=VALUE format, got {expr!r}")
        key, val = expr.split("=", 1)
        key = key.strip()
        if key not in valid_keys:
            raise KeyError(f"Unknown WeightsProfile field {key!r}")
        updates[key] = val.strip()
    merged = {**raw, **updates}
    return WeightsProfile.from_dict(merged, base=base)


def evaluate_profile_on_dataset(
    syn: dict[str, Any],
    wp: WeightsProfile,
    *,
    rebuild_snapshot: bool = True,
) -> dict[str, Any]:
    """Evaluate a WeightsProfile across ALL, WARM, and COLD_START slices."""
    cfg = ConfigV4.from_profile(wp)
    if rebuild_snapshot and "v4_records" in syn:
        snap, _ = build_snapshot_v4_from_records(
            **syn["v4_records"],
            config=cfg,
            version=1,
            weights_profile=wp,
        )
    else:
        base_snap = syn["snap_v4"]
        base_ranker = base_snap.ranker
        if base_ranker is not None:
            cand_ranker = base_ranker.with_updated_priors(
                wp.prior_weights_array(),
                wp.min_prior_ratios_array(),
                wp.max_prior_multiplier,
            )
        else:
            cand_ranker = PositionDebiasedRanker.from_priors(
                prior_weights=wp.prior_weights_array(),
                min_prior_ratios=wp.min_prior_ratios_array(),
                max_prior_multiplier=wp.max_prior_multiplier,
            )
        snap = replace(
            base_snap,
            ranker=cand_ranker,
            weights_profile=wp,
            config_override=cfg,
        )

    rec = CalypsoRecommenderV4(cfg)
    df_test = syn["df_test"]
    cold_ids = syn["cold_product_ids"]
    cold_mask = df_test["purchased"].isin(cold_ids) | df_test["is_new_user"].astype(bool)

    m_all = evaluate_policy(lambda row: rec.recommend(row, snap), df_test, cold_product_ids=cold_ids)
    m_warm = evaluate_policy(
        lambda row: rec.recommend(row, snap), df_test[~cold_mask], cold_product_ids=cold_ids
    )
    m_cold = evaluate_policy(
        lambda row: rec.recommend(row, snap), df_test[cold_mask], cold_product_ids=cold_ids
    )
    return {
        "profile_name": wp.profile_name,
        "objective": conversion_objective(m_all),
        "ALL": m_all,
        "WARM": m_warm,
        "COLD_START": m_cold,
        "snapshot": snap,
    }


def optimize_profile(
    base_profile: WeightsProfile,
    syn: dict[str, Any],
    *,
    iters: int = 16,
    seed: int = 42,
    param_names: list[str] | None = None,
) -> tuple[WeightsProfile, dict[str, Any], list[dict[str, Any]]]:
    """Iterative randomized coordinate hill-climbing maximizing conversion objective."""
    rng = np.random.default_rng(seed)
    active_params = [
        p for p in (param_names or list(OPTIMIZABLE_PARAMS.keys())) if p in OPTIMIZABLE_PARAMS
    ]
    if not active_params:
        active_params = list(OPTIMIZABLE_PARAMS.keys())

    best_wp = base_profile
    best_eval = evaluate_profile_on_dataset(syn, best_wp, rebuild_snapshot=True)
    best_obj = float(best_eval["objective"])
    history: list[dict[str, Any]] = [
        {
            "iter": 0,
            "changed": "baseline",
            "objective": best_obj,
            "hit_rate_buy@10": float(best_eval["ALL"]["hit_rate_buy@10"]),
            "dcg_buy@10": float(best_eval["ALL"]["dcg_buy@10"]),
            "SNIPS_cvr_slot1": float(best_eval["ALL"]["SNIPS_cvr_slot1"]),
            "accepted": True,
        }
    ]

    for step_idx in range(1, max(iters, 0) + 1):
        # Gradually focus from 3-parameter perturbations to fine 1-parameter moves
        temp_frac = max(0.35, 1.0 - (step_idx - 1) / max(iters, 1))
        n_perturb = 3 if step_idx <= max(1, iters // 3) else (2 if step_idx <= (2 * iters) // 3 else 1)
        chosen = list(rng.choice(active_params, size=min(n_perturb, len(active_params)), replace=False))
        cand_updates: dict[str, float] = {}
        desc_parts: list[str] = []
        for pname in chosen:
            lo, hi, scale = OPTIMIZABLE_PARAMS[pname]
            cur_val = float(getattr(best_wp, pname))
            delta = float(rng.normal(0.0, scale * temp_frac))
            new_val = round(min(max(cur_val + delta, lo), hi), 4)
            cand_updates[pname] = new_val
            desc_parts.append(f"{pname}={new_val:.3f}")

        cand_wp = replace(best_wp, **cand_updates)
        cand_eval = evaluate_profile_on_dataset(syn, cand_wp, rebuild_snapshot=False)
        cand_obj = float(cand_eval["objective"])
        accepted = cand_obj > best_obj + 1e-6
        if accepted:
            best_wp = cand_wp
            best_eval = cand_eval
            best_obj = cand_obj

        history.append(
            {
                "iter": step_idx,
                "changed": ", ".join(desc_parts),
                "objective": cand_obj,
                "hit_rate_buy@10": float(cand_eval["ALL"]["hit_rate_buy@10"]),
                "dcg_buy@10": float(cand_eval["ALL"]["dcg_buy@10"]),
                "SNIPS_cvr_slot1": float(cand_eval["ALL"]["SNIPS_cvr_slot1"]),
                "accepted": accepted,
            }
        )

    # Full rebuild verification on the final best profile
    final_name = (
        base_profile.profile_name
        if base_profile.profile_name.endswith("_opt")
        else f"{base_profile.profile_name}_opt"
    )
    best_wp = replace(
        best_wp,
        profile_name=final_name,
        description=f"Optimized iteratively ({iters} iters, seed={seed}) from {base_profile.profile_name}.",
    )
    verified_eval = evaluate_profile_on_dataset(syn, best_wp, rebuild_snapshot=True)
    return best_wp, verified_eval, history


def _format_preset_eval_table(results: list[dict[str, Any]]) -> str:
    header = (
        f"{'preset':<24} {'obj':>8} {'ALL_hit@10':>11} {'ALL_dcg@10':>11} "
        f"{'WARM_hit':>10} {'COLD_hit':>10} {'SNIPS_s1':>10}"
    )
    lines = [header, "-" * len(header)]
    for res in results:
        a = res["ALL"]
        w = res["WARM"]
        c = res["COLD_START"]
        lines.append(
            f"{res['profile_name']:<24} {res['objective']:>8.5f} "
            f"{float(a['hit_rate_buy@10']):>11.5f} {float(a['dcg_buy@10']):>11.5f} "
            f"{float(w['hit_rate_buy@10']):>10.5f} {float(c['hit_rate_buy@10']):>10.5f} "
            f"{float(a['SNIPS_cvr_slot1']):>10.5f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calypso 6.1 Modular Profile Tuner & Optimizer")
    parser.add_argument("--profile-path", type=str, default="", help="Path to weights_profile.json")
    parser.add_argument("--list-presets", action="store_true", help="List available built-in presets")
    parser.add_argument("--show", action="store_true", help="Print current active weights profile JSON")
    parser.add_argument("--preset", type=str, default="", help="Apply a built-in preset to weights_profile.json")
    parser.add_argument("--set", nargs="+", default=[], help="Override profile fields with KEY=VALUE pairs")
    parser.add_argument("--eval-presets", action="store_true", help="Benchmark all built-in presets on synthetic data")
    parser.add_argument("--optimize", action="store_true", help="Run iterative conversion optimization")
    parser.add_argument("--iters", type=int, default=16, help="Optimization iterations")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic benchmark random seed")
    parser.add_argument("--n-test", type=int, default=1200, help="Number of synthetic test requests for tuning")
    parser.add_argument("--save-best", action="store_true", help="Save optimized profile to weights_profile.json")
    args = parser.parse_args()

    cfg = ConfigV4(weights_profile_path=args.profile_path or ConfigV4().weights_profile_path)
    target_path = cfg.resolve_weights_profile_path()

    if args.list_presets:
        print("=== Calypso 6.1 Built-in Presets ===")
        for name in list_presets():
            p = PRESETS[name]
            print(f"- {name:<22}: {p.description}")
        return

    if args.preset or args.set:
        base_wp = get_preset(args.preset) if args.preset else load_active_profile(target_path)
        updated_wp = parse_set_assignments(args.set, base_wp) if args.set else base_wp
        if not args.optimize:
            saved = updated_wp.save_json(target_path)
            print(
                f"[TUNER] Saved profile '{updated_wp.profile_name}' -> {saved} "
                "(live server will hot-reload automatically)"
            )
            return
    else:
        updated_wp = load_active_profile(target_path)

    if args.show:
        print(json.dumps(updated_wp.to_dict(), indent=2, ensure_ascii=False))
        return

    if args.eval_presets:
        t0 = time.perf_counter()
        syn = generate_synthetic_dataset_v4(
            seed=args.seed,
            n_warm_products=400,
            n_new_products=35,
            n_warm_users=900,
            n_history_sessions=3000,
            n_feedback_sessions=800,
            n_test_sessions=args.n_test,
        )
        results = [
            evaluate_profile_on_dataset(syn, get_preset(name), rebuild_snapshot=True)
            for name in list_presets()
        ]
        results.sort(key=lambda r: float(r["objective"]), reverse=True)
        print(f"=== Preset Evaluation (seed={args.seed}, n_test={args.n_test}, {time.perf_counter() - t0:.2f}s) ===")
        print(_format_preset_eval_table(results))
        return

    if args.optimize:
        t0 = time.perf_counter()
        syn = generate_synthetic_dataset_v4(
            seed=args.seed,
            n_warm_products=400,
            n_new_products=35,
            n_warm_users=900,
            n_history_sessions=3000,
            n_feedback_sessions=800,
            n_test_sessions=args.n_test,
        )
        best_wp, verified_eval, history = optimize_profile(
            updated_wp,
            syn,
            iters=args.iters,
            seed=args.seed,
        )
        accepted_steps = [h for h in history if h["accepted"]]
        print(
            f"=== Iterative Conversion Optimization ({args.iters} iters, {time.perf_counter() - t0:.2f}s) ==="
        )
        print(
            f"Baseline: obj={history[0]['objective']:.5f} | "
            f"hit@10={history[0]['hit_rate_buy@10']:.5f} | dcg@10={history[0]['dcg_buy@10']:.5f}"
        )
        print(
            f"Best    : obj={verified_eval['objective']:.5f} | "
            f"hit@10={float(verified_eval['ALL']['hit_rate_buy@10']):.5f} | "
            f"dcg@10={float(verified_eval['ALL']['dcg_buy@10']):.5f} | "
            f"accepted moves={len(accepted_steps) - 1}"
        )
        if args.save_best:
            saved = best_wp.save_json(target_path)
            print(f"[TUNER] Saved optimized profile '{best_wp.profile_name}' -> {saved}")
        return

    # Default action: show brief active profile summary
    print(
        f"Active profile: {updated_wp.profile_name} ({target_path})\n"
        f"Description   : {updated_wp.description}\n"
        "Use --show, --list-presets, --preset NAME, --set K=V, --eval-presets, or --optimize."
    )


if __name__ == "__main__":
    main()
