"""Offline 3-Way Comparison for Calypso 6.2b custom weights profiles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent          # calypso_6_2_b
_SRC = _ROOT / "etapa_4" / "src"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))

from etapa_4.src.config import ConfigV4
from etapa_4.src.recommender import CalypsoRecommenderV4, build_snapshot_v4_from_records, extract_recent_product_ids
from etapa_4.src.weights_profile import WeightsProfile
from etapa_4.src.eval import (
    evaluate_policy,
    is_cold_start_row,
)

# ── Profiles to compare ─────────────────────────────────────────────────
PROFILE_FILES = [
    "weights_6.2.b_conf_inicial.json",
    "weights_6.2.b_radical1.json",
    "weights_6.2.b_radical2.json",
]

# ── Key metrics to extract ──────────────────────────────────────────────
METRICS = [
    "SNIPS_cvr_slot1",
    "SNIPS_ctr_slot1",
    "SNIPS_rev_slot1",
    "hit_rate_buy@10",
    "hit_rate_click@10",
    "dcg_buy@10",
    "cold_item_hit@10",
    "slot1_matches",
]

def _run_single(profile_path: Path, seed: int = 42):
    """Load profile → build snapshot → evaluate on ALL / WARM / COLD_START → return dict."""
    wp = WeightsProfile.load_json(str(profile_path))
    cfg_v4 = ConfigV4.from_profile(wp)

    # Reuse eval.py's synthetic dataset generator
    from etapa_4.src.eval import generate_synthetic_dataset_v4
    data = generate_synthetic_dataset_v4(
        seed=seed, config_v4=cfg_v4, weights_profile=wp
    )

    snap_v4 = data["snap_v4"]
    df_test = data["df_test"]
    cold_ids = data["cold_product_ids"]
    warm_ids = data["warm_user_ids"]

    rec = CalypsoRecommenderV4(cfg_v4)
    policy_fn = lambda row: rec.recommend(row, snap_v4)

    # Split into segments
    cold_mask = [
        is_cold_start_row(row, cold_ids, warm_ids)
        for row in df_test.itertuples(index=False)
    ]
    segments = {
        "ALL": df_test,
        "WARM": df_test[~pd.Series(cold_mask)],
        "COLD_START": df_test[pd.Series(cold_mask)],
    }

    result = {"profile_name": profile_path.name.replace("weights_6.2.b_", "").replace(".json", "")}
    for seg_name, seg_df in segments.items():
        metrics = evaluate_policy(policy_fn, seg_df, cold_product_ids=cold_ids)
        for m in METRICS:
            key = f"{seg_name}_{m}"
            result[key] = metrics.get(m, 0.0)

    return result

def main() -> None:
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 220)

    results: list[dict] = []

    for fname in PROFILE_FILES:
        fpath = _ROOT / fname
        if not fpath.exists():
            print(f"⚠️  Missing {fname}, skipping.")
            continue
        print(f"\n{'─'*60}")
        print(f"🔬 Evaluating: {fname}")
        print(f"{'─'*60}")

        row = _run_single(fpath)
        results.append(row)

    if not results:
        print("❌ No results generated.")
        return

    df = pd.DataFrame(results)
    cols_order = ["profile_name"]
    for seg in ["ALL", "WARM", "COLD_START"]:
        for m in METRICS:
            cols_order.append(f"{seg}_{m}")
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order]

    print(f"\n\n📊 ═══════════════════════════════════════════════════════════════════")
    print(f"   COMPARATIVA OFFLINE EVAL · CALYPSO 6.2b")
    print(f"{'═'*74}\n")
    print(df.round(5).to_string(index=False))
    print(f"\n{'═'*74}\n")

    # ── Quick winner highlight ──────────────────────────────────────────
    best_cvr = df.loc[df["ALL_SNIPS_cvr_slot1"].idxmax(), "profile_name"]
    best_rev = df.loc[df["ALL_SNIPS_rev_slot1"].idxmax(), "profile_name"]
    best_cold = df.loc[df["ALL_cold_item_hit@10"].idxmax(), "profile_name"]
    best_ctr = df.loc[df["ALL_SNIPS_ctr_slot1"].idxmax(), "profile_name"]

    print("🏆  MILLORS PER MÈTRICA (ALL):")
    print(f"   CVR (SNIPS_cvr_slot1)  → {best_cvr}  ({df[df['profile_name']==best_cvr]['ALL_SNIPS_cvr_slot1'].values[0]:.5f})")
    print(f"   CTR (SNIPS_ctr_slot1)  → {best_ctr}  ({df[df['profile_name']==best_ctr]['ALL_SNIPS_ctr_slot1'].values[0]:.5f})")
    print(f"   Revenue (SNIPS_rev_slot1) → {best_rev}  ({df[df['profile_name']==best_rev]['ALL_SNIPS_rev_slot1'].values[0]:.5f})")
    print(f"   Cold Start             → {best_cold}  ({df[df['profile_name']==best_cold]['ALL_cold_item_hit@10'].values[0]:.5f})")
    print()

if __name__ == "__main__":
    main()