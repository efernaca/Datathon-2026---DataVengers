"""Ablación de los 4 cambios de calypso 3.1 respecto a 2.1 sobre el benchmark offline sintético.

Uso (desde calypso_3_1b/):  python etapa_4/ablation_v3.py [--seeds 42 7 123] [--out-csv ablation.csv]
No necesita RawTree: usa generate_synthetic_dataset_v4 de etapa_4/src/eval.py.
El cambio (3) SQL direccional no se puede medir aquí (el mundo sintético ya trae las transiciones);
solo se nota con datos reales.
"""
from __future__ import annotations

import argparse
import os
import sys

# El orden de iteracion de los set() de Python depende del hash -> resultados no reproducibles.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])
from dataclasses import replace
from pathlib import Path

_ETAPA4 = Path(__file__).resolve().parent
_MODEL_ROOT = _ETAPA4.parent
for p in (_MODEL_ROOT, _ETAPA4):
    sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

from etapa_4.src.config import ConfigV4  # noqa: E402
from etapa_4.src.eval import evaluate_stratified, generate_synthetic_dataset_v4  # noqa: E402
from etapa_4.src.recommender import CalypsoRecommenderV4  # noqa: E402

OFF = dict(v3_no_subcat_floor=False, v3_bg_penalty=False, v3_ease_weight=0.0, v3_cold_thresholds=False)
ON = dict(v3_no_subcat_floor=True, v3_bg_penalty=True, v3_ease_weight=0.25, v3_cold_thresholds=True)

VARIANTS: dict[str, dict] = {
    "A_2.1_equivalente": OFF,
    "B_3.1_completo": ON,
    "2.1 + sin_suelo_subcat": {**OFF, "v3_no_subcat_floor": True},
    "2.1 + penaliz_fondo": {**OFF, "v3_bg_penalty": True},
    "2.1 + (suelo+penaliz)": {**OFF, "v3_no_subcat_floor": True, "v3_bg_penalty": True},
    "2.1 + EASE_0.25": {**OFF, "v3_ease_weight": 0.25},
    "2.1 + EASE_0.10": {**OFF, "v3_ease_weight": 0.10},
    "2.1 + cold_umbrales": {**OFF, "v3_cold_thresholds": True},
    "3.1 - (suelo+penaliz)": {**ON, "v3_no_subcat_floor": False, "v3_bg_penalty": False},
    "3.1 - EASE": {**ON, "v3_ease_weight": 0.0},
    "3.1 - cold_umbrales": {**ON, "v3_cold_thresholds": False},
}

METRICS = ["hit_rate_buy@10", "dcg_buy@10", "hit_rate_click@10", "cold_item_hit@10", "SNIPS_cvr_slot1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        for name, flags in VARIANTS.items():
            cfg = replace(ConfigV4(), **flags)
            data = generate_synthetic_dataset_v4(seed=seed, config_v4=cfg)
            rec = CalypsoRecommenderV4(cfg)
            snap = data["snap_v4"]
            rep = evaluate_stratified(
                {name: lambda row, rec=rec, snap=snap: rec.recommend(row, snap)},
                data["df_test"],
                cold_product_ids=data["cold_product_ids"],
                warm_user_ids=data["warm_user_ids"],
            )
            for (seg, pol), r in rep.iterrows():
                rows.append({"seed": seed, "variant": pol, "segment": seg, **{m: r[m] for m in METRICS}})
            print(f"seed={seed} {name:28s} hit_buy(ALL)={rep.loc[('ALL', name), 'hit_rate_buy@10']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    summary = df.groupby(["segment", "variant"], sort=False)[METRICS].mean().round(4)
    print("\n=== Media sobre semillas", args.seeds, "===")
    print(summary)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
