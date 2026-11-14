"""Ablación de las reglas de negocio de calypso 4.1 sobre el benchmark offline sintético.

Uso (desde calypso_4_1/):  python etapa_4/ablation_v4.py [--seeds 42 7 123 2024 99] [--out-csv ablation_v4.csv]
No necesita RawTree: usa generate_synthetic_dataset_v4 de etapa_4/src/eval.py.

Qué se puede medir aquí y qué no:
* Carrito activo / olvidado: el mundo sintético tiene carritos pendientes y a veces el objetivo es uno de ellos.
* Orden por precio: hit_rate@10 no cambia (mismos 10 productos); lo que mide el coste es dcg_buy@10
  (sensible a la posición, como el simulador real, donde los primeros slots se miran mucho más).
* Recurrentes: en este mundo nadie recompra, así que la regla no se activa (no es medible aquí).
"""
from __future__ import annotations

import argparse
import os
import sys

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

OFF = dict(v4_cart_priority=False, v4_forgotten_cart=False, v4_price_ascending=False,
           v4_recurrent=False, v4_live_event_purchases=False)
ON = dict(v4_cart_priority=True, v4_forgotten_cart=True, v4_price_ascending=True,
          v4_recurrent=True, v4_live_event_purchases=True)

VARIANTS: dict[str, dict] = {
    "A_3.1b": OFF,
    "B_4.1_completo": ON,
    "3.1b + carrito_activo": {**OFF, "v4_cart_priority": True},
    "3.1b + carrito_olvidado": {**OFF, "v4_forgotten_cart": True},
    "3.1b + carrito_ambos": {**OFF, "v4_cart_priority": True, "v4_forgotten_cart": True},
    "3.1b + precio_asc": {**OFF, "v4_price_ascending": True},
    "3.1b + recurrentes": {**OFF, "v4_recurrent": True, "v4_live_event_purchases": True},
    "4.1 - precio_asc": {**ON, "v4_price_ascending": False},
}

METRICS = ["hit_rate_buy@10", "dcg_buy@10", "hit_rate_click@10", "cold_item_hit@10"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        base_cfg = ConfigV4()
        data = generate_synthetic_dataset_v4(seed=seed, config_v4=replace(base_cfg, **OFF))
        snap = data["snap_v4"]  # las reglas 4.1 solo actúan en recommend(): mismo snapshot para todas
        for name, flags in VARIANTS.items():
            cfg = replace(base_cfg, **flags)
            rec = CalypsoRecommenderV4(cfg)
            rep = evaluate_stratified(
                {name: lambda row, rec=rec, snap=snap: rec.recommend(row, snap)},
                data["df_test"],
                cold_product_ids=data["cold_product_ids"],
                warm_user_ids=data["warm_user_ids"],
            )
            for (seg, pol), r in rep.iterrows():
                rows.append({"seed": seed, "variant": pol, "segment": seg, **{m: r[m] for m in METRICS}})
            print(
                f"seed={seed} {name:26s} hit_buy={rep.loc[('ALL', name), 'hit_rate_buy@10']:.4f} "
                f"dcg_buy={rep.loc[('ALL', name), 'dcg_buy@10']:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    summary = df.groupby(["segment", "variant"], sort=False)[METRICS].mean().round(4)
    print("\n=== Media sobre semillas", args.seeds, "===")
    print(summary)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
