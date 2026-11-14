"""Análisis de conversión sobre el log real (offline_eval_cache).

Reproduce las tablas de MEDIDAS_CONVERSION.md a partir de la muestra de shop.history_sessions
que descarga run_offline_eval_official.py. No accede a RawTree.

    python calypso_4_1/analisis_conversion.py                 # usa ../offline_eval_cache
    python calypso_4_1/analisis_conversion.py --cache RUTA

Requiere pandas, pyarrow, numpy y scikit-learn.

Qué calcula:
  1. Tasas globales y por página.
  2. Conversión por impresión según la relación del producto con la sesión
     (visto / co-visitado / sin relación) y su puesto en el ranking de ventas.
  3. Modelo logístico por impresión: compra ~ slot + log(puesto de ventas) + visto + co-visitado + home.
     Da la curva de posición y el valor relativo de cada señal.
  4. Calibración a nivel de lista (¿se suman las probabilidades de los 10 productos?).
  5. Comparación de políticas simples bajo ese modelo (estimación, no medida real).
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
POP_EDGES = [-1, 9, 29, 99, 299, 999, 10**9]
POP_LABELS = ["1-10", "11-30", "31-100", "101-300", "301-1000", "1000+"]
MAX_RANK = 3000


def load(cache: Path):
    df = pd.read_parquet(cache / "official_test_data.parquet")
    cov = pd.read_parquet(cache / "covisit.parquet")
    best = json.loads((cache / "best_sellers.json").read_text(encoding="utf-8"))
    for col in ("clicked", "purchased"):
        df[col] = df[col].where(df[col].notna() & (df[col] != ""), None)
    df["page"] = df.context.str["page"]
    covd = cov.groupby("p")["q"].apply(list).to_dict()
    rank = {p: i for i, p in enumerate(best)}
    return df, covd, best, rank


def explode(df: pd.DataFrame, covd: dict, rank: dict) -> pd.DataFrame:
    """Una fila por (petición, slot)."""
    rows = []
    for r in df.itertuples(index=False):
        rv = [v["product_id"] for v in r.recent_views] if r.recent_views is not None else []
        rvset = set(rv)
        nb: set[str] = set()
        for v in rv:
            nb.update(covd.get(v, [])[:20])
        for k, item in enumerate(r.shown):
            in_rv = int(item in rvset)
            rows.append(
                (r.request_id, k + 1, r.page, rank.get(item, 99999), in_rv,
                 int(item in nb and not in_rv), int(r.clicked == item), int(r.purchased == item))
            )
    L = pd.DataFrame(rows, columns=["rid", "slot", "page", "poprank", "visto", "covisitado", "click", "compra"])
    L["pop"] = pd.cut(L.poprank, POP_EDGES, labels=POP_LABELS)
    L["relacion"] = np.where(L.visto == 1, "visto", np.where(L.covisitado == 1, "co-visitado", "sin relacion"))
    L["home"] = (L.page == "home").astype(int)
    L["log_rank"] = np.log(L.poprank.clip(upper=MAX_RANK) + 1.0)
    return L


def rate_table(g) -> pd.DataFrame:
    t = g.agg(impresiones=("compra", "size"), clics=("click", "sum"), compras=("compra", "sum"))
    t["ctr_%"] = 100 * t.clics / t.impresiones
    t["cvr_%"] = 100 * t.compras / t.impresiones
    return t.round(3)


def design(L: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(L.slot.astype(str)).drop(columns=["1"]).astype(float)
    X = X[[str(k) for k in range(2, 11)]]
    X["log_rank"] = L["log_rank"].values
    X["visto"] = L["visto"].values
    X["covisitado"] = L["covisitado"].values
    X["home"] = L["home"].values
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parents[1] / "offline_eval_cache"))
    args = ap.parse_args()
    df, covd, best, rank = load(Path(args.cache))
    L = explode(df, covd, rank)
    n = len(df)

    print(f"\n== 1. Global ({n} peticiones) ==")
    n_click, n_buy = df.clicked.notna().sum(), df.purchased.notna().sum()
    print(f"CTR lista {100*n_click/n:.2f}%  CVR lista {100*n_buy/n:.2f}%  compra|clic {100*n_buy/n_click:.1f}%")
    print("compras sin clic en ese producto:", int((df.purchased.notna() & (df.purchased != df.clicked)).sum()))
    pg = df.groupby("page").agg(
        peticiones=("clicked", "size"),
        ctr=("clicked", lambda s: 100 * s.notna().mean()),
        cvr=("purchased", lambda s: 100 * s.notna().mean()),
        vistas_medias=("recent_views", lambda s: s.str.len().mean()),
    )
    pg["%_compras"] = 100 * df.groupby("page").purchased.apply(lambda s: s.notna().sum()) / n_buy
    print(pg.round(2))

    print("\n== 2. Conversión por impresión: relación con la sesión x puesto de ventas (páginas con sesión) ==")
    S = L[L.home == 0]
    print(rate_table(S.groupby(["relacion", "pop"], observed=True)))
    print("\nPor relación (todas las páginas):")
    print(rate_table(L.groupby("relacion")))
    print("\nReparto de clics y compras por slot (%):")
    print(pd.DataFrame({
        "clics": 100 * L[L.click == 1].slot.value_counts(normalize=True).sort_index(),
        "compras": 100 * L[L.compra == 1].slot.value_counts(normalize=True).sort_index(),
    }).round(1).T)

    print("\n== 3. Modelo logístico por impresión (compra) ==")
    X = design(L)
    model = LogisticRegression(C=1e4, max_iter=3000).fit(X, L.compra)
    coef = pd.Series(model.coef_[0], index=X.columns)
    print("Curva de posición (odds relativas al slot 1):")
    print({1: 1.0, **{int(k): round(float(np.exp(coef[k])), 2) for k in X.columns[:9]}})
    print(f"log(puesto de ventas+1): {coef['log_rank']:.3f}   visto: {coef['visto']:.3f} (x{np.exp(coef['visto']):.1f})"
          f"   co-visitado: {coef['covisitado']:.3f} (x{np.exp(coef['covisitado']):.1f})   home: {coef['home']:.3f}")
    eq_v = np.exp(-coef["visto"] / coef["log_rank"])
    eq_c = np.exp(-coef["covisitado"] / coef["log_rank"])
    print(f"Equivalencia: 'visto' vale dividir el puesto de ventas entre {eq_v:.0f}; 'co-visitado', entre {eq_c:.1f}")

    print("\n== 4. Calibración a nivel de lista (validación cruzada por petición) ==")
    codes = L.rid.astype("category").cat.codes.values
    pred = np.zeros(len(L))
    for fold in (0, 1):
        tr, te = codes % 2 != fold, codes % 2 == fold
        m = LogisticRegression(C=1e4, max_iter=3000).fit(X[tr], L.compra[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    R = L.assign(pred=pred).groupby("rid").agg(suma_pred=("pred", "sum"), compra=("compra", "max"))
    R["decil"] = pd.qcut(R.suma_pred, 10, labels=False, duplicates="drop")
    print((100 * R.groupby("decil").agg(pred=("suma_pred", "mean"), real=("compra", "mean"))).round(2).T)

    print("\n== 5. Políticas simples bajo el modelo (estimación; relativo al baseline) ==")
    slot_c = np.array([0.0] + [coef[str(k)] for k in range(2, 11)])
    b0 = float(model.intercept_[0])

    def logit(item, rvset, nb, home):
        return (b0 + coef["log_rank"] * np.log(min(rank.get(item, 99999), MAX_RANK) + 1.0)
                + coef["visto"] * (item in rvset)
                + coef["covisitado"] * (item in nb and item not in rvset) + coef["home"] * home)

    def value(items, rvset, nb, home):
        z = np.array([logit(i, rvset, nb, home) for i in items]) + slot_c[: len(items)]
        return float((1 / (1 + np.exp(-z))).sum())

    rng = np.random.default_rng(1)
    res: dict[str, list[float]] = {}
    seg = []
    for r in df.itertuples(index=False):
        home = int(r.page == "home")
        rv = [v["product_id"] for v in r.recent_views]
        rvset, nb, sc = set(rv), set(), Counter()
        for k, v in enumerate(reversed(rv)):
            for j, q in enumerate(covd.get(v, [])[:20]):
                nb.add(q)
                sc[q] += 0.7 ** k / (j + 1)
        cands = list(dict.fromkeys(rv + list(nb) + best[:40]))
        s = {i: logit(i, rvset, nb, home) for i in cands}
        base = best[:10]
        coview = [q for q, _ in sc.most_common(10)]
        coview = (coview + [p for p in base if p not in coview])[:10]
        calibrada = sorted(cands, key=lambda i: -s[i])[:10]
        barajada = list(calibrada)
        rng.shuffle(barajada)
        vistos_primero = list(dict.fromkeys(sorted(rvset, key=lambda i: rank.get(i, 99999)) + base))[:10]
        for name, items in (
            ("baseline: top-10 ventas", base),
            ("co-visitas primero (starter)", coview),
            ("vistos primero + top ventas", vistos_primero),
            ("puntuación calibrada", calibrada),
            ("puntuación calibrada, orden al azar", barajada),
        ):
            res.setdefault(name, []).append(value(items, rvset, nb, home))
        seg.append("home" if home else "con sesión")
    P = pd.DataFrame(res)
    P["seg"] = seg
    out = (100 * P.groupby("seg").mean(numeric_only=True)).T
    out["total"] = 100 * P.drop(columns="seg").mean()
    print("Compras esperadas por 100 peticiones:")
    print(out.round(2))
    print("\nRelativo al baseline:")
    print((out / out.iloc[0]).round(2))


if __name__ == "__main__":
    main()
