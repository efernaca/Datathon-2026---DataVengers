"""Odyseus 2.0 · paso 2: entrenamiento del modelo residual (NumPy puro, sin PyTorch).

    python ml_train.py                 # entrena, compara y escribe ml_data/train_report.md
    python ml_train.py --export        # además escribe ml_model.json SI el mejor modelo pasa la puerta

Modelo:  logit(compra) = b + s·prior + f(x)      (prior = score_p de Calypso, s y b aprendidos)
Escalones que se comparan (todos con el mismo offset):
    prior      solo b, s y la posición                      (referencia: Calypso recalibrado)
    lineal     f lineal sobre las 34 variables              (35 parámetros)
    mlp16      34 -> 16 -> 1
    mlp32x16   34 -> 32 -> 16 -> 1
Cabeza auxiliar de clic con peso 0,1. Parada temprana con el bloque de validación.

Métrica de decisión: entre las 10 opciones mostradas en cada petición con compra, ¿en qué puesto
coloca el modelo el producto comprado? (MRR, acierto en el 1.º y en los 3 primeros). Se calcula con
la posición fijada en 0, como al servir, para que el modelo no pueda «copiar» el slot del log.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[2]
CLICK_W = 0.1
FROZEN_FEATURES = ("live_trend_norm",)
LADDER: dict[str, list[int]] = {"lineal": [], "mlp16": [16], "mlp32x16": [32, 16]}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class Net:
    """MLP con offset del prior: logits = MLP(Z) + prior·s. Salidas: [compra, clic]."""

    def __init__(self, n_in: int, hidden: list[int], rng: np.random.Generator, base_logit: np.ndarray, cols: np.ndarray | None = None) -> None:
        self.cols = cols  # subconjunto de variables (modelo «prior»: solo la posición)
        sizes = [n_in if cols is None else len(cols)] + hidden + [2]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            scale = 0.0 if last else np.sqrt(2.0 / sizes[i])
            self.W.append(rng.normal(0.0, 1.0, (sizes[i], sizes[i + 1])) * scale)
            self.b.append(base_logit.copy() if last else np.zeros(sizes[i + 1]))
        self.s = np.ones(2)
        self._adam: dict[str, Any] = {}

    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)) + self.s.size)

    def forward(self, Z: np.ndarray, prior: np.ndarray, cache: bool = False):
        h = Z if self.cols is None else Z[:, self.cols]
        acts = [h]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        logits = h + prior[:, None] * self.s[None, :]
        return (logits, acts) if cache else logits

    def step(self, Z, prior, Y, w, lr: float, wd: float) -> float:
        logits, acts = self.forward(Z, prior, cache=True)
        p = _sigmoid(logits)
        wsum = float(w.sum())
        task_w = np.array([1.0, CLICK_W])
        eps = 1e-9
        loss = float((w[:, None] * task_w * -(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))).sum() / wsum)
        d = (w[:, None] * task_w * (p - Y)) / wsum
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        g_s = (d * prior[:, None]).sum(axis=0)
        for i in range(len(self.W) - 1, -1, -1):
            grads_W[i] = acts[i].T @ d + wd * self.W[i]
            grads_b[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0.0)
        params = self.W + self.b + [self.s]
        grads = grads_W + grads_b + [g_s]
        st = self._adam
        if not st:
            st["m"] = [np.zeros_like(x) for x in params]
            st["v"] = [np.zeros_like(x) for x in params]
            st["t"] = 0
        st["t"] += 1
        b1, b2, t = 0.9, 0.999, st["t"]
        for k, (x, g) in enumerate(zip(params, grads)):
            st["m"][k] = b1 * st["m"][k] + (1 - b1) * g
            st["v"][k] = b2 * st["v"][k] + (1 - b2) * g * g
            x -= lr * (st["m"][k] / (1 - b1**t)) / (np.sqrt(st["v"][k] / (1 - b2**t)) + 1e-8)
        return loss

    def snapshot(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b], self.s.copy()

    def restore(self, snap) -> None:
        self.W, self.b, self.s = [w.copy() for w in snap[0]], [b.copy() for b in snap[1]], snap[2].copy()


def wlogloss(net: Net, Z, prior, y, w) -> float:
    p = _sigmoid(net.forward(Z, prior)[:, 0])
    eps = 1e-9
    return float((w * -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() / w.sum())


def rank_metrics(score: np.ndarray, y_buy: np.ndarray, mask_g: np.ndarray | None = None) -> dict[str, float]:
    """score, y_buy: (n_peticiones, 10). Puesto del producto comprado entre los 10 mostrados."""
    has = y_buy.sum(axis=1) > 0
    if mask_g is not None:
        has &= mask_g
    n = int(has.sum())
    if n == 0:
        return {"n": 0, "mrr": float("nan"), "hit1": float("nan"), "hit3": float("nan")}
    S, Y = score[has], y_buy[has]
    s_pos = (S * Y).sum(axis=1, keepdims=True)
    # empates: puesto medio (un modelo constante no gana por orden de columnas)
    rank = 1.0 + (S > s_pos).sum(axis=1) + 0.5 * ((S == s_pos).sum(axis=1) - 1)
    return {"n": n, "mrr": float((1.0 / rank).mean()), "hit1": float((rank <= 1.0).mean()), "hit3": float((rank <= 3.0).mean())}


def train_one(hidden, cols, seed, D, args) -> tuple[Net, float]:
    rng = np.random.default_rng(seed)
    tr, va = D["tr"], D["va"]
    net = Net(D["Z"].shape[1], hidden, rng, D["base_logit"], cols=cols)
    Ztr, ptr, Ytr, wtr = D["Z"][tr], D["prior"][tr], D["Y"][tr], D["w"][tr]
    Zva, pva, yva, wva = D["Z"][va], D["prior"][va], D["Y"][va][:, 0], D["w"][va]
    n = len(Ztr)
    best, best_snap, bad = float("inf"), net.snapshot(), 0
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            net.step(Ztr[idx], ptr[idx], Ytr[idx], wtr[idx], args.lr, args.wd)
        v = wlogloss(net, Zva, pva, yva, wva)
        if v < best - 1e-6:
            best, best_snap, bad = v, net.snapshot(), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.restore(best_snap)
    return net, best


def ensemble_scores(nets: list[Net], Z: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(score de ranking, logit de compra medio). Ranking = s·prior + f, promediado entre semillas."""
    lg = np.mean([n.forward(Z, prior)[:, 0] for n in nets], axis=0)
    return lg, lg


def main() -> None:
    ap = argparse.ArgumentParser(description="Odyseus 2.0 · entrenamiento del modelo residual")
    ap.add_argument("--data", default="ml_data/dataset.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ips-power", type=float, default=0.5, help="0 = sin IPS; 1 = IPS completo (recortado a [0.2, 5])")
    ap.add_argument("--models", default="lineal,mlp16,mlp32x16")
    ap.add_argument("--export", action="store_true", help="Escribe ml_model.json si el mejor modelo pasa la puerta")
    ap.add_argument("--export-model", default="", help="Fuerza qué escalón exportar (lineal|mlp16|mlp32x16)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--freeze", default="", help="Variables (separadas por comas) que la red no debe usar, además de las fijas")
    ap.add_argument("--model-out", default="ml_model.json", help="Ruta de salida del modelo (relativa a odyseus_3_0/)")
    args = ap.parse_args()

    path = (MODEL_ROOT / args.data).resolve()
    z = np.load(path, allow_pickle=False)
    base_model = str(z["base_model"]) if "base_model" in z.files else "calypso-4.1e"
    if base_model != "calypso-5.1":
        raise SystemExit(
            f"Este dataset se generó con el recomendador {base_model}, no con Calypso 5.1.\n"
            "Relanza `python ml_build_dataset.py` (no vuelve a descargar nada: usa ml_data/raw y ml_data/asof)."
        )
    names = [str(x) for x in z["feature_names"]]
    X = z["X"].astype(np.float64)
    prior = z["prior"].astype(np.float64)
    group = z["group"]
    split_row = z["g_split"][group]
    tr, va, te = split_row == 0, split_row == 1, split_row == 2
    if tr.sum() == 0 or va.sum() == 0:
        raise SystemExit("El dataset no tiene filas de train y val (¿es la prueba de humo?).")
    slot_col = names.index("slot")

    # pesos: deshacer el muestreo de fondo x IPS suavizado (el log favorece a los más vendidos)
    w = z["g_weight"][group].astype(np.float64)
    if args.ips_power > 0:
        prop = np.maximum(z["propensity"].astype(np.float64), 1e-4)
        w *= np.clip((np.median(prop[tr]) / prop) ** args.ips_power, 0.2, 5.0)
    w /= w[tr].mean()

    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0)
    dead = std < 1e-8
    std[dead] = 1.0
    mean[slot_col] = 0.0  # la posición no se centra: al servir vale 0 y debe significar «slot 1»
    std[slot_col] = 1.0
    Z = (X - mean) / std
    lo, hi = Z[tr].min(axis=0), Z[tr].max(axis=0)
    # Variables que la red NO usa (siguen contando dentro del prior): la tendencia en vivo no se puede
    # reproducir «a fecha de» (en vivo son los últimos 60 min; aquí, las 24 h previas al corte).
    for frozen in tuple(FROZEN_FEATURES) + tuple(x.strip() for x in args.freeze.split(",") if x.strip()):
        if frozen in names:
            lo[names.index(frozen)] = hi[names.index(frozen)] = 0.0
    Z = np.clip(Z, lo, hi)
    Y = np.stack([z["y_buy"], z["y_click"]], axis=1).astype(np.float64)
    rate = (w[tr, None] * Y[tr]).sum(axis=0) / w[tr].sum()
    D = {"Z": Z, "prior": prior, "Y": Y, "w": w, "tr": tr, "va": va, "base_logit": np.log(rate / (1 - rate))}

    n_buy_tr = int(Y[tr][:, 0].sum())
    print(f"Dataset: {len(X):,} filas | train {int(tr.sum()):,} ({n_buy_tr:,} compras) | "
          f"val {int(va.sum()):,} ({int(Y[va][:, 0].sum()):,}) | test {int(te.sum()):,} ({int(Y[te][:, 0].sum()):,})")
    print(f"Techo orientativo de parámetros (15 compras/parámetro): {n_buy_tr // 15:,}\n")

    # versión «como al servir»: posición = 0
    Z0 = Z.copy()
    Z0[:, slot_col] = 0.0
    yb10 = z["y_buy"].reshape(-1, 10)
    g_split = z["g_split"]
    g_has_profile = z["g_has_profile"].astype(bool)
    g_has_views = z["g_has_views"].astype(bool)

    def evaluate(score_rows: np.ndarray) -> dict[str, dict[str, float]]:
        S = score_rows.reshape(-1, 10)
        out = {}
        for label, sp in (("val", 1), ("test", 2)):
            gm = g_split == sp
            out[label] = rank_metrics(S, yb10, gm)
            out[label + "_sin_perfil"] = rank_metrics(S, yb10, gm & ~g_has_profile)
            out[label + "_sin_sesion"] = rank_metrics(S, yb10, gm & ~g_has_views)
        return out

    results: dict[str, dict[str, Any]] = {}
    results["slot_del_log"] = {"params": 0, "rank": evaluate(-np.tile(np.arange(10.0), len(yb10))), "val_logloss": float("nan"), "test_logloss": float("nan")}
    results["prior_crudo"] = {"params": 0, "rank": evaluate(prior), "val_logloss": float("nan"), "test_logloss": float("nan")}

    trained: dict[str, list[Net]] = {}
    plan = [("prior", [], np.array([slot_col]))] + [(m, LADDER[m], None) for m in args.models.split(",") if m in LADDER]
    for name, hidden, cols in plan:
        t0 = time.perf_counter()
        nets = [train_one(hidden, cols, 1000 + s, D, args)[0] for s in range(1 if name == "prior" else args.seeds)]
        trained[name] = nets
        lg0 = np.mean([n.forward(Z0, prior)[:, 0] for n in nets], axis=0)   # como al servir
        lg = np.mean([n.forward(Z, prior)[:, 0] for n in nets], axis=0)     # con la posición real
        eps = 1e-9
        ll = {}
        for label, m in (("val", va), ("test", te)):
            p = _sigmoid(lg[m]); y = Y[m][:, 0]; ww = w[m]
            ll[label] = float((ww * -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() / ww.sum()) if m.any() else float("nan")
        results[name] = {"params": nets[0].n_params(), "rank": evaluate(lg0), "val_logloss": ll["val"], "test_logloss": ll["test"],
                         "prior_scale": float(np.mean([n.s[0] for n in nets]))}
        print(f"[{name}] {nets[0].n_params()} parámetros x {len(nets)} semillas en {time.perf_counter() - t0:.0f}s | s={results[name]['prior_scale']:.3f}")

    # ── tabla ────────────────────────────────────────────────────────────────────────────
    def row(name: str) -> str:
        r = results[name]; v, t = r["rank"]["val"], r["rank"]["test"]
        return (f"| {name} | {r['params']} | {v['mrr']:.4f} | {v['hit1']:.3f} | {t['mrr']:.4f} | {t['hit1']:.3f} | {t['hit3']:.3f} | "
                f"{r['rank']['test_sin_perfil']['mrr']:.4f} | {r['rank']['test_sin_sesion']['mrr']:.4f} | {r['val_logloss']:.5f} | {r['test_logloss']:.5f} |")
    header = ("| modelo | parámetros | MRR val | acierto@1 val | MRR test | acierto@1 test | acierto@3 test | MRR test sin perfil | MRR test sin sesión | logloss val | logloss test |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    table = "\n".join([header] + [row(n) for n in results])
    print("\n" + table)
    nv, nt = results["prior"]["rank"]["val"]["n"], results["prior"]["rank"]["test"]["n"]
    se_v, se_t = 0.29 / max(nv, 1) ** 0.5, 0.29 / max(nt, 1) ** 0.5
    print(f"\nPeticiones con compra: val {nv:,} | test {nt:,}. Error típico aproximado del MRR: ±{se_v:.4f} (val), ±{se_t:.4f} (test).")
    print("'slot_del_log' es el orden del sistema antiguo: sale alto por el sesgo de posición y no es alcanzable; no es la referencia.")

    # ── puerta ───────────────────────────────────────────────────────────────────────────
    cands = [n for n in trained if n != "prior"]
    best = args.export_model or max(cands, key=lambda n: results[n]["rank"]["val"]["mrr"])
    b, p0 = results[best], results["prior"]
    gate = {
        "MRR val > prior": b["rank"]["val"]["mrr"] > p0["rank"]["val"]["mrr"] + se_v,
        "MRR test > prior": b["rank"]["test"]["mrr"] > p0["rank"]["test"]["mrr"],
        "logloss val < prior": b["val_logloss"] < p0["val_logloss"],
        "no pierde sin perfil (test)": not (b["rank"]["test_sin_perfil"]["mrr"] < p0["rank"]["test_sin_perfil"]["mrr"] - 2 * 0.29 / max(b["rank"]["test_sin_perfil"]["n"], 1) ** 0.5),
    }
    passed = all(gate.values())
    print(f"\nMejor escalón por MRR de validación: {best}")
    for k, v in gate.items():
        print(f"  [{'OK' if v else 'NO'}] {k}")
    print(f"PUERTA OFFLINE: {'SUPERADA' if passed else 'NO SUPERADA'}")

    # ── importancia por permutación (test, modelo elegido) ───────────────────────────────
    nets = trained[best]
    rng = np.random.default_rng(7)
    base_mrr = b["rank"]["test"]["mrr"]
    te_rows = np.where(te)[0]
    imp = []
    for j, n in enumerate(names):
        if n == "slot":
            continue
        Zp = Z0.copy()
        Zp[te_rows, j] = Z0[rng.permutation(te_rows), j]
        lgp = np.mean([nn.forward(Zp, prior)[:, 0] for nn in nets], axis=0)
        imp.append((n, base_mrr - rank_metrics(lgp.reshape(-1, 10), yb10, g_split == 2)["mrr"]))
    imp.sort(key=lambda kv: -kv[1])

    # ── pesos equivalentes: regresión lineal de f(x) sobre las variables estandarizadas ──
    f_te = np.mean([nn.forward(Z0[te], np.zeros(int(te.sum())))[:, 0] for nn in nets], axis=0)
    A = np.column_stack([Z0[te], np.ones(int(te.sum()))])
    coef, *_ = np.linalg.lstsq(A, f_te, rcond=None)
    s_mean = float(np.mean([nn.s[0] for nn in nets]))
    equiv = sorted(((n, float(coef[j] / s_mean)) for j, n in enumerate(names) if n != "slot"), key=lambda kv: -abs(kv[1]))

    report = [
        "# Odyseus 2.0 · informe de entrenamiento", "",
        f"Generado: {datetime.now(timezone.utc).isoformat()} · dataset: `{path.name}` · ips_power={args.ips_power} · semillas={args.seeds}", "",
        f"Train: {n_buy_tr:,} compras. Peticiones con compra: val {nv:,}, test {nt:,}. Error típico del MRR: ±{se_v:.4f} / ±{se_t:.4f}.", "",
        table, "",
        f"**Mejor escalón:** `{best}` · **Puerta offline:** {'SUPERADA' if passed else 'NO SUPERADA'}", "",
        *[f"- [{'x' if v else ' '}] {k}" for k, v in gate.items()], "",
        "## Importancia por permutación (caída de MRR en test al barajar la variable)", "",
        "| variable | caída de MRR |", "|---|---|", *[f"| {n} | {d:+.4f} |" for n, d in imp[:15]], "",
        "## Pesos equivalentes del residuo", "",
        "Aproximación lineal de f(x)/s por desviación típica de cada variable, en unidades del score de Calypso. "
        "Positivo = la red sube esa señal respecto al prior; negativo = la baja.", "",
        "| variable | corrección por 1 σ |", "|---|---|", *[f"| {n} | {c:+.4f} |" for n, c in equiv[:15]], "",
    ]
    rep_path = path.parent / "train_report.md"
    rep_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nInforme: {rep_path}")

    if args.export:
        if not passed and not args.export_model:
            print("No se exporta ml_model.json: la puerta offline no se ha superado (usa --export-model para forzar).")
            return
        if s_mean <= 1e-6:
            print("No se exporta: la escala del prior aprendida no es positiva.")
            return
        payload = {
            "name": f"odyseus-3.0-{best}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_names": names,
            "mean": mean.tolist(), "std": std.tolist(), "clip_lo": lo.tolist(), "clip_hi": hi.tolist(),
            "prior_scale": s_mean,
            "alpha": args.alpha,
            "nets": [[{"W": W.tolist(), "b": bb.tolist()} for W, bb in zip(nn.W, nn.b)] for nn in nets],
            "metrics": {k: results[k] for k in ("prior", best)},
            "gate_passed": bool(passed),
        }
        out = (MODEL_ROOT / args.model_out).resolve()
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(out)
        print(f"Modelo exportado: {out} (serve.py lo recarga en caliente en la rama B)")


if __name__ == "__main__":
    main()
