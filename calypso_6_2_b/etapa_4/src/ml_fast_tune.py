"""Odyseus 2.0 · ajuste rápido de pesos de Calypso 5.1 sobre el dataset «a fecha de» (offline, sin API).

    python ml_fast_tune.py --minutes 10

Qué hace distinto del optimizador de Odyseus:
  * No llama al recomendador en cada prueba. Las 14 señales de cada (petición, producto) ya están
    calculadas en ml_data/dataset.npz, así que una prueba = unas pocas operaciones NumPy sobre las
    peticiones con compra. Resultado: miles de pruebas por minuto en vez de unas pocas.
  * Sin fuga: las señales de cada bloque solo han visto datos anteriores al bloque.
  * Entrena con los bloques de train, elige con validación y mide una sola vez en test.

Métrica: puesto del producto comprado entre los 10 mostrados (MRR), ponderando cada compra por
1/atención del slot en que ocurrió (curva empírica de 5.1), para no premiar «parecerse al orden antiguo».

La puntuación se recalcula EXACTAMENTE como `_recommend_core` para los parámetros que se ajustan; lo
que no se ajusta (bonus de carrito/recurrentes, misma categoría, bandit) se conserva como término fijo.

Salida: ml_data/weights_profile_tuned_trust<T>.json (NO se aplica solo) y ml_data/fast_tune_report_trust<T>.md,
con <T> = valor de --trust (1.15 -> trust1_15), para que varias ejecuciones no se pisen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .bandits import EMPIRICAL_SLOT_WEIGHTS
from .config import ConfigV4
from .reranker import FEATURE_NAMES

MODEL_ROOT = Path(__file__).resolve().parents[2]
W_FIELDS = ["w_seq_lift", "w_self_cvr_view", "w_intent_score", "w_user_affinity", "w_price_compat", "w_ctx_bayes",
            "w_global_pop", "w_live_trend", "w_is_new_product", "w_reverse_graft", "w_forward_cold",
            "w_log_price_norm", "w_same_top_subcat", "w_same_top_brand"]
SEG_FIELDS = ["seg_known_with_views", "seg_known_with_history", "seg_new_with_views", "seg_known_cold"]
SEG_COLS = ["seq", "intent", "aff", "ctx", "pop", "trend", "cold"]
# No se ajustan: señales de cold start (siempre 0 en el histórico) y tendencia en vivo (no reproducible)
FROZEN_W = {7, 8, 9, 10}
FROZEN_SEG_COLS = {5, 6}


class Evaluator:
    def __init__(self, z, wp, split: int) -> None:
        names = [str(x) for x in z["feature_names"]]
        g_split = z["g_split"]
        yb = z["y_buy"].reshape(-1, 10)
        sel = (g_split == split) & (yb.sum(axis=1) > 0)
        self.n = int(sel.sum())
        X = z["X"].reshape(-1, 10, len(names))[sel].astype(np.float64)
        self.B = X[:, :, : len(FEATURE_NAMES)]
        self.Y = yb[sel].astype(bool)
        self.prior = z["prior"].reshape(-1, 10)[sel].astype(np.float64)
        hp = X[:, 0, names.index("has_profile")] > 0.5
        hv = X[:, 0, names.index("has_views")] > 0.5
        self.hv = hv
        # 0 known_with_views · 1 known_with_history · 2 new_with_views · 3 known_cold (igual que classify_user_segment)
        self.seg = np.where(hp & hv, 0, np.where(hp & ~hv, 1, np.where(~hp & hv, 2, 3)))
        slot = self.Y.argmax(axis=1)
        exam = np.asarray(EMPIRICAL_SLOT_WEIGHTS, dtype=np.float64)
        self.w_req = 1.0 / exam[slot]
        self.w_req /= self.w_req.mean()
        self.same_sc = self.B[:, :, 12] > 0.5
        self.same_br = self.B[:, :, 13] > 0.5
        self.base_params = params_from_profile(wp)
        self.rest = self.prior - self._score(self.base_params)   # términos que no se ajustan

    def _score(self, p: dict) -> np.ndarray:
        B, w = self.B, p["w"]
        m = p["seg"][self.seg][:, None, :]                      # (n, 1, 7)
        s_seq = B[:, :, 0] * m[:, :, 0]
        s_self = B[:, :, 1]
        s_int = B[:, :, 2] * m[:, :, 1]
        s_fwd = B[:, :, 10] * m[:, :, 6]
        bg = w[5] * B[:, :, 5] * m[:, :, 3] + w[6] * B[:, :, 6] * m[:, :, 4] + w[7] * B[:, :, 7] * m[:, :, 5]
        if p["bg_on"]:
            pen = self.hv[:, None] & (s_seq == 0.0) & (s_int == 0.0) & (s_fwd == 0.0) & (s_self == 0.0)
            bg = np.where(pen, bg * np.where(self.same_sc, p["bg_same"], p["bg_diff"]), bg)
        sub_w = w[12] if p["no_floor"] else max(w[12], p["sub_floor"])
        br_w = w[13] if p["no_floor"] else max(w[13], p["br_floor"])
        return (w[0] * s_seq + w[1] * s_self + w[2] * s_int + w[3] * B[:, :, 3] * m[:, :, 2] + w[4] * B[:, :, 4]
                + bg + w[8] * B[:, :, 8] + w[9] * B[:, :, 9] * m[:, :, 6] + w[10] * s_fwd + w[11] * B[:, :, 11]
                + np.where(self.same_sc, sub_w, 0.0) + np.where(self.same_br, br_w, 0.0))

    def metrics(self, p: dict) -> dict[str, float]:
        S = self._score(p) + self.rest
        s_pos = (S * self.Y).sum(axis=1, keepdims=True)
        rank = 1.0 + (S > s_pos).sum(axis=1) + 0.5 * ((S == s_pos).sum(axis=1) - 1)
        rr = 1.0 / rank
        return {"mrr_w": float((rr * self.w_req).mean()), "mrr": float(rr.mean()), "hit1": float((rank <= 1.0).mean()),
                "hit3": float((rank <= 3.0).mean())}


def params_from_profile(wp) -> dict:
    w = np.array([float(getattr(wp, f)) for f in W_FIELDS])
    if wp.w_log_price_norm == 0.0 and wp.tie_break_mode == "conversion":
        w[11] = 0.0
    seg = np.array([[float(x) for x in getattr(wp, f)[:7]] for f in SEG_FIELDS])
    return {"w": w, "seg": seg, "bg_on": bool(wp.v3_bg_penalty), "bg_same": float(wp.bg_penalty_same_subcat),
            "bg_diff": float(wp.bg_penalty_diff_subcat), "no_floor": bool(wp.v3_no_subcat_floor),
            "sub_floor": float(wp.subcat_floor_min), "br_floor": float(wp.brand_floor_min)}


def to_vec(p: dict) -> np.ndarray:
    return np.concatenate([p["w"], p["seg"].ravel(), [p["bg_same"], p["bg_diff"], p["sub_floor"], p["br_floor"]]])


def from_vec(v: np.ndarray, base: dict) -> dict:
    p = dict(base)
    p["w"] = v[:14].copy()
    p["seg"] = v[14:42].reshape(4, 7).copy()
    p["bg_same"], p["bg_diff"], p["sub_floor"], p["br_floor"] = (float(x) for x in v[42:46])
    return p


def vec_names() -> list[str]:
    return W_FIELDS + [f"{f}[{c}]" for f in SEG_FIELDS for c in SEG_COLS] + [
        "bg_penalty_same_subcat", "bg_penalty_diff_subcat", "subcat_floor_min", "brand_floor_min"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Odyseus 2.0 · ajuste rápido de pesos (offline)")
    ap.add_argument("--data", default="ml_data/dataset.npz")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--trust", type=float, default=1.6, help="Cada parámetro se mueve como mucho entre base/trust y base*trust")
    ap.add_argument("--reg", type=float, default=0.01, help="Penalización por alejarse de los pesos actuales")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restarts", type=int, default=4)
    args = ap.parse_args()

    cfg = ConfigV4.from_env()
    wp = cfg.load_weights_profile()
    z = np.load((MODEL_ROOT / args.data).resolve(), allow_pickle=False)
    if "base_model" not in z.files or str(z["base_model"]) != "calypso-5.1":
        raise SystemExit("El dataset no es de Calypso 5.1: relanza `python ml_build_dataset.py`.")
    tr, va, te = Evaluator(z, wp, 0), Evaluator(z, wp, 1), Evaluator(z, wp, 2)
    base = tr.base_params
    v0 = to_vec(base)
    names = vec_names()
    print(f"Peticiones con compra: train {tr.n:,} | val {va.n:,} | test {te.n:,}")
    print(f"Término fijo medio (lo que no se ajusta): {tr.rest.mean():+.4f} (σ {tr.rest.std():.4f})")

    # qué se puede mover: positivos y no congelados; w_log_price_norm (base 0) se mueve en aditivo
    tunable = np.zeros(len(v0), dtype=bool)
    for i in range(14):
        tunable[i] = i not in FROZEN_W and v0[i] > 0.0
    for s in range(4):
        for c in range(7):
            tunable[14 + s * 7 + c] = c not in FROZEN_SEG_COLS and v0[14 + s * 7 + c] > 0.0
    tunable[42:46] = v0[42:46] > 0.0
    add_idx = 11 if v0[11] == 0.0 else -1
    idx = np.where(tunable)[0]
    log_tr = np.log(args.trust)
    print(f"Parámetros ajustables: {len(idx)}{' + w_log_price_norm (aditivo)' if add_idx >= 0 else ''} | zona de confianza x{args.trust}")

    def build(delta: np.ndarray, price: float) -> np.ndarray:
        v = v0.copy()
        v[idx] = v0[idx] * np.exp(delta)
        if add_idx >= 0:
            v[add_idx] = price
        return v

    def objective(delta: np.ndarray, price: float) -> float:
        m = tr.metrics(from_vec(build(delta, price), base))
        return m["mrr_w"] - args.reg * float(np.abs(delta).mean())

    rng = np.random.default_rng(args.seed)
    base_tr, base_va, base_te = tr.metrics(base), va.metrics(base), te.metrics(base)
    t_end = time.time() + args.minutes * 60.0
    n_eval = 0
    t0 = time.time()
    best_va, best_pack = base_va["mrr_w"], (np.zeros(len(idx)), 0.0)
    history = []
    for restart in range(args.restarts):
        delta = np.zeros(len(idx)) if restart == 0 else rng.normal(0.0, 0.15, len(idx)).clip(-log_tr, log_tr)
        price = 0.0
        cur = objective(delta, price)
        sigma = 0.25
        budget_end = t0 + (restart + 1) * (args.minutes * 60.0) / args.restarts
        while time.time() < min(budget_end, t_end):
            # (1+1)-ES con mutación dispersa: mueve pocos parámetros cada vez
            mask = rng.random(len(idx)) < max(2.0 / len(idx), 0.08)
            if not mask.any():
                mask[rng.integers(len(idx))] = True
            cand = np.clip(delta + mask * rng.normal(0.0, sigma, len(idx)), -log_tr, log_tr)
            cprice = float(np.clip(price + (rng.normal(0.0, 0.03) if rng.random() < 0.15 else 0.0), -0.3, 0.3)) if add_idx >= 0 else 0.0
            val = objective(cand, cprice)
            n_eval += 1
            if val > cur:
                delta, price, cur = cand, cprice, val
                sigma = min(sigma * 1.15, 0.5)
            else:
                sigma = max(sigma * 0.985, 0.02)
        p = from_vec(build(delta, price), base)
        m_va = va.metrics(p)
        history.append((restart, tr.metrics(p)["mrr_w"], m_va["mrr_w"]))
        print(f"  reinicio {restart}: MRR ponderado train {history[-1][1]:.4f} | val {m_va['mrr_w']:.4f}")
        if m_va["mrr_w"] > best_va:
            best_va, best_pack = m_va["mrr_w"], (delta.copy(), price)
    elapsed = time.time() - t0
    print(f"\n{n_eval:,} pruebas en {elapsed:.0f} s = {60.0 * n_eval / max(elapsed, 1e-9):,.0f} pruebas/minuto")

    v_best = build(*best_pack)
    p_best = from_vec(v_best, base)
    m_tr, m_va, m_te = tr.metrics(p_best), va.metrics(p_best), te.metrics(p_best)
    se_va, se_te = 0.29 / va.n**0.5, 0.29 / te.n**0.5
    rows = [("actual", base_tr, base_va, base_te), ("ajustado", m_tr, m_va, m_te)]
    table = ["| pesos | MRR pond. train | MRR pond. val | MRR pond. test | MRR test | acierto@1 test | acierto@3 test |", "|---|---|---|---|---|---|---|"]
    for n_, a, b, c in rows:
        table.append(f"| {n_} | {a['mrr_w']:.4f} | {b['mrr_w']:.4f} | {c['mrr_w']:.4f} | {c['mrr']:.4f} | {c['hit1']:.3f} | {c['hit3']:.3f} |")
    print("\n" + "\n".join(table))
    print(f"Error típico aproximado: ±{se_va:.4f} (val), ±{se_te:.4f} (test)")
    gain_va, gain_te = m_va["mrr_w"] - base_va["mrr_w"], m_te["mrr_w"] - base_te["mrr_w"]
    ok = gain_va > 2 * se_va and gain_te > 2 * se_te
    print(f"Mejora: val {gain_va:+.4f} | test {gain_te:+.4f} -> {'por encima del ruido en val y test' if ok else 'NO supera el ruido en val y test: no hay nada que aplicar'}")

    changes = sorted(((names[i], v0[i], v_best[i]) for i in range(len(v0)) if abs(v_best[i] - v0[i]) > 1e-9),
                     key=lambda t: -abs(np.log(max(t[2], 1e-9) / max(t[1], 1e-9))) if t[1] > 0 else -abs(t[2]))
    print("\nCambios (de mayor a menor):")
    for n_, a, b in changes[:20]:
        print(f"  {n_:32s} {a:7.3f} -> {b:7.3f}" + (f"  (x{b / a:.2f})" if a > 0 else ""))

    prof = json.loads(cfg.resolve_weights_profile_path().read_text(encoding="utf-8"))
    for i, f in enumerate(W_FIELDS):
        if abs(v_best[i] - v0[i]) > 1e-9:
            prof[f] = round(float(v_best[i]), 4)
    for s, f in enumerate(SEG_FIELDS):
        arr = [float(x) for x in prof[f]]
        for c in range(7):
            arr[c] = round(float(v_best[14 + s * 7 + c]), 4)
        prof[f] = arr
    for k, f in enumerate(["bg_penalty_same_subcat", "bg_penalty_diff_subcat", "subcat_floor_min", "brand_floor_min"]):
        prof[f] = round(float(v_best[42 + k]), 4)
    prof["profile_name"] = str(prof.get("profile_name", "profile")) + "_fasttuned"
    out_dir = (MODEL_ROOT / args.data).resolve().parent
    tag = f"trust{args.trust:g}".replace(".", "_")           # p. ej. trust1_15, trust2_5
    prof_path = out_dir / f"weights_profile_tuned_{tag}.json"
    rep_path = out_dir / f"fast_tune_report_{tag}.md"
    prof_path.write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# Odyseus 2.0 · ajuste rápido de pesos (offline)", "",
              f"{n_eval:,} pruebas en {elapsed:.0f} s ({60.0 * n_eval / max(elapsed, 1e-9):,.0f} por minuto) · zona de confianza x{args.trust} · reg {args.reg}", "",
              *table, "", f"Error típico: ±{se_va:.4f} (val), ±{se_te:.4f} (test). Mejora val {gain_va:+.4f}, test {gain_te:+.4f}: "
              + ("por encima del ruido." if ok else "no supera el ruido."), "",
              "| parámetro | actual | ajustado | factor |", "|---|---|---|---|",
              *[f"| {n_} | {a:.3f} | {b:.3f} | {('x%.2f' % (b / a)) if a > 0 else ''} |" for n_, a, b in changes], ""]
    rep_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nPerfil propuesto: {prof_path} (no se aplica solo)\nInforme: {rep_path}")


if __name__ == "__main__":
    main()
