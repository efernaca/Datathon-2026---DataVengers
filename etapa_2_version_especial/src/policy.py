"""Política de la Etapa 2 · versión especial.

score(i) = Σ_s w_{segmento}[s] · interruptor[s] · señal_s(i)

* Pesos por segmento (4.3), aprendidos con el feedback propio (4.2); sin aprender, los
  de la Etapa 2.
* Interruptores del experimento factorial (4.1), barajado del top-10 en un 10 % (curva
  de atención medida) y "modo final" en un 20 % (usuario tratado como nuevo, 4.4).
  Todo se decide por hash de `request_id` (reproducible) y se codifica en `model_version`.
* MMR con penalización relativa (N10) y desempate por ingresos lexicográfico (N9).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_WEIGHTS, SEGMENTS, SIGNALS, SWITCHES, Config
from .guardrails import NUM_SLOTS, ensure_valid_submission
from .signals import SIDX, Req, compute_signals, parse_request
from .state import Snapshot

# Qué señal apaga cada interruptor (mmr y rev afectan a la lista, no a una señal)
SWITCH_SIGNAL = {"rg": "injerto_inverso", "int": "intencion", "pf": "precio_fit", "self": "visto_conv"}


def unit_hash(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


@dataclass
class Weights:
    by_segment: dict[str, dict[str, float]]
    version: str = "manual"

    @classmethod
    def default(cls) -> "Weights":
        return cls({seg: dict(DEFAULT_WEIGHTS) for seg in SEGMENTS}, "manual")

    def vector(self, segment: str) -> np.ndarray:
        d = self.by_segment.get(segment) or self.by_segment.get("global") or DEFAULT_WEIGHTS
        return np.array([float(d.get(s, 0.0)) for s in SIGNALS], dtype=np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": self.version, "by_segment": self.by_segment}, indent=1))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Weights":
        if not path.is_file():
            return cls.default()
        d = json.loads(path.read_text())
        return cls(d["by_segment"], d.get("version", "file"))


@dataclass
class Decision:
    request_id: str
    items: list[str]
    segment: str
    switches: dict[str, int]
    shuffled: bool = False
    final_mode: bool = False
    user_id: str = ""
    is_new_user: bool = False
    signals: list[list[float]] = field(default_factory=list)   # señales de los 10 mostrados, en orden
    new_positions: list[int] = field(default_factory=list)      # N8: posiciones de productos nuevos

    def tag(self, base: str, wver: str) -> str:
        sw = "|".join(f"{k}{v}" for k, v in self.switches.items())
        return f"{base}|{wver}|{sw}|shuf{int(self.shuffled)}|fin{int(self.final_mode)}"


def parse_tag(tag: str) -> dict[str, Any]:
    """Inversa de `Decision.tag` (para el análisis de experimentos)."""
    parts = str(tag).split("|")
    out: dict[str, Any] = {"base": parts[0] if parts else "", "wver": parts[1] if len(parts) > 1 else ""}
    for p in parts[2:]:
        for name in (*SWITCHES, "shuf", "fin"):
            if p.startswith(name) and p[len(name):].isdigit():
                out[name] = int(p[len(name):])
    return out


def segment_of(req: Req, has_session: bool, has_new_product: bool, final_mode: bool) -> str:
    if has_new_product:
        return "producto_nuevo"
    new = req.is_new_user or final_mode
    if new:
        return "nuevo_sesion" if has_session else "nuevo_sin_sesion"
    return "conocido_sesion" if has_session else "conocido_sin_sesion"


class Stage2Special:
    def __init__(self, cfg: Config, weights: Weights | None = None,
                 frozen_switches: dict[str, int] | None = None) -> None:
        self.cfg = cfg
        self.weights = weights or Weights.default()
        self.frozen = frozen_switches      # domingo: interruptores fijos según el análisis

    # ------------------------------------------------------------------ asignación (4.1, 4.4)

    def assign(self, rid: str) -> tuple[dict[str, int], bool, bool]:
        cfg = self.cfg
        if not cfg.experiments:
            sw = {s: 1 for s in SWITCHES}
            sw["rev"] = 0                            # N9: apagado hasta medir B4
            if self.frozen:
                sw.update(self.frozen)
            return sw, False, False
        sw = {s: int(unit_hash(f"{rid}:{s}") < cfg.switch_prob) for s in SWITCHES}
        shuffled = unit_hash(f"{rid}:shuffle") < cfg.shuffle_rate
        final = unit_hash(f"{rid}:final") < cfg.final_mode_rate
        return sw, shuffled, final

    # ------------------------------------------------------------------ lista

    def _mmr(self, cand: np.ndarray, score: np.ndarray, snap: Snapshot, k: int, has_session: bool) -> list[int]:
        """N10: valor = exp(score) y penalización relativa valor·(1 − λ_sub·n_sub)(1 − λ_marca·n_marca)."""
        cfg = self.cfg
        cat = snap.catalog
        keep = cfg.mmr_keep_top_session if has_session else cfg.mmr_keep_top_empty
        lam_sub = 0.0 if has_session else cfg.mmr_lambda_subcat
        chosen = [int(i) for i in cand[:keep]]
        pool = [int(i) for i in cand[keep:keep + 60]]
        n_sub: dict[int, int] = {}
        n_br: dict[int, int] = {}
        for i in chosen:
            n_sub[cat.subcategory[i]] = n_sub.get(cat.subcategory[i], 0) + 1
            n_br[cat.brand[i]] = n_br.get(cat.brand[i], 0) + 1
        base = np.exp(score[pool] - score[pool].max()) if pool else np.zeros(0)
        avail = np.ones(len(pool), dtype=bool)
        while len(chosen) < k and avail.any():
            pen = np.array([max(0.0, 1 - lam_sub * n_sub.get(cat.subcategory[i], 0))
                            * max(0.0, 1 - cfg.mmr_lambda_brand * n_br.get(cat.brand[i], 0)) for i in pool])
            val = np.where(avail, base * pen, -1.0)
            j = int(np.argmax(val))
            avail[j] = False
            i = pool[j]
            chosen.append(i)
            n_sub[cat.subcategory[i]] = n_sub.get(cat.subcategory[i], 0) + 1
            n_br[cat.brand[i]] = n_br.get(cat.brand[i], 0) + 1
        return chosen

    @staticmethod
    def _revenue_tiebreak(order: list[int], score: np.ndarray, price: np.ndarray, eps: float) -> list[int]:
        """N9: solo se intercambian vecinos con |Δscore| < ε·|score| y el de abajo es más caro."""
        out = list(order)
        for _ in range(2):
            for a in range(len(out) - 1):
                i, j = out[a], out[a + 1]
                if abs(score[i] - score[j]) < eps * max(abs(score[i]), 1e-9) and price[j] > price[i]:
                    out[a], out[a + 1] = j, i
        return out

    def recommend(self, raw: Any, snap: Snapshot, k: int = NUM_SLOTS, switches: dict[str, int] | None = None,
                  shuffled: bool | None = None, final_mode: bool | None = None) -> Decision:
        cfg = self.cfg
        req = raw if isinstance(raw, Req) else parse_request(raw)
        cat = snap.catalog
        sw, sh, fin = self.assign(req.request_id)
        sw = switches if switches is not None else sw
        sh = sh if shuffled is None else shuffled
        fin = fin if final_mode is None else final_mode

        live = snap.live.get("default")
        S, sess = compute_signals(req, snap, cfg, live=live, use_profile=not fin)
        seg = segment_of(req, len(sess.recent_idx) > 0, sess.has_new_product, fin)
        wv = self.weights.vector(seg).copy()
        for s_name, sig in SWITCH_SIGNAL.items():
            if not sw.get(s_name, 1):
                wv[SIDX[sig]] = 0.0
        score = S @ wv

        excl: set[str] = set()
        prof = snap.history.users.get(req.user_id) if req.user_id else None
        if prof is not None:
            excl |= prof.bought            # incluso en modo final: no recomendar lo ya comprado
        excl |= snap.live_bought.get(req.user_id, frozenset())
        valid = cat.active.copy()
        for pid in excl:
            j = cat.idx.get(pid)
            if j is not None:
                valid[j] = False
        score = np.where(valid, score, -np.inf)
        C = min(200, int(valid.sum()))
        if C > 0:
            cand = np.argpartition(-score, C - 1)[:C]
            cand = cand[np.argsort(-score[cand], kind="stable")]
            cand = cand[np.isfinite(score[cand])]
        else:
            cand = np.zeros(0, dtype=np.int64)

        if sw.get("mmr", 1) and len(cand) > k:
            order = self._mmr(cand, score, snap, k, len(sess.recent_idx) > 0)
        else:
            order = [int(i) for i in cand[:k]]
        if sw.get("rev", 0):
            order = self._revenue_tiebreak(order, score, cat.price, max(cfg.revenue_eps, 0.01))
        items = ensure_valid_submission([cat.ids[i] for i in order], cat.active_ids, cat.global_ranked, excl, k=k)
        if sh:
            rng = np.random.default_rng(int(unit_hash(f"{req.request_id}:perm") * 2**31))
            items = [items[i] for i in rng.permutation(len(items))]
        sel = np.array([cat.idx[p] for p in items])
        return Decision(
            request_id=req.request_id, items=items, segment=seg, switches=sw, shuffled=sh, final_mode=fin,
            user_id=req.user_id, is_new_user=req.is_new_user, signals=np.round(S[sel], 5).tolist(),
            new_positions=[r + 1 for r, p in enumerate(items) if cat.is_new[cat.idx[p]]],
        )


_ = math
