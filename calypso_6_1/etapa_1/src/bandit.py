"""Thompson sampling sobre pocas variantes de mezcla, por segmento (D2).

Con ~400 compras en dos horas no se puede aprender una tasa por producto, pero sí
elegir entre 4 brazos en ~6 segmentos (frío/templado × página). La señal es el
**clic** (más frecuente que la compra). El estado se guarda en disco para sobrevivir
a reinicios.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np


class ThompsonBandit:
    def __init__(self, arms: list[str], prior: tuple[float, float] = (2.0, 18.0),
                 seed: int | None = None) -> None:
        self.arms = list(arms)
        self.prior = prior
        self.state: dict[str, dict[str, list[float]]] = {}
        self.rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

    def _seg(self, segment: str) -> dict[str, list[float]]:
        s = self.state.get(segment)
        if s is None:
            s = {a: [self.prior[0], self.prior[1]] for a in self.arms}
            self.state[segment] = s
        return s

    def choose(self, segment: str) -> str:
        with self._lock:
            s = self._seg(segment)
            draws = {a: self.rng.beta(*s[a]) for a in self.arms}
        return max(draws, key=draws.get)

    def update(self, segment: str, arm: str, reward: float) -> None:
        if arm not in self.arms:
            return
        with self._lock:
            ab = self._seg(segment)[arm]
            ab[0] += reward
            ab[1] += 1.0 - reward

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        with self._lock:
            return {seg: {a: {"n": ab[0] + ab[1] - sum(self.prior), "mean": ab[0] / (ab[0] + ab[1])}
                          for a, ab in arms.items()} for seg, arms in self.state.items()}

    def save(self, path: Path) -> None:
        with self._lock:
            data = {"arms": self.arms, "prior": self.prior, "state": self.state}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)

    def load(self, path: Path) -> bool:
        if not path.is_file():
            return False
        d = json.loads(path.read_text())
        with self._lock:
            for seg, arms in d.get("state", {}).items():
                s = self._seg(seg)
                for a, ab in arms.items():
                    if a in s:
                        s[a] = list(ab)
        return True


def segment_of(is_cold: bool, page: str) -> str:
    return f"{'cold' if is_cold else 'warm'}|{page or 'unknown'}"
