"""Calypso 4.1 · reglas de negocio sobre carrito, compras previas/recurrentes y orden por precio.

Dos piezas:

* ``CartPurchaseIndex``: fusiona tres fuentes por (user_id, product_id)
    - histórico ``shop.history_events`` (calculado una vez, se cachea en disco),
    - eventos en vivo ``shop.events`` (add_to_cart / purchase),
    - compras de nuestro carrusel (``team.feedback``),
  y produce, por usuario, los productos que siguen en el carrito (último add_to_cart posterior a la
  última compra) y los productos comprados >= N veces (recurrentes). La parte histórica se precalcula
  una sola vez; en cada refresco solo se recalculan los usuarios con actividad en vivo.

* ``apply_business_rules``: dado el ranking del modelo, construye la lista final de 10 en tres bloques
    [carrito activo] [carrito olvidado] [recomendados (+ recurrentes garantizados)]
  y, si está activado, ordena cada bloque de más barato a más caro.

Evitar doble conteo: si el simulador escribiera las compras del carrusel también en ``shop.events``,
sumar ambas fuentes convertiría una compra única en "recurrente". Por eso, en vivo se toma
``max(compras_en_events, compras_en_feedback)`` y no la suma.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Estado por (usuario, producto): [n_compras, ultimo_carrito_ms, ultima_compra_ms]
_Entry = list


def _f(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def _rows_to_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        uid = str(row.get("user_id") or "").strip()
        pid = str(row.get("product_id") or "").strip()
        if not uid or not pid:
            continue
        n = _f(row.get("n_purchases"))
        cart = _f(row.get("last_cart_ms"))
        buy = _f(row.get("last_purchase_ms"))
        per_user = out.setdefault(uid, {})
        prev = per_user.get(pid)
        if prev is None:
            per_user[pid] = [n, cart, buy]
        else:  # filas duplicadas (p.ej. carrito + recurrente): combinar
            prev[0] = max(prev[0], n)
            prev[1] = max(prev[1], cart)
            prev[2] = max(prev[2], buy)
    return out


def _derive_user(
    entries: Mapping[str, Sequence[float]],
    min_recurrent: int,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, int], ...]]:
    carts = [(pid, e[1]) for pid, e in entries.items() if e[1] > 0.0 and e[1] > e[2]]
    carts.sort(key=lambda kv: kv[1], reverse=True)
    rec = [(pid, int(e[0])) for pid, e in entries.items() if e[0] >= min_recurrent]
    rec.sort(key=lambda kv: (kv[1], kv[0]), reverse=True)
    return tuple(carts[:12]), tuple(rec[:12])


class CartPurchaseIndex:
    """Índice histórico precalculado + capa en vivo recalculada solo para usuarios activos."""

    def __init__(
        self,
        cart_hist_rows: Iterable[Mapping[str, Any]] = (),
        recurrent_hist_rows: Iterable[Mapping[str, Any]] = (),
        min_recurrent: int = 2,
    ) -> None:
        self.min_recurrent = max(int(min_recurrent), 2)
        self._hist = _rows_to_map(list(cart_hist_rows) + list(recurrent_hist_rows))
        self._base_carts: dict[str, tuple[tuple[str, float], ...]] = {}
        self._base_recurrent: dict[str, tuple[tuple[str, int], ...]] = {}
        for uid, entries in self._hist.items():
            carts, rec = _derive_user(entries, self.min_recurrent)
            if carts:
                self._base_carts[uid] = carts
            if rec:
                self._base_recurrent[uid] = rec
<<<<<<< HEAD
        self._live_base_ev: dict[str, dict[str, list[float]]] = {}
        self._live_base_carts: dict[str, tuple[tuple[str, float], ...]] = dict(self._base_carts)
        self._live_base_rec: dict[str, tuple[tuple[str, int], ...]] = dict(self._base_recurrent)
        self._live_base_bought: dict[str, frozenset[str]] = {}
        self._live_base_merged: dict[str, dict[str, list[float]]] = {}
=======
>>>>>>> cbe71fec6826253cc30a383015bc169527273ef9

    @property
    def n_hist_users(self) -> int:
        return len(self._hist)

<<<<<<< HEAD
    def set_live_base(self, full_live_event_rows: Iterable[Mapping[str, Any]]) -> None:
        """Pre-aggregate full live shop.events stream once so fast ticks only touch tiny deltas."""
        live_ev = _rows_to_map(full_live_event_rows)
        self._live_base_ev = live_ev
        carts = dict(self._base_carts)
        recurrent = dict(self._base_recurrent)
        live_bought: dict[str, frozenset[str]] = {}
        merged_users: dict[str, dict[str, list[float]]] = {}
        for uid, ev_u in live_ev.items():
            merged: dict[str, list[float]] = {
                pid: list(e) for pid, e in self._hist.get(uid, {}).items()
            }
            bought_now: set[str] = set()
            for pid, ev in ev_u.items():
                n_live = ev[0]
                base = merged.get(pid, [0.0, 0.0, 0.0])
                merged[pid] = [
                    base[0] + n_live,
                    max(base[1], ev[1]),
                    max(base[2], ev[2]),
                ]
                if n_live > 0:
                    bought_now.add(pid)
            merged_users[uid] = merged
            u_carts, u_rec = _derive_user(merged, self.min_recurrent)
            if u_carts:
                carts[uid] = u_carts
            else:
                carts.pop(uid, None)
            if u_rec:
                recurrent[uid] = u_rec
            else:
                recurrent.pop(uid, None)
            if bought_now:
                live_bought[uid] = frozenset(bought_now)
        self._live_base_carts = carts
        self._live_base_rec = recurrent
        self._live_base_bought = live_bought
        self._live_base_merged = merged_users

    def build_incremental(
        self,
        recent_event_rows: Iterable[Mapping[str, Any]] = (),
        feedback_purchase_rows: Iterable[Mapping[str, Any]] = (),
    ) -> tuple[
        dict[str, tuple[tuple[str, float], ...]],
        dict[str, tuple[tuple[str, int], ...]],
        dict[str, frozenset[str]],
    ]:
        """Apply only recent event deltas + team feedback onto pre-aggregated live+hist base (<30ms)."""
        live_ev = _rows_to_map(recent_event_rows)
        live_fb = _rows_to_map(feedback_purchase_rows)

        carts = dict(self._live_base_carts)
        recurrent = dict(self._live_base_rec)
        live_bought = dict(self._live_base_bought)

        for uid in set(live_ev) | set(live_fb):
            ev_rec_u = live_ev.get(uid, {})
            ev_base_u = self._live_base_ev.get(uid, {})
            fb_u = live_fb.get(uid, {})
            merged: dict[str, list[float]] = {
                pid: list(e) for pid, e in self._hist.get(uid, {}).items()
            }
            bought_now: set[str] = set()
            for pid in set(ev_base_u) | set(ev_rec_u) | set(fb_u):
                b_ev = ev_base_u.get(pid, (0.0, 0.0, 0.0))
                r_ev = ev_rec_u.get(pid, (0.0, 0.0, 0.0))
                ev0 = max(b_ev[0], r_ev[0])
                ev1 = max(b_ev[1], r_ev[1])
                ev2 = max(b_ev[2], r_ev[2])
                fb = fb_u.get(pid, (0.0, 0.0, 0.0))
                n_live = max(ev0, fb[0])
                base = merged.get(pid, [0.0, 0.0, 0.0])
                merged[pid] = [
                    base[0] + n_live,
                    max(base[1], ev1, fb[1]),
                    max(base[2], ev2, fb[2]),
                ]
                if n_live > 0:
                    bought_now.add(pid)
            u_carts, u_rec = _derive_user(merged, self.min_recurrent)
            if u_carts:
                carts[uid] = u_carts
            else:
                carts.pop(uid, None)
            if u_rec:
                recurrent[uid] = u_rec
            else:
                recurrent.pop(uid, None)
            if bought_now:
                live_bought[uid] = frozenset(bought_now)
        return carts, recurrent, live_bought

=======
>>>>>>> cbe71fec6826253cc30a383015bc169527273ef9
    def build(
        self,
        live_event_rows: Iterable[Mapping[str, Any]] = (),
        feedback_purchase_rows: Iterable[Mapping[str, Any]] = (),
    ) -> tuple[
        dict[str, tuple[tuple[str, float], ...]],
        dict[str, tuple[tuple[str, int], ...]],
        dict[str, frozenset[str]],
    ]:
        """Devuelve (user_carts, user_recurrent, live_bought) con la capa en vivo aplicada."""
        live_ev = _rows_to_map(live_event_rows)
        live_fb = _rows_to_map(feedback_purchase_rows)

        carts = dict(self._base_carts)
        recurrent = dict(self._base_recurrent)
        live_bought: dict[str, frozenset[str]] = {}

        for uid in set(live_ev) | set(live_fb):
            ev_u = live_ev.get(uid, {})
            fb_u = live_fb.get(uid, {})
            merged: dict[str, list[float]] = {
                pid: list(e) for pid, e in self._hist.get(uid, {}).items()
            }
            bought_now: set[str] = set()
            for pid in set(ev_u) | set(fb_u):
                ev = ev_u.get(pid, (0.0, 0.0, 0.0))
                fb = fb_u.get(pid, (0.0, 0.0, 0.0))
                n_live = max(ev[0], fb[0])  # max, no suma: evita contar dos veces la misma compra
                base = merged.get(pid, [0.0, 0.0, 0.0])
                merged[pid] = [
                    base[0] + n_live,
                    max(base[1], ev[1], fb[1]),
                    max(base[2], ev[2], fb[2]),
                ]
                if n_live > 0:
                    bought_now.add(pid)
            u_carts, u_rec = _derive_user(merged, self.min_recurrent)
            if u_carts:
                carts[uid] = u_carts
            else:
                carts.pop(uid, None)
            if u_rec:
                recurrent[uid] = u_rec
            else:
                recurrent.pop(uid, None)
            if bought_now:
                live_bought[uid] = frozenset(bought_now)
        return carts, recurrent, live_bought


def build_cart_and_recurrent_maps(
    cart_hist_rows: Iterable[Mapping[str, Any]] = (),
    recurrent_hist_rows: Iterable[Mapping[str, Any]] = (),
    live_event_rows: Iterable[Mapping[str, Any]] = (),
    feedback_purchase_rows: Iterable[Mapping[str, Any]] = (),
    min_recurrent: int = 2,
) -> tuple[
    dict[str, tuple[tuple[str, float], ...]],
    dict[str, tuple[tuple[str, int], ...]],
    dict[str, frozenset[str]],
]:
    """Atajo sin estado (tests / evaluación offline)."""
    idx = CartPurchaseIndex(cart_hist_rows, recurrent_hist_rows, min_recurrent)
    return idx.build(live_event_rows, feedback_purchase_rows)


def split_cart_items(
    user_cart: Sequence[tuple[str, float]],
    active_ids: frozenset[str] | set[str],
    now_ms: float,
    *,
    use_active: bool,
    use_forgotten: bool,
    active_minutes: float,
    forgotten_max_days: float,
    max_slots: int,
    forgotten_max_slots: int,
) -> tuple[list[str], list[str]]:
    """Separa el carrito del usuario en (activo, olvidado) respetando los topes de slots."""
    active: list[str] = []
    forgotten: list[str] = []
    if not user_cart or max_slots <= 0:
        return active, forgotten
    active_ms = active_minutes * 60_000.0
    max_forgot_ms = forgotten_max_days * 86_400_000.0
    for pid, ts_ms in user_cart:  # ya viene ordenado del más reciente al más antiguo
        if pid not in active_ids:
            continue  # agotado / inactivo: no se puede enseñar
        age = now_ms - ts_ms
        if age < -60_000.0:
            continue  # evento futuro respecto a now_ms (evita fuga temporal)
        if age <= active_ms:
            if use_active:
                active.append(pid)
        elif age <= max_forgot_ms:
            if use_forgotten:
                forgotten.append(pid)
    active = active[:max_slots]
    forgotten = forgotten[: max(0, min(forgotten_max_slots, max_slots - len(active)))]
    return active, forgotten


def guarantee_items(block: list[str], required: Sequence[str]) -> list[str]:
    """Garantiza que ``required`` aparezca en ``block`` sustituyendo las últimas posiciones."""
    out = list(block)
    missing = [pid for pid in required if pid not in out]
    if not missing:
        return out
    protected = set(required)
    pos = len(out) - 1
    for pid in missing:
        while pos >= 0 and out[pos] in protected:
            pos -= 1
        if pos < 0:
            out.append(pid)
            continue
        out[pos] = pid
        pos -= 1
    return out


def sort_by_price(items: Sequence[str], products: Mapping[str, Any]) -> list[str]:
    """Orden estable de más barato a más caro (empate: se respeta el orden del modelo)."""
    order = {pid: i for i, pid in enumerate(items)}

    def _price(pid: str) -> float:
        info = products.get(pid)
        return float(getattr(info, "price_eur", 0.0) or 0.0) if info is not None else 0.0

    return sorted(items, key=lambda pid: (_price(pid), order[pid]))
