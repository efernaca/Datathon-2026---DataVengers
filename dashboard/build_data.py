"""Construye dashboard/index.html (Side Challenge Tukey) a partir de los ficheros del repo.

Uso (desde la raíz del repo o desde dashboard/):
    python dashboard/build_data.py                 # solo datos offline del repo
    python dashboard/export_live.py                # (opcional, necesita red a RawTree) -> dashboard/live/
    python dashboard/build_data.py                 # vuelve a construir incluyendo el feedback en vivo

Solo usa la librería estándar de Python. El HTML resultante es autocontenido
(sin servidor, sin CDN): se abre con doble clic en cualquier portátil.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "calypso_6_2_b"

PRESETS_JSON = MODEL_DIR / "preset_comparison_results.json"
ABLATION_CSV = MODEL_DIR / "etapa_4" / "ablation_v4_results.csv"
LIVE_LOG = MODEL_DIR / "logs" / "live.log"
TRANSITIONS_JSON = ROOT / "gestor_models" / "state" / "model_transitions.json"
RAW_DIR = ROOT / "odyseus_2_0" / "ml_data" / "raw"
LIVE_DIR = HERE / "live"
TEMPLATE = HERE / "template.html"
OUTPUT = HERE / "index.html"

OLD_CAROUSEL_CVR = 1.40  # compras por 100 peticiones del carrusel antiguo (28.053 / 2M, shop.history_sessions)
SLA_MS = 3000


# ----------------------------------------------------------------------------- utilidades
def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def infer_n(values: list[float], lo: int = 30, hi: int = 3000, tol: float = 1e-6) -> int | None:
    """Menor n tal que todas las fracciones son k/n exactas (el JSON no guarda los tamaños)."""
    vals = [v for v in values if v is not None and 0 < v < 1]
    if not vals:
        return None
    for n in range(lo, hi + 1):
        if all(abs(v * n - round(v * n)) < tol * n for v in vals):
            return n
    return None


def note(msg: str) -> None:
    print(f"[build] {msg}")


# ----------------------------------------------------------------------------- panel 2: presets
def panel_presets() -> dict:
    data = json.loads(PRESETS_JSON.read_text(encoding="utf-8"))
    segs = {
        "ALL": "ALL_hit_buy",
        "WARM": "WARM_hit_buy",
        "COLD": "COLD_hit_buy",
        "COLD_ITEM": "COLD_item_hit",
    }
    sizes = {s: infer_n([v.get(k) for v in data.values()]) for s, k in segs.items()}
    rows = []
    for preset, m in data.items():
        for seg, key in segs.items():
            p = m.get(key)
            n = sizes[seg]
            if p is None:
                continue
            lo, hi = wilson(p * n, n) if n else (None, None)
            rows.append({"preset": preset, "segment": seg, "hit": p, "n": n, "lo": lo, "hi": hi})
    ranking = sorted(
        ({"preset": k, "obj": v.get("obj"), "dcg_all": v.get("ALL_dcg_buy"), "latency_ms": v.get("latency_ms"),
          "snips": v.get("SNIPS_s1")} for k, v in data.items()),
        key=lambda r: -(r["obj"] or 0),
    )
    note(f"presets: {len(data)} presets, tamaños inferidos {sizes}")
    return {"rows": rows, "sizes": sizes, "ranking": ranking, "winner": ranking[0]["preset"]}


# ----------------------------------------------------------------------------- panel 3: productos ya vistos
def _rv_ids(rv) -> list[str]:
    out = []
    for x in rv or []:
        if isinstance(x, dict):
            pid = x.get("product_id")
        else:
            pid = x
        if pid:
            out.append(str(pid))
    return out


def panel_viewed(max_files: int | None = None) -> dict | None:
    files = sorted(RAW_DIR.glob("sessions_*.json.gz"))
    if max_files:
        files = files[:max_files]
    if not files:
        note("panel 3: no hay muestra raw, se omite")
        return None
    slot = [[[0, 0], [0, 0]] for _ in range(10)]  # slot -> viewed(0/1) -> [n, compras]
    strata: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # (pid, slot) -> a,b,c,d
    n_req = n_buy = n_req_with_viewed_shown = 0
    buy_viewed = 0  # compras de un producto ya visto
    page_counts: dict[str, list[list[int]]] = defaultdict(lambda: [[0, 0], [0, 0]])
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            reqs = json.load(fh)
        for r in reqs:
            shown = r.get("shown") or []
            viewed = set(_rv_ids(r.get("recent_views")))
            bought = r.get("purchased") or ""
            page = (r.get("page") or (r.get("context") or {}).get("page") or "?")
            n_req += 1
            if bought:
                n_buy += 1
                if bought in viewed:
                    buy_viewed += 1
            any_v = False
            for i, pid in enumerate(shown[:10]):
                v = 1 if pid in viewed else 0
                y = 1 if bought and pid == bought else 0
                any_v = any_v or bool(v)
                slot[i][v][0] += 1
                slot[i][v][1] += y
                page_counts[page][v][0] += 1
                page_counts[page][v][1] += y
                s = strata[(pid, i)]
                if v and y:
                    s[0] += 1
                elif v:
                    s[1] += 1
                elif y:
                    s[2] += 1
                else:
                    s[3] += 1
            n_req_with_viewed_shown += any_v

    def mh(groups) -> tuple[float, float, float]:
        num = den = obs = exp = 0.0
        for a, b, c, d in groups:
            n = a + b + c + d
            if n == 0 or (a + b) == 0 or (c + d) == 0:
                continue
            num += a * d / n
            den += b * c / n
            obs += a
            exp += (a + b) * c / (c + d)
        return (num / den if den else float("nan"), obs, exp)

    or_prod_slot, obs, exp = mh(strata.values())
    or_slot, _, _ = mh(
        [(slot[i][1][1], slot[i][1][0] - slot[i][1][1], slot[i][0][1], slot[i][0][0] - slot[i][0][1]) for i in range(10)]
    )
    rows = []
    for i in range(10):
        for v in (0, 1):
            n, k = slot[i][v]
            lo, hi = wilson(k, n)
            rows.append({"slot": i, "viewed": v, "n": n, "buys": k, "rate": k / n if n else None, "lo": lo, "hi": hi})
    pages = []
    for pg, cnt in sorted(page_counts.items()):
        (n0, k0), (n1, k1) = cnt
        if n0 and n1 and k0:
            pages.append({"page": pg, "rate_viewed": k1 / n1, "rate_not": k0 / n0, "ratio": (k1 / n1) / (k0 / n0), "n_viewed": n1})
    note(f"panel 3: {n_req} peticiones, OR(prod×slot)={or_prod_slot:.2f}, OR(slot)={or_slot:.2f}, obs {obs:.0f} vs esp {exp:.0f}")
    return {
        "files": len(files),
        "n_requests": n_req,
        "n_purchases": n_buy,
        "purchase_share": n_buy / n_req if n_req else None,
        "share_req_with_viewed_shown": n_req_with_viewed_shown / n_req if n_req else None,
        "share_buys_of_viewed": buy_viewed / n_buy if n_buy else None,
        "or_product_slot": or_prod_slot,
        "or_slot": or_slot,
        "observed": obs,
        "expected": exp,
        "rows": rows,
        "pages": pages,
    }


# ----------------------------------------------------------------------------- panel 4: ablation
def panel_ablation() -> dict:
    import csv

    with ABLATION_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    order: list[str] = []
    for r in rows:
        if r["variant"] not in order:
            order.append(r["variant"])
        groups[(r["variant"], r["segment"])].append(r)
    out = []
    for (variant, seg), rs in groups.items():
        for metric in ("hit_rate_buy@10", "dcg_buy@10"):
            vals = [float(r[metric]) for r in rs]
            out.append({
                "variant": variant,
                "segment": seg,
                "metric": metric,
                "mean": statistics.fmean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "seeds": len(vals),
            })
    # diferencia pareada por semilla frente a la base A_3.1b (DCG ALL)
    base = {r["seed"]: float(r["dcg_buy@10"]) for r in rows if r["variant"] == "A_3.1b" and r["segment"] == "ALL"}
    deltas = []
    for v in order:
        d = [float(r["dcg_buy@10"]) - base[r["seed"]] for r in rows
             if r["variant"] == v and r["segment"] == "ALL" and r["seed"] in base]
        if d:
            deltas.append({"variant": v, "mean": statistics.fmean(d), "min": min(d), "max": max(d), "seeds": len(d)})
    seeds = sorted({r["seed"] for r in rows})
    note(f"panel 4: {len(order)} variantes × {len(seeds)} semillas")
    return {"variants": order, "seeds": seeds, "rows": out, "deltas_dcg_all": deltas}


# ----------------------------------------------------------------------------- panel 6: log del servidor
LOG_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \[(\w+)\] (\S+): (.*)$")
SNAP_RE = re.compile(
    r"\[(calypso-[\w.]+) \| profile=(\w+)\] Live snapshot v(\d+) ready & cached in ([\d.]+)s \| active=(\d+) "
    r"\| ml_alpha=([\d.]+) \| regime=(\w+)"
)
SUB_RE = re.compile(r"\[(calypso-[\w.]+)\] Submitted (\d+) fresh answers")
GEST_RE = re.compile(r"===== \[gestor_models\] (Training|Launching) (\S+) \((\S+)\) at (\S+) =====")
LIVE_RE = re.compile(r"LIVE \(team=(\w+), model=([\w.\-]+), profile=(\w+)")


def _err_kind(msg: str) -> str:
    if "Read timed out" in msg:
        return "RawTree: lectura > timeout"
    if "ConnectTimeout" in msg or "Max retries" in msg:
        return "RawTree: sin conexión"
    if "WinError" in msg:
        return "Fichero bloqueado (Windows)"
    return "Otros"


def panel_log() -> dict | None:
    if not LIVE_LOG.exists():
        return None
    snaps, events, subs, errs = [], [], Counter(), []
    profiles: dict[str, str] = {}
    for line in LIVE_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        g = GEST_RE.search(line)
        if g:
            events.append({"ts": g.group(4).replace("T", " "), "kind": g.group(1), "folder": g.group(2), "model": g.group(3)})
            continue
        m = LOG_RE.match(line)
        if not m:
            continue
        ts, level, _src, msg = m.groups()
        s = SNAP_RE.search(msg)
        if s:
            snaps.append({"ts": ts, "model": s.group(1), "profile": s.group(2), "v": int(s.group(3)),
                          "secs": float(s.group(4)), "active": int(s.group(5)), "alpha": float(s.group(6)),
                          "regime": s.group(7)})
            profiles[s.group(1)] = s.group(2)
            continue
        s = SUB_RE.search(msg)
        if s:
            subs[(s.group(1), ts[:16])] += int(s.group(2))
            continue
        s = LIVE_RE.search(msg)
        if s:
            profiles[s.group(2)] = s.group(3)
        if level in ("WARNING", "ERROR"):
            errs.append({"ts": ts, "level": level, "kind": _err_kind(msg)})
    # submissions por minuto del último modelo que aparece en el log
    last_model = snaps[-1]["model"] if snaps else (events[-1]["model"] if events else None)
    per_min = sorted(((k[1], v) for k, v in subs.items() if k[0] == last_model))
    err_hour = Counter((e["ts"][:13], e["kind"]) for e in errs)
    last_snap = snaps[-1] if snaps else None
    model_snaps = [s for s in snaps if s["model"] == last_model]
    note(f"panel 6: {len(snaps)} snapshots, {len(errs)} avisos/errores, último modelo {last_model}")
    return {
        "snapshots": snaps,
        "events": events,
        "submissions_per_min": [{"minute": m, "answers": v} for m, v in per_min],
        "errors_by_hour": [{"hour": h, "kind": k, "count": c} for (h, k), c in sorted(err_hour.items())],
        "last_model": last_model,
        "last_snapshot": last_snap,
        "profiles": profiles,
        "snap_secs_model": [s["secs"] for s in model_snaps],
        "log_end": errs[-1]["ts"] if errs and (not per_min or errs[-1]["ts"][:16] > per_min[-1][0]) else (per_min[-1][0] if per_min else None),
    }


# ----------------------------------------------------------------------------- comprobaciones de coherencia
def checks(presets: dict, log: dict | None) -> list[dict]:
    out = []
    md = MODEL_DIR / "Comparacio_Models_6.2.b.md"
    md_txt = md.read_text(encoding="utf-8", errors="replace") if md.exists() else ""
    md_says_ppf = "0.7187" in md_txt or "0,7187" in md_txt
    out.append({
        "id": "ganador",
        "status": "warn" if md_says_ppf and presets["winner"] != "product_profile_focus" else "ok",
        "title": "Ganador del benchmark de presets",
        "detail": f"El JSON da ganador a {presets['winner']} (obj {presets['ranking'][0]['obj']:.4f}). "
                  + ("Comparacio_Models_6.2.b.md dice product_profile_focus (0,7187): el dashboard usa solo el JSON."
                     if md_says_ppf else "El .md no contradice al JSON."),
    })
    s = presets["sizes"]
    out.append({
        "id": "muestra",
        "status": "warn",
        "title": "Muestra pequeña en el benchmark",
        "detail": f"Tamaños inferidos de las fracciones del JSON: ALL {s['ALL']}, WARM {s['WARM']}, COLD {s['COLD']} sesiones con compra. "
                  "IC 95 % ≈ ±4 pts en ALL y ±8 en COLD: diferencias menores son ruido.",
    })
    stale = []
    for f in (MODEL_DIR / "README_6_2_b.md", MODEL_DIR / "model_manifest.json"):
        if f.exists() and "6.2a" in f.read_text(encoding="utf-8", errors="replace"):
            stale.append(f.name)
    hb = list(MODEL_DIR.glob("**/heartbeat*"))
    for f in hb:
        try:
            if "6.2a" in f.read_text(encoding="utf-8", errors="replace"):
                stale.append(f.relative_to(MODEL_DIR).as_posix())
        except Exception:
            pass
    prof = (log or {}).get("profiles", {}).get("calypso-6.2b")
    out.append({
        "id": "version",
        "status": "warn" if stale or prof else "ok",
        "title": "¿Qué corre exactamente en la final?",
        "detail": (f"Ficheros que aún dicen «6.2a»: {', '.join(stale)}. " if stale else "")
                  + (f"El log arranca calypso-6.2b con profile={prof}; el último commit habla de RADICAL1. Confirmar en la máquina servidora."
                     if prof else ""),
    })
    pin_in_62b = any("pin_viewed_top" in p.read_text(encoding="utf-8", errors="replace")
                     for p in MODEL_DIR.rglob("*.py"))
    out.append({
        "id": "pin",
        "status": "ok" if pin_in_62b else "warn",
        "title": "pin_viewed_top en 6.2b",
        "detail": "Está en el código de 6.2b." if pin_in_62b else
                  "No existe en el código de calypso_6_2_b (solo en odyseus_3_0): el panel 3 es una recomendación, no algo desplegado.",
    })
    out.append({
        "id": "unidades",
        "status": "info",
        "title": "Offline y en vivo no comparten unidad",
        "detail": "hit@10 offline (~70 %) está condicionado a sesiones con compra y a un simulador propio; en vivo medimos compras por 100 peticiones (~1). Nunca se dibujan en el mismo eje.",
    })
    return out


# ----------------------------------------------------------------------------- paneles 1 y 5: feedback en vivo
def _parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    s = s.strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    return None


def panel_live() -> dict | None:
    f = LIVE_DIR / "live_export.json"
    if not f.exists():
        note("paneles 1/5: no hay dashboard/live/live_export.json (ejecutar export_live.py con red a RawTree)")
        return None
    exp = json.loads(f.read_text(encoding="utf-8"))
    rows = exp.get("rows", [])
    new_products = set(exp.get("new_products", []))
    if not rows:
        return {"exported_at": exp.get("exported_at"), "empty": True}

    def user_type(r) -> str:
        if r.get("is_new_user") in (True, 1, "true", "1"):
            return "Nuevo"
        rv = r.get("recent_views")
        if isinstance(rv, str):
            try:
                rv = json.loads(rv)
            except Exception:
                rv = []
        if new_products and any(p in new_products for p in _rv_ids(rv)):
            return "Vio producto nuevo"
        return "Recurrente"

    recs = []
    for r in rows:
        t = _parse_ts(str(r.get("ts", "")))
        if t is None:
            continue
        ut = user_type(r)
        recs.append({
            "t": t,
            "team": r.get("served_by") == "team",
            "reason": r.get("fallback_reason") or "",
            "lat": float(r["latency_ms"]) if r.get("latency_ms") not in (None, "") else None,
            "buy": bool(r.get("purchased")),
            "click": bool(r.get("clicked")),
            "rev": float(r.get("revenue_eur") or 0),
            "page": r.get("page") or "?",
            "ut": ut,
            "cold": ut != "Recurrente",
            "model": r.get("model_version") or "fallback/desconocido",
        })
    recs.sort(key=lambda x: x["t"])
    n = len(recs)

    def rate(sel):
        k = sum(1 for x in sel if x["buy"])
        lo, hi = wilson(k, len(sel))
        return {"n": len(sel), "buys": k, "per100": 100 * k / len(sel) if sel else None, "lo": 100 * lo, "hi": 100 * hi}

    def pct(vals, q):
        if not vals:
            return None
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(q * (len(vals) - 1) + 0.5))]

    lats = [x["lat"] for x in recs if x["lat"] is not None and x["team"]]
    end = recs[-1]["t"]
    last5 = [x for x in recs if x["t"] >= end - timedelta(minutes=5)]
    kpi = {
        "all": rate(recs),
        "cold": rate([x for x in recs if x["cold"]]),
        "team_only": rate([x for x in recs if x["team"]]),
        "fallback_only": rate([x for x in recs if not x["team"]]),
        "on_time": sum(x["team"] for x in recs) / n,
        "on_time_last5": (sum(x["team"] for x in last5) / len(last5)) if last5 else None,
        "lat_p50": pct(lats, 0.5),
        "lat_p95": pct(lats, 0.95),
        "lat_p99": pct(lats, 0.99),
        "fallback_reasons": Counter(x["reason"] for x in recs if not x["team"]).most_common(),
        "window": [recs[0]["t"].isoformat(sep=" "), end.isoformat(sep=" ")],
    }
    grid = []
    for pg in sorted({x["page"] for x in recs}):
        for ut in ("Nuevo", "Vio producto nuevo", "Recurrente"):
            sel = [x for x in recs if x["page"] == pg and x["ut"] == ut]
            if sel:
                grid.append({"page": pg, "user": ut, **rate(sel)})
    bucket = int(exp.get("bucket_minutes", 10))
    series: dict[datetime, list] = defaultdict(list)
    for x in recs:
        b = x["t"].replace(second=0, microsecond=0)
        b -= timedelta(minutes=b.minute % bucket)
        series[b].append(x)
    ts = []
    for b in sorted(series):
        sel = series[b]
        r_ = rate(sel)
        ts.append({"t": b.isoformat(sep=" "), "per100": r_["per100"], "lo": r_["lo"], "hi": r_["hi"], "n": r_["n"],
                   "on_time": sum(x["team"] for x in sel) / len(sel),
                   "model": Counter(x["model"] for x in sel).most_common(1)[0][0]})
    changes = []
    prev = None
    for p in ts:
        if p["model"] != prev:
            changes.append({"t": p["t"], "model": p["model"], "source": "submissions"})
            prev = p["model"]
    if TRANSITIONS_JSON.exists():
        for tr in json.loads(TRANSITIONS_JSON.read_text(encoding="utf-8")):
            t = _parse_ts(tr.get("ts", "")[:19])
            if t and recs[0]["t"] <= t <= end:
                changes.append({"t": t.isoformat(sep=" "), "model": tr.get("model_version"), "source": "gestor_models"})
    by_model = [{"model": m, **rate([x for x in recs if x["model"] == m])} for m in sorted({x["model"] for x in recs})]
    note(f"paneles 1/5: {n} peticiones de feedback, {kpi['all']['per100']:.2f} compras/100")
    return {"exported_at": exp.get("exported_at"), "since": exp.get("since"), "kpi": kpi, "grid": grid,
            "series": ts, "bucket_minutes": bucket, "changes": changes, "by_model": by_model}


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-raw-files", type=int, default=None, help="limitar la muestra raw (pruebas rápidas)")
    args = ap.parse_args()

    presets = panel_presets()
    log = panel_log()
    data = {
        "built_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"),
        "constants": {"old_carousel_per100": OLD_CAROUSEL_CVR, "sla_ms": SLA_MS,
                      "weights": {"conversion": 50, "cold_start": 10, "reliability": 10}},
        "presets": presets,
        "viewed": panel_viewed(args.max_raw_files),
        "ablation": panel_ablation(),
        "log": log,
        "live": panel_live(),
        "checks": checks(presets, log),
    }
    tpl = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = tpl.replace("/*__DASHBOARD_DATA__*/null", payload)
    OUTPUT.write_text(html, encoding="utf-8")
    (HERE / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    note(f"escrito {OUTPUT.relative_to(ROOT)} ({len(html) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
