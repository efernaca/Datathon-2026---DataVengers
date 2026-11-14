"""Run official offline evaluation notebook logic, caching downloaded data locally."""
import os
import sys
import time
from pathlib import Path
from collections import Counter
from collections.abc import Mapping
import numpy as np
import pandas as pd
import requests

# Load env
repo_root = Path(__file__).resolve().parent
env_file = repo_root / "group_API.env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip():
            os.environ.setdefault(k.strip(), v.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
API_KEY = os.environ.get("RAWTREE_API_KEY", "")

cache_dir = repo_root / "offline_eval_cache"
cache_dir.mkdir(exist_ok=True)
data_file = cache_dir / "official_test_data.parquet"
covisit_file = cache_dir / "covisit.parquet"
best_file = cache_dir / "best_sellers.json"

http = requests.Session()
http.headers["Authorization"] = f"Bearer {API_KEY}"

def sql(q: str, timeout: int = 600) -> pd.DataFrame:
    r = http.post(f"{API}/v1/query", json={"sql": q}, timeout=timeout)
    r.raise_for_status()
    return pd.DataFrame(r.json()["data"])

print("[1/4] Comprovant dades locals...")
if not data_file.exists() or not covisit_file.exists() or not best_file.exists():
    print(f"Dades locals no trobades a {cache_dir}. Connectant a RawTree per a la descàrrega única...")
    
    # 1. Trobar data de tall (CUT)
    last_day_df = sql("SELECT toDate(max(ts::DateTime64(3))) AS d FROM shop.history_sessions")
    last_day = last_day_df.d.iloc[0]
    CUT = str((pd.Timestamp(last_day) - pd.Timedelta(days=7)).date())
    print(f"Data màxima a history_sessions: {last_day} | CUT (últims 7 dies): {CUT}")
    
    # 2. Descarregar best sellers fins a CUT
    print("Descarregant best sellers històrics fins a CUT...")
    sales_df = sql(f"""
        SELECT product_id, count() AS n
        FROM shop.history_events
        WHERE event_type = 'purchase' AND ts < '{CUT}'
        GROUP BY product_id
        ORDER BY n DESC, product_id DESC
    """)
    best = sales_df["product_id"].tolist()
    import json
    best_file.write_text(json.dumps(best), encoding="utf-8")
    print(f"Best sellers descarregats i desats ({len(best)} productes).")

    # 3. Descarregar coviews
    print("Descarregant matriu de co-visites (coviews) de RawTree...")
    coviews = sql(f"""
        SELECT a.product_id AS p, b.product_id AS q, count() AS n
        FROM (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view' AND ts < '{CUT}') AS a
        INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view' AND ts < '{CUT}') AS b
          ON a.session_id = b.session_id
        WHERE a.product_id != b.product_id
        GROUP BY p, q ORDER BY n DESC LIMIT 20 BY p
    """)
    coviews.to_parquet(covisit_file)
    print(f"Co-visites desades a {covisit_file} ({len(coviews)} relacions).")

    # 4. Descarregar conjunt de test (sessions dels últims 7 dies)
    # Descarreguem 25.000 impressions de test (mostra estadística molt robusta)
    print("Descarregant impressions de test (shop.history_sessions ts >= CUT, LIMIT 25000)...")
    t0 = time.perf_counter()
    test_df = sql(f"""
        SELECT request_id, ts, user_id, context, recent_views, shown, propensity, clicked, purchased
        FROM shop.history_sessions
        WHERE ts >= '{CUT}'
        ORDER BY rand()
        LIMIT 25000
    """)
    print(f"Descarregades {len(test_df)} impressions en {time.perf_counter() - t0:.2f}s.")
    test_df.to_parquet(data_file)
    print(f"Dades desades correctament a {data_file}.")
else:
    print(f"Carregant dades des del fitxer local ({data_file})... Zero accés a RawTree.")
    import json
    best = json.loads(best_file.read_text(encoding="utf-8"))
    coviews = pd.read_parquet(covisit_file)
    test_df = pd.read_parquet(data_file)
    print(f"Dades carregades amb èxit: {len(test_df)} impressions de test locals.")

# Construir diccionari de covisites
covisit = coviews.groupby("p")["q"].apply(list).to_dict()

# Netejar clicked
test_df["clicked"] = test_df["clicked"].replace("", None)
print(f"Total impressions a avaluar: {len(test_df)} | CTR observat històric: {round(test_df.clicked.notna().mean(), 4):.4f}")

# Definir polítiques oficials del notebook
def baseline(row):
    return best[:10]

def coview(row):
    score = Counter()
    rv = row.recent_views if hasattr(row, "recent_views") else []
    if isinstance(rv, (list, tuple, np.ndarray)):
        for k, v in enumerate(reversed(rv)):
            pid = v.get("product_id") if isinstance(v, dict) else (v["product_id"] if isinstance(v, (dict, Mapping)) or hasattr(v, "__getitem__") else None)
            if pid and pid in covisit:
                for rank, q in enumerate(covisit[pid]):
                    score[q] += 0.7 ** k / (rank + 1)
    items = [q for q, _ in score.most_common(10)]
    return (items + [p for p in baseline(row) if p not in items])[:10]

POLICIES = {
    "official_baseline": baseline,
    "official_coview": coview,
}

# Funció d'avaluació oficial del notebook
def evaluate(policy, df):
    t0 = time.perf_counter()
    recs = [policy(r) for r in df.itertuples(index=False)]
    lat_ms = (time.perf_counter() - t0) * 1000.0 / max(len(df), 1)

    clicked = df.clicked.to_numpy()
    has_click = df.clicked.notna().to_numpy()
    hit = np.array([c in r for c, r in zip(clicked, recs)])[has_click].mean() if has_click.sum() > 0 else 0.0
    
    first_shown = df.shown.str[0].to_numpy()
    match = np.array([r[0] == s for r, s in zip(recs, first_shown)])
    reward = (clicked == first_shown).astype(float)            # click on the slot-1 product
    
    prop_s1 = np.array([float(p[0]) if len(p) > 0 and float(p[0]) > 0 else 1.0 for p in df.propensity])
    w = match / prop_s1
    
    ips = (w * reward).mean()
    snips = (w * reward).sum() / max(w.sum(), 1e-9)
    
    return {
        "hit_rate@10": hit,
        "IPS_ctr_slot1": ips,
        "SNIPS_ctr_slot1": snips,
        "slot1_matches": int(match.sum()),
        "latency_ms_per_req": lat_ms,
    }

print("\n[2/4] Avaluant les polítiques oficials...")
old_slot1 = (test_df.clicked == test_df.shown.str[0]).mean()
print(f"Sistema antic (registre històric) - CTR observat Slot-1: {old_slot1:.4f}")

results = {name: evaluate(p, test_df) for name, p in POLICIES.items()}
res_df = pd.DataFrame(results).T
print("\n" + "="*80)
print("RESULTATS OFICIALS DE L'AVALUACIÓ OFFLINE (Notebook de l'organització)")
print("="*80)
print(res_df.to_string())
print("="*80)
