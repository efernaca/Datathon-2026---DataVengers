# Odyseus (`odyseus-4.1e`) · Optimitzador Iteratiu de Conversió Offline

`odyseus` parteix d'una còpia encapsulada de `calypso_4_1_e` i incorpora un optimitzador iteratiu offline (`optimize.py` / `etapa_4/src/odyseus_optimizer.py`) dissenyat per trobar la combinació òptima de paràmetres que maximitza la taxa de conversió.

## Què optimitza

Explora tot l'espai de `WeightsProfile` (~65 paràmetres) en fases (exploració àmplia, grups funcionals acoblats, refinament local i *momentum*):

- **14 pesos prior del re-ranker:** `w_seq_lift`, `w_self_cvr_view`, `w_intent_score`, `w_user_affinity`, `w_price_compat`, `w_ctx_bayes`, `w_global_pop`, `w_live_trend`, `w_is_new_product`, `w_reverse_graft`, `w_forward_cold`, `w_log_price_norm`, `w_same_top_subcat`, `w_same_top_brand`.
- **Multiplicadors dels 5 segments d'usuari:** `new_with_views`, `new_no_views`, `known_with_views`, `known_with_history`, `known_cold`.
- **Penalització de fons, sòls i diversitat MMR:** `v3_bg_penalty`, `bg_penalty_diff_subcat`, `bg_penalty_same_subcat`, `v3_no_subcat_floor`, `subcat_floor_min`, `brand_floor_min`, `mmr_*`.
- **Exploració i *slot gating*:** `bandit_ucb_alpha`, `cold_explore_start_slot_warm`, `max_cold_explore_in_warm`.
- **Preu i desempat:** `price_std_floor`, `price_cheap_asymmetry`, `revenue_tie_weight`, `tie_break_mode`.
- **Regles de negoci 4.1:** `v4_cart_priority`, `v4_cart_active_minutes`, `v4_cart_max_slots`, `v4_forgotten_cart`, `v4_forgotten_max_slots`, `v4_price_ascending`, `v4_recurrent`, `v4_recurrent_max_slots`, `v4_live_event_purchases`.

## Avaluació offline amb les dades disponibles

1. **Mode `hybrid` (per defecte):** combina:
   - **Dades reals del repositori** (`offline_eval_cache/official_test_data.parquet` + `odyseus/cache/snapshot_v4_latest.pkl`, eliminant les compres futures del conjunt de test de `bought_products` per evitar *leakage*): calcula `hit_rate_buy@10` i `dcg_buy@10` exactes sobre totes les 386 compres reals, `hit_rate_click@10` sobre els 921 clics reals, i `SNIPS_cvr_slot1` no esbiaixat.
   - **Benchmark sintètic estratificat** (`ALL`, `WARM`, `COLD_START`) per protegir els nous usuaris i productes nous (`cold_item_hit@10`).
2. **Mode `real`:** avalua només sobre `official_test_data.parquet` (amb `--full-real` es passen les 25.000 peticions senceres a cada iteració).
3. **Mode `synthetic`:** avalua només sobre el simulador sintètic estratificat.

## Arxiu únic d'historial (`odyseus_history.json`)

Després de **cada iteració**, s'actualitza atòmicament l'arxiu `odyseus/results/odyseus_history.json` amb:
- Metadades globals (`best_iter`, `best_objective`, `baseline_objective`, `best_metrics`, `best_params`).
- L'array `iterations` amb **totes** les iteracions: `iter`, `timestamp`, `strategy`, `accepted`, `objective`, `changed_params`, `metrics` (`real` + `synthetic`) i `params` (tota la configuració completa).

## Comandes d'ús

```powershell
cd odyseus

# 1. Tests i comprovacions ràpides
python -m unittest discover -s etapa_4/tests -v
python serve.py --self-check
python serve.py --dry-run

# 2. Executar optimització iterativa offline (guarda historial a results/odyseus_history.json)
python optimize.py --iters 30 --save-best

# Continuar afegint iteracions sobre el mateix arxiu d'historial
python optimize.py --iters 20 --resume --save-best

# Només dades reals completes (25.000 peticions a cada iteració)
python optimize.py --eval-mode real --full-real --iters 15 --save-best

# També accessible via tuner.py
python tuner.py --optimize --iters 30 --save-best
```
