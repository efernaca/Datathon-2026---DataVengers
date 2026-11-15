# Calypso 5.1 · Model Modular d'Alta Conversió

Versió evolutiva sobre `calypso_4_1_e` (`model_version = "calypso-5.1"`) dissenyada per maximitzar la conversió efectiva a compra sense perdre la fortalesa de cold-start (`~2,5%` CVR) i mantenint governança completa via `weights_profile.json` (*hot-reload* en calent).

---

## Mesures implementades respecte a `calypso_4_1_e`

| Codi | Mesura implementada | Detall tècnic i ubicació |
|---|---|---|
| **M1 / R7** | Sense ordenació per preu ascendent | `v4_price_ascending = false` i `tie_break_mode = "conversion"` per defecte; desempat relatiu `tie_break_eps = 0.015` només dins la mateixa subcategoria quan s'activa `cheap`/`revenue` (`recommender.py`). |
| **M3 / D2** | Taxa base explícita per rànquing de vendes | `use_sales_rank_base_rate = true` (`sales_rank_blend_weight = 0.68`, `w_global_pop = 0.56`) combina `1 - ln(rank)/ln(max_rank)` amb `log1p(weighted_pop)` a `build_snapshot_v4_from_records`. |
| **D3 / R5** | Pool ampliat + Top-35 vendes garantit | `candidate_pool_size = 140` i `guaranteed_top_sellers_in_pool = 35`; `_gather_candidates` reserva els supervendes i contextuals abans d'omplir amb transicions. |
| **D4** | Eliminació del `×0.25` hardcoded al fons | Quan `v3_bg_penalty = false` s'elimina la penalització hardcoded; amb `v3_bg_penalty = true` es suavitzen (`0.85` mateixa subcat / `0.48` altra subcat). |
| **D5** | Co-visites modulades per taxa base de destí | `transition_dest_pop_gamma = 0.50` pondera el *lift* amb `global_pop_norm[dst]` perquè les co-visites de cua no superin productes que venen. |
| **D6 / R1** | "Vist" sense penalització `0.65` en fitxa producte | `view_last_product_page_factor = 1.0`, `view_base_bonus = 0.55` i `w_self_cvr_view = 2.15`. |
| **D8 / M4** | Tendències en viu amb finestra recent de 60 min | `SQL_LIVE_TRENDS` a `serve.py` filtra els últims 60 minuts amb fallback segur (`SQL_LIVE_TRENDS_FALLBACK`), recull `live_purchases` / `live_carts` i els barreja al rànquing (`live_pop_mix_weight = 0.45`). |
| **M5** | Corba d'atenció empírica al Bandit / IPS | `use_empirical_exam_curve = true` aplica `(1.00, 0.78, 0.72, 0.60, 0.58, 0.55, 0.52, 0.50, 0.50, 0.50)` a `bandits.py` i `reranker.py`. |
| **M6 / R3 / R6** | Carret oblidat i recurrents per puntuació | `v4_forgotten_cart = false` i `v4_recurrent = false` eviten bloquejar slots; competeixen mitjançant `cart_forgotten_score_bonus = 0.35`, `recurrent_score_bonus = 0.25` i `allow_recurrent_rebuy = true`. Carret actiu conserva prioritat màx. `2` slots (`v4_cart_max_slots = 2`). |
| **M7 / D11 / X9** | `home` sense diversificació MMR forçada | `mmr_subcat_lambda_home = 0.0` i `mmr_brand_lambda = 0.02`. |
| **R2** | Coherència de categoria al slot 1 + mateixa categoria | `slot1_category_match_bonus = 0.20` i `same_cat_bonus_weight = 0.08` quan coincideix `category`. |
| **R8 / D7** | Protecció del cold-start (`cold_gate` configurable) | `cold_explore_start_slot_cold = 0` per mantenir el 2,5% en cold-start, modificable en calent sense reiniciar. |
| **X10** | Prior de vendes pels productes nous via veïns | `cold_neighbor_pop_prior_weight = 0.80` hereta popularitat mitjana ponderada dels veïns de contingut. |
| **X11** | Reforç per baixades de preu recents | `ProductInfo.discount_ratio` detecta rebaixes a `catalog_updates` i suma `discount_boost_weight = 0.15`. |

---

## Governança ràpida (Hot-Reload)

```powershell
# Avaluar tots els presets (inclou calibrated_m3)
python calypso_5_1/tuner.py --eval-presets

# Canviar paràmetres en calent (serve.py recarrega weights_profile.json automàticament)
python calypso_5_1/tuner.py --set cold_explore_start_slot_cold=7
python calypso_5_1/tuner.py --preset calibrated_m3

# Optimització iterativa
python calypso_5_1/tuner.py --optimize --iters 20 --save-best

# Verificacions locals
python calypso_5_1/serve.py --self-check
python calypso_5_1/serve.py --dry-run
python calypso_5_1/eval.py
```
