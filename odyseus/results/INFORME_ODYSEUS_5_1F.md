# Informe d'Optimització Odyseus — calypso-5.1f

- **Data / Hora actualització:** `2026-11-15T01:18:58+00:00`
- **Mode d'avaluació:** `real` | **Durada execució:** `4201.6s`
- **Iteracions totals:** `609` | **Millor iteració (fast):** `#590`
- **Objectiu Baseline:** `0.395591`
- **Objectiu Millor:** `0.544774`

## Resum per `scoring_mode` (cerca ràpida)

| Scoring Mode | Millor Iter | Objective | Hit@10 Buy | Hit@3 Buy | DCG@10 Buy | Attn@10 | Latència (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hybrid` | 597 | 0.52431 | 0.5959 | 0.4974 | 0.4995 | 0.5321 | 1.92 |
| `calibrated` | 586 | 0.50823 | 0.5829 | 0.4689 | 0.4793 | 0.5140 | 2.04 |
| `legacy` | 566 | 0.46305 | 0.5699 | 0.4249 | 0.4273 | 0.4746 | 1.01 |

## Verificació Exacta Full-Real (totes les 25.000 peticions reals)

| Rank | Candidat | Scoring Mode | Real Obj | Hit@10 Buy | Hit@3 Buy | Hit@1 Buy | DCG@10 | Attn@10 | Product Buy@10 | Category Buy@10 | Home Buy@10 | SNIPS CVR1 | Lat (ms) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `iter_590` | `hybrid` | 0.54477 | 0.6425 | 0.5026 | 0.4016 | 0.5099 | 0.5546 | 0.6230 | 0.6833 | 0.6429 | 0.07035 | 3.60 |
| 2 | `iter_583` | `hybrid` | 0.54447 | 0.6425 | 0.5026 | 0.3990 | 0.5090 | 0.5540 | 0.6230 | 0.6833 | 0.6429 | 0.07035 | 3.67 |
| 3 | `iter_584` | `hybrid` | 0.54442 | 0.6425 | 0.5026 | 0.3990 | 0.5090 | 0.5540 | 0.6230 | 0.6833 | 0.6429 | 0.07037 | 3.68 |
| 4 | `iter_581` | `hybrid` | 0.54061 | 0.6425 | 0.5000 | 0.3886 | 0.5050 | 0.5515 | 0.6230 | 0.6833 | 0.6429 | 0.06284 | 3.41 |
| 5 | `iter_606` | `hybrid` | 0.53980 | 0.6425 | 0.5000 | 0.3808 | 0.5018 | 0.5496 | 0.6230 | 0.6833 | 0.6429 | 0.06481 | 2.43 |
| 6 | `iter_580` | `hybrid` | 0.53980 | 0.6425 | 0.5000 | 0.3834 | 0.5031 | 0.5503 | 0.6230 | 0.6833 | 0.6429 | 0.06280 | 2.31 |
| 7 | `iter_602` | `hybrid` | 0.53978 | 0.6425 | 0.5000 | 0.3808 | 0.5018 | 0.5496 | 0.6230 | 0.6833 | 0.6429 | 0.06471 | 2.23 |
| 8 | `iter_608` | `hybrid` | 0.53971 | 0.6425 | 0.5000 | 0.3808 | 0.5018 | 0.5496 | 0.6230 | 0.6833 | 0.6429 | 0.06444 | 2.36 |
| 9 | `iter_599` | `hybrid` | 0.53837 | 0.6425 | 0.4974 | 0.3756 | 0.4995 | 0.5479 | 0.6230 | 0.6833 | 0.6429 | 0.06418 | 2.31 |
| 10 | `fast_best_iter597` | `hybrid` | 0.53712 | 0.6425 | 0.4974 | 0.3679 | 0.4965 | 0.5462 | 0.6230 | 0.6833 | 0.6429 | 0.06406 | 2.33 |
| 11 | `fast_mode_calibrated_iter586` | `calibrated` | 0.51038 | 0.6218 | 0.4637 | 0.3238 | 0.4637 | 0.5175 | 0.5873 | 0.6833 | 0.7143 | 0.06172 | 2.36 |
| 12 | `fast_mode_legacy_iter566` | `legacy` | 0.46577 | 0.5829 | 0.4326 | 0.2798 | 0.4221 | 0.4771 | 0.5595 | 0.6250 | 0.6429 | 0.02853 | 1.24 |
| 13 | `preset:calibrated_hybrid` | `hybrid` | 0.43795 | 0.5570 | 0.3912 | 0.2280 | 0.3828 | 0.4434 | 0.5238 | 0.6333 | 0.5000 | 0.02742 | 2.94 |
| 14 | `preset:calibrated_logit` | `calibrated` | 0.43715 | 0.5622 | 0.3912 | 0.2176 | 0.3767 | 0.4416 | 0.5317 | 0.6333 | 0.5000 | 0.02682 | 3.22 |
| 15 | `base:calibrated_m3_odyseus_opt` | `legacy` | 0.40799 | 0.5337 | 0.3316 | 0.2228 | 0.3547 | 0.4125 | 0.5040 | 0.6000 | 0.5000 | 0.02032 | 1.80 |
| 16 | `baseline:global_bestsellers` | `baseline` | 0.40532 | 0.5518 | 0.2927 | 0.1658 | 0.3299 | 0.4028 | 0.5357 | 0.5750 | 0.6429 | 0.03618 | 0.00 |
| 17 | `preset:calibrated_m3` | `legacy` | 0.33950 | 0.4171 | 0.3135 | 0.2176 | 0.3104 | 0.3470 | 0.3929 | 0.4583 | 0.5000 | 0.01592 | 1.78 |
| 18 | `preset:conversion_first` | `legacy` | 0.29395 | 0.3575 | 0.2746 | 0.1917 | 0.2706 | 0.3004 | 0.3254 | 0.4083 | 0.5000 | 0.01544 | 1.94 |

## Rendiment per Grup de Paràmetres (Adaptive UCB)

| Grup | Proves | Victòries | Win Rate |
|---|---:|---:|---:|
| `calibrated_logit` | 154 | 34 | 22.1% |
| `segments` | 157 | 16 | 10.2% |
| `session_core` | 77 | 13 | 16.9% |
| `personalization_price` | 77 | 11 | 14.3% |
| `cart_recurrent_rules` | 103 | 7 | 6.8% |
| `base_popularity_5_1` | 85 | 6 | 7.1% |
| `diversity_mmr` | 80 | 6 | 7.5% |
| `cold_start` | 90 | 5 | 5.6% |
| `bandit_ml` | 72 | 3 | 4.2% |

## Top 35 Paràmetres per Impacte i Taxa de Victòria

| Paràmetre | Grup | Proves | Victòries | Win% | Impacte Mitjà | Guany Positiu Total | Valor Campió |
|---|---|---:|---:|---:|---:|---:|---:|
| `recency_decay` | `session_core` | 31 | 8 | 25.8% | 0.00131 | 0.00049 | `0.7688` |
| `cal_w_cart_active` | `calibrated_logit` | 15 | 7 | 46.7% | 0.00399 | 0.00311 | `2.1082` |
| `view_last_product_page_factor` | `session_core` | 37 | 6 | 16.2% | 0.00199 | 0.00176 | `1.2524` |
| `lift_prior_pop` | `session_core` | 33 | 6 | 18.2% | 0.00254 | 0.00091 | `54.2158` |
| `seg_known_with_views__aff` | `segments` | 15 | 5 | 33.3% | 0.00323 | 0.00436 | `2.1419` |
| `tie_break_mode` | `personalization_price` | 29 | 5 | 17.2% | 0.00414 | 0.00338 | `cheap` |
| `mmr_subcat_lambda_session` | `diversity_mmr` | 29 | 5 | 17.2% | 0.00330 | 0.00204 | `0.0024` |
| `tie_break_eps` | `personalization_price` | 25 | 5 | 20.0% | 0.00267 | 0.00203 | `0.038` |
| `cal_home_rank_mult` | `calibrated_logit` | 17 | 5 | 29.4% | 0.00788 | 0.00195 | `0.8094` |
| `price_cheap_asymmetry` | `personalization_price` | 27 | 5 | 18.5% | 0.00248 | 0.00182 | `0.1437` |
| `lift_beta` | `session_core` | 32 | 5 | 15.6% | 0.00313 | 0.00066 | `0.3865` |
| `cal_w_aff` | `calibrated_logit` | 15 | 4 | 26.7% | 0.00944 | 0.00653 | `0.5183` |
| `guaranteed_top_sellers_in_pool` | `base_popularity_5_1` | 19 | 4 | 21.1% | 0.00238 | 0.00304 | `50` |
| `seg_known_with_views__intent` | `segments` | 15 | 4 | 26.7% | 0.00450 | 0.00301 | `0.9441` |
| `cal_w_subcat` | `calibrated_logit` | 13 | 4 | 30.8% | 0.00279 | 0.00255 | `0.3079` |
| `sales_rank_blend_weight` | `base_popularity_5_1` | 24 | 4 | 16.7% | 0.00379 | 0.00212 | `0.8801` |
| `mmr_subcat_lambda_home` | `diversity_mmr` | 25 | 4 | 16.0% | 0.00206 | 0.00202 | `0.1334` |
| `cal_w_recurrent` | `calibrated_logit` | 15 | 4 | 26.7% | 0.00726 | 0.00185 | `0.7018` |
| `cal_rel_cap` | `calibrated_logit` | 10 | 4 | 40.0% | 0.00936 | 0.00044 | `0.7046` |
| `v4_cart_active_minutes` | `cart_recurrent_rules` | 24 | 3 | 12.5% | 0.01365 | 0.00473 | `15.6791` |
| `allow_recurrent_rebuy` | `cart_recurrent_rules` | 17 | 3 | 17.6% | 0.01613 | 0.00440 | `False` |
| `cal_w_rank` | `calibrated_logit` | 17 | 3 | 17.6% | 0.01088 | 0.00261 | `0.5498` |
| `mmr_start_slot_home` | `diversity_mmr` | 28 | 3 | 10.7% | 0.00219 | 0.00197 | `6` |
| `seg_new_with_views__ctx` | `segments` | 17 | 3 | 17.6% | 0.00197 | 0.00148 | `1.6159` |
| `seg_known_with_history__aff` | `segments` | 12 | 3 | 25.0% | 0.00625 | 0.00108 | `0.6926` |
| `cal_w_seen` | `calibrated_logit` | 13 | 3 | 23.1% | 0.00189 | 0.00070 | `2.5501` |
| `v4_recurrent` | `cart_recurrent_rules` | 18 | 2 | 11.1% | 0.01317 | 0.00490 | `False` |
| `seg_known_with_views__pop` | `segments` | 15 | 2 | 13.3% | 0.00529 | 0.00372 | `0.9141` |
| `seg_new_no_views__ctx` | `segments` | 13 | 2 | 15.4% | 0.00583 | 0.00240 | `1.0673` |
| `seg_known_with_views__seq` | `segments` | 12 | 2 | 16.7% | 0.00715 | 0.00222 | `1.6857` |
| `scoring_mode` | `calibrated_logit` | 28 | 2 | 7.1% | 0.03950 | 0.00197 | `hybrid` |
| `fallback_live_trend_weight` | `base_popularity_5_1` | 13 | 2 | 15.4% | 0.00393 | 0.00197 | `0.25` |
| `mmr_brand_lambda` | `diversity_mmr` | 24 | 2 | 8.3% | 0.00281 | 0.00197 | `0.02` |
| `candidate_pool_size` | `base_popularity_5_1` | 22 | 2 | 9.1% | 0.00253 | 0.00197 | `130` |
| `mmr_start_slot_session` | `diversity_mmr` | 26 | 2 | 7.7% | 0.00133 | 0.00197 | `7` |

## Diferències Principals del Campió vs `calypso_5_1_default`

| Paràmetre | Default 5.1 | Campió Odyseus |
|---|---|---|
| `view_last_product_page_factor` | `1.0` | `1.2524` |
| `recency_decay` | `0.7` | `0.7688` |
| `lift_beta` | `0.45` | `0.3865` |
| `lift_prior_pop` | `30.0` | `54.2158` |
| `sales_rank_blend_weight` | `0.32` | `0.8801` |
| `live_pop_mix_weight` | `0.45` | `0.4277` |
| `fallback_live_trend_weight` | `0.6` | `0.25` |
| `transition_dest_pop_gamma` | `0.2` | `0.0` |
| `candidate_pool_size` | `78` | `130` |
| `guaranteed_top_sellers_in_pool` | `14` | `50` |
| `v3_bg_penalty` | `True` | `False` |
| `scoring_mode` | `legacy` | `hybrid` |
| `cal_w_rank` | `0.69` | `0.5498` |
| `cal_w_seen` | `2.3` | `2.5501` |
| `cal_w_rel` | `0.61` | `0.6406` |
| `cal_rel_cap` | `1.0` | `0.7046` |
| `cal_w_cart_active` | `2.6` | `2.1082` |
| `cal_w_cart_forgotten` | `0.75` | `0.6559` |
| `cal_w_recurrent` | `0.45` | `0.7018` |
| `cal_w_intent` | `0.45` | `0.4487` |
| `cal_w_aff` | `0.3` | `0.5183` |
| `cal_w_cold` | `0.45` | `0.2428` |
| `cal_w_subcat` | `0.12` | `0.3079` |
| `cal_w_brand` | `0.06` | `0.136` |
| `cal_w_discount` | `0.18` | `0.294` |
| `cal_home_rank_mult` | `1.25` | `0.8094` |
| `cal_hybrid_ml_weight` | `0.25` | `0.3972` |
| `cold_sparse_cos_weight` | `0.65` | `0.7723` |
| `cold_cat_match_bonus` | `0.15` | `0.0` |
| `cold_brand_match_bonus` | `0.12` | `0.134` |
| `cold_explore_start_slot_warm` | `7` | `8` |
| `price_std_floor` | `0.45` | `0.4652` |
| `price_cheap_asymmetry` | `0.0` | `0.1437` |
| `revenue_tie_weight` | `0.0` | `0.0135` |
| `tie_break_mode` | `conversion` | `cheap` |
| `tie_break_eps` | `0.012` | `0.038` |
| `v4_cart_active_minutes` | `30.0` | `15.6791` |
| `v4_cart_max_slots` | `2` | `1` |
| `v4_recurrent_max_slots` | `2` | `3` |
| `allow_recurrent_rebuy` | `True` | `False` |
| `v4_live_event_purchases` | `True` | `False` |
| `v3_no_subcat_floor` | `False` | `True` |
| `mmr_start_slot_session` | `5` | `7` |
| `mmr_start_slot_home` | `3` | `6` |
| `mmr_subcat_lambda_session` | `0.0` | `0.0024` |
| `mmr_subcat_lambda_home` | `0.06` | `0.1334` |
| `mmr_brand_lambda` | `0.03` | `0.02` |
| `mmr_relative_penalty` | `True` | `False` |
| `bandit_cold_ucb_multiplier` | `1.8` | `1.7273` |
| `use_empirical_exam_curve` | `True` | `False` |
| `v3_ease_weight` | `0.0` | `0.0711` |
| `seg_new_with_views__seq` | `1.12` | `1.2885` |
| `seg_new_with_views__ctx` | `1.15` | `1.6159` |
| `seg_new_no_views__ctx` | `1.45` | `1.0673` |
| `seg_new_no_views__trend` | `1.4` | `1.018` |
| `seg_new_no_views__cold` | `1.2` | `1.2817` |
| `seg_known_with_views__seq` | `1.15` | `1.6857` |
| `seg_known_with_views__intent` | `1.15` | `0.9441` |
| `seg_known_with_views__aff` | `1.1` | `2.1419` |
| `seg_known_with_views__pop` | `0.85` | `0.9141` |
