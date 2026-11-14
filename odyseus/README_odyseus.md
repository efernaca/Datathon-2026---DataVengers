# Odyseus (`calypso-5.1f`) · Optimitzador Adaptatiu sobre Calypso 5.1

`odyseus` utilitza ara **Calypso 5.1** com a model de referència (`model_version = "calypso-5.1f"`) per optimitzar iterativament combinacions de paràmetres orientades a màxima conversió a compra.

## Espai de cerca (`~99` variables en `8` grups)

1. **`base_popularity_5_1`**: `use_sales_rank_base_rate`, `sales_rank_blend_weight`, `live_pop_mix_weight`, `transition_dest_pop_gamma`, `candidate_pool_size`, `guaranteed_top_sellers_in_pool`, `w_ctx_bayes`, `w_global_pop`, `w_live_trend`, `v3_bg_penalty`, `bg_penalty_diff_subcat`, `bg_penalty_same_subcat`.
2. **`session_core`**: `w_seq_lift`, `w_self_cvr_view`, `w_intent_score`, `w_same_top_subcat`, `w_same_top_brand`, `view_last_product_page_factor`, `view_base_bonus`, `recency_decay`, `slot1_category_match_bonus`, `same_cat_bonus_weight`.
3. **`cart_recurrent_rules`**: `v4_cart_priority`, `v4_cart_active_minutes`, `v4_cart_max_slots`, `cart_active_score_bonus`, `v4_forgotten_cart`, `v4_forgotten_max_slots`, `cart_forgotten_score_bonus`, `v4_recurrent`, `v4_recurrent_max_slots`, `recurrent_score_bonus`, `allow_recurrent_rebuy`, `v4_price_ascending`, `v4_live_event_purchases`.
4. **`cold_start`**: `w_is_new_product`, `w_reverse_graft`, `w_forward_cold`, `cold_neighbor_pop_prior_weight`, `cold_sparse_cos_weight`, `cold_subcat_match_bonus`, `cold_cat_match_bonus`, `cold_brand_match_bonus`, `cold_price_prox_bonus`, `cold_explore_start_slot_warm`, `cold_explore_start_slot_cold`, `max_cold_explore_in_warm`.
5. **`personalization_price`**: `w_user_affinity`, `w_price_compat`, `w_log_price_norm`, `price_std_floor`, `price_cheap_asymmetry`, `revenue_tie_weight`, `tie_break_mode`, `tie_break_eps`, `discount_boost_weight`.
6. **`diversity_mmr`**: `v3_no_subcat_floor`, `subcat_floor_min`, `brand_floor_min`, `mmr_start_slot_session`, `mmr_start_slot_home`, `mmr_subcat_lambda_session`, `mmr_subcat_lambda_home`, `mmr_brand_lambda`, `mmr_relative_penalty`.
7. **`bandit_ml`**: `bandit_ucb_alpha`, `bandit_cold_ucb_multiplier`, `bandit_warm_ucb_multiplier`, `use_empirical_exam_curve`, `bandit_exam_power`.
8. **`segments`**: 29 multiplicadors actius dels 5 segments d'usuari (`new_with_views`, `new_no_views`, `known_with_views`, `known_with_history`, `known_cold`).

## Selecció intel·ligent adaptativa (`adaptive_ucb_bandit`)

A cada iteració, Odyseus aprèn de totes les iteracions prèvies (`odyseus_history.json`) mitjançant un selector **UCB per grup i per variable** (`compute_adaptive_intelligence`):
- Dona prioritat més alta als grups i paràmetres que han aportat guany positiu (`wins`, `pos_gain`) o major sensibilitat real en l'objectiu de conversió.
- Manté un terme d'exploració `sqrt(log(N+1)/(trials+1))` perquè les variables noves del 5.1 no provades encara tinguin prioritat inicial alta.
- Aplica *momentum* després d'una millora acceptada i garanteix configuracions úniques (hash SHA-256).

## Comandes principals

```powershell
# Llançar ronda d'iteracions offline (dades reals + sintètiques) i guardar la millor combinació
python odyseus/optimize.py --iters 30 --save-best

# Continuar iterant sobre el mateix historial únic (odyseus/results/odyseus_history.json)
python odyseus/optimize.py --resume --iters 30 --save-best

# Centrar la cerca només en certs grups (ex. base_popularity_5_1 i session_core)
python odyseus/optimize.py --focus-groups base_popularity_5_1 session_core --iters 25 --save-best

# Avaluació 100% real sobre les 25.000 peticions completes
python odyseus/optimize.py --eval-mode real --full-real --iters 20 --save-best
```
