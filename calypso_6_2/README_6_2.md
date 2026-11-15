# Calypso 6.2 · Arquitectura Híbrida Corregida i Optimitzada

Versió evolutiva sobre `calypso_6_1` (`model_version = "calypso-6.2"`), construïda després del diagnòstic empíric sobre 25.000 sessions reals (`static_history_v4.pkl`) per corregir els punts febles estructurals detectats en `6.1`.

## Millores clau respecte Calypso 6.1

1. **Coherència Temporal Real & Senyals d'Intenció Repetida**:
   - Extreu `ts` de la petició i dels objectes `{product_id, ts}` de `recent_views` en lloc d'utilitzar `time.time()` de rellotge real (`eff_now_ms`), evitant el decaïment erroni de carrets.
   - Nou decay temporal continu basat en l'antiguitat de vista (`rv_temporal_recency`) i comptatge de re-visualitzacions (`rv_repeat_count_norm`, fins a ×3–4 d'intenció forta).

2. **50 Features ML + Destil·lació MLP Ponderada per IPS (`reranker.py`)**:
   - **40–49** noves features: `rv_repeat_count_norm`, `rv_temporal_recency`, `subcat_pop_rank_score`, `cat_pop_rank_score`, `price_rel_subcat_compat`, `is_viewed_cart_combo`, `ease_x_same_subcat`, `hop2_x_same_subcat`, `svd_sess_x_subcat`, `ctx_x_no_views`.
   - Pesos morts activats (`w_hop2_seq_score`, `w_covisit_score`, `w_rv_subcat_match_frac`, etc.).
   - `MLPRegressor(64, 32)` entrenat amb **remostreig balancejat i ponderat per IPS** sobre log-odds residuals veritables respecte un logit 1D calibrat.

3. **Candidate Pool Ampliat (92) amb Recuperació Intra-Subcategoria/Categoria (`recommender.py`, `ease.py`, `collab_v6.py`)**:
   - Índex `subcat_top_items` i `cat_top_items` per injectar líders de la subcategoria/categoria explorada (fins i tot sense aresta seqüencial directa).
   - Cap per vista en transicions perquè no ofeguin EASE, 2-hop, co-visit i subcategoria; ítems cold-start ordenats per coincidència subcategoria/categoria + popularitat.
   - EASE amb esmorteïment anti-hub (`(d_a * d_b)^0.04`), `hop2_transitions` fins a `nbs[:16]` i SVD col·laboratiu amb fins a 60.000 columnes d'usuaris.

4. **Contextualització Real i Neteja de Deute Tècnic**:
   - Matrius `page_product_*`, `page_category_*` i `page_home_*` realment diferenciades.
   - Slot-1 separa `same_subcat` (`slot1_category_match_bonus`) de `same_cat_only` (`slot1_cat_only_ratio`).
   - Corregits els 4 noms trencats d'`OPTIMIZABLE_PARAMS` a `tuner.py` i priorització de compres/clics a ClickHouse (`ORDER BY ... DESC LIMIT 12000`).

## Comandos principales

```bash
python calypso_6_2/serve.py --self-check
python calypso_6_2/serve.py --dry-run
python calypso_6_2/tuner.py --eval-presets
python -m unittest discover -s calypso_6_2/etapa_4/tests -p "test_*.py" -v
```
