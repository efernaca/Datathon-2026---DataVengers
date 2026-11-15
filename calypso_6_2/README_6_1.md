# Calypso 6.1 · Modelo Híbrido de Máxima Conversión

Versión evolutiva mayor sobre `calypso_5_1` (`model_version = "calypso-6.1"`), diseñada para maximizar la tasa efectiva de conversión a compra (`hit_rate_buy@10`, `dcg_buy@10`, `SNIPS_cvr_slot1`), proteger la exploración de productos nuevos (*cold-start*), y mantener inferencia en tiempo real muy por debajo del SLA de 3 segundos (< 2 ms/req).

## Arquitectura Calypso 6.1

1. **Vectorización de 40 Features ML** (`calypso_6_1/etapa_4/src/reranker.py`):
   - **0–13**: Base Calypso 5.1 (`seq_lift`, `self_cvr_view`, `intent_score`, `user_affinity`, `price_compat`, `ctx_bayes`, `global_pop`, `live_trend`, `is_new`, `reverse_graft`, `forward_cold`, `log_price_norm`, `same_top_subcat`, `same_top_brand`).
   - **14–18**: Grafos colaborativos y embeddings: `ease_score`, `hop2_score`, `covisit_score`, `svd_session_sim`, `svd_user_sim`.
   - **19–28**: Señales continuas temporales, intención y negocio: `is_last_viewed`, `view_recency_decay`, `cart_active_decay`, `cart_forgotten_decay`, `recurrent_log`, `discount_ratio`, `rv_subcat_coherence`, `rv_brand_coherence`, `rv_price_coherence`, `same_category`.
   - **29–39**: Contexto de página one-hot (`product`/`category`/`home`) e interacciones no lineales explícitas.

2. **Ranker Híbrido de 2 Capas (Lineal IPS + MLP Residual Destilado NumPy)**:
   - **Capa 1**: `LogisticRegression` con ponderación IPS empírica y anclaje de priors con signo (`min_prior_ratios`).
   - **Capa 2**: Teacher `HistGradientBoostingClassifier` destilado a una red `MLPRegressor(48, 24, tanh)` y extraída a tensores `float32` (`nl_W1`, `nl_b1`, `nl_W2`, `nl_b2`, `nl_W3`, `nl_b3`) para inferencia pura en NumPy sin dependencias de runtime.

3. **Fusión EASE + Difusión 2-Hop + Co-Visitation Pre-CUT + SVD 32d** (`ease.py`, `collab_v6.py`):
   - Fusión EASE cerrado (`lambda = 240.0`) con aristas de co-visitation históricas pre-CUT (`covisit.parquet`, sin fuga de datos de test) y difusión dispersa a 2 saltos.
   - Índice colaborativo `TruncatedSVD(32)` en `float32` normalizado L2 sobre taxonomía, transiciones, co-visitation, carritos e intención histórica.

4. **242 Escalares Modulares + Matrices Contextuales por Página y Segmento** (`weights_profile.py`):
   - 3 matrices `page_product_*`, `page_category_*`, `page_home_*` (8 ejes cada una).
   - 5 matrices `seg_*` (`new_cold`, `new_browsing`, `warm_browsing`, `warm_history`, `warm_returning`).
   - Guardrails desacoplados para cold-start (`cold_boost_on_cold_session`, `cold_explore_start_slot_*`, `max_cold_explore_in_warm`).

5. **Entrenamiento Multi-Fuente Cero-Fugas** (`training_data_v6.py`):
   - Combina sesiones de impresión IPS (`Gold`), carritos históricos convertidos vs abandonados, compras recurrentes, intención de usuario y hard-negative mining sobre transiciones secuenciales, sin usar jamás `official_test_data.parquet`.

## Comandos principales

```bash
python calypso_6_1/serve.py --self-check
python calypso_6_1/serve.py --dry-run
python calypso_6_1/tuner.py --eval-presets
python -m unittest discover -s calypso_6_1/etapa_4/tests -p "test_*.py" -v
```
