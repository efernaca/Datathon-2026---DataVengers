# Calypso 4.1.e · Arquitectura Modular Orientada a Conversió i Optimització Iterativa

`calypso_4_1_e` evoluciona `calypso_4_1` separant tots els pesos de scoring, multiplicadors per segment d'usuari, similituds de perfil de producte, regles de negoci (`M1..M5`), objectiu de recompensa (compra vs. clic) i política de desempat en un **perfil JSON centralitzat i recarregable en calent (`weights_profile.json`)**.

## Peces de l'Arquitectura Modular

| Mòdul | Funció |
|---|---|
| `weights_profile.json` | Fitxer JSON actiu al root de `calypso_4_1_e/`. Modificar-lo en calent fa que `serve.py` recarregui pesos i regles en segons sense parar el servei ni perdre peticions. |
| `etapa_4/src/weights_profile.py` | Dataclass `WeightsProfile` + presets integrats + lectura/escriptura atòmica JSON. |
| `etapa_4/src/config.py` | `ConfigV4.from_profile()` i `ConfigV4.from_env()` sincronitzen automàticament `weights_profile.json` (i permeten sobreescriptura per variables d'entorn `CALYPSO_*`). |
| `etapa_4/src/recommender.py` | Classificació en 5 segments d'usuari, asimetria de preu barat, `mmr_relative_penalty`, i `tie_break_mode` (`conversion` / `cheap` / `revenue`). |
| `etapa_4/src/reranker.py` | Clipped IPS `PositionDebiasedRanker` amb suport real de **pesos negatius** (p. ex. `w_log_price_norm < 0`) i control de clic (`ips_purchase_weight` vs. `ips_click_weight`). |
| `etapa_4/src/cold_start.py` | Pesos parametritzats del perfil de producte (`cold_sparse_cos_weight`, `cold_dense_svd_weight`, `cold_subcat_match_bonus`, `cold_cat_match_bonus`, `cold_brand_match_bonus`, `cold_price_prox_bonus`). |
| `etapa_4/src/tuner.py` + `tuner.py` | Eina CLI per llistar/canviar presets, ajustar variables (`--set`), comparar presets (`--eval-presets`) i executar optimització iterativa (`--optimize`). |

## Presets inclosos

- **`conversion_first`** (per defecte): Prioritat absoluta a conversió efectiva a compra (`ips_click_weight=0.10`, `bandit_click_reward=0.05`, `w_log_price_norm=0.0`, `v4_price_ascending=False` per protegir `DCG@10`). Mantiene carret actiu/oblidat (`M1`/`M2`) i compres recurrents (`M4`/`M5`).
- **`cheap_boost`**: Prioritza productes més barats mitjançant senyals suaus (`w_log_price_norm=-0.18`, `price_cheap_asymmetry=0.45`, `tie_break_mode="cheap"`) sense destruir el rànquing del slot 1.
- **`cheap_hard_sort`**: Combina `cheap_boost` amb la regla estricta `v4_price_ascending=True` (`M3` de 4.1) dins cada bloc.
- **`user_trend_focus`**: Amplifica senyals d'afinitat, intenció i tendències contextuals/live per a cada un dels 5 segments d'usuari.
- **`product_profile_focus`**: Amplifica similitud per atributs de producte (subcategoria, marca, preu, descripció) i exploració de nous ítems en cold-start.

## Ús Ràpid (CLI)

```powershell
cd calypso_4_1_e

# Verificació i testos
python -m unittest discover -s etapa_4/tests -v
python serve.py --self-check
python serve.py --dry-run

# Gestió modular de perfils i optimització (tuner.py)
python tuner.py --list-presets
python tuner.py --show
python tuner.py --eval-presets
python tuner.py --preset cheap_boost              # canvi en calent sobre weights_profile.json
python tuner.py --set w_log_price_norm=-0.22 ips_click_weight=0.05
python tuner.py --optimize --iters 20 --save-best

# Servei en viu (amb hot-reload automàtic de weights_profile.json)
python serve.py --watchdog
```
