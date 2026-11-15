# Calypso 6.2a · Iteració A (Multi-Source IPS + Cross-Subcategory Lift + Consensus)

Primera iteració evolutiva sobre `calypso_6_2` (`model_version = "calypso-6.2a"`), focalitzada en:

1. **Aprenentatge Multi-Font activat (`use_multisource_training = True`)**: Combina Gold IPS logs (`1.0`), carritos (`0.18`), recurrents/intents (`0.14`) i transicions seqüencials (`0.12`) amb *hard negatives* alineats al *serving* (competidors seqüencials + subcategoria + populars).
2. **Matriu de transicions creuades entre subcategories (`subcat_transitions`)**: Captura `P(dst_subcat | src_subcat)` per millorar la recuperació i el rànquing de compres complementàries cross-categoria.
3. **56 features d'alta resolució**: Afegeix `cross_subcat_lift`, `pool_rel_seq_norm`, `pool_rel_pop_norm`, `subcat_leader_nonlast`, `multi_channel_consensus` i `price_vs_user_pref_signed`.
4. **Destil·lació Teacher → Student MLP ampliada (`(80, 40)`)**: `HistGradientBoostingClassifier` (`max_depth=5`, `max_iter=180`, `max_bins=128`) destil·lat a MLP NumPy pur amb re-mostreig balancejat IPS fins a 2400 positius.

## Ordres útils

```powershell
python calypso_6_2_a/serve.py --self-check
python calypso_6_2_a/serve.py --dry-run
python calypso_6_2_a/tuner.py --eval-presets
python -m unittest discover -s calypso_6_2_a/etapa_4/tests -p "test_*.py" -v
```
