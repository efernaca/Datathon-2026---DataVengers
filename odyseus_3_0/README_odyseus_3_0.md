# Odyseus 3.0

Copia de `odyseus_2_0` (15/11, 08:45 CET) con un unico cambio de comportamiento: **los productos de
`recent_views` se fijan arriba del carrusel**.

- Parametro nuevo `pin_viewed_top` en `etapa_4/src/weights_profile.py` (por defecto 0 = sin efecto) y
  activado a 10 en `weights_profile.json`. Se recarga en caliente: ponerlo a 0 devuelve el ranking de 2.0.
- Logica en `etapa_4/src/recommender.py`, al final de `_recommend_core`: hasta N vistos (activos y no
  excluidos por compra), ordenados por `self_cvr`, y detras el ranking normal.
- `model_version` = `odyseus-3.0` (ramas A/B: `odyseus-3.0-A` / `-B`).

Por que: en `shop.history_sessions` el mismo producto en el mismo slot se compra ~7x mas si el usuario
ya lo habia visto en la sesion (odds ratio 7,2), sin efecto de recencia. Odyseus 2.0 dejaba fuera del
top-10 el 64,5 % de los productos vistos.

No se han copiado `ml_data/` ni `logs/` (no hacen falta para servir). `cache/` es una copia del snapshot
de 2.0, asi que arranca sin reentrenar. `odyseus_2_0/` queda como estaba.

Uso: `python serve.py --self-check` y despues `python serve.py --watchdog --ab-share 0`.
