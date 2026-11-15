---

## Contradiccio 3: Diversitat MMR a homepage desactivada (però hi ha valor al config.py)

### Lo que diuen los ficheros

| Ficher | Parametre | Valor |
|---|---|---|
| config.py (default) | `mmr_subcat_lambda_home` | `0.15` |
| weights_profile.json (actiu) | `mmr_subcat_lambda_home` | `0.0` |
| config.py (default) | `mmr_brand_lambda` | `0.04` |
| weights_profile.json (actiu) | `mmr_brand_lambda` | `0.02` |

### Sembla contradictori

Config diu `0.15`, profile diu `0.0`.

### Veredicte: NO es contradiccio (sobreescriptura intencional)

README_5.1, mesura M7/D11/X9:

> "home sense diversificacio MMR forçada. `mmr_subcat_lambda_home = 0.0`."

**En llenguatge natural:**
- El MMR (Maximum Marginal Relevance) es un mecanisme de diversitat: evita que els 10 primers slots siguin de la mateixa subcategoria.
- A 4.1, a la homepage s'aplicava `0.15`, obligant a diversificar.
- A 5.1, s'ha **desactivat explicitament** (`0.0`). Si el ML diu que els 10 millors productes per a un usuari son tots de la mateixa subcategoria (ex: tots auriculars bluetooth), forcar diversitat (posar un carregador al slot 3) **baixa la conversio**. Més val deixar que el model decidi sense barres artificials.

**Es correcte?** Si. La diversitat forçada pot perjudicar la CVR. Es mes correcte deixar-ho al ML.

---

## Contradiccio 4: bandit_ucb_alpha menor al profile (0.12) que al config (0.14)

### Lo que diuen los ficheros

| Ficher | Parametre | Valor |
|---|---|---|
| config.py (default) | `bandit_ucb_alpha` | `0.14` |
| weights_profile.json (actiu) | `bandit_ucb_alpha` | `0.12` |

### Sembla contradictori

Xifres diferents.

### Veredicte: NO es contradiccio (sobreescriptura intencional)

El profile sobreescriu el default. `0.12 < 0.14`, significa **menys exploracio en mode WARM** (conservador).

**En llenguatge natural:**
- El UCB (Upper Confidence Bound) es el mecanisme d'exploracio del bandit: quant bonus s'afegeix als productes poc explorats.
- `0.12` es **mes conservador** que `0.14`: s'explora menys i es confia mes en el que ja sabem que converteix.
- Aixo té sentit amb la filosofia de 5.1 ("Prioritat absoluta a la conversio"): si coneixes el que converteix, no experimentis tant.

**Es correcte?** Si. Menys exploracio WARM = mes estabilitat i mes CVR.

---

## Contradiccio 5: v4_cart_max_slots menor al profile (2) que al config (3)

### Lo que diuen los ficheros

| Ficher | Parametre | Valor |
|---|---|---|
| config.py (default) | `v4_cart_max_slots` | `3` |
| weights_profile.json (actiu) | `v4_cart_max_slots` | `2` |

### Veredicte: NO es contradiccio (sobreescriptura intencional)

README_5.1, mesura M6: "Carret actiu conserva prioritat max. 2 slots."

**En llenguatge natural:**
- A 4.1, fins a 3 slots podien ocupar-se amb productes de carrit (actiu + oblidat).
- A 5.1, només 2 slots com a max per al carrit actiu. L'ablation de 4.1 va mostrar que amb 3 slots el DCG baixava (s'expulsen productes ML rellevants). Reduir a 2 es millor.

**Es correcte?** Si. Menys slots reservats al carrit = mes espai per al ML = millor CVR.

---

## Taula resum: Tot correcte?

| # | Parametres | Sembla contradiccio? | Es real? | Per que? |
|---|---|---|---|---|
| 1 | `v4_forgotten_cart=false` + `bonus=0.35` / `v4_recurrent=false` + `bonus=0.25` | Si | **No** | A 5.1 no es bloquegen slots, es competeix per puntuacio. Intencionat. |
| 2 | `v4_price_ascending=false` (config default `True`) | Si | **No** | El profile sobreescriu el default. Desactivat explicitament. |
| 3 | `revenue_tie_weight=0.0` (config default `0.03`) | Si | **No** | Mateix cas: sobreescriptura intencionada. |
| 4 | `mmr_subcat_lambda_home=0.0` (config default `0.15`) | Si | **No** | Diversitat forçada a homepage desactivada. |
| 5 | `bandit_ucb_alpha=0.12` (config default `0.14`) | Si | **No** | Menys exploracio WARM: mes conservador per conversio. |
| 6 | `v4_cart_max_slots=2` (config default `3`) | Si | **No** | Reduit per millorar DCG (ablation ho demostra). |

---

## Veredicte final

**Cap contradiccio real.** Tots els "parametres que no quadren" s'expliquen per dos raons:

1. **El `weights_profile.json` sobreescriu els valors per defecte del `config.py`** → El perfil actiu (`conversion_first`) es la veritat a produccio, no el config.py.
2. **Les mesures M6 i similars desactiven els blocs rigidss i els substitueixen per bonificacions de puntuacio** → Això no es un error, es el canvi arquitectonic central de 5.1: en lloc de "reservar slots", es "competir per slots" mitjançant bonus.

**Es correcte?** Si. Está documentat al README_5.1 y té sentit: mes flexibilitat, menys restriccions artificials, y el model ML decideix quins productes van a quin posacio sense bareres rigides de block-reservation.

---

*Fi del document.*