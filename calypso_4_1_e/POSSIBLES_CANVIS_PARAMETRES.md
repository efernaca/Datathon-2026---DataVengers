# POSSIBLES CANVIS PARAMETRES + IMPORTANTS

## 📊 Escala Qualitativa dels Pesos

Els pesos actuen com a **multiplicadors d'atenció**: com més alt, més "pes" té aquella senyal a l'hora de decidir.

| Valor | Significat Qualitatiu | Com se sent el sistema? |
|---|---|---|
| **> 1.5** | **Prioritat Màxima** 🔴 | El sistema dona molta importància a aquesta senyal. Si està present, el producte puja molt al ranking. |
| **1.0 – 1.5** | **Prioritat Alta** 🟠 | Senyal fort, però no dominant. El sistema hi confia. |
| **0.5 – 1.0** | **Prioritat Mitjana** 🟡 | Senyal moderat. El sistema en fa cas, però no és decisori. |
| **0.1 – 0.5** | **Prioritat Baixa** 🟢 | Senyal secundari. Només fa diferència quan altres signals estan empatades. |
| **~0.0** | **Ignorat** ⚪ | El sistema no en fa cas (o gairebé). |
| **< 0.0** | **Penalització** ⚫ | El sistema **evita** activament aquest producte. |

---

## 🎯 Traducció Qualitativa dels 14 Pesos Principals

### 1. `w_self_cvr_view` (~1.90)
> *"Aquest producte converteix per si sol?"*

| Valor | Què fa el sistema |
|---|---|
| **2.0+** | El sistema prioritza productes que per si sols tenen moltes compres. És el **"candidat estrella"**: si té bona història, guanya gairebé sempre. |
| **1.0–1.5** | El sistema en fa cas, però no decideix sol. |
| **0.5 o menys** | El sistema no li importa tant la conversió pròpia. Busca senyals indirectes (transicions, preferències). |
| **0.0** | El sistema ignora completament la història de compres del producte. |

### 2. `w_seq_lift` (~1.15)
> *"Quants compradors van veure aquest producte després d'altres?"*

| Valor | Què fa el sistema |
|---|---|
| **2.0+** | El sistema busca **seqüències de compra**: si molta gent veu A → B → C, el sistema recomana B després de A. Molt direccional. |
| **1.0** | Comportament equilibrat: el sistema en fa cas, però no obsessiona. |
| **0.5 o menys** | El sistema no li importa el que fan els altres usuaris en seqüència. Més aleatori. |

### 3. `w_intent_score` (~0.60)
> *"L'usuari en vol comprar ara?"*

| Valor | Què fa el sistema |
|---|---|
| **1.4+** | El sistema és **molt sensible a cistelles abandonades**. Si un producte va a la cistella, el torna a recomanar. Ideal per recuperar compres. |
| **0.6** | Comportament normal: ho considera, però no és el factor principal. |
| **0.15 o menys** | El sistema ignora cistelles abandonades. No intenta recuperar compres perdudes. |

### 4. `w_user_affinity` (~0.28)
> *"Aquest usuari prefereix aquesta subcategoria x marca?"*

| Valor | Què fa el sistema |
|---|---|
| **0.85+** | El sistema **personalitza fortament**: si l'usuari compra sovint Nike, li recomana Nike. Pot crear una bombolla de filtratge. |
| **0.28** | Comportament normal: la preferència s'observa, però no domina. |
| **0.05 o menys** | El sistema **no personalitza per marca/subcategoria**. Mostra el mateix a tothom gairebé. |

### 5. `w_ctx_bayes` (~0.30)
> *"Quants usuaris d'aquest perfil (regió x pàgina x dispositiu) comprenen això?"*

| Valor | Què fa el sistema |
|---|---|
| **0.80+** | El sistema adapta fortament el catàleg al context: un usuari a Espanya veurà productes diferents que un a Suècia. |
| **0.30** | Comportament normal: adaptació moderada al context. |
| **0.05 o menys** | El sistema ignora el context. No importa d'on vinguis. |

### 6. `w_global_pop` (~0.22)
> *"Aquest és un producte que compra tothom?"*

| Valor | Què fa el sistema |
|---|---|
| **0.65+** | **Efecte "rich get richer"**: els productes populars es fan més populars. El sistema els mostra més. Pot matar productes nous. |
| **0.22** | Comportament equilibrat: la popularitat ajuda, però no decideix. |
| **0.02 o menys** | El sistema ignora la popularitat global. Dona les mateixes oportunitats a tots els productes. |

### 7. `w_live_trend` (~0.28)
> *"Què es posa de moda ara mateix?"*

| Valor | Què fa el sistema |
|---|---|
| **0.75+** | El sistema és **molt reaccionari**: si avui es posa de moda un producte, el recomana a tothom. Pot crear modes passatgeres. |
| **0.28** | Comportament normal: observa les tendències, però les filtra. |
| **0.02 o menys** | El sistema ignora les tendències del moment. Només mira històric. |