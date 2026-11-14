# 🧠 Què fa cada model? (Explicació senzilla per a tothom)

## 📌 Resum ràpid en una frase

Tenim **4 versions del nostre recomanador**. Cadascuna millora la anterior, però no tots els canvis funcionen igual de bé. El més important és saber **què canvia realment** i **com afecta les recomanacions** que veuran els usuaris.

---

## 🎯 Els 4 models comparats

| Model | Què és? | Rendiment (percentatge d'encerts) | Veredicte |
|-------|---------|----------------------------------|-----------|
| **2.1** | El model base amb intel·ligència artificial | 77.2% | Funciona bé, però es pot millorar |
| **3.1** | El 2.1 amb 4 canvis agressius | 74.5% | ❌ Va pitjor (els canvis eren massa bruscos) |
| **3.1b** | El 3.1 retallat: només el que realment funciona | **79.0%** | ✅ El millor fins ara (+1.8% d'encerts) |
| **3.2** | L'afinament final amb totes les millores refinades | Per confirmar | 🔄 En desenvolupament |

---

## 📖 Explicació detallada

### 🔵 calypso-2.1: El punt de partida

**Què fa:**
És el primer model amb intel·ligència artificial real. Analitza:
- Què ha mirat l'usuari recentment
- Què compra la gent similar
- Quins productes estan de moda
- Les preferències històriques de l'usuari (subcategoria, marca, preu)

**Com tria els 10 productes:**
Cada producte rep un "puntatge" basat en aquestes senyals:
1. **Transicions seqüencials** (si vas per A → compres B): pesa molt
2. **Auto-conversió** (si ja has vist el producte i en vols més): pesa molt
3. **Intenció** (carrets abandonats o moltes vistes sense comprar): pesa moderadament
4. **Afinitat històrica** (subcategoria i marca preferida): pesa poc
5. **Preu adequat** (s'ajusta al pressupost habitual): pesa poc
6. **Context** (què compra gent de la teva regió/dispositiu): pesa moderadament
7. **Popularitat global** (què es ven més): pesa poc
8. **Tendències en viu** (què està de moda ara): pesa moderadament
9. **Productes nous**: bonus molt petit
10. **Similaritat de contingut** (veïns temàtics): pesa moderadament

**Avantatges:**
- Funciona bé en tots els escenaris
- Simple i fàcil d'entendre
- No és massa agressiu

**Desavantatges:**
- No penalitza prou els productes "fantasma" (sense cap motiu per recomanar-los)
- No té en compte el temps de forma precisa (no sap que una vista fa 10 minuts és menys rellevant que una de fa 5 minuts)
---

### 🟠 calypso-3.1: L'intent d'aggressivitat

**Què va intentar fer:**
Va fer 4 canvis grans alhora, pensant que seria millor ser més agresiu:

| Canvi | Què volia fer | Què va passar realment |
|-------|---------------|------------------------|
| **1. "Suelo" de subcategoria** | Forçar a recomanar productes de la mateixa subcategoria, encara que no hi hagi dades | ❌ Va fer pitjor (−2%). Obligar el sistema a recomanar alguna cosa sense proves és un error. |
| **2. Penalització de fons** | Penalitzar fort els productes sense cap senyal que els recomani | ✅ Va funcionar! (−1.8% d'encerts quan es treu) |
| **3. EASE (collaborative filtering)** | Afegir un nou tipus de model per trobar patrons ocults | ⚠️ Neutre (amb pes 0.25 fa mal, amb pes 0.10 no afecta) |
| **4. SQL direccional** | Només contar transicions que van d'antes → després (no al revés) | ⚠️ No es pot mesurar en el laboratori, només amb dades reals |
| **5. Umbrales de cold start** | Acceptar veïns més tous per a productes nous | ⚠️ Neutre |

**Conclusió:**
El 3.1 va intentar fer massa canvis alhora. 3 dels 4 canvis van fer **pitjor** el rendiment. L'únic que va funcionar va ser la **penalització de fons** (canvi #2).
---

### 🟢 calypso-3.1b: L'afinament intel·ligent (RECOMANAT)

**Què és:**
Una versió on cada canvi del 3.1 es pot encendre/apagar amb un interruptor (variable d'entorn). Això permet provar què funciona realment.

**Resultats de les proves (ablació):**

| Combinació de canvis | Rendiment | Què significa |
|---------------------|-----------|---------------|
---

### 🟣 calypso-3.2: L'afinament final (en desenvolupament)

**Què afegeix al 3.1b:**
El 3.2 combina totes les millors idees amb nous mecanismes que encara estem validant:

| Milloa | Què fa | Per què és millor |
|--------|--------|-------------------|
| **Semivida temporal (624 segons)** | Cada 10 minuts, el pes d'una vista es redueix a la meitat | Més fidel a la intenció real del moment. |
| **Bayesià jeràrquic (kappa=50)** | Si no hi ha dades per al context exacte, comparteix amb contextos més generals | No es queda buit en contexts rars. |
| **Cold start més estricte** | Només productes nous amb similitud ≥0.45 I de la mateixa subcategoria | Menys falsos positius. |
| **Slot-gated (slots 1-6 vs 7-10)** | Els 6 primers slots són per el que ja sabem. Del 7 al 10, explora. | Més precisió als primers slots. |
| **Desempat per revenue** | Si dos productes tenen gairebé el mateix score, guanya el més car | Optimitza el tie-breaker del leaderboard. |
| **Menys exploració del bandit** | UCB alpha baixa de 0.22 a 0.14. Confia més en el que ja sabem. | Menys soroll, més estabilitat. |

**Els 5 segments del 3.2:**

| Segment | Qui és | Què li recomana |
|---------|--------|-----------------|
| `conegut_sensio` | Usuari conegut amb productes vistos | Transicions seqüencials + auto-conversió |
| `conegut_sense_sensio` | Usuari conegut però recent_views buit | Perfil d'usuari + popularitat contextual |
| `nou_sensio` | Usuari nou amb productes vistos | Cold start directe + contingut |
| `nou_sense_sensio` | Usuari nou sense productes vistos | Popularitat global + tendències |
| `producte_nou` | Producte nou (new_product) | Cold start invers + veïns de contingut |

---

## 🏆 Resum: Què funciona realment per a la conversió?

### Els 3 canvis més importants (ordre d'impacte):

| Prioritat | Canvi | Impacte estimat | Com afecta el que veurà l'usuari |
|-----------|-------|-----------------|----------------------------------|
| 🔴 **MOLT IMPORTANT** | Penalitzar productes sense senyal | +1.8% d'encerts | Més productes rellevants, menys "farfull" |
| 🟠 **IMPORTANT** | Semivida temporal (10 min) | Millores qualitatives | Les recomanacions s'adapten millor al moment actual |
| 🟡 **MODERAT** | Slot-gated (slots 1-6 vs 7-10) | Més precisió als primers slots | Els 6 primers productes són els millors |
| 🟢 **PETIT** | EASE amb pes baix (0.10) | Neutre | Pot ajudar amb patrons ocults |
| ⚪ **NEUTRE** | Cold thresholds, SQL direccional | No afecta aquí | Es veurà només amb dades reals |

### Què NO funciona (i per què):

| Canvi | Per què no funciona | Què passa |
|-------|---------------------|-----------|
| Suelo de subcategoria | Obligar recomanacions sense dades | El sistema "força" productes que no tenen res que els recomani |
| EASE amb pes alt (0.25) | EASE captura soroll si no està ben ajustat | Pot afegir productes irrelevants amb massa confiança |
| Canvis massa bruscos | Fer massa canvis alhora | Si canvies 5 paràmetres a la vegada, no saps què ha funcionat |
| 2.1 base | 77.2% | Punt de partida |
| 3.1 complet (tot a 1) | 74.5% | ❌ Va pitjor (els canvis eren massa bruscos) |
| 2.1 + penalització de fons | **79.0%** | ✅ **EL MILLOR** (+1.8% d'encerts) |
| 2.1 + EASE 0.25 | 76.4% | ⚠️ Fa mal (pes massa alt) |
| 2.1 + EASE 0.10 | 77.2% | ⚠️ Neutre |
| 2.1 + sense suelo subcat | 75.3% | ❌ El suelo fa mal (−2%) |

**La clau del 3.1b: "No recomanis el que no té motiu"**

> **Què vol dir en llenguatge natural:**
> 
> Abans (2.1), si un producte no tenia cap senyal que el recomanés, es quedava igualment al rànquing perquè no hi havia cap penalització.
> Ara (3.1b), si un producte no té cap motiu per ser recomanat, el sistema li aplica un **multiplicador de penalització**: x0.45 si és de la mateixa subcategoria, x0.18 si és d'una altra.
> Això allibera espai per a productes de veritat rellevants.

**Com funciona el 3.1b per defecte:**

| Interruptor | Valor | Per què? |
|-------------|-------|----------|
| Penalització de fons | ✅ ACTIVA | Aquest és el canvi que funciona (+1.8%) |
| Suelo de subcategoria | ❌ DESACTIVADA | Fa mal (−2%) |
| EASE | 0.0 (desactivat) | Amb pes 0.25 fa mal |
| SQL direccional | ❌ DESACTIVADA | No es pot mesurar aquí |
| Cold thresholds | ❌ DESACTIVADA | Neutre, no cal |