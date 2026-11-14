# Etapa 4 · Blindatge Operatiu Anti-Caigudes, Arrencada Instantània i `calypso-2.1` Slot-Gated

**Versió del model al leaderboard:** `calypso-2.1`  
**Objectiu de l'Etapa 4:** Eliminar les caigudes d'SLA (`ON TIME`), aconseguir una recuperació **en `< 0.5 segons`** després de qualsevol reinici del PC mitjançant **caché atòmica a disc** (`etapa_4/cache/`), garantir enviaments nomès dins la finestra vàlida de 3 segons, i blindar la precisió `WARM` combinada amb exploració *Cold Start* intel·ligent (*Slot-Gated UCB*).

---

## 1. Diagnòstic de les Captures del Leaderboard (`team_32`)

A partir de les captures del leaderboard després de la caiguda de l'ordinador (`0.8257% CONVERSION | 1.3756% COLD START | 10.3% ON TIME | €0.131 | calypso-2.1 · 1.0326%`):

1. **No s'ha perdut l'aprenentatge:** `BEST MODEL: calypso-2.1 · 1.0326%` i `COLD START: 1.3756%` demostren que `calypso-2.1` converteix molt bé quan les respostes arriben dins els 3 segons, i tot el feedback (`team_32.feedback`) i la memòria de `team_32.scratch` continuen intactes a RawTree.
2. **Per què sortia `10.3% ON TIME` i baixava la `CONVERSION` total a `0.8257%`:**
   - Durant els minuts amb el PC aturat, `ON TIME` era `0%` i les peticions comptaven com a *fallback* (`baseline`).
   - En tornar a arrencar l'Etapa 3, `SQL_PULL` demanava **`ORDER BY ts DESC LIMIT 500`** (equivalent a ~165 segons d'històric antic). ~490 d'aquestes 500 peticions **ja havien caducat** (`age_ms > 3.000 ms`), però el bucle perdia ~1,1s puntuant i pujant aquelles 500 respostes caducades (*"Submitted 500 answers"*), fet que retardava les peticions realment noves.
   - Al mateix temps, `--slow-interval 18.0` reexecutava les 11 consultes pesades de 60 dies sobre `shop.history_events` (12M files) cada 18 segons, competint amb el bucle ràpid.

---

## 2. Les 5 Capes de Blindatge de l'Etapa 4

| Capa | Fitxer | Problema que resol | Resultat |
| :--- | :--- | :--- | :--- |
| **1. Caché Atòmica a Disc + Bootstrap** | [disk_cache.py](./src/disk_cache.py), [state.py](./src/state.py) | Un reinici havia d'esperar que acabessin totes les consultes SQL històriques. | Arrencada en **`< 0.4 s`** carregant `snapshot_v4_latest.pkl`, i bootstrap lleuger de fallback si no hi ha caché. |
| **2. Lectura Exclusiva de Petites Finestres SLA** | [serve.py](./src/serve.py), [etapa4_queries.sql](./sql/etapa4_queries.sql) | Es processaven fins a 500 peticions caducades (`> 3s`). | Consulta SQL filtrada `ts >= now64(3) - 2400ms ORDER BY ts ASC LIMIT 25`: zero còmput malgastat en peticions mortes. |
| **3. Confirmació `_seen` Post-Push + Watchdog** | [serve.py](./src/serve.py), [run_watchdog.ps1](./run_watchdog.ps1) | Si fallava un HTTP POST, les peticions es marcaven igualment com a vistes i queien en fallback; si petava el procés no tornava a pujar sol. | `_seen` només s'actualitza **després** d'un `push_submissions` reeixit; `run_watchdog.ps1` reinicia el servei automàticament en `1 s` si cau. |
| **4. Separació Històric Estàtic / Live** | [serve.py](./src/serve.py) | Es re-agregaven 12M de files cada 18s. | L'històric estàtic es calcula/carrega **1 cop** (`static_history_v4.pkl`); el bucle viu només refresca estoc (`1.5s`) i `events`/`feedback`/`catalog_updates` (`15s`). |
| **5. Slot-Gated `calypso-2.1` (< 1 ms/req)** | [recommender.py](./src/recommender.py) | L'exploració `cold_surge` desplaçava ítems `WARM` dels primers slots i trigava `~2.7 ms/req`. | En sessions `WARM`, els slots `0..6` queden blindats amb precisió ML/IPS pura i esmorteïment de fons per coherència de sessió; els ítems nous confirmats pel bandit poden entrar als slots `4..9` i l'exploració UCB general se centra als slots `7..9` (`max 2` nous), precomputant vectors estàtics al snapshot per inferir en `< 1.0 ms/req`. |

---

## 3. Posada en Marxa en Producció (Anti-Caigudes)

Executa el supervisor des de l'arrel del projecte:
```powershell
powershell -ExecutionPolicy Bypass -File etapa_4/run_watchdog.ps1
```
O directament amb el watchdog integrat de Python:
```powershell
python etapa_4/src/serve.py --watchdog
```
I en una segona terminal per controlar la recuperació real de `ON TIME %`:
```powershell
python etapa_4/src/monitor.py --watch 10
```
