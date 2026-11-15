# 🏋️‍♂️ DataVengers — Datathon FME 2026 (UPC Barcelona)

**Sistema de recomanació en temps real per al [RawTree Challenge](https://datathon.rawtree.dev)**

📅 14–15 Novembre 2026 · UPC Barcelona

---

## 📋 Què és aquest projecte?

Un sistema automàtic que, cada vegada que un usuari entra a una botiga online, li **mostra 10 productes recomanats en menys de 3 segons**. L'objectiu és maximitzar les **comandes** (conversió a compra) en un entorn en temps real.

El sistema llegeix peticions de la base de dades compartida (`shop.requests`), analitza el comportament de l'usuari i genera un rànquing personalitzat de 10 productes que escriu a la taula `team_NN.submissions`.

---

## 🗂️ Estructura del projecte

```
Datathon-2026---DataVengers/
├── datathon-fme-2026-main/       # Challenge base (README, starter kit, queries)
├── calypso_2_1/                  # Model base 2.1 — Hybrid + Sequential + IPS Re-ranker
├── calypso_3_1/                  # Intent d'agressivitat (74.5% hit rate)
├── calypso_3_1b/                 # Millor versió abans de 4.1 (79.0% hit rate)
├── calypso_3_2/                  # Afinament final (en desenvolupament)
├── calypso_4_1/                  # Regles de negoci: carrito, recurrents, preu
├── calypso_4_1_e/                # ★ 4.1.e — Modular + Optimització iterativa (ACTIU)
├── calypso_4_2/                  # Versió experimental 4.2
├── calypso_5_1/                  # Versió experimental 5.1
├── calypso_6_2_b/                # 6.2b — Dual-Head + Transfer Learning (64 features)
├── odyseus/                      # Alternativa Odyseus
├── odyseus_2_0/                  # Odyseus v2
├── odyseus_3_0/                  # Odyseus v3
├── gestor_models/                # Manager per canviar/llançar models
├── Presentacion/                 # Materials de presentació
├── Informes/                     # Informes i anàlisis
├── documentacio_IA/              # Documentació IA
├── offline_eval_cache/           # Dades d'avaluació offline (parquet, JSON)
├── comparativa_models_llenguatge_natural.md  # Comparativa senzilla dels models
├── MEDIDAS_IMPLEMENTADAS.md      # Totes les mesures implementades (E1→4.1)
├── .active_model.json            # Model actiu actual (calypso 4.1.e)
├── model_manager.py              # Entry point del gestor
└── run_offline_eval_official.py  # Avaluació oficial offline
```

---

## 🧠 Evolució dels models Calypso

| Model | Hit Rate Buy@10 | Descripció |
|-------|-----------------|------------|
| **2.1** | 77.2% | Model base: transicions, auto-conversió, intenció, afinitat, preu, context, popularitat |
| **3.1** | 74.5% | 4 canvis agressius (va pitjor) |
| **3.1b** | **79.0%** | Només la penalització de fons funciona (+1.8%) |
| **4.1** | — | Regles de negoci: carrito actiu/oblidat, productes recurrents, ordre de preu |
| **4.1.e** | — | ★ **Model actiu**: arquitectura modular, perfils JSON, optimització iterativa |
| **6.2b** | — | Dual-head Slot-1, 64 features, transfer learning, CVR velocity |

> Consulta `calypso_*/README_*.md` i `MEDIDAS_IMPLEMENTADAS.md` per al detall de cada model.

---

## ⚡ Com s'executa?

### 1. Configura les credencials
```bash
cp group_API.env .env   # omple amb el TEAM_ID i RAWTREE_API_KEY
```

### 2. Model actiu
El model actiu es defineix a `.active_model.json`. Actualment és **calypso 4.1.e**.

Per canviar de model:
```powershell
python model_manager.py --set calypso_4_1_e
python calypso_4_1_e/serve.py --watchdog
```

### 3. Avaluació offline
```bash
python run_offline_eval_official.py
```

### 4. Optimització de pesos (4.1.e)
```bash
cd calypso_4_1_e
python tuner.py --optimize --iters 20 --save-best
python tuner.py --eval-presets
```

---

## 📊 Measures clau del sistema

- **Velocitat de resposta**: < 3 segons per petició
- **10 productes** per sol·licitud
- **5 segments d'usuari**: conegut_sensio, conegut_sense_sensio, nou_sensio, nou_sense_sensio, producte_nou
- **Hot-reload**: els pesos es poden canviar sense reiniciar el servei
- **Slot-gating**: slots 1-6 per precisió, slots 7-10 per exploració
- **Bandit UCB**: exploració equilibrada amb explotació
- **Guardrails**: sempre 10 IDs vàlids, sense repeticions de compra

---

## 🏆 Categories de l'avaluació

| Criteri | Punts | Descripció |
|---------|-------|------------|
| Mètode | 10 | Enfoquament fonamentalment justificat i validat offline |
| Aprenentatge del stream | 10 | Ús de feedback i esdeveniments en viu |
| Ús de RawTree | 5 | Consultes SQL, embeddings, pipeline |
| Comunicació | 5 | Entenedor per a un PM |

---

## 📜 Regles

- Tot generat per un simulador amb llavor secreta
- Feedback privat per equip
- Recomanacions per codi, no manualment
- Sistema sense vigilància
- Equips del top share el repositori

---

## 👥 Equip DataVengers

Equip de la **Facultat de Matemàtiques i Estadística (FME), UPC Barcelona**.

---

## 📚 Documentació addicional

- [Challenge base](datathon-fme-2026-main/README.md)
- [Comparativa de models](comparativa_models_llenguatge_natural.md)
- [Mesures implementades](MEDIDAS_IMPLEMENTADAS.md)
- [README Model 4.1.e](calypso_4_1_e/README_4_1_e.md)
- [README Model 6.2b](calypso_6_2_b/README_6_2_b.md)