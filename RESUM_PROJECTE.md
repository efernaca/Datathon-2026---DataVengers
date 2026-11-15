# DataVengers - Resum del Projecte

## Datathon FME 2026 - UPC Barcelona

### Objectiu del Projecte
Desenvolupar un **sistema de recomanació en temps real** per a una botiga online durant la Datathon FME 2026 a la UPC Barcelona. El sistema ha de seleccionar 10 productes per a cada usuari en menys de 3 segons, maximitzant les conversions (comandes).

### Context
- **Esdeveniment**: Datathon FME 2026
- **Dates**: 14-15 Novembre 2026
- **Lloc**: UPC Barcelona
- **Challenge**: RawTree Challenge - Real-Time Recommendation
- **Equip**: DataVengers
- **Plataforma**: RawTree (base de dades analítica per a JSON)

### Com funciona el sistema
El sistema segueix un cicle automàtic:
1. **Llegeix** peticions de recomanació de `shop.requests`
2. **Analitza** el comportament de l'usuari i el context
3. **Calcula** un rànquing de 10 productes en < 3 segons
4. **Escriu** les recomanacions a `team_NN.submissions`
5. **Aprèn** del feedback (`team_NN.feedback`) i les dades en viu

### Dades disponibles
- **Catàleg**: ~5,000 productes en 10 categories, 90 subcategories
- **Historial**: 60 dies de sessions passades (~2 milions de registres)
- **Esdeveniments en viu**: `shop.events`, `shop.catalog_updates`
- **Plataforma**: RawTree - SQL sobre JSON, sense esquema fix

### Arquitectura del projecte

    datathon-fme-2026-main/     <- Challenge base i starter kit
    calypso_2_1/                <- Model base (77.2% hit rate)
    calypso_3_1/                <- Intent d'agressivitat (74.5%)
    calypso_3_1b/               <- Millora selectiva (79.0%)
    calypso_4_1/                <- Regles de negoci
    calypso_4_1_e/              <- Modular + optimitzacio iterativa (ACTIU)
    calypso_6_2_b/              <- Dual-head + transfer learning (64 features)
    odyseus/                    <- Alternativa 1
    odyseus_2_0/                <- Alternativa 2
    odyseus_3_0/                <- Alternativa 3
    gestor_models/              <- Manager per canviar models
    offline_eval_cache/         <- Dades per avaluacio offline

### Models principals

| Model   | Hit Rate | Descripcio                                        |
|---------|----------|---------------------------------------------------|
| 2.1     | 77.2%    | Model base amb ML IPS + Bandit UCB                |
| 3.1b    | 79.0%    | Millor versio abans de 4.1                        |
| 4.1.e   | ~79%+    | Modular, hot-reload, optimitzacio automatica (ACTIU) |
| 6.2.b   | En val.  | 64 features, transfer learning, dual-head Slot-1  |

### Tecnologies utilitzades
- **Llenguatge**: Python 3
- **ML**: Scikit-learn, NumPy, Pandas
- **Bases de dades**: RawTree (SQL sobre JSON)
- **Avaluacio**: Offline evaluation sobre històric, leaderboard en viu
- **Optimitzacio**: Tuning automatic de pesos, presets configurables

### Mesures clau implementades
1. **Transicions seqüencials**: Qui va per A -> compra B
2. **Auto-conversio**: Productes ja vists amb mes probabilitat
3. **Re-ranker IPS**: Correccio de sesg de posicio
4. **Bandit UCB**: Equilibri exploracio/explotacio
5. **MMR diversity**: Diversitat de recomanacions
6. **Cold start**: TF-IDF + SVD per a productes nous
7. **Segmentacio**: 5 segments d'usuari amb pesos especifics
8. **Hot-reload**: Canvi de pesos sense reiniciar (4.1.e en endavant)
9. **Slot-gating**: Precisió als primers slots, exploracio a la cua
10. **Regles de negoci**: Carrito, productes recurrents, preu (4.1.e en endavant)
