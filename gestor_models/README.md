# `gestor_models/` — Gestor Modular de Models i Interfaç en Temps Real

Subcarpeta dedicada exclusivament al control, entrenament, monitoratge en directe i canvi sense talls entre tots els models del projecte (`calypso_3_1` encapsulat i etapes històriques).

---

## Arquitectura modular (on tocar cada cosa)

| Fitxer | Responsabilitat | Quan editar-lo |
|---|---|---|
| [config.py](config.py) | Rutes, port per defecte (`8765`), finestres de temps i càrrega de credencials | Si vols canviar el port web, els minuts de la finestra KPI o afegir carpetes |
| [registry.py](registry.py) | Descobriment automàtic de models (`model_manifest.json` + etapes històriques) i comandes `--train-only` | Si afegeixes nous flags d'entrenament o camps al manifest d'un futur `calypso_3_2` |
| [process_controller.py](process_controller.py) | Detecció de processos (`psutil`), pre-entrenament en segon pla i cutover sense conflictes amb rollback | Si vols ajustar timeouts d'arrencada o estratègies de reinici |
| [telemetry.py](telemetry.py) | Consultes RawTree en directe (`recent_requests`, `timeline` minut a minut, resum per `model_version`) i lectura de logs | Si vols afegir noves mètriques o gràfics SQL de ClickHouse |
| [web_server.py](web_server.py) | API REST (`/api/state`, `/api/train_activate`, `/api/switch`, `/api/stop`) i servidor HTTP local | Si vols afegir nous endpoints o accions des de la UI |
| `static/` (`index.html`, `styles.css`, `app.js`) | Interfaç web en temps real (sense dependències externes) | Si vols modificar el disseny visual, taules o freqüència de refresc (`1500ms`) |
| [tui.py](tui.py) | Monitor en temps real per a la terminal (`python model_manager.py live`) | Si prefereixes veure el flux de sol·licituds sense navegador |
| [cli.py](cli.py) | Comandes CLI (`ui`, `live`, `train`, `switch`, `status`, `list`, `stop`, `logs`, `watchdog`) | Si vols afegir noves subcomandes |

---

## Flux garantit sense conflictes en entrenar/activar un model
Quan selecciones **"Entrenar i Activar"** (des de la UI web o amb `python model_manager.py train <model>`):
1. **Self-check**: valida el nou model en local abans de tocar res.
2. **Pre-entrenament aïllat**: entrena el model en segon pla (p. ex. `--train-only` a `calypso_3_1`, `builder.py` a `etapa_2_version_especial` o `train.py` a `etapa_1_version_especial`) **mentre el model en producció segueix responent el 100% del trànsit**.
3. **Arrencada verificada**: llança el nou `serve.py` i espera que el `heartbeat.json` (o log d'activitat) confirmi `ready=True`.
4. **Tall net (Cutover)**: registra `.active_model.json`, guarda l'esdeveniment a `gestor_models/state/model_transitions.json` i atura ordenadament **tots** els processos antics (`serve`, `trainer`, `watchdog`) perquè mai competeixin dos models alhora. Si falla algun pas, fa **rollback automàtic**.
