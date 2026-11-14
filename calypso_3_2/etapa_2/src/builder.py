"""Proceso CONSTRUCTOR (4.5): todo lo pesado, fuera del proceso que responde.

    python -m src.builder              # en otra terminal, junto a serve.py (bajo supervisor)
    python -m src.builder --once       # calcular el histórico y salir

* Calcula el histórico (8 consultas sobre 60 días) **una vez**, lo guarda en `cache/` con
  escritura atómica y escribe `state/history_version.txt`; el servidor lo recarga solo.
  Si algo falla, el servidor sigue con el último válido.
* Publica los agregados en `team.scratch` (`kind` + `version`) para que otra instancia
  pueda arrancar leyéndolos (`serve.py --from-scratch`).
* Cada hora: reentrena los pesos con el feedback (4.2) y rehace el análisis de los
  experimentos (4.1/4.4). No toca `switches.json` del domingo salvo que se lo pidas.
"""
from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"
    __import__("src")

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from .config import Config
from .history import cache_path, get_or_build_history, publish_aggregates

logger = logging.getLogger("e2esp.builder")


def build_once(client: Any, cfg: Config, rebuild: bool = False, publish: bool = True) -> str:
    h = get_or_build_history(client, cfg, rebuild=rebuild)
    if not h.version:
        h.version = f"h{int(h.meta.get('built_at', time.time()))}"
    state = Path(cfg.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    vfile = state / "history_version.txt"
    prev = vfile.read_text().strip() if vfile.is_file() else ""
    if prev != h.version:
        tmp = vfile.with_suffix(".tmp")
        tmp.write_text(h.version)
        tmp.replace(vfile)
        logger.info("Histórico %s listo en %s", h.version, cache_path(cfg, None))
        if publish:
            try:
                n = publish_aggregates(client, h)
                logger.info("Agregados publicados en scratch: %d filas (version=%s)", n, h.version)
            except Exception as exc:  # noqa: BLE001
                logger.warning("No se pudieron publicar los agregados: %s", exc)
    return h.version


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="recalcular el histórico aunque haya caché")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--learn-every-min", type=float, default=60.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [builder] %(message)s")
    from .analyze import run_analysis
    from .learner import run_learner
    from .rawtree_client import RawTreeClient

    cfg = Config.from_env()
    client = RawTreeClient(cfg.api_url, cfg.api_key, cfg.team_id)
    for attempt in range(1, 100):
        try:
            build_once(client, cfg, rebuild=args.rebuild, publish=not args.no_publish)
            break
        except Exception as exc:  # noqa: BLE001
            logger.error("construcción intento %d: %s", attempt, exc)
            time.sleep(min(60, 5 * attempt))
    if args.once:
        return
    while True:
        time.sleep(args.learn_every_min * 60)
        try:
            run_learner(client, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("learner: %s", exc)
        try:
            res = run_analysis(client, cfg)
            logger.info("Análisis: %d listas, interruptores sugeridos %s", res.get("n_lists", 0), res.get("switches"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("análisis: %s", exc)


if __name__ == "__main__":
    main()
