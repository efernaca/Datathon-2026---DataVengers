"""Encapsulated entrypoint for Calypso 3.1 live RawTree feedback monitor."""
from __future__ import annotations

import sys
from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parent
_STAGE4_ROOT = _MODEL_ROOT / "etapa_4"
sys.path.insert(0, str(_MODEL_ROOT))
sys.path.insert(0, str(_STAGE4_ROOT))

from etapa_4.src.monitor import main  # noqa: E402

if __name__ == "__main__":
    main()
