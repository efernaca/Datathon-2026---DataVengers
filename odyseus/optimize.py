"""Encapsulated CLI entrypoint for the Odyseus offline iterative conversion optimizer (calypso-5.1f)."""
from __future__ import annotations

import sys
from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parent
_STAGE4_ROOT = _MODEL_ROOT / "etapa_4"
for _p in (_STAGE4_ROOT, _MODEL_ROOT):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
sys.path.insert(0, str(_MODEL_ROOT))
sys.path.insert(0, str(_STAGE4_ROOT))

from etapa_4.src.odyseus_optimizer import main  # noqa: E402

if __name__ == "__main__":
    main()
