"""Thin root entrypoint delegating cleanly to the modular gestor_models package."""
from __future__ import annotations

import sys

from gestor_models.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
