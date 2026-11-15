"""Etapa 4 · Calypso 6.2b: Dual-Head Slot-1 Precision, Synthetic->Real Transfer & RawTree CVR Velocity."""
import os

for _blas_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_env, "1")

__all__ = ["__version__"]
__version__ = "calypso-6.2f"
