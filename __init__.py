"""Engram-MultiGRN package.

Reproducibility note:
The deterministic hash multipliers in `model/multigrn.py`
(`DeterministicHash.register_domain`) use Python's built-in ``hash()``,
whose string hashing is randomized across processes UNLESS the environment
variable ``PYTHONHASHSEED`` is fixed (e.g., ``PYTHONHASHSEED=0``).
All manuscript results were produced with ``PYTHONHASHSEED=0``; without it,
domain-hash multipliers differ between runs and results are not reproducible.
"""
import os
import warnings

if os.environ.get("PYTHONHASHSEED", "") != "0":
    warnings.warn(
        "PYTHONHASHSEED is not set to '0': the domain-hash multipliers used by "
        "EngramMultiGRN are randomized across processes, so results are NOT "
        "reproducible. Re-run with PYTHONHASHSEED=0 (e.g., "
        "'PYTHONHASHSEED=0 python scripts/train_full.py').",
        UserWarning,
        stacklevel=2,
    )
