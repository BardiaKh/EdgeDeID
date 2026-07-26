"""EdgeDeID — edge-deployable PHI de-identification.

Lightweight package root: submodules are imported explicitly
(``from edgedeid.eval import compute_metrics``) so that importing one area
(e.g. evaluation or baselines) does not pull in heavy training/RL dependencies.
"""

__version__ = "0.3.0"
