"""Runtime path helpers for notebooks and scripts.

Use this module from notebooks to keep environment detection in one place.
"""

from __future__ import annotations

import os
from pathlib import Path


def detect_runtime_paths(
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
):
    """Detect Kaggle / Colab / local notebook paths.

    Parameters
    ----------
    project_root:
        Root directory of the EdgeDeID repo or mounted dataset. Set this on
        Kaggle or Colab when you want to override auto-detection.
    output_root:
        Directory where checkpoints and artifacts should be written.
        Defaults to ``/kaggle/working`` on Kaggle, otherwise ``project_root``.
    """
    env = "local"
    if Path("/kaggle/working").exists() or Path("/kaggle/input").exists():
        env = "kaggle"
    elif "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists():
        env = "colab"

    if project_root is not None:
        project_root = Path(project_root).expanduser().resolve()
    elif env == "kaggle":
        project_root = Path("/kaggle/input/datasets/sinadaneshgar/edgedeid")
    elif env == "colab":
        project_root = Path.cwd().resolve()
        if project_root.name == "RL":
            project_root = project_root.parent
    else:
        project_root = Path.cwd().resolve()
        if project_root.name == "RL":
            project_root = project_root.parent

    if output_root is not None:
        output_root = Path(output_root).expanduser().resolve()
    elif env == "kaggle":
        output_root = Path("/kaggle/working")
    else:
        output_root = project_root

    return {
        "env": env,
        "project_root": project_root,
        "output_root": output_root,
    }