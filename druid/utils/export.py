"""
druid.utils.export
~~~~~~~~~~~~~~~~~~~

Export DRUID session artifacts: reproducible scripts, reports,
and serialised models.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from druid.core.dataset import DruidDataset


def save_dataset(dataset: DruidDataset, path: str) -> None:
    """
    Serialise a DruidDataset to a pickle file.

    Parameters
    ----------
    dataset : DruidDataset
    path : str
        Output file path (.pkl).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(path: str) -> DruidDataset:
    """
    Load a serialised DruidDataset.

    Parameters
    ----------
    path : str
        Pickle file path.

    Returns
    -------
    DruidDataset
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def export_schema(dataset: DruidDataset, path: str) -> None:
    """
    Export dataset schema as YAML.

    Parameters
    ----------
    dataset : DruidDataset
    path : str
        Output YAML file path.
    """
    profile = dataset.profile()
    schema = {
        "name": profile["name"],
        "target": profile.get("target"),
        "shape": profile["shape"],
        "columns": profile["dtypes"],
        "classification": profile["classification"],
    }

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)


def export_script(dataset: DruidDataset, path: str) -> None:
    """
    Export the session as a standalone Python script.

    Parameters
    ----------
    dataset : DruidDataset
    path : str
    """
    script = dataset.session.to_script()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(script)
