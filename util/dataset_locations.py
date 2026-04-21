"""Shared helpers for loading dataset path overrides from a YAML config file.

Both ``datasets_medical.py`` (classification / segmentation) and
``datasets/coco.py`` (localization) need to resolve paths from the same YAML,
so this module lives in ``util/`` where neither side has an import cycle.

The YAML file is selected via the environment variable
``FOUNDATION_X_DATASET_LOCATIONS_YML`` (default:
``config/dataset_locations_asu_sol.yml``).

YAML structure expected::

    classification:
      candid_ptx:
        use_dicom: true
        images_dicom: /path/to/dicom
        images: /path/to/png
        splits:
          train: ...
          val: ...
          test: ...
          train_json: ...
          val_json: ...
          test_json: ...

    localization:
      candid_ptx:
        images: /path/to/png
        splits:
          train_json: ...
          val_json: ...
          test_json: ...

    segmentation:
      candid_ptx:
        ...
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def _load_dataset_locations() -> dict:
    """Load dataset path overrides from the configured YAML file (cached)."""
    if not hasattr(_load_dataset_locations, "_cache"):
        yml_path = os.environ.get(
            "FOUNDATION_X_DATASET_LOCATIONS_YML",
            "config/dataset_locations_asu_sol.yml",
        )
        yml_file = Path(yml_path)
        if not yml_file.is_absolute():
            # Resolve relative to the *project root* (parent of this file's
            # parent, i.e. Foundation_X+-CUDA-Container/).
            yml_file = Path(__file__).resolve().parent.parent / yml_file
        with yml_file.open("r", encoding="utf-8") as f:
            _load_dataset_locations._cache = yaml.safe_load(f) or {}
    return _load_dataset_locations._cache


def _dataset_location(*keys: str, default=None):
    """Look up a nested key path in the dataset-locations YAML.

    Example::

        path = _dataset_location("localization", "candid_ptx", "images",
                                 default="/fallback/path")
    """
    cfg = _load_dataset_locations()
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
