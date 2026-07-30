from __future__ import annotations

import os
from pathlib import Path


DATASET_FILENAMES = {
    ("quasi", "train"): "indoorH_LS_Data6users_1B32pilot.mat",
    ("quasi", "validation"): "indoorH_LSval_Data6users_1B32pilot.mat",
    ("quasi", "test"): "indoorH_LStest_Data6users_1B32pilot.mat",
    ("mobility", "train"): "OutdoorH_LS_Data6users_60B32pilot.mat",
    ("mobility", "validation"): "OutdoorH_LSval_Data6users_60B32pilot.mat",
    ("mobility", "test"): "OutdoorH_LStest_Data6users_60B32pilot.mat",
}


def default_data_root(project_root: str | Path) -> Path:
    configured = os.environ.get("LPAN_DATA_ROOT")
    return (
        Path(configured).expanduser()
        if configured
        else Path(project_root).resolve() / "data"
    )


def dataset_candidates(
    data_root: str | Path,
    domain: str,
    split: str,
) -> list[Path]:
    key = (domain, split)
    if key not in DATASET_FILENAMES:
        raise ValueError(f"Unsupported dataset selection: {domain}/{split}")
    root = Path(data_root).expanduser().resolve()
    filename = DATASET_FILENAMES[key]
    folder = Path(filename).stem
    legacy_group = "risce" if domain == "quasi" else "risce-0"
    candidates = [
        root / domain / folder / filename,
        root / domain / filename,
        root / folder / filename,
        root / filename,
        root / legacy_group / folder / filename,
    ]
    # Preserve order while removing equivalent paths.
    return list(dict.fromkeys(candidates))


def resolve_dataset_path(
    data_root: str | Path,
    domain: str,
    split: str,
) -> Path:
    candidates = dataset_candidates(data_root, domain, split)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find the {domain} {split} MAT file.\n"
        f"Set --data-root or LPAN_DATA_ROOT, or pass an explicit file path.\n"
        f"Attempted:\n{attempted}"
    )
