from __future__ import annotations

import os
from pathlib import Path


# ~/causrca/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Existing dataset:
# ~/causrca/dataset/
DATA_ROOT = Path(
    os.getenv(
        "CAUS_RCA_DATA_ROOT",
        str(PROJECT_ROOT / "dataset"),
    )
).expanduser().resolve()

OUTPUT_ROOT = Path(
    os.getenv(
        "CAUS_RCA_OUTPUT_ROOT",
        str(PROJECT_ROOT / "output"),
    )
).expanduser().resolve()

PROMPT_ROOT = Path(
    os.getenv(
        "CAUS_RCA_PROMPT_ROOT",
        str(PROJECT_ROOT / "prompts"),
    )
).expanduser().resolve()


RESAMPLE_INTERVAL = 0.5

SUPPORTED_TYPES = {
    "Binary",
    "Alarm",
    "Counter",
    "Continuous",
    "Categorical",
}

REQUIRED_CSV_COLUMNS = {
    "time_s",
    "node",
    "value",
    "type",
}


def ensure_directories() -> None:
    """Create runtime output directories if they don't exist."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "intermediate").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "rca").mkdir(parents=True, exist_ok=True)


def print_config() -> None:
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {DATA_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"Prompt root  : {PROMPT_ROOT}")
