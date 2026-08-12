from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import REQUIRED_CSV_COLUMNS


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """
    Load a causRCA CSV.

    causRCA stores variable semantics directly in each row:

        time_s,node,value,type

    Therefore no external categorical encoding file is required.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        columns = set(reader.fieldnames or [])

        missing = REQUIRED_CSV_COLUMNS - columns

        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                f"{sorted(missing)}"
            )

        records = list(reader)

    return records


def load_json(path: str | Path) -> Any:
    """Load a JSON document."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


def load_expert_graph(path: str | Path):
    """Load a causRCA expert graph."""

    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "NetworkX is required for expert graph processing."
        ) from exc

    return nx.read_gml(path)
