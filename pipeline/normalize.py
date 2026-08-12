from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import SUPPORTED_TYPES


@dataclass
class NormalizationIssue:
    row_index: int
    node: str | None
    variable_type: str | None
    raw_value: Any
    error: str

    def to_dict(self) -> dict:
        return {
            "row_index": self.row_index,
            "node": self.node,
            "type": self.variable_type,
            "raw_value": self.raw_value,
            "error": self.error,
        }


def _finite_number(value: Any) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite numeric value: {value}"
        )

    return value


def _boolean(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    text = str(value).strip().lower()

    if text in {
        "1",
        "true",
        "t",
        "yes",
        "on",
    }:
        return 1

    if text in {
        "0",
        "false",
        "f",
        "no",
        "off",
    }:
        return 0

    raise ValueError(
        f"Cannot interpret {value!r} as boolean."
    )


def normalize_records(
    records: list[dict],
) -> tuple[list[dict], list[NormalizationIssue]]:
    """
    Normalize causRCA records using the `type` field supplied
    directly by the dataset.

    Supported types:

        Binary
        Alarm
        Counter
        Continuous
        Categorical

    Categorical values remain strings intentionally.
    """

    normalized: list[dict] = []
    issues: list[NormalizationIssue] = []

    for index, record in enumerate(records):

        node = record.get("node")

        variable_type = str(
            record.get("type", "")
        ).strip()

        raw_value = record.get("value")

        if variable_type not in SUPPORTED_TYPES:

            issues.append(
                NormalizationIssue(
                    row_index=index,
                    node=node,
                    variable_type=variable_type,
                    raw_value=raw_value,
                    error=(
                        f"Unsupported variable type: "
                        f"{variable_type!r}"
                    ),
                )
            )

            continue

        try:
            time_s = _finite_number(
                record["time_s"]
            )

            if variable_type in {
                "Binary",
                "Alarm",
            }:
                value = _boolean(raw_value)

            elif variable_type in {
                "Counter",
                "Continuous",
            }:
                value = _finite_number(
                    raw_value
                )

            elif variable_type == "Categorical":
                value = str(raw_value)

            else:
                raise ValueError(
                    f"Unhandled variable type: "
                    f"{variable_type}"
                )

            normalized.append(
                {
                    "time_s": time_s,
                    "node": str(node),
                    "value": value,
                    "type": variable_type,
                }
            )

        except Exception as exc:

            issues.append(
                NormalizationIssue(
                    row_index=index,
                    node=node,
                    variable_type=variable_type,
                    raw_value=raw_value,
                    error=str(exc),
                )
            )

    return normalized, issues
