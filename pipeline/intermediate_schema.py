from __future__ import annotations

import json
import math
from typing import Any


REQUIRED_KEYS = {
    "case",
    "time_window",
    "observations",
    "signals",
    "alarms",
    "events",
    "relationships",
    "affected_entities",
    "timeline",
    "evidence",
    "ground_truth_available",
}


def _check_finite(
    value: Any,
    path: str = "$",
) -> None:

    if isinstance(value, float):

        if not math.isfinite(value):

            raise ValueError(
                f"Non-finite value at {path}"
            )

    elif isinstance(value, dict):

        for key, child in value.items():

            _check_finite(
                child,
                f"{path}.{key}",
            )

    elif isinstance(value, list):

        for index, child in enumerate(value):

            _check_finite(
                child,
                f"{path}[{index}]",
            )


def validate_evidence(
    evidence: dict,
) -> dict:

    missing = (
        REQUIRED_KEYS
        - set(evidence.keys())
    )

    if missing:

        raise ValueError(
            "Missing AnalysisEvidence keys: "
            f"{sorted(missing)}"
        )

    if not isinstance(
        evidence["case"],
        dict,
    ):
        raise TypeError(
            "case must be an object"
        )

    if not isinstance(
        evidence["time_window"],
        dict,
    ):
        raise TypeError(
            "time_window must be an object"
        )

    if not isinstance(
        evidence["observations"],
        dict,
    ):
        raise TypeError(
            "observations must be an object"
        )

    for key in (
        "signals",
        "alarms",
        "events",
        "relationships",
        "affected_entities",
        "timeline",
        "evidence",
    ):

        if not isinstance(
            evidence[key],
            list,
        ):

            raise TypeError(
                f"{key} must be an array"
            )

    start = evidence[
        "time_window"
    ].get("start")

    end = evidence[
        "time_window"
    ].get("end")

    if not isinstance(
        start,
        (int, float),
    ):
        raise TypeError(
            "time_window.start must be numeric"
        )

    if not isinstance(
        end,
        (int, float),
    ):
        raise TypeError(
            "time_window.end must be numeric"
        )

    if end < start:

        raise ValueError(
            "time_window.end cannot precede start"
        )

    _check_finite(evidence)

    # Final serialization check.
    json.dumps(
        evidence,
        allow_nan=False,
    )

    return evidence
