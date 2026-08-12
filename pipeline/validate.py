from __future__ import annotations

import json

from .intermediate_schema import (
    validate_evidence,
)


def validate_analysis_evidence(
    evidence: dict,
) -> dict:

    validated = validate_evidence(
        evidence
    )

    # Explicit JSON serialization check.
    json.dumps(
        validated,
        allow_nan=False,
    )

    return validated
