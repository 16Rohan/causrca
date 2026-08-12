from __future__ import annotations

from typing import Any

from .loader import load_json, load_expert_graph


def load_description(
    path: str | None,
) -> dict[str, Any] | None:

    if not path:
        return None

    return load_json(path)


def load_causes(
    path: str | None,
) -> dict[str, Any] | None:

    if not path:
        return None

    return load_json(path)


def load_graph(
    path: str | None,
):

    if not path:
        return None

    return load_expert_graph(path)


def summarize_ground_truth(
    description,
    causes,
) -> dict:

    result = {
        "available": bool(
            description or causes
        ),
    }

    if description:

        result.update(
            {
                "experiment_id": description.get(
                    "exp_id"
                ),
                "group": description.get(
                    "group"
                ),
                "manipulated_variables": (
                    description.get(
                        "manipulatedVars",
                        [],
                    )
                ),
                "alarms": description.get(
                    "alarms",
                    [],
                ),
                "diagnoses": description.get(
                    "diagnoses",
                    [],
                ),
            }
        )

    if causes:

        for key in (
            "cause_start_at",
            "alarms_detected_at",
            "diagnosis_at",
            "cause_end_at",
            "alarms_resolved_at",
        ):

            if key in causes:

                result[key] = causes[key]

    return result
