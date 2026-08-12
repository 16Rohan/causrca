# pipeline/intermediate_schema.py
"""
Validation for the compact causRCA intermediate representation.

This schema is INTERNAL to the Python -> LLM pipeline.

It is NOT the final RCA output schema.

The final output contract in:

    prompts/output_example.json

must remain unchanged.

Design goals
------------
1. Validate structure before NIM.
2. Reject malformed intermediate evidence early.
3. Prevent raw record dumps from accidentally reaching the LLM.
4. Allow numeric values to be int or float.
5. Reject NaN and Infinity.
6. Keep the intermediate representation compact.
"""

from __future__ import annotations

import json
import math
from typing import Any

# =============================================================================
# REQUIRED TOP-LEVEL KEYS
# =============================================================================

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


# =============================================================================
# SIZE LIMITS
# =============================================================================

MAX_SIGNALS = 100
MAX_ALARMS = 100
MAX_EVENTS = 100
MAX_RELATIONSHIPS = 200
MAX_AFFECTED_ENTITIES = 100
MAX_TIMELINE = 100
MAX_EVIDENCE = 150

MAX_EVENT_CATALOG = 100
MAX_OCCURRENCE_ANALYSIS = 100
MAX_STATE_TRANSITIONS = 100
MAX_SIGNIFICANT_CHANGES = 100
MAX_CANDIDATE_CAUSES = 100


# =============================================================================
# BASIC TYPE HELPERS
# =============================================================================


def _is_number(
    value: Any,
) -> bool:
    """
    JSON numeric value.

    bool is deliberately excluded because Python considers bool
    to be a subclass of int.
    """

    return isinstance(
        value,
        (int, float),
    ) and not isinstance(
        value,
        bool,
    )


def _check_finite(
    value: Any,
    path: str = "$",
) -> None:
    """
    Recursively reject NaN and Infinity.
    """

    if isinstance(value, float):

        if not math.isfinite(value):

            raise ValueError(f"Non-finite value at {path}")

        return

    if isinstance(value, dict):

        for key, child in value.items():

            _check_finite(
                child,
                f"{path}.{key}",
            )

        return

    if isinstance(value, list):

        for index, child in enumerate(value):

            _check_finite(
                child,
                f"{path}[{index}]",
            )


def _require_object(
    value: Any,
    name: str,
) -> None:

    if not isinstance(
        value,
        dict,
    ):

        raise TypeError(f"{name} must be an object")


def _require_array(
    value: Any,
    name: str,
) -> None:

    if not isinstance(
        value,
        list,
    ):

        raise TypeError(f"{name} must be an array")


def _require_string(
    value: Any,
    name: str,
) -> None:

    if not isinstance(
        value,
        str,
    ):

        raise TypeError(f"{name} must be a string")


def _require_number(
    value: Any,
    name: str,
) -> None:

    if not _is_number(value):

        raise TypeError(f"{name} must be numeric")


def _require_bool(
    value: Any,
    name: str,
) -> None:

    if not isinstance(
        value,
        bool,
    ):

        raise TypeError(f"{name} must be boolean")


def _check_array_size(
    value: list,
    maximum: int,
    name: str,
) -> None:

    if len(value) > maximum:

        raise ValueError(
            f"{name} contains {len(value)} items; " f"maximum allowed is {maximum}"
        )


# =============================================================================
# CASE
# =============================================================================


def _validate_case(
    case: dict,
) -> None:

    if "id" in case:
        _require_string(
            case["id"],
            "case.id",
        )

    for key in (
        "name",
        "type",
        "source",
        "dataset",
        "path",
    ):

        if key in case:
            _require_string(
                case[key],
                f"case.{key}",
            )


# =============================================================================
# TIME WINDOW
# =============================================================================


def _validate_time_window(
    time_window: dict,
) -> None:

    if "start" not in time_window:
        raise ValueError("time_window.start is required")

    if "end" not in time_window:
        raise ValueError("time_window.end is required")

    _require_number(
        time_window["start"],
        "time_window.start",
    )

    _require_number(
        time_window["end"],
        "time_window.end",
    )

    start = float(time_window["start"])

    end = float(time_window["end"])

    if end < start:

        raise ValueError("time_window.end cannot precede start")


# =============================================================================
# OBSERVATIONS
# =============================================================================


def _validate_observations(
    observations: dict,
) -> None:

    if "record_count" in observations:

        if not isinstance(
            observations["record_count"],
            int,
        ) or isinstance(
            observations["record_count"],
            bool,
        ):

            raise TypeError("observations.record_count " "must be an integer")

        if observations["record_count"] < 0:

            raise ValueError("observations.record_count " "cannot be negative")

    if "unique_nodes" in observations:

        _require_array(
            observations["unique_nodes"],
            "observations.unique_nodes",
        )

        for index, node in enumerate(observations["unique_nodes"]):

            _require_string(
                node,
                f"observations.unique_nodes[{index}]",
            )

    if "analysis" in observations:

        _require_object(
            observations["analysis"],
            "observations.analysis",
        )


# =============================================================================
# SIGNALS
# =============================================================================


def _validate_signals(
    signals: list,
) -> None:

    _check_array_size(
        signals,
        MAX_SIGNALS,
        "signals",
    )

    for index, signal in enumerate(signals):

        path = f"signals[{index}]"

        _require_object(
            signal,
            path,
        )

        if "id" in signal:
            _require_string(
                signal["id"],
                f"{path}.id",
            )

        if "type" in signal:
            _require_string(
                signal["type"],
                f"{path}.type",
            )

        for key in (
            "count",
            "min",
            "max",
            "mean",
            "median",
            "stdev",
        ):

            if key in signal:

                _require_number(
                    signal[key],
                    f"{path}.{key}",
                )

        if "alarm_activity" in signal:

            _require_object(
                signal["alarm_activity"],
                f"{path}.alarm_activity",
            )


# =============================================================================
# ALARMS
# =============================================================================


def _validate_alarms(
    alarms: list,
) -> None:

    _check_array_size(
        alarms,
        MAX_ALARMS,
        "alarms",
    )

    for index, alarm in enumerate(alarms):

        path = f"alarms[{index}]"

        _require_object(
            alarm,
            path,
        )

        if "node" in alarm:
            _require_string(
                alarm["node"],
                f"{path}.node",
            )

        for key in (
            "first",
            "last",
            "occurrences",
            "frequency",
        ):

            if key in alarm:

                _require_number(
                    alarm[key],
                    f"{path}.{key}",
                )


# =============================================================================
# EVENTS
# =============================================================================


def _validate_events(
    events: list,
) -> None:
    """
    Validate grouped events.

    Individual raw observations are deliberately NOT allowed here.
    """

    _check_array_size(
        events,
        MAX_EVENTS,
        "events",
    )

    for index, event in enumerate(events):

        path = f"events[{index}]"

        _require_object(
            event,
            path,
        )

        for key in (
            "id",
            "event_id",
            "node",
            "type",
            "pattern",
        ):

            if key in event:

                _require_string(
                    event[key],
                    f"{path}.{key}",
                )

        for key in (
            "occurrences",
            "first_seen",
            "last_seen",
            "duration",
        ):

            if key in event:

                _require_number(
                    event[key],
                    f"{path}.{key}",
                )

        if "occurrences" in event and event["occurrences"] < 0:

            raise ValueError(f"{path}.occurrences " "cannot be negative")


# =============================================================================
# RELATIONSHIPS
# =============================================================================


def _validate_relationships(
    relationships: list,
) -> None:

    _check_array_size(
        relationships,
        MAX_RELATIONSHIPS,
        "relationships",
    )

    allowed_types = {
        "TEMPORAL_PRECEDENCE",
        "CO_OCCURRENCE",
        "SHARED_ENTITY",
        "SIGNAL_EVENT_ASSOCIATION",
    }

    for index, relationship in enumerate(relationships):

        path = f"relationships[{index}]"

        _require_object(
            relationship,
            path,
        )

        if "source" in relationship:

            _require_string(
                relationship["source"],
                f"{path}.source",
            )

        if "target" in relationship:

            _require_string(
                relationship["target"],
                f"{path}.target",
            )

        if "relationship" in relationship:

            _require_string(
                relationship["relationship"],
                f"{path}.relationship",
            )

            if relationship["relationship"] not in allowed_types:

                raise ValueError(
                    f"{path}.relationship "
                    f"contains unsupported type: "
                    f"{relationship['relationship']}"
                )

        for key in (
            "lag",
            "overlap_duration",
            "source_first_seen",
            "source_last_seen",
            "target_first_seen",
            "target_last_seen",
        ):

            if key in relationship:

                _require_number(
                    relationship[key],
                    f"{path}.{key}",
                )


# =============================================================================
# AFFECTED ENTITIES
# =============================================================================


def _validate_affected_entities(
    entities: list,
) -> None:

    _check_array_size(
        entities,
        MAX_AFFECTED_ENTITIES,
        "affected_entities",
    )

    for index, entity in enumerate(entities):

        path = f"affected_entities[{index}]"

        _require_object(
            entity,
            path,
        )

        for key in (
            "id",
            "type",
        ):

            if key in entity:

                _require_string(
                    entity[key],
                    f"{path}.{key}",
                )

        for key in (
            "event_count",
            "event_groups",
            "alarm_occurrences",
        ):

            if key in entity:

                _require_number(
                    entity[key],
                    f"{path}.{key}",
                )


# =============================================================================
# TIMELINE
# =============================================================================


def _validate_timeline(
    timeline: list,
) -> None:

    _check_array_size(
        timeline,
        MAX_TIMELINE,
        "timeline",
    )

    for index, item in enumerate(timeline):

        path = f"timeline[{index}]"

        _require_object(
            item,
            path,
        )

        if "group_id" in item:

            _require_string(
                item["group_id"],
                f"{path}.group_id",
            )

        for key in (
            "first_seen",
            "last_seen",
            "occurrences",
        ):

            if key in item:

                _require_number(
                    item[key],
                    f"{path}.{key}",
                )


# =============================================================================
# EVIDENCE
# =============================================================================


def _validate_evidence_items(
    evidence: list,
) -> None:

    _check_array_size(
        evidence,
        MAX_EVIDENCE,
        "evidence",
    )

    for index, item in enumerate(evidence):

        path = f"evidence[{index}]"

        _require_object(
            item,
            path,
        )

        if "type" in item:

            _require_string(
                item["type"],
                f"{path}.type",
            )

        for key in (
            "group_id",
            "event_id",
            "node",
            "pattern",
            "source",
            "target",
            "relationship",
        ):

            if key in item:

                _require_string(
                    item[key],
                    f"{path}.{key}",
                )

        for key in (
            "occurrences",
            "first_seen",
            "last_seen",
            "lag",
            "overlap_duration",
            "time_s",
            "previous",
            "current",
            "delta",
        ):

            if key in item:

                _require_number(
                    item[key],
                    f"{path}.{key}",
                )


# =============================================================================
# OPTIONAL COMPACT STRUCTURES
# =============================================================================


def _validate_event_catalog(
    catalog: Any,
) -> None:

    if catalog is None:
        return

    _require_array(
        catalog,
        "event_catalog",
    )

    _check_array_size(
        catalog,
        MAX_EVENT_CATALOG,
        "event_catalog",
    )

    for index, item in enumerate(catalog):

        path = f"event_catalog[{index}]"

        _require_object(
            item,
            path,
        )

        for key in (
            "id",
            "node",
            "type",
            "pattern",
        ):

            if key in item:

                _require_string(
                    item[key],
                    f"{path}.{key}",
                )


def _validate_occurrence_analysis(
    occurrence_analysis: Any,
) -> None:

    if occurrence_analysis is None:
        return

    _require_array(
        occurrence_analysis,
        "occurrence_analysis",
    )

    _check_array_size(
        occurrence_analysis,
        MAX_OCCURRENCE_ANALYSIS,
        "occurrence_analysis",
    )

    for index, item in enumerate(occurrence_analysis):

        path = f"occurrence_analysis[{index}]"

        _require_object(
            item,
            path,
        )

        if "group_id" in item:

            _require_string(
                item["group_id"],
                f"{path}.group_id",
            )

        for key in (
            "count",
            "frequency",
            "mean_interval",
            "mean_change_magnitude",
            "max_change_magnitude",
        ):

            if key in item:

                _require_number(
                    item[key],
                    f"{path}.{key}",
                )


def _validate_state_transitions(
    transitions: Any,
) -> None:

    if transitions is None:
        return

    _require_array(
        transitions,
        "state_transitions",
    )

    _check_array_size(
        transitions,
        MAX_STATE_TRANSITIONS,
        "state_transitions",
    )

    for index, item in enumerate(transitions):

        path = f"state_transitions[{index}]"

        _require_object(
            item,
            path,
        )

        for key in (
            "node",
            "from",
            "to",
        ):

            if key in item:

                _require_string(
                    item[key],
                    f"{path}.{key}",
                )

        for key in (
            "count",
            "first_seen",
            "last_seen",
        ):

            if key in item:

                _require_number(
                    item[key],
                    f"{path}.{key}",
                )


def _validate_significant_changes(
    changes: Any,
) -> None:

    if changes is None:
        return

    _require_array(
        changes,
        "significant_changes",
    )

    _check_array_size(
        changes,
        MAX_SIGNIFICANT_CHANGES,
        "significant_changes",
    )

    for index, item in enumerate(changes):

        path = f"significant_changes[{index}]"

        _require_object(
            item,
            path,
        )

        if "node" in item:

            _require_string(
                item["node"],
                f"{path}.node",
            )

        for key in (
            "time_s",
            "previous",
            "current",
            "delta",
            "absolute_delta",
        ):

            if key in item:

                _require_number(
                    item[key],
                    f"{path}.{key}",
                )


def _validate_candidate_causes(
    candidates: Any,
) -> None:

    if candidates is None:
        return

    _require_array(
        candidates,
        "candidate_causes",
    )

    _check_array_size(
        candidates,
        MAX_CANDIDATE_CAUSES,
        "candidate_causes",
    )

    for index, item in enumerate(candidates):

        path = f"candidate_causes[{index}]"

        _require_object(
            item,
            path,
        )

        for key in (
            "group_id",
            "node",
            "pattern",
            "role",
        ):

            if key in item:

                _require_string(
                    item[key],
                    f"{path}.{key}",
                )

        for key in (
            "temporal_successors",
            "temporal_predecessors",
        ):

            if key in item:

                _require_number(
                    item[key],
                    f"{path}.{key}",
                )


# =============================================================================
# MAIN VALIDATOR
# =============================================================================


def validate_evidence(
    evidence: dict,
) -> dict:
    """
    Validate the compact AnalysisEvidence object.

    This function intentionally validates the intermediate format,
    not the final RCA output schema.
    """

    if not isinstance(
        evidence,
        dict,
    ):

        raise TypeError("AnalysisEvidence must be an object")

    # -------------------------------------------------------------------------
    # Required keys.
    # -------------------------------------------------------------------------

    missing = REQUIRED_KEYS - set(evidence.keys())

    if missing:

        raise ValueError("Missing AnalysisEvidence keys: " f"{sorted(missing)}")

    # -------------------------------------------------------------------------
    # Required structures.
    # -------------------------------------------------------------------------

    _require_object(
        evidence["case"],
        "case",
    )

    _require_object(
        evidence["time_window"],
        "time_window",
    )

    _require_object(
        evidence["observations"],
        "observations",
    )

    _require_array(
        evidence["signals"],
        "signals",
    )

    _require_array(
        evidence["alarms"],
        "alarms",
    )

    _require_array(
        evidence["events"],
        "events",
    )

    _require_array(
        evidence["relationships"],
        "relationships",
    )

    _require_array(
        evidence["affected_entities"],
        "affected_entities",
    )

    _require_array(
        evidence["timeline"],
        "timeline",
    )

    _require_array(
        evidence["evidence"],
        "evidence",
    )

    _require_bool(
        evidence["ground_truth_available"],
        "ground_truth_available",
    )

    # -------------------------------------------------------------------------
    # Individual structure validation.
    # -------------------------------------------------------------------------

    _validate_case(evidence["case"])

    _validate_time_window(evidence["time_window"])

    _validate_observations(evidence["observations"])

    _validate_signals(evidence["signals"])

    _validate_alarms(evidence["alarms"])

    _validate_events(evidence["events"])

    _validate_relationships(evidence["relationships"])

    _validate_affected_entities(evidence["affected_entities"])

    _validate_timeline(evidence["timeline"])

    _validate_evidence_items(evidence["evidence"])

    # -------------------------------------------------------------------------
    # Optional compact analytical structures.
    # -------------------------------------------------------------------------

    _validate_event_catalog(evidence.get("event_catalog"))

    _validate_occurrence_analysis(evidence.get("occurrence_analysis"))

    _validate_state_transitions(evidence.get("state_transitions"))

    _validate_significant_changes(evidence.get("significant_changes"))

    _validate_candidate_causes(evidence.get("candidate_causes"))

    # -------------------------------------------------------------------------
    # Semantic sanity checks.
    # -------------------------------------------------------------------------

    time_window = evidence["time_window"]

    start = float(time_window["start"])

    end = float(time_window["end"])

    # Event groups must fall inside the analysis window.
    for index, event in enumerate(evidence["events"]):

        first_seen = event.get("first_seen")

        last_seen = event.get("last_seen")

        if first_seen is not None:

            if float(first_seen) < start or float(first_seen) > end:

                raise ValueError(
                    f"events[{index}].first_seen " "falls outside time_window"
                )

        if last_seen is not None:

            if float(last_seen) < start or float(last_seen) > end:

                raise ValueError(
                    f"events[{index}].last_seen " "falls outside time_window"
                )

        if (
            first_seen is not None
            and last_seen is not None
            and float(last_seen) < float(first_seen)
        ):

            raise ValueError(f"events[{index}].last_seen " "cannot precede first_seen")

    # -------------------------------------------------------------------------
    # Final strict JSON serialization check.
    # -------------------------------------------------------------------------

    _check_finite(evidence)

    try:

        json.dumps(
            evidence,
            ensure_ascii=False,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "AnalysisEvidence cannot be " "serialized as strict JSON"
        ) from exc

    return evidence


__all__ = [
    "validate_evidence",
    "REQUIRED_KEYS",
]
