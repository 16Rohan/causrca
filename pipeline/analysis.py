# pipeline/analysis.py
"""
Deterministic analysis for the causRCA time-series dataset.

This module operates on normalized/resampled records and produces
compact analytical primitives for the downstream RCA pipeline.

Important design rule:

    RAW OBSERVATIONS
          ↓
    aggregate repeated behavior
          ↓
    event definitions
    event groups
    occurrence statistics
    state transitions
    significant changes
          ↓
    compact evidence

The module does NOT perform LLM reasoning and does NOT declare causal
relationships. Causality is handled by the dedicated causal stage.

No arbitrary anomaly thresholds are introduced here.
A measured deviation is evidence, not automatically an anomaly.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List

Record = Dict[str, Any]


# =============================================================================
# CONSTANTS
# =============================================================================

# Prevent pathological intermediate objects on very large cases.
# This is a representation limit, not an anomaly threshold.
MAX_EVENT_GROUPS = 500
MAX_SIGNIFICANT_CHANGES = 250


# =============================================================================
# BASIC HELPERS
# =============================================================================


def _safe_float(value: Any) -> float | None:
    """Return a finite float or None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        result = float(value)
    except TypeError, ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def _sorted_records(
    records: Iterable[Record],
) -> List[Record]:
    """Return records in deterministic time/node order."""

    return sorted(
        records,
        key=lambda r: (
            float(r.get("time_s", 0.0)),
            str(r.get("node", "")),
        ),
    )


def _group_by_node(
    records: List[Record],
) -> Dict[str, List[Record]]:
    """Group records by signal/node."""

    groups: Dict[str, List[Record]] = defaultdict(list)

    for record in records:
        node = str(record.get("node", ""))
        groups[node].append(record)

    for node in groups:
        groups[node].sort(key=lambda r: float(r.get("time_s", 0.0)))

    return dict(groups)


def _percentage(
    value: int | float,
    total: int | float,
) -> float:
    """Return a human-readable percentage."""

    if not total:
        return 0.0

    return round(
        (float(value) / float(total)) * 100.0,
        4,
    )


# =============================================================================
# NUMERIC STATISTICS
# =============================================================================


def _numeric_statistics(
    node_groups: Dict[str, List[Record]],
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate deterministic descriptive statistics for numeric signals.

    No anomaly threshold is applied.
    """

    result: Dict[str, Dict[str, Any]] = {}

    for node, values in node_groups.items():

        numeric_values = [
            value
            for value in (_safe_float(record.get("value")) for record in values)
            if value is not None
        ]

        if not numeric_values:
            continue

        count = len(numeric_values)

        result[node] = {
            "count": count,
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "stdev": (statistics.stdev(numeric_values) if count > 1 else 0.0),
        }

    return result


# =============================================================================
# ALARM STATISTICS
# =============================================================================


def _is_active_alarm(
    record: Record,
) -> bool:
    """
    Determine whether a normalized record represents an active alarm.

    Alarm/Binary values are expected to have been normalized to a
    numeric representation by the previous preprocessing stage.
    """

    record_type = str(record.get("type", ""))

    if record_type not in {
        "Alarm",
        "Binary",
    }:
        return False

    value = record.get("value")

    if isinstance(value, bool):
        return value

    numeric = _safe_float(value)

    return numeric == 1.0


def _alarm_statistics(
    node_groups: Dict[str, List[Record]],
) -> List[Dict[str, Any]]:
    """Aggregate alarm activity by node."""

    result: List[Dict[str, Any]] = []

    for node, records in node_groups.items():

        active = [record for record in records if _is_active_alarm(record)]

        if not active:
            continue

        times = [float(record["time_s"]) for record in active]

        first = min(times)
        last = max(times)

        duration = max(
            last - first,
            0.0,
        )

        frequency = len(active) / duration if duration > 0 else float(len(active))

        result.append(
            {
                "node": node,
                "first": first,
                "last": last,
                "occurrences": len(active),
                "frequency": frequency,
            }
        )

    result.sort(
        key=lambda item: (
            -item["occurrences"],
            item["node"],
        )
    )

    return result


# =============================================================================
# EVENT SIGNATURES
# =============================================================================


def _event_signature(
    previous: Any,
    current: Any,
    record_type: str,
) -> str | None:
    """
    Convert a value transition into a compact semantic event type.

    Continuous values are NOT stored repeatedly.

    Instead:

        increase
        decrease
        value_change

    is recorded once as an event definition and occurrences are
    aggregated separately.
    """

    if previous is None:
        return "INITIAL_VALUE"

    previous_num = _safe_float(previous)
    current_num = _safe_float(current)

    if previous_num is not None and current_num is not None:

        if current_num > previous_num:
            return "INCREASE"

        if current_num < previous_num:
            return "DECREASE"

        return None

    if previous != current:
        if record_type in {
            "Alarm",
            "Binary",
        }:
            return "STATE_CHANGE"

        return "VALUE_CHANGE"

    return None


# =============================================================================
# EVENT CATALOG
# =============================================================================


def _build_event_catalog(
    node_groups: Dict[str, List[Record]],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Build a deduplicated event catalog and aggregated event groups.

    Returns:

        event_catalog
        event_groups
        occurrence_analysis
    """

    # Signature -> catalog entry
    catalog: Dict[
        tuple[str, str, str],
        Dict[str, Any],
    ] = {}

    # Signature -> occurrence information
    occurrences: Dict[
        tuple[str, str, str],
        Dict[str, Any],
    ] = {}

    # -------------------------------------------------------------------------
    # Scan each signal once.
    # -------------------------------------------------------------------------

    for node, records in node_groups.items():

        previous_value: Any = None
        previous_time: float | None = None

        for record in records:

            current_value = record.get("value")
            record_type = str(record.get("type", ""))
            current_time = float(record.get("time_s", 0.0))

            signature_type = _event_signature(
                previous_value,
                current_value,
                record_type,
            )

            if signature_type is None:
                previous_value = current_value
                previous_time = current_time
                continue

            signature = (
                node,
                record_type,
                signature_type,
            )

            # -------------------------------------------------------------
            # Create definition exactly once.
            # -------------------------------------------------------------

            if signature not in catalog:

                event_id = f"E{len(catalog) + 1:04d}"

                catalog[signature] = {
                    "id": event_id,
                    "node": node,
                    "type": record_type,
                    "pattern": signature_type,
                }

                occurrences[signature] = {
                    "event_id": event_id,
                    "count": 0,
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "intervals": [],
                    "magnitudes": [],
                }

            occurrence = occurrences[signature]

            occurrence["count"] += 1
            occurrence["first_seen"] = min(
                occurrence["first_seen"],
                current_time,
            )
            occurrence["last_seen"] = max(
                occurrence["last_seen"],
                current_time,
            )

            if previous_time is not None:
                interval = current_time - previous_time

                if interval >= 0:
                    occurrence["intervals"].append(interval)

            previous_numeric = _safe_float(previous_value)
            current_numeric = _safe_float(current_value)

            if previous_numeric is not None and current_numeric is not None:
                occurrence["magnitudes"].append(abs(current_numeric - previous_numeric))

            previous_value = current_value
            previous_time = current_time

    # -------------------------------------------------------------------------
    # Build deterministic output arrays.
    # -------------------------------------------------------------------------

    catalog_items = list(catalog.values())

    catalog_items.sort(key=lambda item: item["id"])

    # Limit the representation, not the analysis.
    # Important event groups are selected by occurrence count.
    occurrence_items = list(occurrences.items())

    occurrence_items.sort(
        key=lambda item: (
            -item[1]["count"],
            item[0][0],
            item[0][2],
        )
    )

    selected = occurrence_items[:MAX_EVENT_GROUPS]

    event_groups: List[Dict[str, Any]] = []
    occurrence_analysis: List[Dict[str, Any]] = []

    for index, (
        signature,
        occurrence,
    ) in enumerate(selected, start=1):

        node, record_type, pattern = signature

        group_id = f"G{index:04d}"

        count = occurrence["count"]
        first = occurrence["first_seen"]
        last = occurrence["last_seen"]

        duration = max(
            last - first,
            0.0,
        )

        frequency = count / duration if duration > 0 else float(count)

        intervals = occurrence["intervals"]
        magnitudes = occurrence["magnitudes"]

        event_id = occurrence["event_id"]

        event_groups.append(
            {
                "id": group_id,
                "event_id": event_id,
                "node": node,
                "type": record_type,
                "pattern": pattern,
                "occurrences": count,
                "first_seen": first,
                "last_seen": last,
                "duration": duration,
            }
        )

        analysis: Dict[str, Any] = {
            "group_id": group_id,
            "count": count,
            "frequency": frequency,
        }

        if intervals:
            analysis["mean_interval"] = statistics.mean(intervals)

        if magnitudes:
            analysis["mean_change_magnitude"] = statistics.mean(magnitudes)

            analysis["max_change_magnitude"] = max(magnitudes)

        occurrence_analysis.append(analysis)

    return (
        catalog_items,
        event_groups,
        occurrence_analysis,
    )


# =============================================================================
# STATE TRANSITIONS
# =============================================================================


def _build_state_transitions(
    node_groups: Dict[str, List[Record]],
) -> List[Dict[str, Any]]:
    """
    Aggregate repeated state transitions.

    Example:

        OFF → ON
        ON → OFF

    is represented once with occurrence counts.
    """

    transitions: Dict[
        tuple[str, str, str],
        Dict[str, Any],
    ] = {}

    for node, records in node_groups.items():

        previous_value: Any = None

        for record in records:

            current_value = record.get("value")

            if previous_value is None:
                previous_value = current_value
                continue

            if previous_value == current_value:
                continue

            previous_text = str(previous_value)
            current_text = str(current_value)

            key = (
                node,
                previous_text,
                current_text,
            )

            entry = transitions.setdefault(
                key,
                {
                    "node": node,
                    "from": previous_text,
                    "to": current_text,
                    "count": 0,
                    "first_seen": float(record["time_s"]),
                    "last_seen": float(record["time_s"]),
                },
            )

            entry["count"] += 1
            entry["last_seen"] = float(record["time_s"])

            previous_value = current_value

    result = list(transitions.values())

    result.sort(
        key=lambda item: (
            -item["count"],
            item["node"],
            item["from"],
            item["to"],
        )
    )

    return result[:MAX_EVENT_GROUPS]


# =============================================================================
# SIGNIFICANT CHANGES
# =============================================================================


def _build_significant_changes(
    node_groups: Dict[str, List[Record]],
) -> List[Dict[str, Any]]:
    """
    Rank the largest observed numeric changes.

    This is a compact representation of change magnitude.

    It does NOT label these changes as anomalies.
    """

    changes: List[Dict[str, Any]] = []

    for node, records in node_groups.items():

        previous_value: float | None = None

        for record in records:

            current_value = _safe_float(record.get("value"))

            if previous_value is None or current_value is None:
                previous_value = current_value
                continue

            delta = current_value - previous_value

            if delta == 0:
                previous_value = current_value
                continue

            changes.append(
                {
                    "node": node,
                    "time_s": float(record["time_s"]),
                    "previous": previous_value,
                    "current": current_value,
                    "delta": delta,
                    "absolute_delta": abs(delta),
                }
            )

            previous_value = current_value

    changes.sort(
        key=lambda item: (
            -item["absolute_delta"],
            item["node"],
            item["time_s"],
        )
    )

    return changes[:MAX_SIGNIFICANT_CHANGES]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def compute_statistics(
    records: List[Record],
) -> Dict[str, Any]:
    """
    Compute deterministic RCA-oriented statistics.

    The returned object intentionally contains aggregated structures
    rather than record-level event dumps.

    Existing keys are retained for compatibility while the new compact
    structures are added for the next evidence-building stage.
    """

    if not records:
        return {
            "total_records": 0,
            "unique_nodes": [],
            "time_span_secs": (
                0.0,
                0.0,
            ),
            "numeric_stats": {},
            "alarm_stats": [],
            "event_catalog": [],
            "event_groups": [],
            "occurrence_analysis": [],
            "state_transitions": [],
            "significant_changes": [],
            "summary": {
                "numeric_variables": 0,
                "alarm_variables": 0,
                "event_definitions": 0,
                "event_groups": 0,
            },
        }

    ordered_records = _sorted_records(records)

    node_groups = _group_by_node(ordered_records)

    times = [float(record["time_s"]) for record in ordered_records]

    min_time = min(times)
    max_time = max(times)

    numeric_stats = _numeric_statistics(node_groups)

    alarm_stats = _alarm_statistics(node_groups)

    (
        event_catalog,
        event_groups,
        occurrence_analysis,
    ) = _build_event_catalog(node_groups)

    state_transitions = _build_state_transitions(node_groups)

    significant_changes = _build_significant_changes(node_groups)

    numeric_variables = len(numeric_stats)

    alarm_variables = len(alarm_stats)

    result: Dict[str, Any] = {
        # ------------------------------------------------------------------
        # Existing compatibility fields.
        # ------------------------------------------------------------------
        "total_records": len(ordered_records),
        "unique_nodes": list(node_groups.keys()),
        "time_span_secs": (
            min_time,
            max_time,
        ),
        "numeric_stats": numeric_stats,
        "alarm_stats": alarm_stats,
        # ------------------------------------------------------------------
        # New compact RCA representation.
        # ------------------------------------------------------------------
        "event_catalog": event_catalog,
        "event_groups": event_groups,
        "occurrence_analysis": (occurrence_analysis),
        "state_transitions": (state_transitions),
        "significant_changes": (significant_changes),
        # ------------------------------------------------------------------
        # High-level deterministic summary.
        # ------------------------------------------------------------------
        "summary": {
            "numeric_variables": (numeric_variables),
            "alarm_variables": (alarm_variables),
            "event_definitions": len(event_catalog),
            "event_groups": len(event_groups),
        },
    }

    return result


__all__ = [
    "compute_statistics",
]
