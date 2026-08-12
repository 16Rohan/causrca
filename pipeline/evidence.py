# pipeline/evidence.py
"""
Compact AnalysisEvidence builder for causRCA.

Purpose
-------
Convert deterministic Python analysis into a compact representation
suitable for LLM-based root-cause analysis.

The LLM must NOT receive the raw dataset or record-level event history.

Instead, this module sends:

    case metadata
    time window
    compact signal statistics
    unique event definitions
    grouped event occurrences
    causal/temporal relationships
    affected entities
    significant changes
    deterministic evidence
    ground-truth availability
    data-quality information

Repeated observations are represented once and referenced by ID.

Final LLM output schema:
    UNCHANGED.

This module only controls the intermediate representation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_SIGNALS = 100
MAX_EVENTS = 100
MAX_EVENT_GROUPS = 100
MAX_RELATIONSHIPS = 200
MAX_AFFECTED_ENTITIES = 100
MAX_EVIDENCE = 150
MAX_CANDIDATE_CAUSES = 100
MAX_CHANGES = 100


# =============================================================================
# HELPERS
# =============================================================================

def _safe_float(
    value: Any,
) -> float | None:
    """Return a finite float or None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _clean_number(
    value: Any,
) -> Any:
    """
    Convert finite numeric values into JSON-safe values.

    Integer-valued floats are converted to integers only to keep the
    intermediate JSON compact and readable.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        if value.is_integer():
            return int(value)

        return value

    return value


def _clean(
    value: Any,
) -> Any:
    """Recursively sanitize values for strict JSON."""

    if isinstance(value, dict):
        return {
            str(key): _clean(child)
            for key, child in value.items()
        }

    if isinstance(value, list):
        return [
            _clean(child)
            for child in value
        ]

    return _clean_number(value)


def _limit(
    items: List[Any],
    limit: int,
) -> List[Any]:
    """Return a deterministic bounded list."""

    return items[:limit]


# =============================================================================
# CASE INFORMATION
# =============================================================================

def _build_case(
    case: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Preserve useful case metadata without copying raw dataset information.
    """

    if not case:
        return {
            "id": "unknown",
            "source": "causRCA",
        }

    result: Dict[str, Any] = {}

    for key in (
        "id",
        "name",
        "type",
        "source",
        "dataset",
        "path",
    ):
        if key in case:
            result[key] = case[key]

    if "id" not in result:
        result["id"] = "unknown"

    return result


# =============================================================================
# TIME WINDOW
# =============================================================================

def _build_time_window(
    records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Determine the relative analysis time window."""

    if not records:
        return {
            "start": 0.0,
            "end": 0.0,
        }

    times = []

    for record in records:

        value = _safe_float(
            record.get("time_s")
        )

        if value is not None:
            times.append(value)

    if not times:
        return {
            "start": 0.0,
            "end": 0.0,
        }

    return {
        "start": min(times),
        "end": max(times),
    }


# =============================================================================
# SIGNALS
# =============================================================================

def _build_signals(
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Produce compact signal definitions.

    Full raw signal histories are intentionally excluded.
    """

    numeric_stats = analysis.get(
        "numeric_stats",
        {},
    )

    alarm_stats = analysis.get(
        "alarm_stats",
        [],
    )

    alarm_by_node = {
        str(item.get("node")): item
        for item in alarm_stats
    }

    signals: List[
        Dict[str, Any]
    ] = []

    for node, stats in numeric_stats.items():

        signal: Dict[str, Any] = {
            "id": str(node),
            "type": "numeric",
            "count": stats.get("count", 0),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "mean": stats.get("mean"),
            "median": stats.get("median"),
            "stdev": stats.get("stdev"),
        }

        if node in alarm_by_node:
            signal["alarm_activity"] = (
                alarm_by_node[node]
            )

        signals.append(signal)

    # Add alarm-only entities.
    existing = {
        signal["id"]
        for signal in signals
    }

    for item in alarm_stats:

        node = str(
            item.get("node", "")
        )

        if not node or node in existing:
            continue

        signals.append(
            {
                "id": node,
                "type": "alarm",
                "alarm_activity": item,
            }
        )

    signals.sort(
        key=lambda item: (
            str(item.get("id", ""))
        )
    )

    return _limit(
        signals,
        MAX_SIGNALS,
    )


# =============================================================================
# EVENT CATALOG
# =============================================================================

def _build_event_catalog(
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Keep each unique event definition exactly once.
    """

    catalog = analysis.get(
        "event_catalog",
        [],
    )

    result: List[
        Dict[str, Any]
    ] = []

    for event in catalog:

        result.append(
            {
                "id": event.get("id"),
                "node": event.get("node"),
                "type": event.get("type"),
                "pattern": event.get("pattern"),
            }
        )

    result.sort(
        key=lambda item: str(
            item.get("id", "")
        )
    )

    return _limit(
        result,
        MAX_EVENTS,
    )


# =============================================================================
# EVENT GROUPS
# =============================================================================

def _build_event_groups(
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Aggregate event occurrences.

    No individual event records are copied.
    """

    groups = analysis.get(
        "event_groups",
        [],
    )

    result: List[
        Dict[str, Any]
    ] = []

    for group in groups:

        result.append(
            {
                "id": group.get("id"),
                "event_id": group.get(
                    "event_id"
                ),
                "node": group.get("node"),
                "type": group.get("type"),
                "pattern": group.get(
                    "pattern"
                ),
                "occurrences": group.get(
                    "occurrences",
                    0,
                ),
                "first_seen": group.get(
                    "first_seen"
                ),
                "last_seen": group.get(
                    "last_seen"
                ),
                "duration": group.get(
                    "duration"
                ),
            }
        )

    result.sort(
        key=lambda item: (
            -float(
                item.get(
                    "occurrences",
                    0,
                )
                or 0
            ),
            str(
                item.get(
                    "id",
                    "",
                )
            ),
        )
    )

    return _limit(
        result,
        MAX_EVENT_GROUPS,
    )


# =============================================================================
# OCCURRENCE ANALYSIS
# =============================================================================

def _build_occurrence_analysis(
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Keep aggregated occurrence statistics only.

    The individual observations that produced these statistics are
    deliberately discarded.
    """

    occurrences = analysis.get(
        "occurrence_analysis",
        [],
    )

    result = [
        dict(item)
        for item in occurrences
        if isinstance(item, dict)
    ]

    result.sort(
        key=lambda item: (
            -float(
                item.get(
                    "count",
                    0,
                )
                or 0
            ),
            str(
                item.get(
                    "group_id",
                    "",
                )
            ),
        )
    )

    return _limit(
        result,
        MAX_EVENT_GROUPS,
    )


# =============================================================================
# STATE TRANSITIONS
# =============================================================================

def _build_state_transitions(
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Include aggregated state transitions, not raw transition records.
    """

    transitions = analysis.get(
        "state_transitions",
        [],
    )

    result = [
        {
            "node": item.get("node"),
            "from": item.get("from"),
            "to": item.get("to"),
            "count": item.get("count"),
            "first_seen": item.get(
                "first_seen"
            ),
            "last_seen": item.get(
                "last_seen"
            ),
        }
        for item in transitions
        if isinstance(item, dict)
    ]

    result.sort(
        key=lambda item: (
            -float(
                item.get(
                    "count",
                    0,
                )
                or 0
            ),
            str(
                item.get(
                    "node",
                    "",
                )
            ),
        )
    )

    return _limit(
        result,
        MAX_EVENT_GROUPS,
    )


# =============================================================================
# SIGNIFICANT CHANGES
# =============================================================================

def _build_significant_changes(
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Include only the most significant deterministic changes.

    These are evidence of change, not automatically anomalies.
    """

    changes = analysis.get(
        "significant_changes",
        [],
    )

    result = [
        {
            "node": item.get("node"),
            "time_s": item.get("time_s"),
            "previous": item.get(
                "previous"
            ),
            "current": item.get(
                "current"
            ),
            "delta": item.get(
                "delta"
            ),
            "absolute_delta": item.get(
                "absolute_delta"
            ),
        }
        for item in changes
        if isinstance(item, dict)
    ]

    result.sort(
        key=lambda item: (
            -float(
                item.get(
                    "absolute_delta",
                    0,
                )
                or 0
            ),
            str(
                item.get(
                    "node",
                    "",
                )
            ),
        )
    )

    return _limit(
        result,
        MAX_CHANGES,
    )


# =============================================================================
# RELATIONSHIPS
# =============================================================================

def _build_relationships(
    causal: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Build the compact relationship graph."""

    if not causal:
        return []

    relationships = causal.get(
        "relationships",
        [],
    )

    result = [
        dict(item)
        for item in relationships
        if isinstance(item, dict)
    ]

    return _limit(
        result,
        MAX_RELATIONSHIPS,
    )


# =============================================================================
# CANDIDATE CAUSES
# =============================================================================

def _build_candidate_causes(
    causal: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Include deterministic upstream candidates."""

    if not causal:
        return []

    candidates = causal.get(
        "candidate_causes",
        [],
    )

    result = [
        dict(item)
        for item in candidates
        if isinstance(item, dict)
    ]

    return _limit(
        result,
        MAX_CANDIDATE_CAUSES,
    )


# =============================================================================
# AFFECTED ENTITIES
# =============================================================================

def _build_affected_entities(
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build entity summaries from compact statistics.

    No raw records are included.
    """

    entity_counts: Dict[
        str,
        Dict[str, Any],
    ] = defaultdict(
        lambda: {
            "event_count": 0,
            "event_groups": 0,
            "alarm_occurrences": 0,
        }
    )

    for group in analysis.get(
        "event_groups",
        [],
    ):

        node = str(
            group.get("node", "")
        )

        if not node:
            continue

        entity_counts[node][
            "event_count"
        ] += int(
            group.get(
                "occurrences",
                0,
            )
            or 0
        )

        entity_counts[node][
            "event_groups"
        ] += 1

    for alarm in analysis.get(
        "alarm_stats",
        [],
    ):

        node = str(
            alarm.get("node", "")
        )

        if not node:
            continue

        entity_counts[node][
            "alarm_occurrences"
        ] += int(
            alarm.get(
                "occurrences",
                0,
            )
            or 0
        )

    result: List[
        Dict[str, Any]
    ] = []

    for node, stats in entity_counts.items():

        result.append(
            {
                "id": node,
                "type": "signal",
                **stats,
            }
        )

    result.sort(
        key=lambda item: (
            -int(
                item.get(
                    "event_count",
                    0,
                )
            ),
            str(
                item.get(
                    "id",
                    "",
                )
            ),
        )
    )

    return _limit(
        result,
        MAX_AFFECTED_ENTITIES,
    )


# =============================================================================
# COMPACT EVIDENCE ITEMS
# =============================================================================

def _build_evidence_items(
    analysis: Dict[str, Any],
    event_groups: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    changes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Construct a small collection of high-value evidence items.

    Each item should communicate one analytical fact.

    Repetition is avoided by referencing IDs.
    """

    evidence: List[
        Dict[str, Any]
    ] = []

    # -------------------------------------------------------------------------
    # Event-group evidence.
    # -------------------------------------------------------------------------

    for group in event_groups:

        occurrences = int(
            group.get(
                "occurrences",
                0,
            )
            or 0
        )

        if occurrences <= 0:
            continue

        evidence.append(
            {
                "type": "EVENT_GROUP",
                "group_id": group.get(
                    "id"
                ),
                "event_id": group.get(
                    "event_id"
                ),
                "node": group.get(
                    "node"
                ),
                "pattern": group.get(
                    "pattern"
                ),
                "occurrences": occurrences,
                "first_seen": group.get(
                    "first_seen"
                ),
                "last_seen": group.get(
                    "last_seen"
                ),
            }
        )

    # -------------------------------------------------------------------------
    # Relationship evidence.
    # -------------------------------------------------------------------------

    for relationship in relationships:

        relationship_type = (
            relationship.get(
                "relationship"
            )
        )

        if relationship_type not in {
            "TEMPORAL_PRECEDENCE",
            "CO_OCCURRENCE",
        }:
            continue

        item = {
            "type": "RELATIONSHIP",
            "source": relationship.get(
                "source"
            ),
            "target": relationship.get(
                "target"
            ),
            "relationship": (
                relationship_type
            ),
        }

        if "lag" in relationship:
            item["lag"] = relationship[
                "lag"
            ]

        if "overlap_duration" in relationship:
            item[
                "overlap_duration"
            ] = relationship[
                "overlap_duration"
            ]

        evidence.append(item)

    # -------------------------------------------------------------------------
    # Significant changes.
    # -------------------------------------------------------------------------

    for change in changes:

        evidence.append(
            {
                "type": "SIGNAL_CHANGE",
                "node": change.get(
                    "node"
                ),
                "time_s": change.get(
                    "time_s"
                ),
                "previous": change.get(
                    "previous"
                ),
                "current": change.get(
                    "current"
                ),
                "delta": change.get(
                    "delta"
                ),
            }
        )

    return _limit(
        evidence,
        MAX_EVIDENCE,
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def build_analysis_evidence(
    *,
    records: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    causal: Dict[str, Any] | None = None,
    case: Dict[str, Any] | None = None,
    ground_truth_available: bool = False,
    data_quality: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build the compact AnalysisEvidence object.

    Parameters
    ----------
    records:
        Normalized/resampled records. Used ONLY for determining the
        analysis time window. Raw records are never copied into the
        resulting evidence.

    analysis:
        Output from pipeline.analysis.compute_statistics().

    causal:
        Output from pipeline.causal.compute_causal_relationships().

    case:
        Case metadata.

    ground_truth_available:
        Whether ground-truth information exists.

    data_quality:
        Optional preprocessing quality information.
    """

    event_catalog = _build_event_catalog(
        analysis
    )

    event_groups = _build_event_groups(
        analysis
    )

    occurrence_analysis = (
        _build_occurrence_analysis(
            analysis
        )
    )

    state_transitions = (
        _build_state_transitions(
            analysis
        )
    )

    significant_changes = (
        _build_significant_changes(
            analysis
        )
    )

    relationships = _build_relationships(
        causal
    )

    candidate_causes = (
        _build_candidate_causes(
            causal
        )
    )

    affected_entities = (
        _build_affected_entities(
            analysis
        )
    )

    evidence_items = (
        _build_evidence_items(
            analysis,
            event_groups,
            relationships,
            significant_changes,
        )
    )

    time_window = _build_time_window(
        records
    )

    compact_analysis: Dict[
        str,
        Any,
    ] = {
        "record_count": analysis.get(
            "total_records",
            len(records),
        ),
        "unique_nodes": len(
            analysis.get(
                "unique_nodes",
                [],
            )
        ),
        "event_definitions": len(
            event_catalog
        ),
        "event_groups": len(
            event_groups
        ),
        "relationship_count": len(
            relationships
        ),
        "candidate_cause_count": len(
            candidate_causes
        ),
        "significant_change_count": len(
            significant_changes
        ),
    }

    evidence: Dict[str, Any] = {
        "case": _build_case(
            case
        ),

        "time_window": time_window,

        "observations": {
            "record_count": analysis.get(
                "total_records",
                len(records),
            ),
            "unique_nodes": analysis.get(
                "unique_nodes",
                [],
            ),
            "analysis": compact_analysis,
        },

        # ------------------------------------------------------------------
        # Compact analytical representation.
        # ------------------------------------------------------------------

        "signals": _build_signals(
            analysis
        ),

        "alarms": _limit(
            [
                dict(item)
                for item in analysis.get(
                    "alarm_stats",
                    [],
                )
                if isinstance(item, dict)
            ],
            MAX_SIGNALS,
        ),

        "events": event_groups,

        "relationships": relationships,

        "affected_entities": affected_entities,

        "timeline": _limit(
            [
                {
                    "group_id": group.get(
                        "id"
                    ),
                    "first_seen": group.get(
                        "first_seen"
                    ),
                    "last_seen": group.get(
                        "last_seen"
                    ),
                    "occurrences": group.get(
                        "occurrences"
                    ),
                }
                for group in event_groups
            ],
            MAX_EVENT_GROUPS,
        ),

        "evidence": evidence_items,

        # ------------------------------------------------------------------
        # Additional compact RCA context.
        # ------------------------------------------------------------------

        "event_catalog": event_catalog,

        "occurrence_analysis": (
            occurrence_analysis
        ),

        "state_transitions": (
            state_transitions
        ),

        "significant_changes": (
            significant_changes
        ),

        "candidate_causes": candidate_causes,

        "trends": {
            "time_window": time_window,
            "event_groups": len(
                event_groups
            ),
        },

        "numeric_statistics": {
            str(node): dict(stats)
            for node, stats in analysis.get(
                "numeric_stats",
                {},
            ).items()
        },

        "categorical_statistics": {},

        "anomalies": [],

        "baseline_comparisons": [],

        "severity_counts": {},

        "ground_truth_available": (
            bool(
                ground_truth_available
            )
        ),

        "data_quality": (
            data_quality
            if data_quality is not None
            else {}
        ),
    }

    return _clean(
        evidence
    )


# =============================================================================
# COMPATIBILITY ALIAS
# =============================================================================

def build_evidence(
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compatibility wrapper.

    Existing code can continue calling build_evidence().
    """

    return build_analysis_evidence(
        *args,
        **kwargs,
    )


__all__ = [
    "build_analysis_evidence",
    "build_evidence",
]