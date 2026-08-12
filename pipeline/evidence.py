from __future__ import annotations

from collections import defaultdict
from typing import Any

from .analysis import compare_windows


# =============================================================================
# SIGNAL INVENTORY
# =============================================================================

def _build_signal_inventory(
    statistics: dict,
) -> list[dict]:

    inventory: list[dict] = []

    for node, stats in statistics.get(
        "numeric",
        {},
    ).items():

        inventory.append(
            {
                "id": node,
                "type": "signal",
                "variable_type": stats.get(
                    "type"
                ),
                "statistics": stats,
                "source": (
                    "deterministic_python_analysis"
                ),
            }
        )

    for node, stats in statistics.get(
        "categorical",
        {},
    ).items():

        inventory.append(
            {
                "id": node,
                "type": "signal",
                "variable_type": stats.get(
                    "type"
                ),
                "statistics": stats,
                "source": (
                    "deterministic_python_analysis"
                ),
            }
        )

    return inventory


# =============================================================================
# ALARM EVIDENCE
# =============================================================================

def _build_alarm_evidence(
    statistics: dict,
) -> list[dict]:

    evidence: list[dict] = []

    for index, alarm in enumerate(
        statistics.get("alarms", []),
        start=1,
    ):

        evidence.append(
            {
                "event_id": (
                    f"ALARM-EVENT-{index:04d}"
                ),
                "timestamp": alarm.get(
                    "first"
                ),
                "node": alarm.get(
                    "node"
                ),
                "type": "Alarm",
                "value": 1,
                "occurrences": alarm.get(
                    "count",
                    0,
                ),
                "first_detected": alarm.get(
                    "first"
                ),
                "last_detected": alarm.get(
                    "last"
                ),
                "frequency": alarm.get(
                    "frequency",
                    0.0,
                ),
                "source": (
                    "deterministic_python_analysis"
                ),
            }
        )

    return evidence


# =============================================================================
# SIGNIFICANT EVENTS
# =============================================================================

def _build_events(
    records: list[dict],
    anomalies: list[dict],
) -> list[dict]:

    """
    Build the event array expected by AnalysisEvidence.

    We deliberately do NOT copy every resampled observation into this array.

    Events are significant observations that are useful for RCA:
      - alarm activations
      - observations belonging to deterministic anomaly windows
      - signal changes associated with detected anomalies

    The complete raw/resampled data remains outside the NIM contract.
    """

    events: list[dict] = []

    # -------------------------------------------------------------------------
    # Index anomaly windows by affected entity.
    # -------------------------------------------------------------------------

    anomaly_windows: dict[
        str,
        list[tuple[float, float, str]],
    ] = defaultdict(list)

    for anomaly in anomalies:

        start = anomaly.get(
            "first_detected"
        )

        end = anomaly.get(
            "last_detected"
        )

        if start is None or end is None:
            continue

        for entity in anomaly.get(
            "affected_entities",
            [],
        ):

            anomaly_windows[
                entity
            ].append(
                (
                    float(start),
                    float(end),
                    anomaly["id"],
                )
            )

    # -------------------------------------------------------------------------
    # Extract meaningful records.
    # -------------------------------------------------------------------------

    for record in records:

        node = record.get(
            "node"
        )

        timestamp = record.get(
            "time_s"
        )

        value = record.get(
            "value"
        )

        variable_type = record.get(
            "type"
        )

        if node is None or timestamp is None:
            continue

        timestamp = float(timestamp)

        related_anomalies: list[str] = []

        for (
            start,
            end,
            anomaly_id,
        ) in anomaly_windows.get(
            node,
            [],
        ):

            if start <= timestamp <= end:

                related_anomalies.append(
                    anomaly_id
                )

        # Alarm activations are always significant.
        is_alarm_event = (
            variable_type == "Alarm"
            and value == 1
        )

        # Non-alarm records are included only when they fall
        # inside a deterministic anomaly window.
        if (
            not is_alarm_event
            and not related_anomalies
        ):
            continue

        events.append(
            {
                "event_id": (
                    f"EVENT-{len(events) + 1:06d}"
                ),
                "timestamp": timestamp,
                "node": node,
                "type": variable_type,
                "value": value,
                "related_anomalies": (
                    related_anomalies
                ),
                "source": (
                    "deterministic_python_analysis"
                ),
            }
        )

    return events


# =============================================================================
# RELATIONSHIPS
# =============================================================================

def _build_relationships(
    anomalies: list[dict],
) -> list[dict]:

    """
    Build temporal relationships between detected anomalies.

    These relationships indicate temporal overlap only.
    They do NOT assert causality.
    """

    relationships: list[dict] = []

    for index, left in enumerate(
        anomalies
    ):

        left_start = left.get(
            "first_detected"
        )

        left_end = left.get(
            "last_detected"
        )

        if (
            left_start is None
            or left_end is None
        ):
            continue

        for right in anomalies[
            index + 1:
        ]:

            right_start = right.get(
                "first_detected"
            )

            right_end = right.get(
                "last_detected"
            )

            if (
                right_start is None
                or right_end is None
            ):
                continue

            overlap_start = max(
                left_start,
                right_start,
            )

            overlap_end = min(
                left_end,
                right_end,
            )

            if overlap_start <= overlap_end:

                relationships.append(
                    {
                        "relationship_id": (
                            f"REL-{len(relationships) + 1:04d}"
                        ),
                        "type": (
                            "temporal_overlap"
                        ),
                        "source": left["id"],
                        "target": right["id"],
                        "overlap_start": (
                            overlap_start
                        ),
                        "overlap_end": (
                            overlap_end
                        ),
                        "supports_causation": False,
                        "source_basis": (
                            "deterministic_temporal_analysis"
                        ),
                    }
                )

    return relationships


# =============================================================================
# TIMELINE
# =============================================================================

def _build_timeline(
    anomalies: list[dict],
) -> list[dict]:

    timeline: list[dict] = []

    for anomaly in anomalies:

        first = anomaly.get(
            "first_detected"
        )

        last = anomaly.get(
            "last_detected"
        )

        if first is not None:

            timeline.append(
                {
                    "timestamp": first,
                    "event_type": "anomaly_start",
                    "event_id": anomaly[
                        "id"
                    ],
                    "severity": anomaly[
                        "severity"
                    ],
                    "entity_ids": anomaly.get(
                        "affected_entities",
                        [],
                    ),
                }
            )

        if last is not None:

            timeline.append(
                {
                    "timestamp": last,
                    "event_type": "anomaly_end",
                    "event_id": anomaly[
                        "id"
                    ],
                    "severity": anomaly[
                        "severity"
                    ],
                    "entity_ids": anomaly.get(
                        "affected_entities",
                        [],
                    ),
                }
            )

    timeline.sort(
        key=lambda item: item[
            "timestamp"
        ]
    )

    return timeline


# =============================================================================
# EVIDENCE ITEMS
# =============================================================================

def _build_evidence_items(
    anomalies: list[dict],
    comparisons: list[dict],
) -> list[dict]:

    evidence: list[dict] = []

    # -------------------------------------------------------------------------
    # Evidence directly produced by deterministic anomaly detection.
    # -------------------------------------------------------------------------

    for anomaly in anomalies:

        for item in anomaly.get(
            "evidence",
            [],
        ):

            evidence.append(
                {
                    **item,
                    "anomaly_id": anomaly[
                        "id"
                    ],
                    "source": (
                        "deterministic_python_analysis"
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # Baseline comparisons.
    # -------------------------------------------------------------------------

    for index, comparison in enumerate(
        comparisons,
        start=1,
    ):

        evidence.append(
            {
                "evidence_id": (
                    f"BASELINE-EVID-{index:04d}"
                ),
                "type": "baseline_comparison",
                "entity": comparison[
                    "node"
                ],
                "baseline_mean": comparison[
                    "baseline_mean"
                ],
                "incident_mean": comparison[
                    "incident_mean"
                ],
                "absolute_deviation": comparison[
                    "absolute_deviation"
                ],
                "relative_deviation": comparison[
                    "relative_deviation"
                ],
                "baseline_count": comparison[
                    "baseline_count"
                ],
                "incident_count": comparison[
                    "incident_count"
                ],
                "source": (
                    "deterministic_python_analysis"
                ),
            }
        )

    return evidence


# =============================================================================
# AFFECTED ENTITIES
# =============================================================================

def _build_affected_entities(
    statistics: dict,
) -> list[dict]:

    """
    Use the canonical affected_entities generated by analysis.py.

    Do not independently reconstruct entity IDs here.
    """

    return [
        dict(entity)
        for entity in statistics.get(
            "affected_entities",
            [],
        )
    ]


# =============================================================================
# INCIDENT WINDOW
# =============================================================================

def _derive_incident_window(
    anomalies: list[dict],
) -> tuple[
    float,
    float,
] | None:

    starts = [
        float(
            anomaly["first_detected"]
        )
        for anomaly in anomalies
        if anomaly.get(
            "first_detected"
        ) is not None
    ]

    ends = [
        float(
            anomaly["last_detected"]
        )
        for anomaly in anomalies
        if anomaly.get(
            "last_detected"
        ) is not None
    ]

    if not starts or not ends:
        return None

    return (
        min(starts),
        max(ends),
    )


# =============================================================================
# MAIN EVIDENCE BUILDER
# =============================================================================

def build_evidence(
    *,
    records: list[dict],
    statistics: dict,
    case_manifest: dict,
    incident_window: tuple[
        float,
        float,
    ] | None = None,
) -> dict:

    """
    Construct the AnalysisEvidence object.

    This is the deterministic Python -> NIM boundary.

    IMPORTANT:
        This function does not expose ground truth.

    IMPORTANT:
        This function does not alter the final frontend schema.

    IMPORTANT:
        The top-level AnalysisEvidence structure conforms to
        pipeline/intermediate_schema.py.
    """

    anomalies = statistics.get(
        "anomalies",
        [],
    )

    # -------------------------------------------------------------------------
    # Incident window
    # -------------------------------------------------------------------------

    if incident_window is None:

        incident_window = (
            _derive_incident_window(
                anomalies
            )
        )

    # -------------------------------------------------------------------------
    # Baseline comparisons
    # -------------------------------------------------------------------------

    if incident_window is not None:

        comparisons = compare_windows(
            records,
            incident_window[0],
            incident_window[1],
        )

    else:

        comparisons = []

    # -------------------------------------------------------------------------
    # Signal inventory
    # -------------------------------------------------------------------------

    signals = _build_signal_inventory(
        statistics
    )

    # -------------------------------------------------------------------------
    # Alarm events
    # -------------------------------------------------------------------------

    alarms = _build_alarm_evidence(
        statistics
    )

    # -------------------------------------------------------------------------
    # Significant events
    # -------------------------------------------------------------------------

    events = _build_events(
        records,
        anomalies,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    relationships = _build_relationships(
        anomalies
    )

    # -------------------------------------------------------------------------
    # Timeline
    # -------------------------------------------------------------------------

    timeline = _build_timeline(
        anomalies
    )

    # -------------------------------------------------------------------------
    # Evidence items
    # -------------------------------------------------------------------------

    evidence_items = _build_evidence_items(
        anomalies,
        comparisons,
    )

    # -------------------------------------------------------------------------
    # Observations
    #
    # Keep aggregate facts here. These are internal AnalysisEvidence
    # facts and do not alter the frontend schema.
    # -------------------------------------------------------------------------

    total_records = statistics.get(
        "total_records",
        len(records),
    )

    anomaly_count = len(
        anomalies
    )

    normal_count = max(
        0,
        total_records - anomaly_count,
    )

    anomaly_rate = (
        anomaly_count
        / total_records
        * 100.0
        if total_records
        else 0.0
    )

    observations = {
        "total_records": total_records,
        "variables": len(
            statistics.get(
                "unique_nodes",
                [],
            )
        ),
        "numeric_variables": len(
            statistics.get(
                "numeric",
                {},
            )
        ),
        "categorical_variables": len(
            statistics.get(
                "categorical",
                {},
            )
        ),
        "alarm_variables": len(
            statistics.get(
                "alarms",
                [],
            )
        ),
        "normal_observations": normal_count,
        "anomalous_observations": anomaly_count,
        "anomaly_rate": anomaly_rate,
    }

    # -------------------------------------------------------------------------
    # Additional deterministic facts
    # -------------------------------------------------------------------------

    trends = statistics.get(
        "trends",
        {
            "event_rate": [],
            "anomaly_rate": [],
        },
    )

    # -------------------------------------------------------------------------
    # Final AnalysisEvidence
    # -------------------------------------------------------------------------

    evidence = {
        "case": {
            "dataset": "causRCA",
            "case_type": case_manifest.get(
                "case_type"
            ),
            "family": case_manifest.get(
                "family"
            ),
            "experiment_id": case_manifest.get(
                "experiment_id"
            ),
            "run_id": case_manifest.get(
                "run_id"
            ),
            "source_file": case_manifest.get(
                "csv_path"
            ),
        },

        "time_window": {
            "start": statistics[
                "time_span"
            ]["start"],
            "end": statistics[
                "time_span"
            ]["end"],
        },

        "observations": observations,

        "signals": signals,

        "alarms": alarms,

        # MUST remain an ARRAY because this is required by
        # intermediate_schema.py.
        "events": events,

        "relationships": relationships,

        "affected_entities": (
            _build_affected_entities(
                statistics
            )
        ),

        "timeline": timeline,

        "evidence": evidence_items,

        # These additional fields are intentionally allowed by the
        # current AnalysisEvidence validator. They provide NIM with
        # deterministic aggregate context without modifying the
        # required contract.
        "trends": trends,

        "numeric_statistics": statistics.get(
            "numeric",
            {},
        ),

        "categorical_statistics": statistics.get(
            "categorical",
            {},
        ),

        "anomalies": anomalies,

        "baseline_comparisons": comparisons,

        "severity_counts": _severity_counts(
            anomalies
        ),

        # Ground truth is deliberately unavailable to the inference model.
        "ground_truth_available": False,
    }

    return evidence


# =============================================================================
# SEVERITY COUNTS
# =============================================================================

def _severity_counts(
    anomalies: list[dict],
) -> dict[str, int]:

    counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    for anomaly in anomalies:

        severity = str(
            anomaly.get(
                "severity",
                "LOW",
            )
        ).upper()

        if severity in counts:

            counts[
                severity
            ] += 1

    return counts
