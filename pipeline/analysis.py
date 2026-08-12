from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of temporal buckets used for trend generation.
# This is deliberately independent of the frontend schema.
TREND_BUCKETS = 12

# Minimum relative deviation before a numeric signal is considered
# potentially anomalous.
LOW_DEVIATION_THRESHOLD = 0.10
MEDIUM_DEVIATION_THRESHOLD = 0.25
HIGH_DEVIATION_THRESHOLD = 0.50

# Minimum number of observations required before using statistics
# for anomaly classification.
MIN_BASELINE_SAMPLES = 3


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(values)


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _percentage(
    value: float,
    total: float,
) -> float:
    if total == 0:
        return 0.0
    return (value / total) * 100.0


def group_by_node(
    records: list[dict],
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        grouped[record["node"]].append(record)

    for values in grouped.values():
        values.sort(key=lambda x: x["time_s"])

    return grouped


# ---------------------------------------------------------------------------
# Numeric statistics
# ---------------------------------------------------------------------------

def _numeric_statistics(
    values: list[float],
) -> dict:

    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "first": None,
            "last": None,
            "change_count": 0,
        }

    changes = sum(
        1
        for previous, current in zip(
            values,
            values[1:],
        )
        if previous != current
    )

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": _safe_mean(values),
        "median": statistics.median(values),
        "stdev": _safe_stdev(values),
        "first": values[0],
        "last": values[-1],
        "change_count": changes,
    }


def _categorical_statistics(
    values: list[str],
) -> dict:

    if not values:
        return {
            "count": 0,
            "unique_values": 0,
            "distribution": [],
            "first": None,
            "last": None,
            "change_count": 0,
        }

    counts = Counter(values)

    return {
        "count": len(values),
        "unique_values": len(counts),
        "distribution": [
            {
                "value": value,
                "count": count,
                "percentage": _percentage(
                    count,
                    len(values),
                ),
            }
            for value, count in counts.most_common()
        ],
        "first": values[0],
        "last": values[-1],
        "change_count": sum(
            1
            for previous, current in zip(
                values,
                values[1:],
            )
            if previous != current
        ),
    }


# ---------------------------------------------------------------------------
# Overall dataset statistics
# ---------------------------------------------------------------------------

def compute_statistics(
    records: list[dict],
) -> dict:

    if not records:
        return {
            "total_records": 0,
            "unique_nodes": [],
            "time_span": {
                "start": 0.0,
                "end": 0.0,
            },
            "numeric": {},
            "categorical": {},
            "alarms": [],
            "trends": {
                "event_rate": [],
                "anomaly_rate": [],
            },
            "anomalies": [],
            "affected_entities": [],
        }

    grouped = group_by_node(records)

    timestamps = [
        float(record["time_s"])
        for record in records
        if _finite(record.get("time_s"))
    ]

    if not timestamps:
        raise ValueError(
            "No valid timestamps found in records."
        )

    start = min(timestamps)
    end = max(timestamps)

    numeric: dict = {}
    categorical: dict = {}
    alarms: list[dict] = []

    # ---------------------------------------------------------------
    # Per-variable statistics
    # ---------------------------------------------------------------

    for node, node_records in grouped.items():

        variable_type = node_records[0]["type"]

        if variable_type in {
            "Binary",
            "Counter",
            "Continuous",
        }:

            values = [
                float(record["value"])
                for record in node_records
                if _finite(record.get("value"))
            ]

            if values:
                numeric[node] = {
                    "type": variable_type,
                    **_numeric_statistics(values),
                }

        elif variable_type == "Categorical":

            values = [
                str(record["value"])
                for record in node_records
            ]

            if values:
                categorical[node] = {
                    "type": variable_type,
                    **_categorical_statistics(values),
                }

        # Alarm processing is based on the actual Alarm type.
        if variable_type == "Alarm":

            alarm_events = [
                record
                for record in node_records
                if record["value"] == 1
            ]

            if alarm_events:

                alarm_times = [
                    float(record["time_s"])
                    for record in alarm_events
                ]

                alarms.append(
                    {
                        "node": node,
                        "count": len(alarm_events),
                        "first": min(alarm_times),
                        "last": max(alarm_times),
                        "frequency": (
                            len(alarm_events)
                            / max(
                                max(alarm_times)
                                - min(alarm_times),
                                1e-9,
                            )
                        ),
                    }
                )

    # ---------------------------------------------------------------
    # Temporal trends
    # ---------------------------------------------------------------

    trends = compute_temporal_trends(
        records,
        start,
        end,
    )

    # ---------------------------------------------------------------
    # Deterministic anomaly detection
    # ---------------------------------------------------------------

    anomalies = detect_anomalies(
        records,
        start,
        end,
    )

    # ---------------------------------------------------------------
    # Affected entities
    # ---------------------------------------------------------------

    affected_entities = build_affected_entities(
        records,
        anomalies,
    )

    return {
        "total_records": len(records),

        "unique_nodes": sorted(
            grouped.keys()
        ),

        "time_span": {
            "start": start,
            "end": end,
        },

        "numeric": numeric,

        "categorical": categorical,

        "alarms": alarms,

        "trends": trends,

        "anomalies": anomalies,

        "affected_entities": affected_entities,
    }


# ---------------------------------------------------------------------------
# Temporal trends
# ---------------------------------------------------------------------------

def compute_temporal_trends(
    records: list[dict],
    start: float,
    end: float,
) -> dict:

    duration = end - start

    if duration <= 0:
        return {
            "event_rate": [
                {
                    "timestamp": start,
                    "value": len(records),
                }
            ],
            "anomaly_rate": [
                {
                    "timestamp": start,
                    "value": 0.0,
                }
            ],
        }

    bucket_count = min(
        TREND_BUCKETS,
        max(1, len(records)),
    )

    bucket_width = duration / bucket_count

    buckets = [
        {
            "start": start + i * bucket_width,
            "end": (
                start + (i + 1) * bucket_width
                if i < bucket_count - 1
                else end
            ),
            "records": [],
        }
        for i in range(bucket_count)
    ]

    for record in records:

        timestamp = float(
            record["time_s"]
        )

        if timestamp >= end:
            index = bucket_count - 1
        else:
            index = int(
                (timestamp - start)
                / bucket_width
            )

            index = max(
                0,
                min(
                    index,
                    bucket_count - 1,
                ),
            )

        buckets[index]["records"].append(
            record
        )

    event_rate = []
    anomaly_rate = []

    for bucket in buckets:

        bucket_duration = max(
            bucket["end"] - bucket["start"],
            1e-9,
        )

        count = len(
            bucket["records"]
        )

        event_rate_value = (
            count / bucket_duration
        )

        anomalous = 0

        for record in bucket["records"]:

            # Deterministic anomaly indicator:
            # only records associated with an actual alarm
            # are counted here. Signal deviations are handled
            # separately by detect_anomalies().
            if (
                record["type"] == "Alarm"
                and record["value"] == 1
            ):
                anomalous += 1

        anomaly_percentage = _percentage(
            anomalous,
            count,
        )

        event_rate.append(
            {
                "timestamp": bucket["start"],
                "value": round(
                    event_rate_value,
                    6,
                ),
            }
        )

        anomaly_rate.append(
            {
                "timestamp": bucket["start"],
                "value": round(
                    anomaly_percentage,
                    6,
                ),
            }
        )

    return {
        "event_rate": event_rate,
        "anomaly_rate": anomaly_rate,
    }


# ---------------------------------------------------------------------------
# Baseline calculation
# ---------------------------------------------------------------------------

def compute_signal_baseline(
    records: list[dict],
    node: str,
    incident_start: float | None = None,
) -> dict | None:

    values = [
        float(record["value"])
        for record in records
        if (
            record["node"] == node
            and record["type"]
            in {
                "Binary",
                "Counter",
                "Continuous",
            }
            and _finite(record.get("value"))
            and (
                incident_start is None
                or float(record["time_s"])
                < incident_start
            )
        )
    ]

    if len(values) < MIN_BASELINE_SAMPLES:
        return None

    mean = _safe_mean(values)
    stdev = _safe_stdev(values)

    return {
        "count": len(values),
        "mean": mean,
        "stdev": stdev,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


# ---------------------------------------------------------------------------
# Deterministic anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    records: list[dict],
    start: float,
    end: float,
) -> list[dict]:

    grouped = group_by_node(records)

    anomalies: list[dict] = []

    # ---------------------------------------------------------------
    # Alarm anomalies
    # ---------------------------------------------------------------

    for node, node_records in grouped.items():

        if node_records[0]["type"] != "Alarm":
            continue

        active = [
            record
            for record in node_records
            if record["value"] == 1
        ]

        if not active:
            continue

        timestamps = [
            float(record["time_s"])
            for record in active
        ]

        first = min(timestamps)
        last = max(timestamps)

        duration = max(
            last - first,
            1e-9,
        )

        frequency = len(active) / duration

        anomalies.append(
            {
                "id": (
                    f"ALARM-{len(anomalies) + 1:03d}"
                ),
                "severity": (
                    "HIGH"
                    if len(active) >= 10
                    else "MEDIUM"
                ),
                "type": "ALARM_CLUSTER",
                "title": (
                    f"Repeated activity in alarm "
                    f"signal {node}"
                ),
                "description": (
                    f"Alarm signal {node} was active "
                    f"{len(active)} times during the "
                    f"analysed window."
                ),
                "first_detected": first,
                "last_detected": last,
                "occurrences": len(active),
                "affected_entities": [node],
                "evidence": [
                    {
                        "evidence_id": (
                            f"EVID-{len(anomalies) + 1:04d}"
                        ),
                        "metric": "alarm_occurrences",
                        "value": len(active),
                        "baseline": 0,
                        "unit": "events",
                    },
                    {
                        "evidence_id": (
                            f"EVID-{len(anomalies) + 1:04d}-FREQ"
                        ),
                        "metric": "alarm_frequency",
                        "value": frequency,
                        "baseline": 0,
                        "unit": "events_per_second",
                    },
                ],
                "possible_causes": [],
                "recommendations": [],
                "confidence": 1.0,
                "detection_basis": (
                    "deterministic_alarm_activity"
                ),
            }
        )

    # ---------------------------------------------------------------
    # Numeric signal anomalies
    # ---------------------------------------------------------------

    for node, node_records in grouped.items():

        if node_records[0]["type"] not in {
            "Counter",
            "Continuous",
        }:
            continue

        values = [
            float(record["value"])
            for record in node_records
            if _finite(record.get("value"))
        ]

        if len(values) < MIN_BASELINE_SAMPLES:
            continue

        # Use the first half as a baseline candidate.
        # This avoids using the entire incident distribution
        # as its own baseline.
        midpoint = len(node_records) // 2

        baseline_records = node_records[:midpoint]

        baseline_values = [
            float(record["value"])
            for record in baseline_records
            if _finite(record.get("value"))
        ]

        if len(baseline_values) < MIN_BASELINE_SAMPLES:
            continue

        incident_values = [
            float(record["value"])
            for record in node_records[midpoint:]
            if _finite(record.get("value"))
        ]

        if not incident_values:
            continue

        baseline_mean = _safe_mean(
            baseline_values
        )

        incident_mean = _safe_mean(
            incident_values
        )

        absolute_deviation = (
            incident_mean - baseline_mean
        )

        # If baseline is near zero, relative deviation becomes
        # mathematically unstable. Use absolute deviation relative
        # to the baseline standard deviation instead.
        baseline_stdev = _safe_stdev(
            baseline_values
        )

        if abs(baseline_mean) > 1e-12:

            relative_deviation = (
                absolute_deviation
                / abs(baseline_mean)
            )

        elif baseline_stdev > 1e-12:

            relative_deviation = (
                absolute_deviation
                / baseline_stdev
            )

        else:

            relative_deviation = (
                0.0
                if abs(absolute_deviation) < 1e-12
                else float("inf")
            )

        abs_relative = abs(
            relative_deviation
        )

        if abs_relative < LOW_DEVIATION_THRESHOLD:
            continue

        if abs_relative >= HIGH_DEVIATION_THRESHOLD:
            severity = "HIGH"
        elif abs_relative >= MEDIUM_DEVIATION_THRESHOLD:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        timestamps = [
            float(record["time_s"])
            for record in node_records[midpoint:]
        ]

        anomaly_id = (
            f"SIG-{len(anomalies) + 1:03d}"
        )

        evidence_id = (
            f"EVID-SIG-{len(anomalies) + 1:04d}"
        )

        anomalies.append(
            {
                "id": anomaly_id,
                "severity": severity,
                "type": "SIGNAL_DEVIATION",
                "title": (
                    f"Deviation detected in signal {node}"
                ),
                "description": (
                    f"Signal {node} changed from a "
                    f"baseline mean of "
                    f"{baseline_mean:.6g} to an "
                    f"incident mean of "
                    f"{incident_mean:.6g}."
                ),
                "first_detected": min(timestamps),
                "last_detected": max(timestamps),
                "occurrences": len(incident_values),
                "affected_entities": [node],
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "metric": "baseline_mean",
                        "value": baseline_mean,
                        "baseline": baseline_mean,
                        "unit": "dataset_units",
                    },
                    {
                        "evidence_id": (
                            f"{evidence_id}-INCIDENT"
                        ),
                        "metric": "incident_mean",
                        "value": incident_mean,
                        "baseline": baseline_mean,
                        "unit": "dataset_units",
                    },
                    {
                        "evidence_id": (
                            f"{evidence_id}-DEV"
                        ),
                        "metric": "relative_deviation",
                        "value": (
                            relative_deviation
                            if math.isfinite(
                                relative_deviation
                            )
                            else None
                        ),
                        "baseline": 0.0,
                        "unit": "ratio",
                    },
                    {
                        "evidence_id": (
                            f"{evidence_id}-STD"
                        ),
                        "metric": "baseline_stdev",
                        "value": baseline_stdev,
                        "baseline": 0.0,
                        "unit": "dataset_units",
                    },
                ],
                "possible_causes": [],
                "recommendations": [],
                "confidence": min(
                    1.0,
                    0.5
                    + min(
                        abs_relative,
                        1.0,
                    )
                    * 0.5,
                ),
                "detection_basis": (
                    "deterministic_baseline_deviation"
                ),
            }
        )

    return anomalies


# ---------------------------------------------------------------------------
# Affected entities
# ---------------------------------------------------------------------------

def build_affected_entities(
    records: list[dict],
    anomalies: list[dict],
) -> list[dict]:

    grouped = group_by_node(records)

    anomaly_counts = Counter()

    for anomaly in anomalies:

        for entity in anomaly[
            "affected_entities"
        ]:

            anomaly_counts[entity] += 1

    affected = []

    for node in sorted(
        anomaly_counts.keys()
    ):

        node_records = grouped.get(
            node,
            [],
        )

        anomaly_count = anomaly_counts[
            node
        ]

        if anomaly_count >= 1:
            status = "WARNING"
        else:
            status = "NORMAL"

        affected.append(
            {
                "id": node,
                "type": "signal",
                "event_count": len(
                    node_records
                ),
                "anomaly_count": anomaly_count,
                "status": status,
            }
        )

    return affected


# ---------------------------------------------------------------------------
# Window comparison
# ---------------------------------------------------------------------------

def compare_windows(
    records: list[dict],
    incident_start: float,
    incident_end: float,
) -> list[dict]:

    before: dict[str, list[float]] = defaultdict(list)
    during: dict[str, list[float]] = defaultdict(list)

    for record in records:

        if record["type"] not in {
            "Binary",
            "Counter",
            "Continuous",
        }:
            continue

        if not _finite(record.get("value")):
            continue

        timestamp = float(
            record["time_s"]
        )

        node = record["node"]
        value = float(record["value"])

        if timestamp < incident_start:

            before[node].append(value)

        elif (
            incident_start
            <= timestamp
            <= incident_end
        ):

            during[node].append(value)

    comparisons = []

    for node in sorted(
        set(before) & set(during)
    ):

        if (
            len(before[node])
            < MIN_BASELINE_SAMPLES
        ):
            continue

        if not during[node]:
            continue

        baseline = _safe_mean(
            before[node]
        )

        incident = _safe_mean(
            during[node]
        )

        absolute = incident - baseline

        if abs(baseline) > 1e-12:

            relative = (
                absolute
                / abs(baseline)
            )

        else:

            relative = None

        comparisons.append(
            {
                "node": node,
                "baseline_mean": baseline,
                "incident_mean": incident,
                "absolute_deviation": absolute,
                "relative_deviation": relative,
                "baseline_count": len(
                    before[node]
                ),
                "incident_count": len(
                    during[node]
                ),
            }
        )

    return comparisons
