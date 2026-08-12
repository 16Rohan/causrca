# pipeline/causal.py
"""
Deterministic causal/relationship analysis for causRCA.

Purpose
-------
Convert compact analytical primitives into a small relationship graph
that can be consumed by the downstream evidence builder.

This module does NOT:
    - call an LLM
    - declare a definitive root cause
    - invent causal relationships
    - inspect raw CSV files
    - duplicate individual observations

It produces candidate relationships backed by measurable evidence.

Design
------
    event_groups
         +
    occurrence_analysis
         +
    signal statistics
         +
    state transitions
         ↓
    compact relationship graph

Relationship types currently produced:

    TEMPORAL_PRECEDENCE
    CO_OCCURRENCE
    SHARED_ENTITY
    SIGNAL_EVENT_ASSOCIATION

These are evidence relationships, not final causal conclusions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum number of relationships exposed downstream.
# This is a representation limit, not a causal threshold.
MAX_RELATIONSHIPS = 250

# Maximum number of candidate causes.
MAX_CANDIDATE_CAUSES = 100

# A small tolerance used only for timestamp comparisons.
TIME_EPSILON = 1e-9


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


def _overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    """Return whether two time intervals overlap."""

    return (
        start_a <= end_b + TIME_EPSILON
        and start_b <= end_a + TIME_EPSILON
    )


def _duration(
    item: Dict[str, Any],
) -> float:
    """Return duration represented by an event group."""

    start = _safe_float(
        item.get("first_seen")
    )
    end = _safe_float(
        item.get("last_seen")
    )

    if start is None or end is None:
        return 0.0

    return max(
        end - start,
        0.0,
    )


# =============================================================================
# TEMPORAL RELATIONSHIPS
# =============================================================================

def _temporal_relationships(
    event_groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Find event groups that occur in a consistent temporal order.

    This does NOT say A causes B.

    It records:

        A occurs before B

    so the LLM can use the ordering as RCA evidence.
    """

    relationships: List[Dict[str, Any]] = []

    groups = [
        group
        for group in event_groups
        if _safe_float(
            group.get("first_seen")
        ) is not None
    ]

    groups.sort(
        key=lambda item: (
            _safe_float(
                item.get("first_seen")
            )
            or 0.0,
            str(item.get("id", "")),
        )
    )

    for index, source in enumerate(groups):

        source_first = _safe_float(
            source.get("first_seen")
        )

        source_last = _safe_float(
            source.get("last_seen")
        )

        if (
            source_first is None
            or source_last is None
        ):
            continue

        for target in groups[index + 1:]:

            target_first = _safe_float(
                target.get("first_seen")
            )

            target_last = _safe_float(
                target.get("last_seen")
            )

            if (
                target_first is None
                or target_last is None
            ):
                continue

            # Target must begin after source.
            if (
                target_first
                <= source_last + TIME_EPSILON
            ):
                continue

            lag = (
                target_first
                - source_last
            )

            relationships.append(
                {
                    "source": source["id"],
                    "target": target["id"],
                    "relationship": (
                        "TEMPORAL_PRECEDENCE"
                    ),
                    "lag": lag,
                    "source_first_seen": (
                        source_first
                    ),
                    "source_last_seen": (
                        source_last
                    ),
                    "target_first_seen": (
                        target_first
                    ),
                    "target_last_seen": (
                        target_last
                    ),
                }
            )

    # Prefer relationships with shorter temporal gaps.
    relationships.sort(
        key=lambda item: (
            item["lag"],
            item["source"],
            item["target"],
        )
    )

    return relationships


# =============================================================================
# CO-OCCURRENCE
# =============================================================================

def _cooccurrence_relationships(
    event_groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Find event groups whose active windows overlap.

    Overlap indicates association, not causation.
    """

    relationships: List[Dict[str, Any]] = []

    groups = []

    for group in event_groups:

        start = _safe_float(
            group.get("first_seen")
        )

        end = _safe_float(
            group.get("last_seen")
        )

        if start is None or end is None:
            continue

        groups.append(
            (
                start,
                end,
                group,
            )
        )

    groups.sort(
        key=lambda item: (
            item[0],
            str(item[2].get("id", "")),
        )
    )

    for index, (
        start_a,
        end_a,
        group_a,
    ) in enumerate(groups):

        for (
            start_b,
            end_b,
            group_b,
        ) in groups[index + 1:]:

            # Once the next interval begins after A ends,
            # later intervals cannot overlap A.
            if start_b > end_a:
                break

            if not _overlap(
                start_a,
                end_a,
                start_b,
                end_b,
            ):
                continue

            overlap_start = max(
                start_a,
                start_b,
            )

            overlap_end = min(
                end_a,
                end_b,
            )

            overlap_duration = max(
                overlap_end - overlap_start,
                0.0,
            )

            relationships.append(
                {
                    "source": group_a["id"],
                    "target": group_b["id"],
                    "relationship": "CO_OCCURRENCE",
                    "overlap_duration": (
                        overlap_duration
                    ),
                }
            )

    return relationships


# =============================================================================
# SHARED ENTITY
# =============================================================================

def _shared_entity_relationships(
    event_groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Connect event groups belonging to the same physical/logical entity.

    This is useful for distinguishing:

        same signal, different behavior

    from:

        unrelated signals.
    """

    by_node: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for group in event_groups:

        node = str(
            group.get("node", "")
        )

        if node:
            by_node[node].append(
                group
            )

    relationships: List[Dict[str, Any]] = []

    for node, groups in by_node.items():

        if len(groups) < 2:
            continue

        groups.sort(
            key=lambda item: str(
                item.get("id", "")
            )
        )

        for index, source in enumerate(groups):

            for target in groups[index + 1:]:

                relationships.append(
                    {
                        "source": source["id"],
                        "target": target["id"],
                        "relationship": (
                            "SHARED_ENTITY"
                        ),
                        "entity": node,
                    }
                )

    return relationships


# =============================================================================
# SIGNAL / EVENT ASSOCIATION
# =============================================================================

def _signal_event_associations(
    event_groups: List[Dict[str, Any]],
    numeric_stats: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Associate event groups with available signal statistics.

    This is intentionally lightweight.

    The event group already identifies its node, so the relationship
    simply records that deterministic statistics exist for that entity.
    """

    relationships: List[Dict[str, Any]] = []

    for group in event_groups:

        node = str(
            group.get("node", "")
        )

        if node not in numeric_stats:
            continue

        relationships.append(
            {
                "source": group["id"],
                "target": node,
                "relationship": (
                    "SIGNAL_EVENT_ASSOCIATION"
                ),
                "statistics_available": True,
            }
        )

    return relationships


# =============================================================================
# RELATIONSHIP DEDUPLICATION
# =============================================================================

def _relationship_key(
    relationship: Dict[str, Any],
) -> tuple[Any, ...]:
    """Generate a deterministic relationship identity."""

    return (
        relationship.get("source"),
        relationship.get("target"),
        relationship.get("relationship"),
        relationship.get("entity"),
    )


def _deduplicate_relationships(
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove duplicate relationships."""

    seen: set[tuple[Any, ...]] = set()
    result: List[Dict[str, Any]] = []

    for relationship in relationships:

        key = _relationship_key(
            relationship
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(
            relationship
        )

    return result


# =============================================================================
# CANDIDATE CAUSES
# =============================================================================

def _candidate_causes(
    event_groups: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Produce conservative candidate cause structures.

    These are NOT assertions that a particular event is the root cause.

    A candidate is simply an event group that:

        1. occurs before another group, and
        2. therefore warrants causal investigation.
    """

    incoming: Dict[
        str,
        int,
    ] = defaultdict(int)

    outgoing: Dict[
        str,
        int,
    ] = defaultdict(int)

    for relationship in relationships:

        if relationship.get(
            "relationship"
        ) != "TEMPORAL_PRECEDENCE":
            continue

        source = relationship.get(
            "source"
        )

        target = relationship.get(
            "target"
        )

        if source:
            outgoing[source] += 1

        if target:
            incoming[target] += 1

    group_lookup = {
        group["id"]: group
        for group in event_groups
        if group.get("id")
    }

    candidates: List[Dict[str, Any]] = []

    for group_id, outgoing_count in (
        outgoing.items()
    ):

        group = group_lookup.get(
            group_id
        )

        if group is None:
            continue

        # Only describe the evidence.
        # Do not fabricate a probability.
        candidates.append(
            {
                "group_id": group_id,
                "node": group.get(
                    "node"
                ),
                "pattern": group.get(
                    "pattern"
                ),
                "temporal_successors": (
                    outgoing_count
                ),
                "temporal_predecessors": (
                    incoming.get(
                        group_id,
                        0,
                    )
                ),
                "role": (
                    "upstream_candidate"
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            -item[
                "temporal_successors"
            ],
            item["group_id"],
        )
    )

    return candidates[
        :MAX_CANDIDATE_CAUSES
    ]


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_causal_relationships(
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the compact deterministic relationship graph.

    Input:
        Output from pipeline.analysis.compute_statistics()

    Output:
        {
            "relationships": [...],
            "candidate_causes": [...]
        }
    """

    event_groups = list(
        analysis.get(
            "event_groups",
            [],
        )
    )

    numeric_stats = dict(
        analysis.get(
            "numeric_stats",
            {},
        )
    )

    if not event_groups:
        return {
            "relationships": [],
            "candidate_causes": [],
        }

    relationships: List[
        Dict[str, Any]
    ] = []

    relationships.extend(
        _temporal_relationships(
            event_groups
        )
    )

    relationships.extend(
        _cooccurrence_relationships(
            event_groups
        )
    )

    relationships.extend(
        _shared_entity_relationships(
            event_groups
        )
    )

    relationships.extend(
        _signal_event_associations(
            event_groups,
            numeric_stats,
        )
    )

    relationships = (
        _deduplicate_relationships(
            relationships
        )
    )

    # Keep the representation compact.
    #
    # Temporal relationships are generally the most useful for RCA,
    # followed by co-occurrence, shared-entity, and simple associations.
    priority = {
        "TEMPORAL_PRECEDENCE": 0,
        "CO_OCCURRENCE": 1,
        "SHARED_ENTITY": 2,
        "SIGNAL_EVENT_ASSOCIATION": 3,
    }

    relationships.sort(
        key=lambda item: (
            priority.get(
                item.get(
                    "relationship"
                ),
                99,
            ),
            item.get("lag", 0.0),
            str(
                item.get(
                    "source",
                    "",
                )
            ),
            str(
                item.get(
                    "target",
                    "",
                )
            ),
        )
    )

    relationships = relationships[
        :MAX_RELATIONSHIPS
    ]

    candidates = _candidate_causes(
        event_groups,
        relationships,
    )

    return {
        "relationships": relationships,
        "candidate_causes": candidates,
    }


__all__ = [
    "compute_causal_relationships",
]