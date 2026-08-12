from __future__ import annotations

import math
from collections import defaultdict

from .config import RESAMPLE_INTERVAL


def _make_grid(
    start: float,
    end: float,
) -> list[float]:

    if end < start:
        return []

    count = int(
        math.floor(
            (end - start)
            / RESAMPLE_INTERVAL
            + 1e-9
        )
    )

    return [
        round(
            start + i * RESAMPLE_INTERVAL,
            9,
        )
        for i in range(count + 1)
    ]


def resample_records(
    records: list[dict],
) -> list[dict]:

    if not records:
        return []

    grouped = defaultdict(list)

    for record in records:

        grouped[
            record["node"]
        ].append(record)

    output = []

    for node, node_records in grouped.items():

        node_records.sort(
            key=lambda x: x["time_s"]
        )

        start = float(
            node_records[0]["time_s"]
        )

        end = float(
            node_records[-1]["time_s"]
        )

        timeline = _make_grid(
            start,
            end,
        )

        cursor = 0
        current_value = node_records[0]["value"]

        variable_type = node_records[0]["type"]

        for timestamp in timeline:

            while (
                cursor < len(node_records)
                and node_records[cursor]["time_s"]
                <= timestamp + 1e-9
            ):

                current_value = (
                    node_records[cursor]["value"]
                )

                cursor += 1

            output.append(
                {
                    "time_s": timestamp,
                    "node": node,
                    "value": current_value,
                    "type": variable_type,
                }
            )

    output.sort(
        key=lambda x: (
            x["time_s"],
            x["node"],
        )
    )

    return output
