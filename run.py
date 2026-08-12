from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

from nim_client import call_nim
from pipeline.analysis import compute_statistics
from pipeline.config import (
    OUTPUT_ROOT,
    PROMPT_ROOT,
    ensure_directories,
    print_config,
)
from pipeline.discover import (
    discover_cases,
    get_case,
    print_cases,
)
from pipeline.evidence import build_evidence
from pipeline.loader import load_csv
from pipeline.normalize import normalize_records
from pipeline.resample import resample_records
from pipeline.validate import validate_analysis_evidence


def save_json(
    path: Path,
    data: dict,
) -> None:
    """
    Save a dictionary as strict JSON.

    allow_nan=False prevents invalid JSON values such as
    NaN and Infinity from silently being written.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def post_to_website(
    result: dict,
) -> None:
    """
    POST the final RCA JSON to the configured website endpoint.

    This is intentionally optional. If no endpoint is configured,
    the pipeline simply keeps the locally saved RCA result.
    """

    endpoint = os.getenv(
        "WEBSITE_RCA_ENDPOINT"
    )

    if not endpoint:
        print(
            "WEBSITE_RCA_ENDPOINT not set. "
            "Skipping website POST."
        )
        return

    headers = {
        "Content-Type": "application/json",
    }

    website_token = os.getenv(
        "WEBSITE_API_KEY"
    )

    if website_token:
        headers[
            "Authorization"
        ] = f"Bearer {website_token}"

    response = requests.post(
        endpoint,
        headers=headers,
        json=result,
        timeout=30,
    )

    response.raise_for_status()

    print(
        f"Website POST successful: "
        f"{response.status_code}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run the causRCA Industrial Equipment "
            "Root-Cause Analysis pipeline."
        )
    )

    parser.add_argument(
        "--case",
        type=int,
        default=0,
        help=(
            "Index of the causRCA case to analyze. "
            "Use --list-cases to see available cases."
        ),
    )

    parser.add_argument(
        "--list-cases",
        action="store_true",
        help=(
            "List all discovered causRCA cases "
            "and exit."
        ),
    )

    parser.add_argument(
        "--no-nim",
        action="store_true",
        help=(
            "Run only the deterministic Python pipeline. "
            "Do not call NVIDIA NIM."
        ),
    )

    parser.add_argument(
        "--no-post",
        action="store_true",
        help=(
            "Do not POST the final RCA JSON "
            "to the website."
        ),
    )

    args = parser.parse_args()

    # =========================================================
    # INITIALIZATION
    # =========================================================

    ensure_directories()

    print_config()

    # =========================================================
    # 1. DISCOVER
    # =========================================================

    print(
        "\n[1/8] Discovering causRCA cases..."
    )

    cases = discover_cases()

    print(
        f"Discovered {len(cases)} cases."
    )

    if args.list_cases:
        print_cases(cases)
        return

    case = get_case(
        cases,
        args.case,
    )

    print(
        f"Selected: {case.csv_path}"
    )

    # =========================================================
    # 2. LOAD
    # =========================================================

    print(
        "\n[2/8] Loading data..."
    )

    # causRCA stores the variable type directly in each CSV row:
    #
    #     time_s,node,value,type
    #
    # Therefore there is NO dependency on an external
    # categorical_encoding.json file.

    raw_records = load_csv(
        case.csv_path
    )

    print(
        f"Loaded {len(raw_records)} raw records."
    )

    # =========================================================
    # 3. NORMALIZE
    # =========================================================

    print(
        "\n[3/8] Normalizing..."
    )

    normalized, issues = normalize_records(
        raw_records,
    )

    print(
        f"Normalized records: "
        f"{len(normalized)}"
    )

    print(
        f"Normalization issues: "
        f"{len(issues)}"
    )

    if issues:
        print(
            "Warning: some records could not be normalized."
        )

        # Show only the first few issues so a large dataset
        # does not flood the terminal.
        for issue in issues[:5]:
            print(
                f"  Row {issue.row_index}: "
                f"{issue.error}"
            )

        if len(issues) > 5:
            print(
                f"  ... and "
                f"{len(issues) - 5} more issues."
            )

    if not normalized:
        raise RuntimeError(
            "No valid records remained "
            "after normalization."
        )

    # =========================================================
    # 4. RESAMPLE
    # =========================================================

    print(
        "\n[4/8] Resampling..."
    )

    resampled = resample_records(
        normalized
    )

    print(
        f"Resampled records: "
        f"{len(resampled)}"
    )

    if not resampled:
        raise RuntimeError(
            "Resampling produced no records."
        )

    # =========================================================
    # 5. ANALYSIS
    # =========================================================

    print(
        "\n[5/8] Running deterministic analysis..."
    )

    statistics = compute_statistics(
        resampled
    )

    print(
        f"Unique variables: "
        f"{len(statistics['unique_nodes'])}"
    )

    print(
        f"Numeric variables: "
        f"{len(statistics['numeric'])}"
    )

    print(
        f"Categorical variables: "
        f"{len(statistics['categorical'])}"
    )

    print(
        f"Alarm variables with events: "
        f"{len(statistics['alarms'])}"
    )

    # =========================================================
    # 6. BUILD AND VALIDATE EVIDENCE
    # =========================================================

    print(
        "\n[6/8] Building AnalysisEvidence..."
    )

    evidence = build_evidence(
        records=resampled,
        statistics=statistics,
        case_manifest=case.to_dict(),
    )

    # Attach deterministic data-quality information.
    evidence["data_quality"] = {
        "raw_records": len(raw_records),
        "normalized_records": len(normalized),
        "resampled_records": len(resampled),
        "normalization_issues": len(issues),
    }

    # Ground truth must NOT be exposed to the LLM.
    #
    # It may exist in the dataset and may later be used for
    # evaluation, but the inference input remains blind.
    evidence[
        "ground_truth_available"
    ] = False

    # Validate the complete intermediate contract before
    # anything is sent to the cloud model.
    evidence = validate_analysis_evidence(
        evidence
    )

    # ---------------------------------------------------------
    # Save deterministic intermediate output
    # ---------------------------------------------------------

    case_name = Path(
        case.csv_path
    ).stem

    intermediate_path = (
        OUTPUT_ROOT
        / "intermediate"
        / f"{case_name}.json"
    )

    save_json(
        intermediate_path,
        evidence,
    )

    print(
        f"Evidence saved to: "
        f"{intermediate_path}"
    )

    # =========================================================
    # NIM DISABLED
    # =========================================================

    if args.no_nim:

        print(
            "\nNIM disabled."
        )

        print(
            "Deterministic Python pipeline "
            "completed successfully."
        )

        print(
            f"Intermediate output: "
            f"{intermediate_path}"
        )

        return

    # =========================================================
    # 7. NVIDIA NIM
    # =========================================================

    print(
        "\n[7/8] Sending evidence to NVIDIA NIM..."
    )

    result = call_nim(
        evidence=evidence,

        system_prompt_path=(
            PROMPT_ROOT
            / "system_prompt.txt"
        ),

        output_example_path=(
            PROMPT_ROOT
            / "output_example.json"
        ),
    )

    # ---------------------------------------------------------
    # Save final RCA JSON locally
    # ---------------------------------------------------------

    rca_path = (
        OUTPUT_ROOT
        / "rca"
        / f"{case_name}.json"
    )

    save_json(
        rca_path,
        result,
    )

    print(
        f"RCA JSON saved to: "
        f"{rca_path}"
    )

    # =========================================================
    # WEBSITE POST
    # =========================================================

    if args.no_post:

        print(
            "\nWebsite POST disabled."
        )

        print(
            "Final RCA JSON remains available at:"
        )

        print(
            f"  {rca_path}"
        )

        return

    # =========================================================
    # 8. WEBSITE
    # =========================================================

    print(
        "\n[8/8] Posting RCA JSON to website..."
    )

    post_to_website(
        result
    )

    print(
        "\nPipeline completed successfully."
    )

    print(
        f"Final RCA output: "
        f"{rca_path}"
    )


if __name__ == "__main__":
    main()
