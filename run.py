from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

from inference_client import call_inference_model

from pipeline.analysis import compute_statistics
from pipeline.causal import compute_causal_relationships
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
from pipeline.evidence import build_analysis_evidence
from pipeline.loader import load_csv
from pipeline.normalize import normalize_records
from pipeline.resample import resample_records
from pipeline.validate import validate_analysis_evidence

# =============================================================================
# JSON
# =============================================================================


def save_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


# =============================================================================
# WEBSITE
# =============================================================================


def post_to_website(
    result: dict,
) -> None:

    endpoint = os.getenv("WEBSITE_RCA_ENDPOINT")

    if not endpoint:

        print("WEBSITE_RCA_ENDPOINT not set. " "Skipping website POST.")

        return

    headers = {
        "Content-Type": "application/json",
    }

    token = os.getenv("WEBSITE_API_KEY")

    if token:

        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(
        endpoint,
        headers=headers,
        json=result,
        timeout=30,
    )

    response.raise_for_status()

    print(f"Website POST successful: " f"{response.status_code}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=("Run the causRCA root-cause " "analysis pipeline.")
    )

    parser.add_argument(
        "--case",
        type=int,
        default=0,
        help=("Index of the causRCA case " "to analyze."),
    )

    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List discovered cases and exit.",
    )

    parser.add_argument(
        "--no-inference",
        action="store_true",
        help=(
            "Run the Python preprocessing and "
            "analysis pipeline without calling "
            "the inference model."
        ),
    )

    parser.add_argument(
        "--no-post",
        action="store_true",
        help=(
            "Run inference and save the final "
            "RCA JSON, but do not post it "
            "to the website."
        ),
    )

    args = parser.parse_args()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    ensure_directories()
    print_config()

    # =========================================================================
    # 1. DISCOVERY
    # =========================================================================

    print("\n[1/8] Discovering causRCA cases...")

    cases = discover_cases()

    print(f"Discovered {len(cases)} cases.")

    if args.list_cases:

        print_cases(cases)

        return

    case = get_case(
        cases,
        args.case,
    )

    print(f"Selected: {case.csv_path}")

    # =========================================================================
    # 2. LOAD
    # =========================================================================

    print("\n[2/8] Loading data...")

    raw_records = load_csv(case.csv_path)

    print(f"Loaded {len(raw_records)} raw records.")

    # =========================================================================
    # 3. NORMALIZE
    # =========================================================================

    print("\n[3/8] Normalizing...")

    normalized, issues = normalize_records(raw_records)

    print(f"Normalized records: " f"{len(normalized)}")

    print(f"Normalization issues: " f"{len(issues)}")

    if not normalized:

        raise RuntimeError("No valid records remained " "after normalization.")

    # =========================================================================
    # 4. RESAMPLE
    # =========================================================================

    print("\n[4/8] Resampling...")

    resampled = resample_records(normalized)

    print(f"Resampled records: " f"{len(resampled)}")

    if not resampled:

        raise RuntimeError("Resampling produced no records.")

    # =========================================================================
    # 5. DETERMINISTIC ANALYSIS
    # =========================================================================

    print("\n[5/8] Running deterministic analysis...")

    analysis = compute_statistics(resampled)

    unique_nodes = analysis.get(
        "unique_nodes",
        [],
    )

    numeric_stats = analysis.get(
        "numeric_stats",
        {},
    )

    alarm_stats = analysis.get(
        "alarm_stats",
        [],
    )

    event_catalog = analysis.get(
        "event_catalog",
        [],
    )

    event_groups = analysis.get(
        "event_groups",
        [],
    )

    print(f"Unique variables: " f"{len(unique_nodes)}")

    print(f"Numeric variables: " f"{len(numeric_stats)}")

    print(f"Alarm variables with events: " f"{len(alarm_stats)}")

    print(f"Unique event definitions: " f"{len(event_catalog)}")

    print(f"Event groups: " f"{len(event_groups)}")

    # =========================================================================
    # CAUSAL ANALYSIS
    # =========================================================================

    print("\n    Building causal relationships...")

    causal = compute_causal_relationships(analysis)

    relationships = causal.get(
        "relationships",
        [],
    )

    candidate_causes = causal.get(
        "candidate_causes",
        [],
    )

    print(f"Relationships: " f"{len(relationships)}")

    print(f"Candidate upstream causes: " f"{len(candidate_causes)}")

    # =========================================================================
    # 6. COMPACT EVIDENCE
    # =========================================================================

    print("\n[6/8] Building compact AnalysisEvidence...")

    if hasattr(
        case,
        "to_dict",
    ):

        case_manifest = case.to_dict()

    else:

        case_manifest = {
            "id": Path(case.csv_path).stem,
            "source": "causRCA",
            "path": str(case.csv_path),
        }

    evidence = build_analysis_evidence(
        records=resampled,
        analysis=analysis,
        causal=causal,
        case=case_manifest,
        ground_truth_available=False,
        data_quality={
            "raw_records": len(raw_records),
            "normalized_records": len(normalized),
            "resampled_records": len(resampled),
            "normalization_issues": len(issues),
        },
    )

    # Validate compact intermediate format.
    evidence = validate_analysis_evidence(evidence)

    case_name = Path(case.csv_path).stem

    intermediate_path = OUTPUT_ROOT / "intermediate" / f"{case_name}.json"

    save_json(
        intermediate_path,
        evidence,
    )

    print(f"Compact evidence saved to: " f"{intermediate_path}")

    print(f"Compact evidence size: " f"{intermediate_path.stat().st_size:,} bytes")

    # =========================================================================
    # STOP BEFORE MODEL
    # =========================================================================

    if args.no_inference:

        print("\nInference model disabled.")

        print("Deterministic Python pipeline " "completed successfully.")

        print(f"Intermediate output: " f"{intermediate_path}")

        return

    # =========================================================================
    # 7. INFERENCE
    # =========================================================================

    print("\n[7/8] Sending compact evidence " "to inference model...")

    result = call_inference_model(
        evidence=evidence,
        system_prompt_path=(PROMPT_ROOT / "system_prompt.txt"),
        output_example_path=(PROMPT_ROOT / "output_example.json"),
    )

    # =========================================================================
    # SAVE FINAL OUTPUT
    # =========================================================================

    rca_path = OUTPUT_ROOT / "rca" / f"{case_name}.json"

    save_json(
        rca_path,
        result,
    )

    print(f"RCA JSON saved to: " f"{rca_path}")

    # =========================================================================
    # STOP BEFORE WEBSITE
    # =========================================================================

    if args.no_post:

        print("\nWebsite POST disabled.")

        print(f"Final RCA output: " f"{rca_path}")

        return

    # =========================================================================
    # 8. WEBSITE
    # =========================================================================

    print("\n[8/8] Posting RCA JSON to website...")

    post_to_website(result)

    print("\nPipeline completed successfully.")

    print(f"Final RCA output: " f"{rca_path}")


if __name__ == "__main__":
    main()
