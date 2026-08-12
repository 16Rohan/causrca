from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import DATA_ROOT


@dataclass
class CaseManifest:
    case_type: str
    csv_path: str

    family: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None

    description_path: str | None = None
    causes_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def discover_cases() -> list[CaseManifest]:
    """
    Discover every analyzable causRCA case.

    Expert graphs are dataset-level resources, not individual cases.
    """

    cases: list[CaseManifest] = []

    # ---------------------------------------------------------
    # Real operation
    # ---------------------------------------------------------

    real_op = DATA_ROOT / "real_op"

    if real_op.exists():
        for csv_path in sorted(real_op.glob("*.csv")):

            cases.append(
                CaseManifest(
                    case_type="real_op",
                    csv_path=str(csv_path.resolve()),
                )
            )

    # ---------------------------------------------------------
    # Digital twin
    # ---------------------------------------------------------

    digital_twin = DATA_ROOT / "dig_twin"

    if digital_twin.exists():

        for family_dir in sorted(digital_twin.iterdir()):

            if not family_dir.is_dir():
                continue

            # Example:
            #
            # exp_coolant
            # exp_hydraulics
            # exp_probe

            for experiment_dir in sorted(family_dir.iterdir()):

                if not experiment_dir.is_dir():
                    continue

                description_candidates = list(
                    experiment_dir.glob("*_description.json")
                )

                description_path = (
                    description_candidates[0]
                    if description_candidates
                    else None
                )

                for run_dir in sorted(experiment_dir.iterdir()):

                    if not run_dir.is_dir():
                        continue

                    causes_path = run_dir / "causes.json"

                    csv_files = sorted(
                        run_dir.glob("faultDataset_*.csv")
                    )

                    for csv_path in csv_files:

                        cases.append(
                            CaseManifest(
                                case_type="fault",
                                csv_path=str(csv_path.resolve()),
                                family=family_dir.name,
                                experiment_id=experiment_dir.name,
                                run_id=run_dir.name,
                                description_path=(
                                    str(description_path.resolve())
                                    if description_path
                                    else None
                                ),
                                causes_path=(
                                    str(causes_path.resolve())
                                    if causes_path.exists()
                                    else None
                                ),
                            )
                        )

    return cases


def get_case(
    cases: list[CaseManifest],
    index: int,
) -> CaseManifest:

    if not cases:
        raise RuntimeError("No causRCA cases were discovered.")

    if index < 0 or index >= len(cases):
        raise IndexError(
            f"Case index {index} is invalid. "
            f"Available cases: 0-{len(cases) - 1}"
        )

    return cases[index]


def print_cases(cases: list[CaseManifest]) -> None:

    for index, case in enumerate(cases):

        print(
            f"[{index}] "
            f"{case.case_type} | "
            f"{case.family or 'real_op'} | "
            f"{case.experiment_id or '-'} | "
            f"{case.run_id or '-'} | "
            f"{Path(case.csv_path).name}"
        )
