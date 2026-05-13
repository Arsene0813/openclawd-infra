from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DICTIONARY_PATH = ROOT / "retail_ops/data/DATA_DICTIONARY.md"
DEMO2_OUTPUT_PATH = ROOT / "retail_ops/outputs/demo2_cross_store_comparability_output.csv"
RESULT_PATH = ROOT / "eval/results/retail_data_contract_validation_result.txt"

REQUIRED_DEMO2_FIELDS = [
    "store_id",
    "period_month",
    "region_type",
    "store_type",
    "comparison_scope_flag",
    "comparison_limit_notes",
]

REQUIRED_DICTIONARY_TERMS = [
    "region_type",
    "weak context",
    "not a hard market-area classification",
    "activity_cost_ratio_pct",
    "not traditional ROI",
    "estimated_income_proxy",
    "not audited profit",
    "order_conversion_rate_pct",
    "order_users / entry_users",
    "top3_sku_transaction_amount_share_pct",
    "not full product-category share",
    "Pairwise comparability-gate fields are not currently implemented",
]

STALE_TERM_PARTS = [
    ("demo3_", "pairwise"),
    ("demo_3_", "pairwise"),
    ("search_entry_", "structure"),
    ("activity_", "transfer"),
    ("eval_retail_", "demo3"),
    ("answer_", "demo3"),
    ("run_", "demo3"),
    ("validate_", "demo3"),
]

CHECKED_DOCS = [
    "README.md",
    "PROJECT_STATUS.md",
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
    "retail_ops/README.md",
    "retail_ops/COMPARABILITY_GATE_V0.md",
    "retail_ops/data/DATA_DICTIONARY.md",
]


def stale_terms() -> list[str]:
    return [left + right for left, right in STALE_TERM_PARTS]


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    return fieldnames, rows


def main() -> int:
    failures: list[str] = []

    if not DATA_DICTIONARY_PATH.exists():
        failures.append(f"Missing data dictionary: {DATA_DICTIONARY_PATH}")

    if DATA_DICTIONARY_PATH.exists():
        dictionary = DATA_DICTIONARY_PATH.read_text(encoding="utf-8")
        for term in REQUIRED_DICTIONARY_TERMS:
            if term not in dictionary:
                failures.append(f"DATA_DICTIONARY.md missing required term: {term}")

    try:
        fieldnames, rows = read_csv_rows(DEMO2_OUTPUT_PATH)
    except Exception as exc:
        failures.append(str(exc))
        fieldnames, rows = [], []

    for field in REQUIRED_DEMO2_FIELDS:
        if field not in fieldnames:
            failures.append(f"Demo2 output missing required field: {field}")

    if not rows:
        failures.append("Demo2 output has no rows.")

    if rows:
        store_ids = {row.get("store_id", "") for row in rows}
        expected_demo2_store_ids = {"B", "C", "D", "E", "F"}
        missing = expected_demo2_store_ids - store_ids
        if missing:
            failures.append(f"Demo2 output missing expected store_ids: {sorted(missing)}")

    for rel in CHECKED_DOCS:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"Missing checked document: {rel}")
            continue

        text = path.read_text(encoding="utf-8").lower()
        for term in stale_terms():
            if term.lower() in text:
                failures.append(
                    f"{rel} still contains stale Demo3 term: {term}"
                )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if failures:
        output = ["Retail data contract validation FAILED.", ""]
        output.extend(f"[FAIL] {failure}" for failure in failures)
        RESULT_PATH.write_text("\n".join(output) + "\n", encoding="utf-8")
        print("\n".join(output))
        return 1

    output = [
        "Retail data contract validation PASSED.",
        "",
        "Checked current implemented scope: Demo 1 and Demo 2.",
        "Checked Demo2 output required fields.",
        "Checked DATA_DICTIONARY.md boundaries.",
        "Checked comparability gate is future work, not implemented Demo3.",
    ]

    RESULT_PATH.write_text("\n".join(output) + "\n", encoding="utf-8")
    print("\n".join(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
