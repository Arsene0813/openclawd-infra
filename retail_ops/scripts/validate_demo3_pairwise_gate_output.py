from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = ROOT / "retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv"
DICTIONARY_PATH = ROOT / "retail_ops/data/DATA_DICTIONARY.md"
FIELD_USAGE_PATH = ROOT / "retail_ops/FIELD_USAGE_REVIEW.md"
GATE_PATH = ROOT / "retail_ops/COMPARABILITY_GATE_V0.md"

EXPECTED_FIELDS = [
    "reference_store_id",
    "candidate_store_id",
    "period_month",
    "comparison_question_type",
    "reference_region_type",
    "candidate_region_type",
    "region_type_comparison_note",
    "reference_store_type",
    "candidate_store_type",
    "store_type_comparison_note",
    "search_entry_share_gap_pct",
    "activity_order_share_gap_pct",
    "activity_cost_ratio_gap_pct",
    "refund_pressure_gap_pct",
    "invalid_order_pressure_gap_pct",
    "top3_sku_concentration_gap_pct",
    "pairwise_comparison_decision",
    "pairwise_limit_notes",
]

EXPECTED_QUESTION_TYPES = {
    "search_entry_structure",
    "activity_transfer",
    "order_quality_pressure",
}

EXPECTED_DECISIONS = {
    "comparable_with_limits",
    "partially_comparable",
    "not_comparable_for_strategy_transfer",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != EXPECTED_FIELDS:
            raise SystemExit(
                "Demo 3 output header mismatch.\n"
                f"Expected: {EXPECTED_FIELDS}\n"
                f"Actual: {reader.fieldnames}"
            )

        return list(reader)


def main() -> int:
    rows = read_rows(OUTPUT_PATH)

    if len(rows) != 30:
        raise SystemExit(f"Expected 30 pairwise question rows, got {len(rows)}")

    question_counts = Counter(row["comparison_question_type"] for row in rows)
    expected_counts = {
        "search_entry_structure": 10,
        "activity_transfer": 10,
        "order_quality_pressure": 10,
    }

    if question_counts != expected_counts:
        raise SystemExit(
            "Unexpected question type counts.\n"
            f"Expected: {expected_counts}\n"
            f"Actual: {dict(question_counts)}"
        )

    pair_counts = Counter(
        (row["reference_store_id"], row["candidate_store_id"])
        for row in rows
    )

    if len(pair_counts) != 10:
        raise SystemExit(f"Expected 10 store pairs, got {len(pair_counts)}")

    if any(count != 3 for count in pair_counts.values()):
        raise SystemExit(f"Each store pair should have 3 question rows: {dict(pair_counts)}")

    decisions = {row["pairwise_comparison_decision"] for row in rows}
    unknown_decisions = decisions - EXPECTED_DECISIONS

    if unknown_decisions:
        raise SystemExit(f"Unknown pairwise decisions: {sorted(unknown_decisions)}")

    region_note_text = "\n".join(row["region_type_comparison_note"] for row in rows)
    limit_note_text = "\n".join(row["pairwise_limit_notes"] for row in rows)
    combined_output_text = region_note_text + "\n" + limit_note_text

    required_output_terms = [
        "not_market_classification",
        "do_not_rank_stores_or_transfer_strategy_directly",
    ]

    for term in required_output_terms:
        if term not in combined_output_text:
            raise SystemExit(f"Missing required limitation term in output: {term}")

    forbidden_terms = [
        "city_center",
        "county",
        "community",
        "warehouse_area",
        "best_store",
        "worst_store",
        "traditional_roi",
        "profit_margin",
    ]

    lowered_output_text = combined_output_text.lower()

    for term in forbidden_terms:
        if term in lowered_output_text:
            raise SystemExit(f"Forbidden premature or misleading term found: {term}")

    docs_text = (
        DICTIONARY_PATH.read_text(encoding="utf-8")
        + "\n"
        + FIELD_USAGE_PATH.read_text(encoding="utf-8")
        + "\n"
        + GATE_PATH.read_text(encoding="utf-8")
    )

    for field in EXPECTED_FIELDS:
        if field not in docs_text:
            raise SystemExit(f"Field is not documented before use: {field}")

    normalized_docs_text = docs_text.replace("`", "").lower()

    if "region_type is only weak market-context evidence" not in normalized_docs_text:
        raise SystemExit("Missing region_type weak-context boundary in docs.")

    if "not a market-area classification" not in normalized_docs_text:
        raise SystemExit("Missing market-area classification boundary in docs.")

    print("Demo 3 pairwise gate validation PASSED.")
    print("Checked output rows: 30")
    print("Checked store pairs: 10")
    print("Checked question types: 3 per pair")
    print("Checked pairwise decision values.")
    print("Checked region_type is not treated as market-area classification.")
    print("Checked Demo 3 output fields are documented before use.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
