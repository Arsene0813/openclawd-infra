from __future__ import annotations

import csv
from pathlib import Path

OUTPUT_PATH = Path("retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv")
DICTIONARY_PATH = Path("retail_ops/data/DATA_DICTIONARY.md")
FIELD_USAGE_PATH = Path("retail_ops/FIELD_USAGE_REVIEW.md")
GATE_PATH = Path("retail_ops/COMPARABILITY_GATE_V0.md")
RESULTS_PATH = Path("eval/results/eval_retail_demo3_pairwise_gate_result.txt")


def read_rows() -> list[dict[str, str]]:
    if not OUTPUT_PATH.exists():
        raise SystemExit(f"Missing Demo 3 output: {OUTPUT_PATH}")

    with OUTPUT_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    rows = read_rows()

    dictionary_text = DICTIONARY_PATH.read_text(encoding="utf-8")
    field_usage_text = FIELD_USAGE_PATH.read_text(encoding="utf-8")
    gate_text = GATE_PATH.read_text(encoding="utf-8")

    combined_docs = dictionary_text + "\n" + field_usage_text + "\n" + gate_text
    normalized_docs = combined_docs.replace("`", "").lower()

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "pairwise_output_has_three_question_types",
            {row["comparison_question_type"] for row in rows}
            == {"search_entry_structure", "activity_transfer", "order_quality_pressure"},
            "Demo 3 should test three narrow comparison question types.",
        )
    )

    checks.append(
        (
            "pairwise_output_has_expected_row_count",
            len(rows) == 30,
            "Demo 3 should produce 30 rows: 10 store pairs times 3 question types.",
        )
    )

    checks.append(
        (
            "activity_transfer_can_refuse_strategy_transfer",
            any(
                row["comparison_question_type"] == "activity_transfer"
                and row["pairwise_comparison_decision"] == "not_comparable_for_strategy_transfer"
                for row in rows
            ),
            "Activity strategy transfer should be refused for pairs with large gaps or context differences.",
        )
    )

    checks.append(
        (
            "search_entry_structure_stays_narrow",
            all(
                "compare_search_entry_structure_only" in row["pairwise_limit_notes"]
                and "do_not_rank_stores_or_transfer_strategy_directly" in row["pairwise_limit_notes"]
                for row in rows
                if row["comparison_question_type"] == "search_entry_structure"
            ),
            "Search-entry structure comparison should stay narrow and should not become full strategy transfer.",
        )
    )

    checks.append(
        (
            "order_quality_pressure_stays_narrow",
            all(
                "compare_refund_and_invalid_order_pressure_only" in row["pairwise_limit_notes"]
                and "do_not_rank_stores_or_transfer_strategy_directly" in row["pairwise_limit_notes"]
                for row in rows
                if row["comparison_question_type"] == "order_quality_pressure"
            ),
            "Order-quality pressure comparison should stay limited to refund and invalid-order pressure.",
        )
    )

    checks.append(
        (
            "region_type_boundary_is_preserved_in_output",
            all(
                "not_market_classification" in row["region_type_comparison_note"]
                or "market_classification_is_unresolved" in row["region_type_comparison_note"]
                for row in rows
            ),
            "region_type should remain weak context, not market-area classification.",
        )
    )

    checks.append(
        (
            "region_type_boundary_is_preserved_in_docs",
            "region_type is only weak market-context evidence" in normalized_docs
            and "not a market-area classification" in normalized_docs,
            "Docs should explicitly preserve the region_type weak-context boundary.",
        )
    )

    checks.append(
        (
            "no_pairwise_decision_is_best_store_ranking",
            not any(
                "best" in row["pairwise_comparison_decision"].lower()
                or "worst" in row["pairwise_comparison_decision"].lower()
                for row in rows
            ),
            "Pairwise decisions must not rank stores as best or worst.",
        )
    )

    checks.append(
        (
            "new_pairwise_fields_are_documented",
            all(
                field in combined_docs
                for field in [
                    "reference_store_id",
                    "candidate_store_id",
                    "comparison_question_type",
                    "reference_region_type",
                    "candidate_region_type",
                    "region_type_comparison_note",
                    "reference_store_type",
                    "candidate_store_type",
                    "store_type_comparison_note",
                    "pairwise_comparison_decision",
                    "pairwise_limit_notes",
                ]
            ),
            "New Demo 3 pairwise fields should be documented before use.",
        )
    )

    passed = 0
    failed = 0
    lines: list[str] = []

    for name, ok, detail in checks:
        if ok:
            passed += 1
            lines.append(f"PASS {name}")
        else:
            failed += 1
            lines.append(f"FAIL {name}: {detail}")

    summary = [
        f"Retail Demo 3 pairwise gate eval result: {passed}/{len(checks)} passed",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
        *lines,
        "",
    ]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
