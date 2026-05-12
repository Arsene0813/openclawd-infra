from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retail_ops.scripts.answer_demo3_pairwise_gate import answer_demo3_pairwise_question  # noqa: E402


OUTPUT_PATH = REPO_ROOT / "retail_ops" / "outputs" / "demo3_pairwise_comparability_gate_output.csv"


def load_rows() -> list[dict[str, str]]:
    with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_row(question_type: str, decision: str | None = None) -> dict[str, str]:
    for row in load_rows():
        if row.get("comparison_question_type") != question_type:
            continue
        if decision is not None and row.get("pairwise_comparison_decision") != decision:
            continue
        return row
    raise AssertionError(f"No row found for question_type={question_type!r}, decision={decision!r}")


def assert_contains(text: str, expected: str) -> None:
    if expected not in text:
        raise AssertionError(f"Expected {expected!r} in answer:\n{text}")


def test_activity_transfer_answer_preserves_gate_boundary() -> None:
    row = find_row("activity_transfer")
    question = (
        f"Can Store {row['reference_store_id']} and Store {row['candidate_store_id']} "
        "be compared for activity_transfer?"
    )
    result = answer_demo3_pairwise_question(question)

    assert result["status"] == "answered"
    assert result["comparison_question_type"] == "activity_transfer"
    assert result["pairwise_comparison_decision"] == row["pairwise_comparison_decision"]
    assert "activity_order_share_gap_pct" in result["relevant_gap_fields"]
    assert "activity_cost_ratio_gap_pct" in result["relevant_gap_fields"]
    assert_contains(result["answer"], "comparability-gate answer")
    assert_contains(result["answer"], "not a final operating recommendation")


def test_search_entry_structure_answer_stays_narrow() -> None:
    row = find_row("search_entry_structure")
    question = (
        f"Can Store {row['candidate_store_id']} and Store {row['reference_store_id']} "
        "be compared for search_entry_structure?"
    )
    result = answer_demo3_pairwise_question(question)

    assert result["status"] == "answered"
    assert result["comparison_question_type"] == "search_entry_structure"
    assert "search_entry_share_gap_pct" in result["relevant_gap_fields"]
    assert_contains(result["answer"], "The stored pairwise row")
    assert_contains(result["answer"], "not a final operating recommendation")


def test_order_quality_pressure_answer_includes_quality_gaps() -> None:
    row = find_row("order_quality_pressure")
    question = (
        f"Can Store {row['reference_store_id']} and Store {row['candidate_store_id']} "
        "be compared for order_quality_pressure?"
    )
    result = answer_demo3_pairwise_question(question)

    assert result["status"] == "answered"
    assert result["comparison_question_type"] == "order_quality_pressure"
    assert "refund_pressure_gap_pct" in result["relevant_gap_fields"]
    assert "invalid_order_pressure_gap_pct" in result["relevant_gap_fields"]
    assert_contains(result["answer"], "Limit notes")


def test_broad_48_store_ranking_is_refused() -> None:
    result = answer_demo3_pairwise_question("Rank all 48 stores by operating quality.")

    assert result["status"] == "refused"
    assert result["reason"] == "unsupported_broad_request"
    assert_contains(result["answer"], "does not support full 48-store ranking")


def test_missing_store_pair_is_reported() -> None:
    result = answer_demo3_pairwise_question("Can Store A and Store Z be compared for activity_transfer?")

    assert result["status"] == "pair_not_found"
    assert result["requested_store_ids"] == ["A", "Z"]
    assert_contains(result["answer"], "current Demo 3 output only covers the existing Demo 2 B-F sample")


def test_missing_question_type_is_reported() -> None:
    result = answer_demo3_pairwise_question("Can Store B and Store E be compared?")

    assert result["status"] == "needs_supported_question_type"
    assert "activity_transfer" in result["supported_question_types"]


def run() -> int:
    tests = [
        test_activity_transfer_answer_preserves_gate_boundary,
        test_search_entry_structure_answer_stays_narrow,
        test_order_quality_pressure_answer_includes_quality_gaps,
        test_broad_48_store_ranking_is_refused,
        test_missing_store_pair_is_reported,
        test_missing_question_type_is_reported,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
            failed += 1

    print(f"\nRetail Demo 3 pairwise answer-path eval result: {passed}/{len(tests)} passed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
