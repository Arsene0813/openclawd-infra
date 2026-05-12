from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GATE_PATH = Path("retail_ops/COMPARABILITY_GATE_V0.md")
DICTIONARY_PATH = Path("retail_ops/data/DATA_DICTIONARY.md")
FIELD_USAGE_PATH = Path("retail_ops/FIELD_USAGE_REVIEW.md")
DEMO2_FACTS_PATH = Path("retail_ops/outputs/generated_demo2_retail_memory_facts.json")
PROJECT_STATUS_PATH = Path("PROJECT_STATUS.md")
DEMO3_OUTPUT_PATH = Path("retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv")
RESULTS_PATH = Path("eval/results/eval_retail_demo2_comparability_gate_result.txt")


def load_facts(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("facts"), list):
        return data["facts"]

    raise SystemExit(f"Unexpected Demo 2 facts JSON structure: {path}")


def fact_as_text(fact: dict[str, Any]) -> str:
    return json.dumps(fact, ensure_ascii=False).lower()


def fact_preserves_demo2_scope(fact: dict[str, Any]) -> bool:
    text = fact_as_text(fact)

    has_demo2_context = "demo2" in text or "demo 2" in text
    has_cross_store_context = "cross_store" in text or "cross-store" in text
    has_same_period_context = "same_period" in text or "same-period" in text or "2026-03" in text

    return has_demo2_context and has_cross_store_context and has_same_period_context


def fact_preserves_limits(fact: dict[str, Any]) -> bool:
    text = fact_as_text(fact)

    limit_terms = [
        "limit",
        "limitation",
        "boundary",
        "guard",
        "not_comparable",
        "do_not",
        "must not",
        "not a final",
        "not full",
        "not sufficient",
        "cannot",
    ]

    return any(term in text for term in limit_terms)


def main() -> int:
    gate_text = GATE_PATH.read_text(encoding="utf-8")
    dictionary_text = DICTIONARY_PATH.read_text(encoding="utf-8")
    field_usage_text = FIELD_USAGE_PATH.read_text(encoding="utf-8")
    project_status_text = PROJECT_STATUS_PATH.read_text(encoding="utf-8")
    facts = load_facts(DEMO2_FACTS_PATH)

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "implemented_sql_scope_flags_are_documented",
            all(
                term in gate_text
                for term in [
                    "comparison_scope_flag",
                    "same_period_diagnostic_ready",
                    "not_comparable_period_mismatch",
                    "insufficient_data",
                ]
            ),
            "Implemented Demo 2 SQL scope flags should be documented.",
        )
    )

    checks.append(
        (
            "demo3_pairwise_outcomes_are_implemented_separately",
            "pairwise_comparison_decision" in gate_text
            and "pairwise_limit_notes" in gate_text
            and str(DEMO3_OUTPUT_PATH) in project_status_text
            and DEMO3_OUTPUT_PATH.exists(),
            "Demo 3 pairwise outcomes should be implemented separately from the Demo 2 row-level comparison_scope_flag.",
        )
    )

    checks.append(
        (
            "period_granularity_is_defined_in_dictionary",
            "period_granularity" in dictionary_text
            and "month" in dictionary_text,
            "period_granularity should be defined in the data dictionary.",
        )
    )

    checks.append(
        (
            "period_granularity_is_reviewed_without_rename",
            "period_granularity" in field_usage_text
            and "Period Granularity Field Review" in field_usage_text
            and "No. Keep unchanged." in field_usage_text,
            "period_granularity should be reviewed without renaming existing fields.",
        )
    )

    checks.append(
        (
            "generated_demo2_facts_preserve_scope_and_limits",
            bool(facts)
            and all(fact_preserves_demo2_scope(fact) for fact in facts)
            and all(fact_preserves_limits(fact) for fact in facts),
            "Generated Demo 2 facts should preserve Demo 2 same-period cross-store scope and interpretation limits.",
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
        f"Retail Demo 2 comparability-gate consistency eval result: {passed}/{len(checks)} passed",
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
