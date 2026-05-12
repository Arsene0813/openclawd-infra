from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(".")

FACTS_PATH = ROOT / "retail_ops/outputs/generated_demo2_retail_memory_facts.json"
GATE_PATH = ROOT / "retail_ops/COMPARABILITY_GATE_V0.md"
DICTIONARY_PATH = ROOT / "retail_ops/data/DATA_DICTIONARY.md"
FIELD_REVIEW_PATH = ROOT / "retail_ops/FIELD_USAGE_REVIEW.md"
RESULTS_PATH = ROOT / "eval/results/eval_retail_demo2_comparability_gate_result.txt"

checks = []

gate_text = GATE_PATH.read_text(encoding="utf-8")
dictionary_text = DICTIONARY_PATH.read_text(encoding="utf-8")
field_review_text = FIELD_REVIEW_PATH.read_text(encoding="utf-8")
facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
facts_text = json.dumps(facts, ensure_ascii=False)

checks.append((
    "implemented_sql_scope_flags_are_documented",
    all(term in gate_text for term in [
        "## Implemented SQL Scope Flags",
        "same_period_diagnostic_ready",
        "not_comparable_period_mismatch",
        "insufficient_data",
    ]),
    "COMPARABILITY_GATE_V0.md should document current comparison_scope_flag values as implemented SQL scope flags.",
))

checks.append((
    "future_pairwise_outcomes_are_separate",
    all(term in gate_text for term in [
        "## Future Pairwise Gate Outcomes",
        "not current SQL output columns",
        "comparable_with_limits",
        "partially_comparable",
        "not_comparable_for_strategy_transfer",
    ]),
    "Future pairwise gate outcomes should be separated from current SQL output fields.",
))

checks.append((
    "period_granularity_is_defined_in_dictionary",
    all(term in dictionary_text for term in [
        "period_start",
        "period_end",
        "period_granularity",
        "not a direct Meituan backend metric",
    ]),
    "DATA_DICTIONARY.md should define period_start, period_end, and period_granularity as memory-fact period metadata.",
))

checks.append((
    "period_granularity_is_reviewed_without_rename",
    "| period_granularity |" in field_review_text and "Generated retail memory facts" in field_review_text and "| No. |" in field_review_text,
    "FIELD_USAGE_REVIEW.md should include period_granularity and confirm no rename.",
))

checks.append((
    "generated_demo2_facts_preserve_scope_and_limits",
    all(term in facts_text for term in [
        "comparison_scope_flag",
        "comparison_limit_notes",
        "period_granularity",
    ]),
    "Generated Demo 2 memory facts should preserve comparison scope, limit notes, and period granularity.",
))

passed = 0
failed = 0
lines = []

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

sys.exit(0 if failed == 0 else 1)
