from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "PROJECT_STATUS.md",
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
    "retail_ops/README.md",
    "retail_ops/ARCHITECTURE.md",
    "retail_ops/EXPERIMENT_RESULTS.md",
    "retail_ops/COMPARABILITY_GATE_V0.md",
    "retail_ops/FIELD_USAGE_REVIEW.md",
    "retail_ops/LINEAGE.md",
    "retail_ops/data/DATA_DICTIONARY.md",
    "retail_ops/sql/01_store_a_month_over_month_diagnostic.sql",
    "retail_ops/sql/02_demo2_cross_store_comparability.sql",
    "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
    "eval/eval_retail_demo2_comparability_gate.py",
]

CURRENT_SCOPE_REQUIRED_TERMS = {
    "README.md": [
        "The current implemented retail scope stops at Demo 2",
        "Future Work: Comparability Gate",
    ],
    "PROJECT_STATUS.md": [
        "Demo 1",
        "Demo 2",
        "Comparability gate",
        "Future work",
    ],
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md": [
        "A pairwise comparability gate is planned as the next stage",
        "not presented as a finished demo",
    ],
    "retail_ops/README.md": [
        "The current implemented retail scope stops at Demo 2",
        "Future Work: Comparability Gate",
    ],
    "retail_ops/COMPARABILITY_GATE_V0.md": [
        "A pairwise comparability gate is not currently implemented as a finished demo",
        "transaction order volume",
        "transaction amount",
        "activity or promotion",
        "region and market context",
        "competition environment",
        "repeated reporting windows",
    ],
    "retail_ops/data/DATA_DICTIONARY.md": [
        "Pairwise comparability-gate fields are not currently implemented",
        "region_type remains weak context only",
        "not a hard market-area classification",
    ],
}

STALE_TERM_PARTS = [
    ("eval_retail_", "demo3"),
    ("run_", "demo3"),
    ("validate_", "demo3"),
    ("answer_", "demo3"),
    ("demo_3_", "pairwise"),
    ("demo3_", "pairwise"),
    ("03_demo2_", "pairwise_comparability_gate"),
    ("search_entry_", "structure"),
    ("activity_", "transfer"),
]

SCAN_SUFFIXES = {
    ".md",
    ".py",
    ".sql",
    ".csv",
    ".json",
    ".txt",
    ".yml",
    ".yaml",
}

SKIP_STALE_SCAN_FILES = {
    "scripts/validate_project_consistency.py",
    "retail_ops/scripts/validate_retail_data_contract.py",
}


def stale_terms() -> list[str]:
    return [left + right for left, right in STALE_TERM_PARTS]


def git_tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        ROOT / line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in SCAN_SUFFIXES


def main() -> int:
    failures: list[str] = []

    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"Missing required current-scope file: {rel}")

    for rel, required_terms in CURRENT_SCOPE_REQUIRED_TERMS.items():
        path = ROOT / rel
        if not path.exists():
            failures.append(f"Missing file for content check: {rel}")
            continue

        text = read_text(path)
        for term in required_terms:
            if term not in text:
                failures.append(f"{rel} missing required current-scope term: {term}")

    deleted_terms = stale_terms()

    for path in git_tracked_files():
        if not is_text_file(path):
            continue

        rel = path.relative_to(ROOT).as_posix()

        if rel in SKIP_STALE_SCAN_FILES:
            continue

        try:
            text = read_text(path)
        except UnicodeDecodeError:
            continue

        lowered = text.lower()

        for term in deleted_terms:
            if term.lower() in lowered:
                failures.append(
                    f"{rel} still contains deleted Demo3 artifact term: {term}"
                )

    dictionary = read_text(ROOT / "retail_ops/data/DATA_DICTIONARY.md")

    dictionary_required_boundaries = [
        "activity_cost_ratio_pct",
        "not traditional ROI",
        "estimated_income_proxy",
        "not audited profit",
        "order_conversion_rate_pct",
        "order_users / entry_users",
        "top3_sku_transaction_amount_share_pct",
        "not full product-category share",
    ]

    for term in dictionary_required_boundaries:
        if term not in dictionary:
            failures.append(f"DATA_DICTIONARY.md missing boundary term: {term}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("[PASS] Project consistency validation passed.")
    print("[PASS] Current implemented retail scope stops at Demo 2.")
    print("[PASS] Comparability gate is documented as future work.")
    print("[PASS] Deleted Demo3 artifact references are absent from project docs and artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
