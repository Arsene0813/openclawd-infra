from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TERMS = [
    "store_visitors",
    "store_entry_count",
    "search_visitors",
    "activity_gross_revenue",
    "estimated_order_income",
    "paid_users",
    "paid_amount",
    "store_a_demo1_summary.csv",
    "generated_memory_facts.json",
    "demo_retail_decision_support.md",
    "cross_store_comparison_report.md",
    "eval_retail_decision_cases.json",
]

FORBIDDEN_RETAIL_ROUTING_PATTERNS = [
    'slots.append("refund_pressure_improved")',
    "slots.append('refund_pressure_improved')",
    '"slot": "refund_pressure_improved"',
    "'slot': 'refund_pressure_improved'",
]

REQUIRED_FILES = [
    "README.md",
    "retail_ops/README.md",
    "PROJECT_STATUS.md",
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
    "retail_ops/data/DATA_DICTIONARY.md",
    "retail_ops/LINEAGE.md",

    "retail_ops/scripts/validate_retail_data_contract.py",
    "retail_ops/outputs/retail_data_contract_validation_result.txt",
    "retail_ops/outputs/generated_retail_memory_facts.json",
    "eval/eval_retail_report.md",
    "eval/eval_report.md",
]

REQUIRED_DOC_REFERENCES = {
    "README.md": [
        "retail_ops/scripts/validate_retail_data_contract.py",
        "retail_ops/outputs/retail_data_contract_validation_result.txt",
        "retail_ops/outputs/generated_retail_memory_facts.json",
        "eval/eval_retail_report.md",
        "/chat_livestream_kb",
        "/chat_retail_ops_kb",
    ],
    "retail_ops/README.md": [
        "retail_ops/scripts/validate_retail_data_contract.py",
        "retail_ops/outputs/retail_data_contract_validation_result.txt",
        "generated_retail_memory_facts.json",
        "eval/eval_retail_report.md",
    ],
    "PROJECT_STATUS.md": [
        "Current Implemented Scope",
        "Current Boundary",
        "Not Yet Implemented",
        "Validation Commands",
        "retail_ops/data/DATA_DICTIONARY.md",
        "retail_ops/LINEAGE.md",
        "validate_retail_data_contract.py",
        "eval/eval_retail.py",
    ],
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md": [
        "DATA_DICTIONARY.md",
        "LINEAGE.md",
        "validate_retail_data_contract.py",
        "retail_data_contract_validation_result.txt",
    ],
}

ENDPOINT_IMPLEMENTATION_CHECKS = {
    "/chat_mem": '@app.post("/chat_mem")',
    "/chat_livestream_kb": '@app.post("/chat_livestream_kb")',
    "/chat_retail_ops_kb": '@app.post("/chat_retail_ops_kb")',
}


def tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    paths = []
    for line in result.stdout.splitlines():
        path = ROOT / line.strip()
        if path.suffix.lower() in {
            ".md",
            ".py",
            ".sql",
            ".csv",
            ".json",
            ".txt",
            ".yml",
            ".yaml",
        }:
            paths.append(path)

    return paths


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    failures: list[str] = []

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            failures.append(f"Missing required file: {rel}")

    validator_self = (ROOT / "scripts/validate_project_consistency.py").resolve()
    retail_validator = (
        ROOT / "retail_ops/scripts/validate_retail_data_contract.py"
    ).resolve()

    tracked_text = {}

    for path in tracked_text_files():
        if path.resolve() in {validator_self, retail_validator}:
            continue

        try:
            text = read_text(path)
        except UnicodeDecodeError:
            continue

        rel_path = path.relative_to(ROOT).as_posix()
        tracked_text[rel_path] = text

        for term in FORBIDDEN_TERMS:
            if term in text:
                failures.append(f"Forbidden stale term `{term}` found in {rel_path}")

    for rel, required_terms in REQUIRED_DOC_REFERENCES.items():
        path = ROOT / rel
        if not path.exists():
            continue

        text = read_text(path)
        for term in required_terms:
            if term not in text:
                failures.append(f"`{rel}` is missing required reference: {term}")

    api_main = ROOT / "api/main.py"

    if api_main.exists():
        api_text = read_text(api_main)
        all_tracked_text = "\n".join(tracked_text.values())

        for endpoint, implementation_marker in ENDPOINT_IMPLEMENTATION_CHECKS.items():
            endpoint_is_claimed = endpoint in all_tracked_text
            endpoint_is_implemented = implementation_marker in api_text

            if endpoint_is_claimed and not endpoint_is_implemented:
                failures.append(
                    f"Endpoint `{endpoint}` is referenced in project files "
                    f"but not implemented in api/main.py"
                )

        for pattern in FORBIDDEN_RETAIL_ROUTING_PATTERNS:
            if pattern in api_text:
                failures.append(
                    "Non-canonical retail retrieval slot found in api/main.py: "
                    "`refund_pressure_improved` is a SQL-derived supporting observation, "
                    "not a canonical retrieval slot. Use `order_quality_pressure_profile` "
                    "for refund / invalid-order pressure retrieval."
                )
    else:
        failures.append("Missing required API file: api/main.py")

    facts_path = ROOT / "retail_ops/outputs/generated_retail_memory_facts.json"

    if facts_path.exists():
        try:
            facts = json.loads(read_text(facts_path))
        except json.JSONDecodeError as exc:
            failures.append(f"generated_retail_memory_facts.json is invalid JSON: {exc}")
        else:
            if not isinstance(facts, list):
                failures.append("generated_retail_memory_facts.json should contain a list")

    if failures:
        print("Project consistency validation FAILED.")
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("Project consistency validation PASSED.")
    print(f"Checked required files: {len(REQUIRED_FILES)}")
    print(f"Checked documentation reference groups: {len(REQUIRED_DOC_REFERENCES)}")
    print("Checked documented API endpoints against api/main.py.")
    print("Checked retail routing slots against canonical retail memory slot rules.")
    print("No stale field aliases or stale file paths found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
