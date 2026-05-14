from pathlib import Path

SPEC_PATH = Path("retail_ops/COMPARABILITY_GATE_V0.md")

REQUIRED_SPEC_TERMS = [
    "reference_store_id",
    "candidate_store_id",
    "comparison_question_type",
    "comparison_decision",
    "comparable",
    "comparable_with_limits",
    "not_comparable",
    "insufficient_evidence",
    "supporting_fields",
    "blocking_or_limiting_factors",
    "allowed_interpretation",
    "unsupported_interpretation",
]


def main() -> int:
    spec = SPEC_PATH.read_text(encoding="utf-8")

    missing = [term for term in REQUIRED_SPEC_TERMS if term not in spec]
    if missing:
        print("Future comparability gate contract stub FAILED.")
        print(f"Missing terms: {missing}")
        return 1

    print("[SKIP] Future comparability gate is documented but not implemented.")
    print("[PASS] Planned input triple, output enum, and output fields are present in COMPARABILITY_GATE_V0.md.")
    print("[PASS] This stub freezes the future contract without claiming an implemented pairwise gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
