from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "retail_ops" / "outputs" / "demo3_pairwise_comparability_gate_output.csv"

SUPPORTED_QUESTION_TYPES = {
    "search_entry_structure",
    "activity_transfer",
    "order_quality_pressure",
}

GAP_FIELDS_BY_QUESTION_TYPE = {
    "search_entry_structure": [
        "search_entry_share_gap_pct",
        "activity_order_share_gap_pct",
        "refund_pressure_gap_pct",
        "invalid_order_pressure_gap_pct",
        "top3_sku_concentration_gap_pct",
    ],
    "activity_transfer": [
        "activity_order_share_gap_pct",
        "activity_cost_ratio_gap_pct",
        "refund_pressure_gap_pct",
        "invalid_order_pressure_gap_pct",
        "top3_sku_concentration_gap_pct",
    ],
    "order_quality_pressure": [
        "refund_pressure_gap_pct",
        "invalid_order_pressure_gap_pct",
        "activity_order_share_gap_pct",
        "activity_cost_ratio_gap_pct",
    ],
}


def load_rows(path: Path = DEFAULT_INPUT_PATH) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Demo 3 pairwise output not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Demo 3 pairwise output has no rows: {path}")

    return rows


def normalize_store_id(value: str) -> str:
    return value.strip().upper()


def infer_question_type(message: str) -> str | None:
    text = message.lower()

    explicit = re.search(
        r"\b(search_entry_structure|activity_transfer|order_quality_pressure)\b",
        text,
    )
    if explicit:
        return explicit.group(1)

    search_terms = [
        "search",
        "entry",
        "traffic",
        "visibility",
        "seen",
        "entered",
        "曝光",
        "入店",
        "搜索",
        "流量",
    ]
    activity_terms = [
        "activity",
        "promotion",
        "subsidy",
        "transfer",
        "活动",
        "补贴",
        "促销",
        "迁移",
        "复制",
    ]
    order_quality_terms = [
        "refund",
        "invalid",
        "cancel",
        "quality",
        "退款",
        "无效订单",
        "取消",
        "订单质量",
        "履约",
    ]

    if any(term in text for term in activity_terms):
        return "activity_transfer"

    if any(term in text for term in order_quality_terms):
        return "order_quality_pressure"

    if any(term in text for term in search_terms):
        return "search_entry_structure"

    return None


def extract_store_ids(message: str) -> list[str]:
    text = message.upper()

    found: list[str] = []

    patterns = [
        r"\bSTORE\s*([A-Z])\b",
        r"\bSTORES\s*([A-Z])\b",
        r"\b([A-Z])\s*/\s*([A-Z])\b",
        r"\b([A-Z])\s+(?:AND|VS|VERSUS)\s+([A-Z])\b",
        r"\b([A-Z])\s*(?:和|与|对比)\s*([A-Z])\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            groups = [g for g in match.groups() if g]
            for group in groups:
                store_id = normalize_store_id(group)
                if store_id not in found:
                    found.append(store_id)

    return found


def is_unsupported_broad_request(message: str) -> bool:
    text = message.lower()
    broad_terms = [
        "rank all",
        "ranking all",
        "all 48",
        "48 stores",
        "best store",
        "worst store",
        "market classification",
        "classify markets",
        "causal effect",
        "prove causality",
        "最终建议",
        "全部48",
        "48家",
        "排名所有",
        "最佳门店",
        "最差门店",
        "市场分类",
        "因果",
    ]
    return any(term in text for term in broad_terms)


def find_pair_row(
    rows: list[dict[str, str]],
    store_a: str,
    store_b: str,
    question_type: str,
) -> dict[str, str] | None:
    a = normalize_store_id(store_a)
    b = normalize_store_id(store_b)

    for row in rows:
        reference = normalize_store_id(row.get("reference_store_id", ""))
        candidate = normalize_store_id(row.get("candidate_store_id", ""))
        row_question_type = row.get("comparison_question_type", "")

        same_pair = {reference, candidate} == {a, b}
        same_question = row_question_type == question_type

        if same_pair and same_question:
            return row

    return None


def selected_gap_fields(row: dict[str, str], question_type: str) -> dict[str, str]:
    fields = GAP_FIELDS_BY_QUESTION_TYPE.get(question_type, [])
    return {
        field: row.get(field, "")
        for field in fields
        if field in row and str(row.get(field, "")).strip() != ""
    }


def build_supported_answer(row: dict[str, str], requested_store_ids: list[str], question_type: str) -> dict[str, Any]:
    reference_store_id = row.get("reference_store_id", "")
    candidate_store_id = row.get("candidate_store_id", "")
    decision = row.get("pairwise_comparison_decision", "")
    limit_notes = row.get("pairwise_limit_notes", "")
    gaps = selected_gap_fields(row, question_type)

    pair_note = (
        f"The stored pairwise row is {reference_store_id}/{candidate_store_id}. "
        f"The requested pair was {requested_store_ids[0]}/{requested_store_ids[1]}."
    )

    boundary_note = (
        "This is a comparability-gate answer, not a final operating recommendation. "
        "It does not prove causality, rank stores, classify market areas, or decide that one store's strategy should be copied to the other."
    )

    lines = [
        pair_note,
        f"For `{question_type}`, the pairwise decision is `{decision}`.",
    ]

    if gaps:
        gap_text = ", ".join(f"{k}={v}" for k, v in gaps.items())
        lines.append(f"Relevant gap fields: {gap_text}.")

    if limit_notes:
        lines.append(f"Limit notes: {limit_notes}")

    lines.append(boundary_note)

    return {
        "status": "answered",
        "reference_store_id": reference_store_id,
        "candidate_store_id": candidate_store_id,
        "requested_store_ids": requested_store_ids,
        "comparison_question_type": question_type,
        "pairwise_comparison_decision": decision,
        "relevant_gap_fields": gaps,
        "pairwise_limit_notes": limit_notes,
        "answer": "\n".join(lines),
    }


def answer_demo3_pairwise_question(message: str, input_path: Path = DEFAULT_INPUT_PATH) -> dict[str, Any]:
    if is_unsupported_broad_request(message):
        return {
            "status": "refused",
            "reason": "unsupported_broad_request",
            "answer": (
                "Demo 3 only supports pairwise comparability checks for selected store pairs and supported question types. "
                "It does not support full 48-store ranking, best-store selection, market-area classification, causal promotion-effect analysis, or final operating recommendations."
            ),
        }

    question_type = infer_question_type(message)
    if question_type is None:
        return {
            "status": "needs_supported_question_type",
            "supported_question_types": sorted(SUPPORTED_QUESTION_TYPES),
            "answer": (
                "Demo 3 needs one supported comparison question type: "
                "`search_entry_structure`, `activity_transfer`, or `order_quality_pressure`."
            ),
        }

    if question_type not in SUPPORTED_QUESTION_TYPES:
        return {
            "status": "unsupported_question_type",
            "supported_question_types": sorted(SUPPORTED_QUESTION_TYPES),
            "answer": (
                f"`{question_type}` is not supported. "
                "Demo 3 currently supports `search_entry_structure`, `activity_transfer`, and `order_quality_pressure`."
            ),
        }

    store_ids = extract_store_ids(message)
    if len(store_ids) != 2:
        return {
            "status": "needs_two_store_ids",
            "answer": (
                "Demo 3 needs exactly two store IDs, for example: "
                "`Can Store B and Store E be compared for activity_transfer?`"
            ),
        }

    rows = load_rows(input_path)
    row = find_pair_row(rows, store_ids[0], store_ids[1], question_type)

    if row is None:
        return {
            "status": "pair_not_found",
            "requested_store_ids": store_ids,
            "comparison_question_type": question_type,
            "answer": (
                f"No Demo 3 pairwise row was found for {store_ids[0]}/{store_ids[1]} "
                f"with `{question_type}`. The current Demo 3 output only covers the existing Demo 2 B-F sample."
            ),
        }

    return build_supported_answer(row, store_ids, question_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a Demo 3 pairwise comparability question from CSV output.")
    parser.add_argument("message", help="Question to answer.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to demo3_pairwise_comparability_gate_output.csv.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full structured answer as JSON.",
    )
    args = parser.parse_args()

    result = answer_demo3_pairwise_question(args.message, Path(args.input))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])


if __name__ == "__main__":
    main()
