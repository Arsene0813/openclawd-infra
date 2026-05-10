from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_ALIASES = {
    "store_visitors",
    "store_entry_count",
    "search_visitors",
    "activity_gross_revenue",
    "estimated_order_income",
    "paid_users",
    "paid_amount",
}

EXPECTED_SOURCE_HEADERS = [
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "region_type",
    "store_type",
    "transaction_amount",
    "transaction_orders",
    "valid_orders",
    "invalid_orders",
    "estimated_income_proxy",
    "average_order_value",
    "exposure_users",
    "exposure_times",
    "store_average_rank",
    "entry_conversion_rate_pct",
    "entry_users",
    "entry_times",
    "order_users",
    "order_times",
    "order_conversion_rate_pct",
    "order_amount",
    "payment_users",
    "payment_amount",
    "payment_conversion_rate_pct",
    "search_exposure_users",
    "search_average_rank",
    "search_entry_users",
    "merchant_list_exposure_users",
    "merchant_list_average_rank",
    "merchant_list_entry_users",
    "activity_zone_exposure_users",
    "activity_zone_entry_users",
    "order_page_exposure_users",
    "order_page_entry_users",
    "other_exposure_users",
    "other_entry_users",
    "activity_original_transaction_amount",
    "activity_orders",
    "activity_cost",
    "merchant_subsidy_amount",
    "platform_subsidy_amount",
    "activity_cost_ratio_pct",
    "refund_amount",
    "full_refund_orders",
    "refund_orders_all_or_partial",
]

EXPECTED_TOP_SKU_HEADERS = [
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "sku_rank",
    "sku_name",
    "sku_transaction_amount",
    "sales_volume",
    "sku_category_note",
]

EXPECTED_SQL_OUTPUT_HEADERS = [
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "transaction_amount",
    "transaction_orders",
    "valid_orders",
    "invalid_orders",
    "estimated_income_proxy",
    "average_order_value",
    "exposure_users",
    "exposure_times",
    "store_average_rank",
    "search_exposure_users",
    "search_average_rank",
    "entry_users",
    "entry_times",
    "search_entry_users",
    "order_users",
    "order_times",
    "order_amount",
    "payment_users",
    "payment_amount",
    "activity_original_transaction_amount",
    "activity_orders",
    "activity_cost",
    "merchant_subsidy_amount",
    "platform_subsidy_amount",
    "refund_amount",
    "full_refund_orders",
    "refund_orders_all_or_partial",
    "top3_sku_transaction_amount",
    "entry_conversion_rate_pct",
    "order_conversion_rate_pct",
    "payment_conversion_rate_pct",
    "search_exposure_share_pct",
    "search_entry_share_pct",
    "search_entry_rate_pct",
    "estimated_income_proxy_ratio_pct",
    "activity_order_share_pct",
    "activity_cost_ratio_pct",
    "merchant_subsidy_share_of_activity_cost_pct",
    "refund_pressure_pct",
    "refund_order_pressure_pct",
    "invalid_order_pressure_pct",
    "top3_sku_transaction_amount_share_pct",
    "transaction_amount_mom_pct",
    "transaction_orders_mom_pct",
    "estimated_income_proxy_mom_pct",
    "exposure_users_mom_pct",
    "search_exposure_users_mom_pct",
    "entry_users_mom_pct",
    "search_entry_users_mom_pct",
    "order_users_mom_pct",
    "payment_users_mom_pct",
    "average_order_value_mom_pct",
    "refund_amount_mom_pct",
    "store_average_rank_change",
    "search_average_rank_change",
    "transaction_recovered_with_conversion_aov_tradeoff",
    "refund_pressure_improved",
]

METADATA_FIELDS = {
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "region_type",
    "store_type",
}

# These are SQL diagnostic outputs, not direct Meituan backend labels.
# They are allowed in SQL output but should not be silently reused as backend source fields.
# They should support comparison-preparation and operating-signal profiles, not unsupported fixed operating labels.
SQL_DIAGNOSTIC_OUTPUT_FIELDS = {
    "search_exposure_share_pct",
    "search_entry_share_pct",
    "search_entry_rate_pct",
    "estimated_income_proxy_ratio_pct",
    "activity_order_share_pct",
    "merchant_subsidy_share_of_activity_cost_pct",
    "refund_pressure_pct",
    "refund_order_pressure_pct",
    "invalid_order_pressure_pct",
    "top3_sku_transaction_amount_share_pct",
    "transaction_amount_mom_pct",
    "transaction_orders_mom_pct",
    "estimated_income_proxy_mom_pct",
    "exposure_users_mom_pct",
    "search_exposure_users_mom_pct",
    "entry_users_mom_pct",
    "search_entry_users_mom_pct",
    "order_users_mom_pct",
    "payment_users_mom_pct",
    "average_order_value_mom_pct",
    "refund_amount_mom_pct",
    "store_average_rank_change",
    "search_average_rank_change",
    "transaction_recovered_with_conversion_aov_tradeoff",
    "refund_pressure_improved",
}

CANONICAL_RETAIL_SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "order_quality_pressure_profile",
    "single_metric_attribution_guard",
    "top3_sku_product_mix_note",
}

REQUIRED_FILES = [
    "retail_ops/data/DATA_DICTIONARY.md",
    "retail_ops/LINEAGE.md",

    "retail_ops/data/store_a_monthly_metrics.csv",
    "retail_ops/data/store_a_top_skus.csv",
    "retail_ops/sql/01_store_a_month_over_month_diagnostic.sql",
    "retail_ops/outputs/store_a_demo1_sql_output.csv",
    "retail_ops/outputs/store_a_demo1_interpretation_summary.csv",
    "retail_ops/outputs/generated_retail_memory_facts.json",
]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def read_csv_headers(relative_path: str) -> list[str]:
    with (ROOT / relative_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def read_csv_rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def extract_backticked_fields(text: str) -> set[str]:
    return set(re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", text))


def load_json(relative_path: str) -> Any:
    return json.loads(read_text(relative_path))


def validate_exact_headers(
    *,
    failures: list[str],
    label: str,
    actual: list[str],
    expected: list[str],
) -> None:
    if actual != expected:
        failures.append(
            f"{label} header mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def expected_entity_id_from_store_id(store_id: str) -> str:
    return f"store_{store_id}"


def collect_store_ids(rows: list[dict[str, str]]) -> set[str]:
    return {row["store_id"] for row in rows if row.get("store_id")}


def main() -> int:
    failures: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).exists():
            failures.append(f"Missing required file: {relative_path}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    dictionary = read_text("retail_ops/data/DATA_DICTIONARY.md")
    lineage = read_text("retail_ops/LINEAGE.md")
    slot_contract = read_text("retail_ops/data/DATA_DICTIONARY.md")
    sql = read_text("retail_ops/sql/01_store_a_month_over_month_diagnostic.sql")

    source_headers = read_csv_headers("retail_ops/data/store_a_monthly_metrics.csv")
    top_sku_headers = read_csv_headers("retail_ops/data/store_a_top_skus.csv")
    output_headers = read_csv_headers("retail_ops/outputs/store_a_demo1_sql_output.csv")
    interpretation_headers = read_csv_headers(
        "retail_ops/outputs/store_a_demo1_interpretation_summary.csv"
    )

    source_rows = read_csv_rows("retail_ops/data/store_a_monthly_metrics.csv")
    top_sku_rows = read_csv_rows("retail_ops/data/store_a_top_skus.csv")
    output_rows = read_csv_rows("retail_ops/outputs/store_a_demo1_sql_output.csv")

    source_header_set = set(source_headers)
    top_sku_header_set = set(top_sku_headers)
    output_header_set = set(output_headers)

    documented_fields = extract_backticked_fields(dictionary)

    validate_exact_headers(
        failures=failures,
        label="store_a_monthly_metrics.csv",
        actual=source_headers,
        expected=EXPECTED_SOURCE_HEADERS,
    )
    validate_exact_headers(
        failures=failures,
        label="store_a_top_skus.csv",
        actual=top_sku_headers,
        expected=EXPECTED_TOP_SKU_HEADERS,
    )
    validate_exact_headers(
        failures=failures,
        label="store_a_demo1_sql_output.csv",
        actual=output_headers,
        expected=EXPECTED_SQL_OUTPUT_HEADERS,
    )

    if "## Store Entity ID Convention / 门店实体 ID 规则" not in dictionary:
        failures.append("DATA_DICTIONARY.md missing Store Entity ID Convention section")

    if '`entity_id = "store_" + store_id`' not in dictionary:
        failures.append('DATA_DICTIONARY.md missing exact entity_id generation rule: `entity_id = "store_" + store_id`')

    if "## Source Metrics vs SQL-Derived Diagnostics / 后台原始指标与 SQL 派生诊断边界" not in dictionary:
        failures.append("DATA_DICTIONARY.md missing Source Metrics vs SQL-Derived Diagnostics boundary section")

    required_boundary_terms = [
        "Most canonical fields in this dictionary are normalized representations of metrics observed directly from the Meituan merchant backend.",
        "The current Demo 1 SQL does not claim to create or infer those backend metrics.",
        "order_conversion_rate_pct",
        "valid_orders / entry_users",
    ]

    for term in required_boundary_terms:
        if term not in dictionary:
            failures.append(f"DATA_DICTIONARY.md missing source-vs-SQL boundary term: {term}")

    # Source and top-SKU fields must either be metadata or documented in DATA_DICTIONARY.
    for field in source_header_set | top_sku_header_set:
        if field in METADATA_FIELDS:
            continue
        if field not in documented_fields:
            failures.append(f"Field `{field}` appears in source CSVs but is not documented in DATA_DICTIONARY.md")

    # SQL output fields must be documented, metadata, or explicitly classified as SQL diagnostic outputs.
    for field in output_header_set:
        if field in METADATA_FIELDS or field in SQL_DIAGNOSTIC_OUTPUT_FIELDS:
            continue
        if field not in documented_fields:
            failures.append(f"SQL output field `{field}` is not documented or classified as a diagnostic output")

    # Required canonical fields must appear in source/output and in SQL.
    required_canonical_fields = [
        "entry_users",
        "entry_times",
        "search_entry_users",
        "payment_users",
        "payment_amount",
        "activity_original_transaction_amount",
        "estimated_income_proxy",
        "exposure_users",
        "exposure_times",
        "order_users",
        "order_times",
        "order_amount",
        "transaction_amount",
        "transaction_orders",
        "average_order_value",
        "activity_orders",
        "activity_cost",
        "merchant_subsidy_amount",
        "platform_subsidy_amount",
        "refund_amount",
        "full_refund_orders",
        "refund_orders_all_or_partial",
        "valid_orders",
        "invalid_orders",
    ]

    for field in required_canonical_fields:
        if field not in source_header_set and field not in output_header_set:
            failures.append(f"Required canonical field `{field}` missing from source/output CSV headers")
        if field not in documented_fields:
            failures.append(f"Required canonical field `{field}` missing from DATA_DICTIONARY.md")
        if field not in sql:
            failures.append(f"Required canonical field `{field}` missing from SQL diagnostic")

    # Top-SKU guard: prevent store-level transaction_amount from being reused at SKU grain.
    forbidden_top_sku_fields = {"rank", "transaction_amount"}
    for field in forbidden_top_sku_fields:
        if field in top_sku_header_set:
            failures.append(f"Forbidden top-SKU field `{field}` found in store_a_top_skus.csv")

    # Forbidden aliases should not appear in tracked project text files, except inside validators.
    validator_paths = {
        (ROOT / "retail_ops/scripts/validate_retail_data_contract.py").resolve(),
        (ROOT / "scripts/validate_project_consistency.py").resolve(),
    }

    for path in tracked_files():
        if path.resolve() in validator_paths:
            continue
        if path.suffix.lower() not in {".md", ".py", ".sql", ".csv", ".json", ".txt", ".yml", ".yaml"}:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for alias in FORBIDDEN_ALIASES:
            if alias in text:
                failures.append(f"Forbidden alias `{alias}` found in {path.relative_to(ROOT)}")

    # DATA_DICTIONARY.md must contain every canonical retail memory slot.
    for slot in CANONICAL_RETAIL_SLOTS:
        if f"`{slot}`" not in slot_contract:
            failures.append(f"Canonical slot `{slot}` missing from DATA_DICTIONARY.md")

    # Generated facts must use canonical slots, valid entity_id mapping, and valid source fields.
    facts = load_json("retail_ops/outputs/generated_retail_memory_facts.json")
    if not isinstance(facts, list):
        failures.append("generated_retail_memory_facts.json must contain a list of facts")
        facts = []

    known_store_ids = collect_store_ids(source_rows) | collect_store_ids(top_sku_rows) | collect_store_ids(output_rows)
    expected_entity_ids = {expected_entity_id_from_store_id(store_id) for store_id in known_store_ids}

    csv_headers_by_path = {
        "retail_ops/data/store_a_monthly_metrics.csv": source_header_set,
        "retail_ops/data/store_a_top_skus.csv": top_sku_header_set,
        "retail_ops/outputs/store_a_demo1_sql_output.csv": output_header_set,
    }

    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            failures.append(f"Fact at index {index} is not an object")
            continue

        entity_id = fact.get("entity_id")
        slot = fact.get("slot")
        source_path = fact.get("source_path")
        source_fields = fact.get("source_fields", [])
        period_label = fact.get("period_label")
        period_start = fact.get("period_start")
        period_end = fact.get("period_end")
        period_granularity = fact.get("period_granularity")

        if not period_label or not period_start or not period_end:
            failures.append(
                f"Fact slot `{slot}` must include period_label, period_start, and period_end"
            )

        if period_label and "_to_" in period_label:
            if period_granularity != "month_range":
                failures.append(
                    f"Fact slot `{slot}` uses range period_label `{period_label}` "
                    f"but period_granularity is `{period_granularity}`, expected `month_range`"
                )

        if period_label and "_to_" not in period_label:
            if period_granularity not in {"month", "unknown"}:
                failures.append(
                    f"Fact slot `{slot}` uses non-range period_label `{period_label}` "
                    f"but period_granularity is `{period_granularity}`"
                )


        if entity_id not in expected_entity_ids:
            failures.append(
                f"Fact slot `{slot}` has invalid entity_id `{entity_id}`. "
                f"Expected one of {sorted(expected_entity_ids)} from store_id mapping."
            )

        if slot not in CANONICAL_RETAIL_SLOTS:
            failures.append(f"Fact has non-canonical slot `{slot}`")

        if source_path not in csv_headers_by_path:
            failures.append(f"Fact slot `{slot}` uses unsupported or unknown source_path `{source_path}`")
        else:
            valid_headers = csv_headers_by_path[source_path]
            for field in source_fields:
                if field not in valid_headers:
                    failures.append(
                        f"Fact slot `{slot}` references source_field `{field}` "
                        f"not present in `{source_path}`"
                    )

        if fact.get("lineage_path") and fact["lineage_path"] != "retail_ops/LINEAGE.md":
            failures.append(f"Fact slot `{slot}` uses unexpected lineage_path `{fact['lineage_path']}`")

    # Interpretation summary may be a reporting subset of generated facts,
    # but any declared summary_type/slot must still be canonical.
    interpretation_rows = read_csv_rows("retail_ops/outputs/store_a_demo1_interpretation_summary.csv")
    if interpretation_headers:
        summary_key = None
        if "summary_type" in interpretation_headers:
            summary_key = "summary_type"
        elif "slot" in interpretation_headers:
            summary_key = "slot"

        if summary_key is None:
            failures.append(
                "store_a_demo1_interpretation_summary.csv must include either `summary_type` or `slot` "
                "so its reporting rows can be checked against canonical slots"
            )
        else:
            for row in interpretation_rows:
                summary_value = row.get(summary_key, "")
                if summary_value and summary_value not in CANONICAL_RETAIL_SLOTS:
                    failures.append(
                        f"Interpretation summary uses non-canonical {summary_key} `{summary_value}`"
                    )

    # Core lineage files should mention the most important consistency guardrails.
    lineage_required_terms = [
        "visibility_entry_profile",
        "activity_lever_profile",
        "transaction_conversion_profile",
        "order_quality_pressure_profile",
        "single_metric_attribution_guard",
        "top3_sku_product_mix_note",
    ]
    for term in lineage_required_terms:
        if term not in lineage:
            failures.append(f"LINEAGE.md missing required retail lineage term `{term}`")

    if failures:
        print("Retail data contract validation FAILED.")
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("Retail data contract validation PASSED.")
    print(f"Checked source CSV headers: {len(source_headers)}")
    print(f"Checked top-SKU CSV headers: {len(top_sku_headers)}")
    print(f"Checked SQL output headers: {len(output_headers)}")
    print(f"Checked generated retail memory facts: {len(facts)}")
    print(f"Checked known store_id values: {sorted(known_store_ids)}")
    print(f"Checked expected entity_id values: {sorted(expected_entity_ids)}")
    print("No forbidden alias fields found.")
    print("Entity ID convention is documented and validated.")
    print("Retail fact period metadata is documented and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
