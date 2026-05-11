import csv
import json
from pathlib import Path

OUTPUT_DIR = Path("retail_ops/outputs")
DATA_DIR = Path("retail_ops/data")

COMPARABILITY_OUTPUT = OUTPUT_DIR / "demo2_cross_store_comparability_output.csv"
TOP_SEARCH_TERMS = DATA_DIR / "demo2_top_search_terms.csv"
TOP_SKUS_BY_AMOUNT = DATA_DIR / "demo2_top_skus_by_transaction_amount.csv"

OUTPUT_PATH = OUTPUT_DIR / "generated_demo2_retail_memory_facts.json"

PERIOD_LABEL = "2026-03"
PERIOD_START = "2026-03-01"
PERIOD_END = "2026-03-31"

SOURCE_PATH = "retail_ops/outputs/demo2_cross_store_comparability_output.csv"
LINEAGE_PATH = "retail_ops/LINEAGE.md"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    return round(float(value), 2)


def as_int(value):
    return int(float(value))


def group_by_store(rows):
    grouped = {}

    for row in rows:
        grouped.setdefault(row["store_id"], []).append(row)

    return grouped


comparability_rows = read_csv(COMPARABILITY_OUTPUT)
top_search_by_store = group_by_store(read_csv(TOP_SEARCH_TERMS))
top_skus_by_store = group_by_store(read_csv(TOP_SKUS_BY_AMOUNT))

facts = []

for row in comparability_rows:
    store_id = row["store_id"]
    entity_id = f"store_{store_id}"

    top_search_terms = [
        {
            "search_term": item["search_term"],
            "search_term_en": item["search_term_en"],
            "search_term_exposure_times": as_int(item["search_term_exposure_times"]),
            "search_term_click_times": as_int(item["search_term_click_times"]),
            "search_term_order_times": as_int(item["search_term_order_times"]),
        }
        for item in top_search_by_store.get(store_id, [])
    ]

    top_skus_by_amount = [
        {
            "sku_name": item["sku_name"],
            "sku_name_en": item["sku_name_en"],
            "sku_transaction_amount": as_float(item["sku_transaction_amount"]),
        }
        for item in top_skus_by_store.get(store_id, [])
    ]

    common_limitations = [
        "cross-store demo",
        "same-period March 2026 comparison only",
        "traffic-source users may overlap",
        "not causal attribution",
        "compare with region, store type, activity, and refund limits",
    ]

    facts.append(
        {
            "kind": "retail_memory_fact",
            "type": "retail_metric_profile",
            "entity_id": entity_id,
            "slot": "visibility_entry_profile",
            "period_label": PERIOD_LABEL,
            "period_start": PERIOD_START,
            "period_end": PERIOD_END,
            "value": (
                f"Store {store_id}'s March 2026 visibility and entry profile can be described from exposure, rank, entry, "
                f"and search-entry metrics. The record supports cautious comparison of whether the store was being seen "
                f"and entered, but it does not prove causal performance differences."
            ),
            "observed_values": {
                "region_type": row["region_type"],
                "store_type": row["store_type"],
                "exposure_users": as_int(row["exposure_users"]),
                "exposure_times": as_int(row["exposure_times"]),
                "store_average_rank": as_float(row["store_average_rank"]),
                "entry_users": as_int(row["entry_users"]),
                "entry_conversion_rate_pct": as_float(row["entry_conversion_rate_pct"]),
                "search_exposure_users": as_int(row["search_exposure_users"]),
                "search_average_rank": as_float(row["search_average_rank"]),
                "search_entry_users": as_int(row["search_entry_users"]),
                "search_entry_rate_pct": as_float(row["search_entry_rate_pct"]),
                "search_entry_share_pct": as_float(row["search_entry_share_pct"]),
                "top_search_terms": top_search_terms,
            },
            "calculation": "search_entry_rate_pct = search_entry_users / search_exposure_users * 100; search_entry_share_pct = search_entry_users / entry_users * 100",
            "source_fields": [
                "exposure_users",
                "exposure_times",
                "store_average_rank",
                "entry_users",
                "entry_conversion_rate_pct",
                "search_exposure_users",
                "search_average_rank",
                "search_entry_users",
                "search_entry_rate_pct",
                "search_entry_share_pct",
                "search_term",
                "search_term_en",
                "search_term_exposure_times",
                "search_term_click_times",
                "search_term_order_times",
            ],
            "confidence": "high",
            "source_path": SOURCE_PATH,
            "lineage_path": LINEAGE_PATH,
            "limitations": common_limitations,
            "is_active": True,
            "period_granularity": "month",
        }
    )

    facts.append(
        {
            "kind": "retail_memory_fact",
            "type": "retail_metric_profile",
            "entity_id": entity_id,
            "slot": "activity_lever_profile",
            "period_label": PERIOD_LABEL,
            "period_start": PERIOD_START,
            "period_end": PERIOD_END,
            "value": (
                f"Store {store_id}'s March 2026 activity metrics describe activity involvement and subsidy cost structure. "
                f"These metrics should be treated as operating-tool evidence, not as proof that activity caused the store's result."
            ),
            "observed_values": {
                "activity_original_transaction_amount": as_float(row["activity_original_transaction_amount"]),
                "activity_orders": as_int(row["activity_orders"]),
                "transaction_orders": as_int(row["transaction_orders"]),
                "activity_cost": as_float(row["activity_cost"]),
                "merchant_subsidy_amount": as_float(row["merchant_subsidy_amount"]),
                "platform_subsidy_amount": as_float(row["platform_subsidy_amount"]),
                "activity_cost_ratio_pct": as_float(row["activity_cost_ratio_pct"]),
                "activity_order_share_pct": as_float(row["activity_order_share_pct"]),
            },
            "calculation": "activity_order_share_pct = activity_orders / transaction_orders * 100; activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100",
            "source_fields": [
                "activity_original_transaction_amount",
                "activity_orders",
                "transaction_orders",
                "activity_cost",
                "merchant_subsidy_amount",
                "platform_subsidy_amount",
                "activity_cost_ratio_pct",
                "activity_order_share_pct",
            ],
            "confidence": "high",
            "source_path": SOURCE_PATH,
            "lineage_path": LINEAGE_PATH,
            "limitations": [
                "cross-store demo",
                "activity mechanism details are not included",
                "promotion cycle dates are unknown",
                "activity metrics describe tool usage, not causal proof",
                "short-term ROI may not be the primary target in every operating stage",
            ],
            "is_active": True,
            "period_granularity": "month",
        }
    )

    facts.append(
        {
            "kind": "retail_memory_fact",
            "type": "retail_metric_profile",
            "entity_id": entity_id,
            "slot": "order_quality_pressure_profile",
            "period_label": PERIOD_LABEL,
            "period_start": PERIOD_START,
            "period_end": PERIOD_END,
            "value": (
                f"Store {store_id}'s March 2026 refund and invalid-order metrics provide order-quality pressure signals. "
                f"They should constrain direct performance comparison when refund or invalid-order pressure is elevated."
            ),
            "observed_values": {
                "refund_amount": as_float(row["refund_amount"]),
                "transaction_amount": as_float(row["transaction_amount"]),
                "refund_pressure_pct": as_float(row["refund_pressure_pct"]),
                "valid_orders": as_int(row["valid_orders"]),
                "invalid_orders": as_int(row["invalid_orders"]),
                "transaction_orders": as_int(row["transaction_orders"]),
                "invalid_order_pressure_pct": as_float(row["invalid_order_pressure_pct"]),
                "full_refund_orders": as_int(row["full_refund_orders"]),
                "refund_orders_all_or_partial": as_int(row["refund_orders_all_or_partial"]),
            },
            "calculation": "refund_pressure_pct = refund_amount / transaction_amount * 100; invalid_order_pressure_pct = invalid_orders / transaction_orders * 100",
            "source_fields": [
                "refund_amount",
                "transaction_amount",
                "refund_pressure_pct",
                "valid_orders",
                "invalid_orders",
                "transaction_orders",
                "invalid_order_pressure_pct",
                "full_refund_orders",
                "refund_orders_all_or_partial",
            ],
            "confidence": "medium",
            "source_path": SOURCE_PATH,
            "lineage_path": LINEAGE_PATH,
            "limitations": [
                "cross-store demo",
                "refund amount is counted by refund-success date",
                "not an exact original-order cohort refund rate",
                "invalid order reason is not included",
            ],
            "is_active": True,
            "period_granularity": "month",
        }
    )

    facts.append(
        {
            "kind": "retail_memory_fact",
            "type": "retail_metric_profile",
            "entity_id": entity_id,
            "slot": "top3_sku_product_mix_note",
            "period_label": PERIOD_LABEL,
            "period_start": PERIOD_START,
            "period_end": PERIOD_END,
            "value": (
                f"Store {store_id}'s March 2026 top-3 SKU transaction-amount evidence is retained as lightweight product-mix evidence. "
                f"It should not be treated as full SKU category-share analysis."
            ),
            "observed_values": {
                "top3_sku_transaction_amount": as_float(row["top3_sku_transaction_amount"]),
                "transaction_amount": as_float(row["transaction_amount"]),
                "top3_sku_transaction_amount_share_pct": as_float(row["top3_sku_transaction_amount_share_pct"]),
                "top_skus_by_transaction_amount": top_skus_by_amount,
            },
            "calculation": "top3_sku_transaction_amount_share_pct = top3_sku_transaction_amount / transaction_amount * 100",
            "source_fields": [
                "sku_name",
                "sku_name_en",
                "sku_transaction_amount",
                "transaction_amount",
                "top3_sku_transaction_amount",
                "top3_sku_transaction_amount_share_pct",
                "sku_category_note",
            ],
            "confidence": "medium",
            "source_path": "retail_ops/data/demo2_top_skus_by_transaction_amount.csv",
            "lineage_path": LINEAGE_PATH,
            "limitations": [
                "top-3 SKU evidence only",
                "not full SKU category-share analysis",
                "manual category inference should not be overclaimed",
                "English SKU names are helper translations, not backend source values",
            ],
            "is_active": True,
            "period_granularity": "month",
        }
    )

    facts.append(
        {
            "kind": "retail_memory_fact",
            "type": "retail_metric_profile",
            "entity_id": entity_id,
            "slot": "single_metric_attribution_guard",
            "period_label": PERIOD_LABEL,
            "period_start": PERIOD_START,
            "period_end": PERIOD_END,
            "value": (
                f"Store {store_id}'s March 2026 metrics should not be interpreted from a single metric alone. "
                f"The comparison scope is {row['comparison_scope_flag']}, with limit notes: {row['comparison_limit_notes']}."
            ),
            "observed_values": {
                "transaction_amount": as_float(row["transaction_amount"]),
                "transaction_orders": as_int(row["transaction_orders"]),
                "average_order_value": as_float(row["average_order_value"]),
                "store_average_rank": as_float(row["store_average_rank"]),
                "entry_conversion_rate_pct": as_float(row["entry_conversion_rate_pct"]),
                "order_conversion_rate_pct": as_float(row["order_conversion_rate_pct"]),
                "search_entry_share_pct": as_float(row["search_entry_share_pct"]),
                "activity_order_share_pct": as_float(row["activity_order_share_pct"]),
                "activity_cost_ratio_pct": as_float(row["activity_cost_ratio_pct"]),
                "refund_pressure_pct": as_float(row["refund_pressure_pct"]),
                "invalid_order_pressure_pct": as_float(row["invalid_order_pressure_pct"]),
                "top3_sku_transaction_amount_share_pct": as_float(row["top3_sku_transaction_amount_share_pct"]),
                "comparison_scope_flag": row["comparison_scope_flag"],
                "comparison_limit_notes": row["comparison_limit_notes"],
            },
            "calculation": "comparison_limit_notes are derived from search, activity, refund, invalid-order, and top-3 SKU concentration thresholds in the Demo 2 SQL output",
            "source_fields": [
                "transaction_amount",
                "transaction_orders",
                "average_order_value",
                "store_average_rank",
                "entry_conversion_rate_pct",
                "order_conversion_rate_pct",
                "search_entry_share_pct",
                "activity_order_share_pct",
                "activity_cost_ratio_pct",
                "refund_pressure_pct",
                "invalid_order_pressure_pct",
                "top3_sku_transaction_amount_share_pct",
                "comparison_scope_flag",
                "comparison_limit_notes",
            ],
            "confidence": "high",
            "source_path": SOURCE_PATH,
            "lineage_path": LINEAGE_PATH,
            "limitations": common_limitations,
            "is_active": True,
            "period_granularity": "month",
        }
    )

OUTPUT_PATH.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"Wrote {len(facts)} facts to {OUTPUT_PATH}")
