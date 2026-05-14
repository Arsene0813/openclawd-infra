# Field Usage Review Before Comparability-Gate Expansion

This file records the field-name review before expanding the retail comparability narrative.

Current decision: **no existing field is renamed in this patch**.

The purpose of this review is to protect the Meituan backend metric contract. Backend-derived fields, SQL-derived diagnostic fields, and retrieval-facing memory slots should not be mixed or renamed casually.

## Field Review Table

| Existing field | Dictionary definition / boundary | Current use location | Rename? |
|---|---|---|---|
| `store_id` | Canonical store identifier used in source CSV files, SQL diagnostics, and metric outputs. | Source CSVs, SQL outputs, demo outputs. | No. |
| `entity_id` | Retrieval-layer identifier generated from `store_id` using `entity_id = "store_" + store_id`. | Generated retail memory facts. | No. |
| `region_type` | Weak region or market-context metadata. It is not a store-stage label, mature market-area classification, or sufficient comparability condition by itself. | Demo 2 source metrics, SQL output, generated facts, comparability review. | No. Keep unchanged. |
| `store_type` | Store operating-type field used as context for comparison. | Source CSVs, SQL output, generated facts. | No. |
| `transaction_amount` | Backend transaction amount for the selected store-period scope. It must not be mixed with `gross_revenue`, `estimated_income_proxy`, or SKU-level `sku_transaction_amount`. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `transaction_orders` | Backend transaction-order count for the selected store-period scope. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `average_order_value` | Backend average-order-value field read with `transaction_amount` and `transaction_orders`. | Source CSVs, SQL output. | No. |
| `estimated_income_proxy` | Platform-displayed estimated income / estimated order income proxy. It is not audited profit. | Source CSVs, SQL output, transaction/conversion facts, evidence-boundary docs. | No. |
| `entry_users` | Backend-reported users entering the store during the selected period. | Source CSVs, SQL output, visibility and conversion facts. | No. |
| `search_entry_users` | Backend-reported users entering from search during the selected period. | Source CSVs, SQL output, visibility facts. | No. |
| `order_users` | Backend order-user metric used in the backend order-conversion formula. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `order_conversion_rate_pct` | Backend formula field: `order_users / entry_users * 100`. It must not be recomputed from `valid_orders / entry_users`. | Source CSVs, SQL output, lineage, transaction/conversion facts. | No. |
| `payment_users` | Backend successful-payment-user metric. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `payment_conversion_rate_pct` | Backend payment-conversion metric. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `activity_original_transaction_amount` | Original transaction amount of orders that used activities. | Source CSVs, SQL output, activity facts. | No. |
| `activity_orders` | Backend activity-driven order count. | Source CSVs, SQL output, activity facts. | No. |
| `activity_cost` | Backend activity-cost field. | Source CSVs, SQL output, activity facts. | No. |
| `merchant_subsidy_amount` | Merchant-borne subsidy amount. | Source CSVs, SQL output, activity facts. | No. |
| `platform_subsidy_amount` | Platform-borne subsidy amount. | Source CSVs, SQL output, activity facts. | No. |
| `activity_cost_ratio_pct` | Activity cost divided by activity original transaction amount. It is activity-cost-ratio evidence, not traditional ROI. | Source CSVs, SQL output, activity facts, lineage. | No. |
| `activity_order_share_pct` | SQL-derived activity-order share. It shows activity involvement, not full campaign status or causal effect. | Demo 2 SQL output, generated facts, comparability review. | No. |
| `refund_amount` | Backend refund amount counted by refund-success date. | Source CSVs, SQL output, order-quality facts. | No. |
| `refund_pressure_pct` | SQL-derived refund-pressure signal based on `refund_amount / transaction_amount * 100`. | SQL output and order-quality facts. | No. |
| `valid_orders` | Backend accepted-and-not-cancelled order count. It is an order-status metric, not a user-funnel metric. | Source CSVs, SQL output. | No. |
| `invalid_orders` | Backend cancelled-order count. | Source CSVs, SQL output, invalid-order pressure. | No. |
| `invalid_order_pressure_pct` | SQL-derived invalid-order pressure based on `invalid_orders / (valid_orders + invalid_orders) * 100`. | SQL output and order-quality facts. | No. |
| `sku_transaction_amount` | SKU-level transaction amount. It must not be confused with store-level `transaction_amount`. | Top-SKU source files and top-SKU evidence. | No. |
| `top3_sku_transaction_amount_share_pct` | SQL-derived lightweight top-SKU concentration evidence. It is not full product-category sales share. | SQL output and top-SKU memory note. | No. |
| `comparison_scope_flag` | SQL-derived data-readiness and comparison-scope guardrail for Demo 2. It is not a pairwise store-comparability decision. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| `comparison_limit_notes` | SQL-derived interpretation-boundary notes for Demo 2. It records constraints from search, activity, refund, order-quality, region/store context, and product-mix evidence. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| `visibility_entry_profile` | Retrieval-facing memory slot for exposure, ranking, entry, and search-entry structure. | Generated retail memory facts. | No. |
| `activity_lever_profile` | Retrieval-facing memory slot for activity orders, activity cost, subsidy, and activity-cost ratio. | Generated retail memory facts. | No. |
| `transaction_conversion_profile` | Retrieval-facing memory slot for transaction scale, order conversion, payment, and average order value. | Generated retail memory facts. | No. |
| `order_quality_pressure_profile` | Retrieval-facing memory slot for refund pressure, refund-order pressure, invalid-order pressure, and related order-quality signals. | Generated retail memory facts. | No. |
| `single_metric_attribution_guard` | Retrieval-facing memory slot that prevents unsupported interpretation from one metric alone. | Generated retail memory facts. | No. |
| `top3_sku_product_mix_note` | Retrieval-facing memory slot for limited top-SKU evidence. It is not full category-share analysis. | Generated retail memory facts. | No. |

## Future Comparability-Gate Field Review

Pairwise comparability-gate fields are outside the current implemented retail scope.

A reliable future gate should consider transaction order volume, transaction amount, activity status, activity involvement, activity intensity, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

At the current sample size, `region_type` remains weak context only. It must not be used as a hard market-area classification, store-stage label, or peer-store grouping rule.

Possible future fields such as `market_area_type`, `market_area_type_source`, `market_area_type_confidence`, `comparison_question_type`, or `comparison_decision` should only be added after they are documented in `retail_ops/data/DATA_DICTIONARY.md` and linked through `retail_ops/LINEAGE.md`.

## Field Rename Gate for Future Changes

Any future field-name change must pass this review before implementation:

| Existing field | Dictionary definition | Current use location | Rename decision |
|---|---|---|---|
| Proposed field to change | Must be checked against `retail_ops/data/DATA_DICTIONARY.md`. | Must list CSV, SQL, output, memory-fact, lineage, README, admissions, and eval usage. | Do not rename unless the whole path is migrated together. |

Current decision: **no current source CSV field, SQL output field, generated memory slot, or evaluation field is renamed in this patch**.
