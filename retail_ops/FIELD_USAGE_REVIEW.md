# Field Usage Review Before Comparability-Gate Expansion

This file records the field-name review before expanding the retail comparability narrative.

Current decision: no existing field is renamed before comparability-gate expansion.

The purpose of this review is to protect the Meituan backend metric contract. Existing backend-derived fields, SQL-derived diagnostic fields, and retrieval-facing memory slots should not be mixed or renamed casually.

## Field Review Table

| Existing field | Dictionary definition / boundary | Current use location | Rename? |
|---|---|---|---|
| `store_id` | Canonical store identifier used in source CSV files, SQL diagnostics, and metric outputs. | Source CSVs, SQL outputs, demo outputs. | No. |
| `entity_id` | Retrieval-layer identifier generated from `store_id` using the convention `store_` + `store_id`. | Generated retail memory facts. | No. |
| `region_type` | Weak region or market-context metadata from available store evidence. It is not a store-stage label, not a mature market-area classification, and not a sufficient comparability condition by itself. | Demo 2 source metrics, SQL output, generated facts, comparability review. | No. |
| `store_type` | Store operating-type field used as comparison context. | Source metrics, SQL output, generated facts. | No. |
| `business_district_rank` | Backend ranking-related field when available. It is supplementary context, not a global market ranking. | Demo 2 source/output evidence. | No. |
| `transaction_amount` | Store-period transaction amount following the backend transaction metric definition. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `transaction_orders` | Store-period transaction order count following the backend transaction metric definition. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `average_order_value` | Backend average-order-value field read with `transaction_amount` and `transaction_orders`. | Source CSVs, SQL output. | No. |
| `estimated_income_proxy` | Platform-displayed estimated income proxy; not audited profit. | Source CSVs, SQL output, evidence-boundary docs. | No. |
| `entry_users` | Backend-reported users entering the store during the selected period. | Source CSVs, SQL output, visibility/entry facts. | No. |
| `search_entry_users` | Backend-reported users entering from search during the selected period. | Source CSVs, SQL output, search-entry diagnostics. | No. |
| `order_users` | Backend-reported users who submitted orders during the selected period. | Source CSVs, SQL output, order-conversion definition. | No. |
| `order_conversion_rate_pct` | Backend formula field: `order_users / entry_users * 100`; must not be recomputed from `valid_orders / entry_users`. | Source CSVs, SQL output, lineage, admissions summary. | No. |
| `payment_users` | Backend-reported users who submitted and successfully paid. | Source CSVs, SQL output. | No. |
| `payment_amount` | Backend-reported commodity paid amount for paid orders. | Source CSVs, SQL output. | No. |
| `payment_conversion_rate_pct` | Backend payment-conversion metric. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `activity_original_transaction_amount` | Original transaction amount of orders that used activities. | Source CSVs, SQL output, activity facts. | No. |
| `activity_orders` | Number of orders brought by marketing activities. | Source CSVs, SQL output, activity-order share. | No. |
| `activity_cost` | `merchant_subsidy_amount + platform_subsidy_amount`. | Source CSVs, SQL output, activity facts. | No. |
| `merchant_subsidy_amount` | Merchant-borne promotion subsidy. | Source CSVs, SQL output. | No. |
| `platform_subsidy_amount` | Platform-borne promotion subsidy. | Source CSVs, SQL output. | No. |
| `activity_cost_ratio_pct` | Activity cost divided by activity original transaction amount; activity-cost-ratio evidence, not traditional ROI. | Source CSVs, SQL output, activity facts. | No. |
| `activity_order_share_pct` | SQL-derived activity-order share. | SQL output and activity facts. | No. |
| `refund_amount` | Successful actual refund amount in the selected period, dated by refund success date. | Source CSVs, SQL output, order-quality facts. | No. |
| `valid_orders` | Accepted and not-cancelled order count; order-status metric, not user-level funnel numerator. | Source CSVs, SQL output. | No. |
| `invalid_orders` | Cancelled order count. | Source CSVs, SQL output, invalid-order pressure. | No. |
| `refund_pressure_pct` | SQL-derived refund-pressure signal based on refund amount and transaction amount. | SQL output and order-quality facts. | No. |
| `invalid_order_pressure_pct` | SQL-derived invalid-order-pressure signal based on invalid and valid order counts. | SQL output and order-quality facts. | No. |
| `sku_transaction_amount` | SKU-level transaction amount; must not be confused with store-level `transaction_amount`. | Top-SKU source files and top-SKU evidence. | No. |
| `top3_sku_transaction_amount_share_pct` | SQL-derived top-three SKU concentration signal; not full product-category sales share. | SQL output, top-SKU facts, evidence-boundary docs. | No. |
| `comparison_scope_flag` | SQL-derived data-readiness and comparison-scope guardrail for Demo 2. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| `comparison_limit_notes` | SQL-derived interpretation-boundary notes for Demo 2. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| `period_start` | Memory-fact period metadata; not a direct Meituan backend metric. | Generated retail memory facts. | No. |
| `period_end` | Memory-fact period metadata; not a direct Meituan backend metric. | Generated retail memory facts. | No. |
| `period_granularity` | Memory-fact period metadata; not a direct Meituan backend metric. | Generated retail memory facts. | No. |
| `visibility_entry_profile` | Retrieval-facing memory slot for exposure, ranking, entry, and search-entry structure. | Generated retail memory facts and retail evals. | No. |
| `activity_lever_profile` | Retrieval-facing memory slot for activity orders, activity cost, subsidy, and activity-cost ratio. | Generated retail memory facts and retail evals. | No. |
| `transaction_conversion_profile` | Retrieval-facing memory slot for transaction scale, order conversion, payment, and average order value. | Generated retail memory facts and retail evals. | No. |
| `order_quality_pressure_profile` | Retrieval-facing memory slot for refund pressure, invalid-order pressure, and related order-quality signals. | Generated retail memory facts and retail evals. | No. |
| `single_metric_attribution_guard` | Retrieval-facing memory slot that prevents unsupported interpretation from one metric alone. | Generated retail memory facts and retail evals. | No. |
| `top3_sku_product_mix_note` | Retrieval-facing memory slot for limited top-SKU evidence. | Generated retail memory facts and retail evals. | No. |

## Additional Current Fields to Protect Before Future Expansion

The fields below already appear in current Demo 2 outputs or generated memory facts, but they should not be promoted into new labels or renamed during future comparability-gate work.

| Existing field | Definition / boundary | Current use location | Rename? |
|---|---|---|---|
| `order_amount` | Backend order-submission amount field; read with `order_users`, `order_times`, and `order_conversion_rate_pct`; not profit and not the same as `transaction_amount`. | Demo 2 source metrics, SQL output, generated facts. | No. |
| `entry_conversion_rate_pct` | Backend-reported entry conversion rate; should be read as a backend metric, not recomputed from unrelated order-status fields. | Demo 2 source metrics and SQL output. | No. |
| `search_entry_rate_pct` | SQL-derived search-entry rate: `search_entry_users / search_exposure_users * 100`. | Demo 2 SQL output and visibility-entry facts. | No. |
| `search_entry_share_pct` | SQL-derived search-entry share: `search_entry_users / entry_users * 100`; directional source-entry structure only because traffic-source users may overlap. | Demo 2 SQL output and visibility-entry facts. | No. |
| `full_refund_orders` | Backend refund-order count field; used as order-quality context. | Demo 2 source metrics, SQL output, order-quality facts. | No. |
| `refund_orders_all_or_partial` | Backend refund-order count including full and partial refund cases; used as order-quality context. | Demo 2 source metrics, SQL output, order-quality facts. | No. |
| `top3_sku_transaction_amount` | Sum of top-three SKU transaction amount in the current lightweight product-mix evidence. | Demo 2 SQL output and top-SKU facts. | No. |
| `source_fields` | Retrieval-facing metadata listing fields supporting a generated memory fact. | Generated retail memory facts. | No. |
| `observed_values` | Retrieval-facing metadata carrying observed values used by a generated memory fact. | Generated retail memory facts. | No. |
| `calculation` | Retrieval-facing explanation of the calculation or backend formula used by a generated memory fact. | Generated retail memory facts. | No. |
| `confidence` | Evidence confidence for the generated memory fact; not causal confidence. | Generated retail memory facts. | No. |
| `limitations` | Retrieval-facing limitation list that constrains later answers. | Generated retail memory facts. | No. |
| `source_path` | Retrieval-facing path to the supporting output file. | Generated retail memory facts. | No. |
| `lineage_path` | Retrieval-facing path to lineage documentation. | Generated retail memory facts. | No. |

## Current Path-Level Naming Decision

The current SQL and output file names remain unchanged.

| Existing path | Reason to keep |
|---|---|
| `retail_ops/sql/02_demo2_cross_store_comparability.sql` | Already referenced by docs, outputs, and eval files. The file represents Demo 2 same-period diagnostic structure with comparability guardrails, not a finished pairwise gate. |
| `retail_ops/outputs/demo2_cross_store_comparability_output.csv` | Already used by generated Demo 2 memory facts and evaluations. |
| `retail_ops/outputs/retail_data_contract_validation_result.txt` | Canonical saved validation output. |

## Future Field Gate

Any future field-name change must pass this review format before implementation:

| Proposed field | Dictionary definition / boundary | Planned use location | Rename or new field decision |
|---|---|---|---|

Possible future market-context fields such as `market_area_type`, `market_area_type_source`, and `market_area_type_confidence` are not implemented in the current repository. They should only be added after broader store data and repeated reporting windows provide enough evidence to support the classification.

