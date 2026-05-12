# Field Usage Review Before Comparability-Gate Expansion

This file records the field-name review before expanding the retail comparability narrative.

Current decision: no existing field is renamed in this patch.

The purpose of this review is to protect the Meituan backend metric contract. Existing backend-derived fields, SQL-derived diagnostic fields, and retrieval-facing memory slots should not be mixed or renamed casually.

## Field Review Table

| Existing field | Dictionary definition / boundary | Current use location | Rename? |
|---|---|---|---|
| store_id | Canonical store identifier used in source CSV files, SQL diagnostics, and metric outputs. | Source CSVs, SQL outputs, demo outputs. | No. |
| entity_id | Retrieval-layer identifier generated from store_id using entity_id = "store_" + store_id. | Generated retail memory facts. | No. |
| region_type | Weak region or market-context metadata from available store evidence. It is not a store-stage label, not a mature market-area classification, and not a sufficient comparability condition by itself. | Demo source data, Demo 2 comparability output, Demo 3 pairwise context note. | No. Keep unchanged. |
| store_type | Store metadata field used to separate operating formats. | Source CSVs and SQL output. | No. |
| business_district_rank | Backend-provided ranking-related field when available in the Demo 2 source. | Demo 2 source/output evidence. | No. |
| transaction_amount | Transaction amount for same-day paid and same-day not-cancelled orders, following the backend transaction metric page. | Source CSVs, SQL output, transaction/conversion profile. | No. |
| transaction_orders | Transaction order count for same-day paid and same-day not-cancelled orders. | Source CSVs, SQL output, transaction/conversion profile. | No. |
| average_order_value | Backend transaction metric page average order value, read with transaction_amount and transaction_orders. | Source CSVs, SQL output. | No. |
| estimated_income_proxy | Platform-displayed estimated income / estimated order income proxy. It is not audited profit. | Source CSVs, SQL output, admissions summary boundary. | No. |
| entry_users | Backend-reported users entering the store during the selected period. | Source CSVs, SQL output, visibility/entry profile. | No. |
| search_entry_users | Backend-reported users entering from search during the selected period. | Source CSVs, SQL output, search-entry diagnostics. | No. |
| order_users | Backend-reported users who submitted orders during the selected period. | Source CSVs, SQL output, order conversion definition. | No. |
| order_conversion_rate_pct | Backend formula: order_users / entry_users * 100. It must not be recomputed as valid_orders / entry_users. | Source CSVs, SQL output, lineage, admissions summary. | No. |
| payment_users | Backend-reported users who submitted and successfully paid. | Source CSVs, SQL output. | No. |
| payment_amount | Backend-reported commodity paid amount for paid orders. | Source CSVs, SQL output. | No. |
| activity_original_transaction_amount | Original transaction amount of orders that enjoyed activities. | Source CSVs, SQL output, activity profile. | No. |
| activity_orders | Number of orders brought by marketing activities. | Source CSVs, SQL output, activity-order share. | No. |
| activity_cost | merchant_subsidy_amount + platform_subsidy_amount. | Source CSVs, SQL output, activity profile. | No. |
| merchant_subsidy_amount | Merchant-borne promotion subsidy. | Source CSVs, SQL output. | No. |
| platform_subsidy_amount | Platform-borne promotion subsidy. | Source CSVs, SQL output. | No. |
| activity_cost_ratio_pct | Backend-formula field: activity cost divided by activity original transaction amount. It should not be described as traditional ROI. | Source CSVs, SQL output, Demo 2 summary. | No. |
| refund_amount | Successful actual refund amount in the selected period, dated by refund success date. | Source CSVs, SQL output, order-quality profile. | No. |
| valid_orders | Accepted and not-cancelled order count. It is an order-status metric, not a user-funnel metric. | Source CSVs, SQL output. | No. |
| invalid_orders | Cancelled order count. | Source CSVs, SQL output, invalid-order pressure. | No. |
| sku_transaction_amount | SKU-level transaction amount. It must not be confused with store-level transaction_amount. | Top-SKU source files and top-SKU evidence. | No. |
| search_entry_rate_pct | SQL-derived diagnostic signal, not a raw Meituan backend metric. | SQL output and Demo 2 diagnostics. | No. |
| search_entry_share_pct | SQL-derived diagnostic signal for search-entry structure. | SQL output and visibility/entry profile. | No. |
| activity_order_share_pct | SQL-derived diagnostic signal for activity-order involvement. | SQL output and activity profile. | No. |
| refund_pressure_pct | SQL-derived diagnostic signal based on refund amount and transaction amount. | SQL output and order-quality profile. | No. |
| invalid_order_pressure_pct | SQL-derived diagnostic signal based on invalid and valid order counts. | SQL output and order-quality profile. | No. |
| top3_sku_transaction_amount_share_pct | SQL-derived lightweight product-mix evidence. It is not full product-category sales share. | SQL output and top-SKU memory note. | No. |
| comparison_scope_flag | SQL-derived data-readiness and comparison-scope guardrail for Demo 2. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| comparison_limit_notes | SQL-derived interpretation-boundary notes for Demo 2. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| period_granularity | Memory-fact metadata recording the time grain of the fact, currently `month`; it is not a Meituan backend metric. | Generated retail memory facts. | No. |
| visibility_entry_profile | Retrieval-facing memory slot for exposure, ranking, entry, and search-entry structure. | Generated retail memory facts. | No. |
| activity_lever_profile | Retrieval-facing memory slot for activity orders, activity cost, subsidy, and activity-cost ratio. | Generated retail memory facts. | No. |
| transaction_conversion_profile | Retrieval-facing memory slot for transaction scale, order conversion, payment, and average order value. | Generated retail memory facts. | No. |
| order_quality_pressure_profile | Retrieval-facing memory slot for refund pressure, refund-order pressure, invalid-order pressure, and related order-quality signals. | Generated retail memory facts. | No. |
| single_metric_attribution_guard | Retrieval-facing memory slot that prevents unsupported interpretation from one metric alone. | Generated retail memory facts. | No. |
| top3_sku_product_mix_note | Retrieval-facing memory slot for limited top-SKU evidence. | Generated retail memory facts. | No. |

## Demo 3 New Field Review Table

These fields are new SQL-derived pairwise comparability-gate outputs. They do not rename existing fields.

| Existing / SQL-derived field | Dictionary definition / boundary | Use location | Rename? |
|---|---|---|---|
| reference_store_id | First store in a pairwise comparison; uses existing store_id value. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| candidate_store_id | Second store in a pairwise comparison; uses existing store_id value. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| comparison_question_type | Narrow comparison question being tested: search_entry_structure, activity_transfer, or order_quality_pressure. | Demo 3 pairwise SQL output and eval. | No. Keep unchanged. |
| reference_region_type | Existing region_type value for the reference store in a pairwise output; weak context only, not market-area classification. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| candidate_region_type | Existing region_type value for the candidate store in a pairwise output; weak context only, not market-area classification. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| region_type_comparison_note | Compares existing region_type values while explicitly preserving that region_type is not a market-area classification. | Demo 3 pairwise SQL output and eval. | No. Keep unchanged. |
| reference_store_type | Existing store_type value for the reference store in a pairwise output; operating-format context, not a performance label. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| candidate_store_type | Existing store_type value for the candidate store in a pairwise output; operating-format context, not a performance label. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| store_type_comparison_note | Compares existing store_type values as operating-format context, not as a performance label. | Demo 3 pairwise SQL output and eval. | No. Keep unchanged. |
| search_entry_share_gap_pct | Absolute gap between two stores' search_entry_share_pct values. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| activity_order_share_gap_pct | Absolute gap between two stores' activity_order_share_pct values. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| activity_cost_ratio_gap_pct | Absolute gap between two stores' activity_cost_ratio_pct values. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| refund_pressure_gap_pct | Absolute gap between two stores' refund_pressure_pct values. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| invalid_order_pressure_gap_pct | Absolute gap between two stores' invalid_order_pressure_pct values. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| top3_sku_concentration_gap_pct | Absolute gap between two stores' top3_sku_transaction_amount_share_pct values; still not full product-category share. | Demo 3 pairwise SQL output. | No. Keep unchanged. |
| pairwise_comparison_decision | SQL-derived gate outcome for the selected comparison question. | Demo 3 pairwise SQL output and eval. | No. Keep unchanged. |
| pairwise_limit_notes | Interpretation notes for why the pair is comparable, partially comparable, or not comparable for strategy transfer. | Demo 3 pairwise SQL output and eval. | No. Keep unchanged. |

## Patch Rule

This patch does not rename existing fields, does not introduce new source CSV fields, and does not introduce new canonical retail memory slots.

This patch does introduce documented Demo 3 SQL-derived pairwise output fields. Those fields are listed in DATA_DICTIONARY.md and in the Demo 3 New Field Review Table above before being used in SQL output, validation, or evaluation.

Conceptual terms such as comparable, partially comparable, or not comparable for strategy transfer are now implemented only as Demo 3 pairwise gate outputs. They must not be treated as store-stage labels or final operating recommendations.

## Field Rename Gate for Future Changes

Any future field-name change must pass this table before implementation. The current decision is to keep all implemented canonical names unchanged and preserve the definitions in `retail_ops/data/DATA_DICTIONARY.md`.

| Existing field | Dictionary / implemented definition boundary | Current use position | Rename decision |
|---|---|---|---|
| `store_id` | Canonical store identifier used in source CSV files, SQL diagnostics, and metric outputs. | Source CSVs, SQL outputs, validation scripts. | No. Keep unchanged. |
| `entity_id` | Retrieval-layer identifier used in generated retail memory facts; current convention is `store_` + `store_id`. | Generated retail memory facts and retrieval/evaluation logic. | No. Keep unchanged. |
| `region_type` | Coarse operating-context label used for current staged comparison; not a full geographic segmentation model. | Demo 2 source metrics, SQL output, generated facts, comparability review. | No. Keep unchanged. |
| `store_type` | Store operating-type field used as a comparison boundary. | Source metrics, SQL output, generated facts, comparability review. | No. Keep unchanged. |
| `business_district_rank` | Meituan backend ranking field; not a global market ranking. | Source metrics and SQL output. | No. Keep unchanged. |
| `exposure_users` | Backend exposure-user metric. | Source metrics, SQL output, visibility memory facts. | No. Keep unchanged. |
| `entry_users` | Backend entry-user metric. | Source metrics, SQL output, visibility and conversion memory facts. | No. Keep unchanged. |
| `search_entry_users` | Backend search-entry-user metric. | Source metrics, SQL output, visibility memory facts. | No. Keep unchanged. |
| `order_users` | Backend order-user metric used in the backend order-conversion formula. | Source metrics, SQL output, transaction/conversion facts. | No. Keep unchanged. |
| `order_conversion_rate_pct` | Backend formula field: `order_users / entry_users * 100`; must not be recomputed from `valid_orders / entry_users`. | Source metrics, SQL output, transaction/conversion facts, lineage. | No. Keep unchanged. |
| `payment_users` | Backend successful-payment-user metric. | Source metrics, SQL output, transaction/conversion facts. | No. Keep unchanged. |
| `payment_conversion_rate_pct` | Backend payment-conversion metric. | Source metrics, SQL output, transaction/conversion facts. | No. Keep unchanged. |
| `transaction_amount` | Backend transaction amount for the selected period and scope. | Source metrics, SQL output, transaction/conversion facts. | No. Keep unchanged. |
| `transaction_orders` | Backend transaction-order count for the selected period and scope. | Source metrics, SQL output, activity and transaction facts. | No. Keep unchanged. |
| `average_order_value` | Backend average-order-value field. | Source metrics, SQL output, transaction/conversion facts. | No. Keep unchanged. |
| `estimated_income_proxy` | Platform-displayed estimated income / estimated order income proxy; not audited profit. | Source metrics, SQL output, transaction/conversion facts, evidence-boundary docs. | No. Keep unchanged. |
| `activity_original_transaction_amount` | Backend original transaction amount for orders that used activities. | Source metrics, SQL output, activity facts. | No. Keep unchanged. |
| `activity_orders` | Backend activity-driven order count. | Source metrics, SQL output, activity facts. | No. Keep unchanged. |
| `activity_cost` | Backend activity cost field. | Source metrics, SQL output, activity facts. | No. Keep unchanged. |
| `merchant_subsidy_amount` | Merchant-borne subsidy amount. | Source metrics, SQL output, activity facts. | No. Keep unchanged. |
| `platform_subsidy_amount` | Platform-borne subsidy amount. | Source metrics, SQL output, activity facts. | No. Keep unchanged. |
| `activity_cost_ratio_pct` | Activity cost divided by activity original transaction amount; activity-cost-ratio evidence, not traditional ROI. | Source metrics, SQL output, activity facts, lineage. | No. Keep unchanged. |
| `activity_order_share_pct` | SQL-derived activity-order share. | Demo 2 SQL output, generated facts, comparability review. | No. Keep unchanged. |
| `refund_amount` | Backend refund amount counted by refund-success date. | Source metrics, SQL output, order-quality facts. | No. Keep unchanged. |
| `valid_orders` | Backend valid-order count; order-status metric, not user-level order-conversion denominator. | Source metrics, SQL output, order-quality facts. | No. Keep unchanged. |
| `invalid_orders` | Backend invalid-order count. | Source metrics, SQL output, order-quality facts. | No. Keep unchanged. |
| `refund_pressure_pct` | SQL-derived refund-pressure signal. | SQL output, order-quality facts, comparability review. | No. Keep unchanged. |
| `invalid_order_pressure_pct` | SQL-derived invalid-order-pressure signal. | SQL output, order-quality facts, comparability review. | No. Keep unchanged. |
| `sku_transaction_amount` | SKU-level transaction amount; not store-period total transaction amount. | Top-SKU source tables and product-mix facts. | No. Keep unchanged. |
| `top3_sku_transaction_amount_share_pct` | SQL-derived top-three SKU concentration signal; not full product-category sales share. | SQL output, top-SKU facts, evidence-boundary docs. | No. Keep unchanged. |
| `comparison_scope_flag` | SQL-derived comparison-scope flag for current Demo 2 diagnostic readiness. | Demo 2 SQL output, generated facts, comparability-gate eval. | No. Keep unchanged. |
| `comparison_limit_notes` | SQL-derived comparison-limit notes. | Demo 2 SQL output, generated facts, comparability-gate eval. | No. Keep unchanged. |
| `period_start` | Memory-fact period metadata; not a direct Meituan backend metric. | Generated retail memory facts. | No. Keep unchanged. |
| `period_end` | Memory-fact period metadata; not a direct Meituan backend metric. | Generated retail memory facts. | No. Keep unchanged. |
| `period_granularity` | Memory-fact period metadata; not a direct Meituan backend metric. | Generated retail memory facts. | No. Keep unchanged. |
| `visibility_entry_profile` | Canonical retail memory slot. | Generated retail memory facts and retail evals. | No. Keep unchanged. |
| `activity_lever_profile` | Canonical retail memory slot. | Generated retail memory facts and retail evals. | No. Keep unchanged. |
| `transaction_conversion_profile` | Canonical retail memory slot. | Generated retail memory facts and retail evals. | No. Keep unchanged. |
| `order_quality_pressure_profile` | Canonical retail memory slot for refund and invalid-order pressure. | Generated retail memory facts and retail evals. | No. Keep unchanged. |
| `single_metric_attribution_guard` | Canonical retail memory slot for attribution boundary. | Generated retail memory facts and retail evals. | No. Keep unchanged. |
| `top3_sku_product_mix_note` | Canonical retail memory slot for lightweight top-SKU evidence. | Generated retail memory facts and retail evals. | No. Keep unchanged. |

## Demo 2 Carry-Through Review: `order_amount` and `payment_amount`

This review patch does not rename any field.

`order_amount` and `payment_amount` are already defined in `retail_ops/data/DATA_DICTIONARY.md` and already exist in `retail_ops/data/demo2_store_period_metrics.csv`.

This patch carries them through the Demo 2 SQL output and Demo 2 `transaction_conversion_profile` memory facts so that the order-submission and payment-funnel amount fields remain visible alongside `order_users`, `payment_users`, `order_conversion_rate_pct`, and `payment_conversion_rate_pct`.

| Existing field | Dictionary definition / boundary | Current use location after this patch | Rename? |
|---|---|---|---|
| `order_amount` | Backend-reported total actual paid commodity amount of submitted orders during the selected period. | Demo 2 source CSV, Demo 2 SQL output, Demo 2 `transaction_conversion_profile` memory facts. | No. Keep unchanged. |
| `payment_amount` | Backend-reported total actual paid commodity amount of paid orders during the selected period. | Demo 2 source CSV, Demo 2 SQL output, Demo 2 `transaction_conversion_profile` memory facts. | No. Keep unchanged. |

These fields should not be confused with `transaction_amount`.

`order_amount` and `payment_amount` belong to the order-submission and payment funnel view, while `transaction_amount` follows the backend transaction metric definition documented in the dictionary.

## Period Granularity Field Review

This review patch does not rename any field.

`period_granularity` is a documentation and validation field used to describe the reporting level of a row, such as monthly store-period data. It is not a replacement for `period_start`, `period_end`, `store_id`, or any Meituan backend metric.

| Existing / SQL-derived field | Dictionary definition / boundary | Use location | Rename? |
|---|---|---|---|
| `period_granularity` | Describes the reporting granularity of the row, for example a monthly store-period row. | Data dictionary, Demo 2 comparability-gate consistency evaluation, documentation boundary checks. | No. Keep unchanged. |

`period_granularity` should be used only to clarify the time/reporting level of a row. It should not be used to merge mismatched periods or to imply that two stores are comparable when activity intensity, refunds, ranking, or competition may differ.
