# Field Usage Review Before Comparability-Gate Expansion

This file records the field-name review before expanding the retail comparability narrative.

Current decision: no existing field is renamed in this patch.

The purpose of this review is to protect the Meituan backend metric contract. Existing backend-derived fields, SQL-derived diagnostic fields, and retrieval-facing memory slots should not be mixed or renamed casually.

## Field Review Table

| Existing field | Dictionary definition / boundary | Current use location | Rename? |
|---|---|---|---|
| store_id | Canonical store identifier used in source CSV files, SQL diagnostics, and metric outputs. | Source CSVs, SQL outputs, demo outputs. | No. |
| entity_id | Retrieval-layer identifier generated from store_id using entity_id = "store_" + store_id. | Generated retail memory facts. | No. |
| region_type | Coarse region or market-context label from available store evidence. It is not a fully normalized city/county/community classification. | Demo source data and Demo 2 comparability output. | No. |
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

## Patch Rule

This patch does not introduce new SQL output fields, new source CSV fields, or new canonical retail memory slots.

The new comparability-gate documentation may use conceptual terms such as comparable, partially comparable, or insufficient evidence, but those terms are not current SQL columns or memory slots.

If any of them later become implemented fields, they must first be documented in:

1. retail_ops/data/DATA_DICTIONARY.md
2. retail_ops/LINEAGE.md
3. SQL output documentation
4. generated memory fact logic
5. validation and evaluation cases
