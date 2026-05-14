# Demo 2: Same-Period Cross-Store Diagnostic

<!-- stable-demo2-scope-boundary -->
## Scope Boundary

Demo 2 is a same-period cross-store diagnostic, not a pairwise comparability gate.

`comparison_scope_flag` only means that a store-period row is ready for the current diagnostic structure. It does not mean that two stores are fully comparable for pricing, promotion, SKU, ranking, or strategy-transfer decisions.

`comparison_limit_notes` records interpretation limits such as activity involvement, search-entry dependence, refund pressure, invalid-order pressure, and SKU concentration.

Demo 2 organizes five same-period store-period records into a shared diagnostic structure.

The purpose is to place selected Meituan backend metrics under the same reporting window and data contract, while preserving the limits that a later pairwise comparability gate would need to respect.

At this stage, comparability means row-level same-period diagnostic readiness. It does not mean pairwise store matching, store ranking, or strategy-transfer approval.

## Purpose

This demo tests whether five anonymized instant-retail store records can be placed under the same reporting window and field contract before making any operating interpretation.

The purpose is to structure selected backend metrics into a same-period diagnostic format and record the limits that should constrain interpretation.

## Business Problem

Meituan's merchant backend provides detailed store-level metrics, but the backend is mainly designed for reviewing one store at a time.

With many stores, the harder problem is deciding which stores can be compared, under what conditions they can be compared, and which signals are strong enough to support cautious operating judgment.

In this project, instant-retail competition is understood through this operating chain:

    being seen -> being entered -> being ordered -> being selected again or maintaining share

Promotion, subsidy, price, SKU mix, ranking position, and fulfillment conditions are operating levers inside this chain. They should be interpreted through the store's current operating state and comparison limits, not as isolated goals.

## Scope

| Item | Value |
|---|---|
| Stores | B, C, D, E, F |
| Reporting window | 2026-03-01 to 2026-03-31 |
| Period label | 2026-03 |
| Source | Manually structured Meituan merchant-backend metrics for instant-retail store operations. |
| Processing method | Offline SQL diagnostic. |
| SQL file | `retail_ops/sql/02_demo2_cross_store_comparability.sql` |
| SQL output | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` |
| Memory facts | `retail_ops/outputs/generated_demo2_retail_memory_facts.json` |

Some source traffic-channel fields are retained in the structured source file but not carried into the current Demo 2 diagnostic output. Demo 2 focuses on selected same-period diagnostic signals rather than exhaustive traffic-source decomposition.

`region_type` is kept as weak regional context only. In the current sample, it should not be read as a mature market-area classification or as a hard comparability condition.

A future market-area field would require broader store coverage and external or data-supported evidence such as local consumption level, competitive density, price pressure, and SKU demand structure.

## What the SQL Checks

The SQL prepares a same-period diagnostic output with:

- period alignment;
- region and store-type context;
- exposure, ranking, entry, and search-entry structure;
- activity-order share and activity-cost structure;
- refund pressure;
- invalid-order pressure;
- top-3 SKU transaction-amount concentration;
- comparison-scope and comparison-limit notes.

## Key Diagnostic Fields

### `comparison_scope_flag`

This field records whether the row is inside the current Demo 2 comparison scope.

In the current Demo 2 output, all B-F stores use the same March 2026 reporting window and are marked:

    same_period_diagnostic_ready

This means the rows are ready for the current same-period diagnostic. It does not mean the stores are fully comparable in every business sense.

### `comparison_limit_notes`

This field records the main reasons why direct cross-store interpretation should be constrained.

Examples include:

- high search-entry dependence;
- high or moderate activity involvement;
- high or moderate refund pressure;
- high or moderate invalid-order pressure;
- top-3 SKU transaction-amount concentration;
- the need to compare with region, store type, activity, refund, order-quality, and product-mix limits.

These notes are interpretation guardrails.

## What This Demo Supports

This demo supports cautious same-period diagnostic comparison.

It can help identify whether a store's metrics should be read with extra caution because of activity involvement, search-entry dependence, refund pressure, invalid-order pressure, or SKU concentration.

## What This Demo Does Not Support

This demo does not support:

- ranking stores as simply best or worst;
- causal claims about why one store performed better than another;
- final subsidy, pricing, or SKU decisions;
- full 48-store grouping;
- profit or margin analysis;
- complete SKU category-share analysis;
- automated Meituan backend ingestion.

## Why This Matters for the Memory Layer

The memory layer should not answer cross-store questions by retrieving isolated metrics.

It should preserve each store's period, evidence, comparison scope, and interpretation limits.

For this reason, Demo 2 converts SQL diagnostics into generated retail memory facts using the existing retail slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `order_quality_pressure_profile`
- `top3_sku_product_mix_note`
- `single_metric_attribution_guard`

The memory facts are currently file-backed for Demo 2. This is enough to test the data contract, SQL diagnostic output, fact generation, and limitation-preserving answer behavior, but it is not yet a full 48-store decision-support system.

## What the Current Demo 2 Output Shows

The current output should be read as row-level diagnostic evidence, not as a pairwise store-comparability decision.

The raw `comparison_limit_notes` column below uses the contract strings produced by the SQL output and documented in `DATA_DICTIONARY.md`.

The readable summary is only a human-facing explanation.

| Store | Raw `comparison_limit_notes` from current output | Readable summary |
|---|---|---|
| B | `high_search_entry_dependence; high_activity_involvement; moderate_refund_pressure; compare_with_region_store_type_activity_refund_limits` | Search entry is highly dominant; activity involvement is high; refund pressure is moderate; comparison should stay limited by region, store type, activity, refund, order-quality, and product-mix context. |
| C | `moderate_activity_involvement; top3_sku_amount_concentration; compare_with_region_store_type_activity_refund_limits` | Activity involvement is moderate; top-3 SKU transaction amount is concentrated; comparison should stay limited by region, store type, activity, refund, order-quality, and product-mix context. |
| D | `high_search_entry_dependence; high_activity_involvement; compare_with_region_store_type_activity_refund_limits` | Search entry is highly dominant; activity involvement is high; comparison should stay limited by region, store type, activity, refund, order-quality, and product-mix context. |
| E | `high_search_entry_dependence; moderate_activity_involvement; high_refund_pressure; high_invalid_order_pressure; compare_with_region_store_type_activity_refund_limits` | Search entry is highly dominant; activity involvement is moderate; refund pressure and invalid-order pressure are high; comparison should stay limited by region, store type, activity, refund, order-quality, and product-mix context. |
| F | `high_search_entry_dependence; high_activity_involvement; moderate_refund_pressure; moderate_invalid_order_pressure; compare_with_region_store_type_activity_refund_limits` | Search entry is highly dominant; activity involvement is high; refund pressure and invalid-order pressure are moderate; comparison should stay limited by region, store type, activity, refund, order-quality, and product-mix context. |

This table does not rank stores. It only makes the SQL-derived interpretation limits easier to read before future pairwise gate work is added.

## Derived-Metric Scope Note

Demo 2 intentionally keeps the same-period cross-store output narrower than Demo 1.

Demo 1 is a month-over-month diagnostic for one store, so it includes more month-level derived indicators. Demo 2 is a same-period B-F diagnostic, so it focuses on field-contract consistency, selected cross-store evidence, and comparison-boundary behavior.

For that reason, Demo 2 does not expand every derived field defined in `retail_ops/data/DATA_DICTIONARY.md`. Fields such as `refund_order_pressure_pct` and `search_exposure_share_pct` remain valid dictionary definitions, but they are not required columns in the current Demo 2 output. Demo 2 mainly uses source refund-order fields, `refund_pressure_pct`, search-entry structure, activity evidence, order volume, transaction amount, store type, and top-SKU concentration evidence to support its current scope.
