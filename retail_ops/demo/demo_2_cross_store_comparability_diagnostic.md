# Demo 2: Same-Period Cross-Store Comparability Diagnostic

In this demo, comparability means row-level same-period diagnostic readiness. It does not mean pairwise store matching, store ranking, or strategy-transfer approval.

## Purpose

This demo tests whether five anonymized Meituan instant-retail stores can be compared under the same reporting window before making any operating interpretation.

The purpose is to structure backend metrics into a comparable diagnostic format and record the limits that should constrain interpretation.

## Business Problem

Meituan's merchant backend provides detailed store-level metrics, but the backend is mainly designed for reviewing one store at a time.

With many stores, the harder problem is deciding which stores can be compared, under what conditions they can be compared, and which signals are strong enough to support cautious operating judgment.

In this project, instant-retail competition is understood through the operating chain:

being seen -> being entered -> being ordered -> being selected again or maintaining share.

Promotion, subsidy, price, SKU mix, ranking position, and fulfillment conditions are operating levers inside this chain. They should not be interpreted as isolated goals.

## Scope

| Item | Value |
|---|---|
| Stores | B, C, D, E, F |
| Reporting window | 2026-03-01 to 2026-03-31 |
| Period label | 2026-03 |
| Source | Manually structured Meituan merchant-backend metrics |
| Processing method | Offline SQL diagnostic |
| SQL file | `retail_ops/sql/02_demo2_cross_store_comparability.sql` |
| SQL output | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` |
| Memory facts | `retail_ops/outputs/generated_demo2_retail_memory_facts.json` |

Some source traffic-channel fields are retained in the structured source file but not carried into the current Demo 2 diagnostic output. Demo 2 focuses on selected same-period diagnostic signals rather than exhaustive traffic-source decomposition.

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

`same_period_diagnostic_ready`

This means the rows are ready for the current same-period diagnostic. It does not mean the stores are fully comparable in every business sense.

### `comparison_limit_notes`

This field records the main reasons why direct cross-store interpretation should be constrained.

Examples include:

- high search-entry dependence;
- high or moderate activity involvement;
- high or moderate refund pressure;
- high or moderate invalid-order pressure;
- top-3 SKU transaction-amount concentration;
- the need to compare with region, store type, activity, and refund limits.

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

The memory facts are currently file-backed for Demo 2.

This is enough to test the data contract, SQL diagnostic output, fact generation, and limitation-preserving answer behavior, but it is not yet a full 48-store decision-support system.

## What the Current Demo 2 Output Shows

The current output should be read as row-level diagnostic evidence, not as a pairwise store-comparability decision.

| Store | Main scope / limit notes from current output |
|---|---|
| B | high search-entry dependence; high activity involvement; moderate refund pressure; compare with region, store type, activity, refund, order-quality, and product-mix limits |
| C | moderate activity involvement; top-3 SKU amount concentration; compare with region, store type, activity, refund, order-quality, and product-mix limits |
| D | high search-entry dependence; high activity involvement; compare with region, store type, activity, refund, order-quality, and product-mix limits |
| E | high search-entry dependence; moderate activity involvement; high refund pressure; high invalid-order pressure; compare with region, store type, activity, refund, order-quality, and product-mix limits |
| F | high search-entry dependence; high activity involvement; moderate refund pressure; moderate invalid-order pressure; compare with region, store type, activity, refund, order-quality, and product-mix limits |

This table does not rank stores. It only makes the SQL-derived interpretation limits easier to read before future pairwise gate work is added.
