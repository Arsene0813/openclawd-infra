# Project Summary for Admissions Review

## Project Title

Lifecycle-Aware AI Memory Layer for Retail Decision Support

Repository: `livestream-agent-memory-layer`

## One-Minute Summary

This project grew out of a practical problem in Meituan instant-retail operations. The merchant backend gives many useful single-store metrics, but a multi-store operation needs another layer: deciding which store-period records can be inspected together, which metric definitions must stay fixed, and where a comparison should stop before it becomes unsupported advice.

I built a staged local prototype around that problem. The metric dictionary keeps the original Meituan backend meanings for fields such as order conversion rate, transaction amount, order amount, payment amount, refund amount, and activity cost ratio. SQL then turns selected store-period records into diagnostic outputs, and generated memory facts carry the period, source fields, observed values, source paths, evidence-trace confidence, and limitations into retrieval-facing checks.

The current repository implements two limited retail demos: Store A month-over-month diagnosis, and a B-F same-period diagnostic review. The pairwise comparability gate remains future work, not a finished demo. That next stage should judge whether two store-period records can be compared for a specific operating question before suggesting whether a pricing, subsidy, SKU, ranking, or fulfillment action can transfer.

This retail path extends the earlier livestream memory-layer work in a limited, practical way:

| Reused mechanism | Retail-specific addition | Still not implemented |
|---|---|---|
| Typed memory facts with source fields | Meituan metric dictionary and SQL-derived diagnostic fields | Full 48-store automation |
| Source-bounded retrieval | Store-period evidence with `comparison_limit_notes` | Pairwise comparability gate |
| Scenario-based evaluation | Answer-boundary checks for metric misuse and strategy-transfer overreach | Automatic operating recommendations |


## Business Problem

The current demos do not directly measure repeat purchase, customer cohorts, or long-term market share. Those are operating goals that motivate future data collection; the implemented evidence mainly covers visibility, entry, order, payment, activity, refund, invalid-order, and top-SKU signals.

In Meituan instant retail, store competition plays out through a local operating chain:

being seen -> being entered -> being ordered -> being selected again or maintaining share

| Operating step | Practical meaning |
|---|---|
| Being seen | The store or product can be discovered by consumers in a specific local context. |
| Being entered | Exposure turns into store visits. |
| Being ordered | Visits turn into submitted and paid orders. |
| Being selected again or maintaining share | The store can sustain demand, trust, and local visibility over time. |

Promotion, subsidy, price adjustment, SKU arrangement, ranking position, and fulfillment stability are tools inside this chain. Their meaning depends on the store state. A new store may need activity support to gain first exposure and first orders. A store under local price pressure may use pricing or subsidy tools to defend visibility and market share. A store with high search exposure can still underperform if entry quality, order conversion, refund pressure, invalid orders, or SKU concentration create friction.

The decision-support question is: which store-period records can be compared, for which operating question, under which metric definitions and limitations?

## Technical Approach

The retail prototype has four layers.

| Layer | What it does |
|---|---|
| Metric dictionary | Preserves the original meaning of Meituan backend metrics and maps Chinese backend labels to canonical project field names. |
| SQL diagnostics | Turns selected backend exports into store-period diagnostic outputs. |
| Memory facts | Store evidence, period, source fields, observed values, and limitations in a retrieval-facing structure. |
| Evaluation and answer-boundary checks | Test whether later answers preserve definitions, limits, and refusal behavior instead of producing unsupported advice. |

The SQL layer is not used to paste labels onto stores. It is used to organize backend metrics into a structure that makes cautious comparison possible.

The memory layer is not used to make final decisions. It is used to remember what was observed, what the evidence depends on, and where a comparison should stop.

## Current Implemented Retail Path

### Demo 1: Store A Month-over-Month Diagnostic

Demo 1 analyzes Store A across February, March, and April 2026. It shows why one metric alone is not enough.

For example, April 2026 showed recovery in traffic and transaction scale, but order conversion and average order value declined. Refund pressure and invalid-order pressure improved at the same time. The purpose is to preserve a careful operating profile, not to label a month as simply good or bad.

Main file:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

### Demo 2: Same-Period Cross-Store Diagnostic

Demo 2 extends the analysis to selected Stores B-F under the same March 2026 reporting window.

The point is not to rank the stores. The point is to place selected store-period fields under the same reporting window and field contract before any operating interpretation is made.

Store type, activity involvement, refund pressure, invalid-order pressure, search-entry structure, top-SKU evidence, and weak region context can all limit what a cross-store comparison means.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

### Future Work: Comparability Gate

A pairwise comparability gate is planned as the next stage, but it is not presented as a finished demo in the current repository.

The gate should judge whether selected store-period records can be compared for a specific operating question. It should consider transaction order volume, transaction amount, current activity involvement and intensity based on existing activity fields, explicit activity status or campaign-calendar evidence if available, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

The current demo sample is still small. For that reason, store locations should not be classified by subjective experience, intuition, or habitual labels.

Taking `region_type` as an example, the current project does not use it to decide that one store belongs to a fixed market-area type. A more reliable classification should wait until more store data is available and can be judged together with data comparability, actual local consumption level, competition environment, and other operating conditions.

## Field Contract Examples

The project keeps field names consistent with `retail_ops/data/DATA_DICTIONARY.md`.

| Field | Boundary |
|---|---|
| `order_conversion_rate_pct` | Follows the backend formula `order_users / entry_users * 100`; it must not be recomputed from `valid_orders / entry_users`. |
| `activity_cost_ratio_pct` | Means activity cost divided by activity original transaction amount; it should not be described as traditional ROI. |
| `estimated_income_proxy` | Treated as a platform-displayed estimated income proxy, not audited profit. |
| `refund_pressure_pct` | A selected-period refund-pressure signal, not a perfect original-order cohort refund rate. |
| `region_type` | Weak region or market-context evidence; not a mature market-area classification and not a hard peer-grouping rule. |
| `top3_sku_transaction_amount_share_pct` | Lightweight top-SKU concentration evidence, not full product-category sales share. |

## Current Scope

This is an ongoing prototype, not a finished 48-store operating platform.

Current scope:

- selected Meituan backend data has been manually structured into canonical CSV files;
- Demo 1 and Demo 2 are implemented as local staged diagnostics;
- the comparability gate is intentionally left as future work;
- automated Meituan backend ingestion is not implemented;
- full 48-store automated decision support is not implemented;
- causal attribution of sales growth to search ranking, promotion, or conversion change is not claimed;
- full SKU category classification across the catalogue is not implemented.

At this stage, the value is in showing how messy but real backend data can be turned into a cautious, traceable, and testable decision-support prototype.
