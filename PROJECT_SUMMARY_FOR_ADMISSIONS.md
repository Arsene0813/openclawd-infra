# Project Summary for Admissions Review

## Project Title

Meituan Instant-Retail Decision Support Prototype

Repository: `livestream-agent-memory-layer`

## Project Summary

This project grew from a real operational problem in a 48-store Meituan instant-retail business.

Although the Meituan merchant backend provides extensive operational metrics, the platform is fundamentally designed for single-store monitoring rather than cross-store comparative analysis. As store count increased, the marginal value of further optimizing isolated store-level metrics began to decline.

At the current stage of the business, a more valuable problem is understanding why certain stores consistently outperform others, and whether those operational advantages are transferable.

The central challenge is comparability:

Which stores can be meaningfully compared?
Under what operational conditions are comparisons valid?
Which observed differences reflect actionable operational patterns rather than environmental noise?

This project explores a possible framework for combining data science methods and AI-assisted reasoning systems to:

experiment with comparability analysis across stores and operating periods,
preserve metric semantics and operational context,
investigate whether operational patterns from high-performing stores may be transferable,
and support scalable decision-making for future business expansion.

The long-term goal is not merely improving individual store metrics, but building a decision-support framework capable of discovering repeatable operational patterns that can support large-scale replication of the business model.

I built a staged local prototype around that problem. The current implementation manually structures selected Meituan backend metrics, preserves their platform-specific meanings in `retail_ops/data/DATA_DICTIONARY.md`, uses SQL to generate limited diagnostic outputs, and converts diagnostic evidence into memory facts with source fields, observed values, source paths, confidence, and limitations.

The current repository implements two retail demos:

1. Store A month-over-month diagnostic across February, March, and April 2026.
2. Stores B-F same-period diagnostic for March 2026.

The project does not yet implement a pairwise comparability gate. That is the next planned stage. The gate should judge whether two store-period records can be compared for a specific operating question before any pricing, subsidy, SKU, ranking, or fulfillment action is transferred.

## Business Problem

In Meituan instant retail, store competition is not only about having products online. For standardized products such as contact lenses and care solutions, many stores compete around a local operating chain:

Being seen -> being entered -> being ordered -> being selected again or maintaining share.

| Operating step | Practical meaning |
|---|---|
| Being seen | Whether the store and products receive enough exposure through search, ranking, listing position, and platform traffic. |
| Being entered | Whether exposure turns into store visits or search-related visits. |
| Being ordered | Whether visits turn into submitted and paid orders under current product, price, activity, and fulfillment conditions. |
| Being selected again or maintaining share | The longer-term operating goal that motivates future data collection, but is not directly measured in the current demos. |

Promotion, subsidy, price adjustment, SKU arrangement, ranking optimization, and fulfillment stability are tools inside this chain. Their meaning depends on store state and local competition. For example, a new store may need activity support to gain first exposure and first orders. A store under local price pressure may use pricing or subsidy tools to defend visibility and market share. A store with high exposure may still underperform if entry quality, order conversion, refund pressure, invalid orders, or SKU concentration create friction.

The decision-support question is:

Which store-period records can be compared, for which operating question, under which metric definitions and limitations?

## Technical Approach

The retail prototype has four layers.

| Layer | What it does |
|---|---|
| Metric dictionary | Preserves Meituan backend metric meanings and maps Chinese backend labels to canonical project field names. |
| SQL diagnostics | Turns selected backend exports into store-period diagnostic outputs. |
| Memory facts | Store evidence, period, source fields, observed values, source paths, confidence, and limitations in a retrieval-facing structure. |
| Evaluation and answer-boundary checks | Test whether later answers preserve definitions, limitations, and refusal behavior instead of producing unsupported advice. |

The SQL layer is not used to paste subjective labels onto stores. It is used to organize backend metrics into a structure that makes cautious comparison possible.

The memory layer is not used to make final operating decisions. It is used to remember what was observed, what the evidence depends on, and where an interpretation should stop.

## Current Implemented Retail Path

### Demo 1: Store A Month-over-Month Diagnostic

Demo 1 analyzes Store A across February, March, and April 2026. It shows why one metric alone is not enough. April 2026 showed recovery in traffic and transaction scale, but order conversion and average order value declined. Refund pressure and invalid-order pressure improved at the same time.

The purpose is to preserve a careful operating profile, not to label a month as simply good or bad.

Main file:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

### Demo 2: Same-Period B-F Diagnostic

Demo 2 extends the analysis to selected Stores B-F under the same March 2026 reporting window. The point is not to rank the stores. The point is to place selected store-period fields under the same reporting window and field contract before any stronger operating interpretation is made.

Store type, order volume, transaction amount, activity involvement, activity intensity, search-entry structure, refund pressure, invalid-order pressure, and top-SKU evidence can all affect what a cross-store comparison means.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

## Region and Market-Context Boundary

The current demo sample is too small to support a reliable regional or market-area classification.

For that reason, the project deliberately avoids classifying store locations by subjective experience, intuition, address impression, or habitual labels. Taking `region_type` as the example, the current project keeps it only as weak contextual signals rather than definitive grouping labels.

A more reliable market-area classification should wait until more store data and more reporting windows are available. It should be judged together with data comparability, actual local consumption level, competition environment, activity conditions, SKU structure, refund pressure, invalid-order pressure, and other operating evidence.

If the project later introduces fields such as `market_area_type`, those fields should be added as new documented fields rather than silently changing the meaning of `region_type`.

## Future Work: Pairwise Comparability Gate

The next planned stage is a pairwise comparability gate.

The gate should answer a narrow question:

Can these two store-period records be compared for this specific operating question?

It should consider at least:

- transaction order volume;
- transaction amount;
- activity involvement;
- activity intensity;
- explicit activity status or campaign-calendar evidence if available;
- store type;
- region and market context;
- competition environment;
- SKU structure;
- refund pressure;
- invalid-order pressure;
- repeated reporting windows.

The gate should not produce one global store ranking or one universal comparability score. A store pair may be comparable for search-entry structure but not comparable for promotion transfer, pricing pressure, SKU strategy, or fulfillment interpretation.

## Field Contract Examples

The project keeps field names consistent with:

- `retail_ops/data/DATA_DICTIONARY.md`

| Field | Boundary |
|---|---|
| `order_conversion_rate_pct` | Follows the backend formula `order_users / entry_users * 100`; it must not be recomputed from `valid_orders / entry_users`. |
| `activity_cost_ratio_pct` | Means activity cost divided by activity original transaction amount; it should not be described as traditional ROI. |
| `estimated_income_proxy` | Treated as a platform-displayed estimated income proxy, not audited profit. |
| `refund_pressure_pct` | A selected-period refund-pressure signal, not a perfect original-order cohort refund rate. |
| `region_type` | Weak region or market-context evidence; not a mature market-area classification and not a hard peer-grouping rule. |
| `top3_sku_transaction_amount_share_pct` | Lightweight top-SKU concentration evidence, not full product-category sales share. |

## Current Scope

Current implemented scope:

- selected Meituan backend data has been manually structured into canonical CSV files;
- Demo 1 and Demo 2 are implemented as local staged diagnostics;
- SQL outputs are converted into generated retail memory facts;
- scenario-based checks test whether metric definitions and comparison limits are preserved;
- the pairwise comparability gate is documented as future work;
- automated Meituan backend ingestion is not implemented;
- full 48-store automated decision support is not implemented;
- causal attribution of sales growth to search ranking, promotion, or conversion change is not claimed;
- full SKU category classification across the catalogue is not implemented.

At this stage, the repository should be read as a staged prototype with clear scope boundaries: it shows how selected single-store backend metrics can be reorganized into a cautious, traceable, and testable decision-support workflow for future multi-store comparison.
