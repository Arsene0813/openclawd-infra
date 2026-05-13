# Project Summary for Admissions Review

## Project title

**Lifecycle-Aware AI Memory Layer for Retail Decision Support**

Repository: `livestream-agent-memory-layer`

## One-minute summary

This project comes from a concrete operating problem in my Meituan instant-retail stores. It uses metrics manually structured from the Meituan merchant-backend / Waimai merchant-backend UI. As the operation expanded across many stores, the backend data became less useful for cross-store decisions: it showed detailed single-store metrics, but it did not tell me whether two stores were comparable before I copied a pricing, subsidy, SKU, or ranking action from one store to another.

I built a staged local prototype. First, a metric dictionary preserves the original Meituan backend definitions. Then SQL turns selected store-period records into diagnostic outputs. Finally, generated memory facts carry the period, source fields, observed values, and limitations into later retrieval and answer-boundary checks.

The current repository implements two limited demos: Store A month-over-month diagnosis and a B-F same-period diagnostic review. The next stage is a pairwise comparability gate, but that should only be implemented after more store periods and stronger market-context evidence are available.

## Business problem

In Meituan instant retail, store competition is not only about whether a store has products online. It is about whether the store can move through an operating chain:

| Operating step | Practical meaning |
|---|---|
| Being seen | The store or product can be discovered by consumers in a specific local context. |
| Being entered | Exposure turns into store visits. |
| Being ordered | Visits turn into submitted and paid orders. |
| Being selected again / maintaining share | The store can sustain demand, trust, and local visibility over time. |

Promotion, subsidy, price adjustment, SKU arrangement, ranking position, and fulfillment stability are tools inside this chain. They are not isolated goals.

In this framework, activity cost is not treated as a simple ROI problem. A new store may need stronger activity support to gain first exposure and first orders. A store under local price pressure may need pricing or subsidy tools to defend visibility and market share. A store with high search exposure may still have weak results if entry, order conversion, refund pressure, invalid orders, or SKU concentration create friction.

The decision-support problem is therefore not: "Which store is best?"

The better question is:

> Which stores are comparable, for which question, under which metric definitions and limitations?

## Technical approach

The retail prototype has four layers.

| Layer | What it does |
|---|---|
| Metric dictionary | Preserves the original meaning of Meituan backend metrics and maps Chinese backend labels to canonical project field names. |
| SQL diagnostics | Turns selected backend exports into store-period diagnostic outputs. |
| Memory facts | Store evidence, period, source fields, observed values, and limitations in a retrieval-facing structure. |
| Evaluation and answer boundary checks | Test whether later answers preserve definitions, limits, and refusal behavior instead of producing unsupported advice. |

The SQL layer is not used to paste labels onto stores. It is used to organize backend metrics into a structure that makes cautious comparison possible.

The memory layer is not used to make final decisions for me. It is used to remember what was observed, what the evidence depends on, and where a comparison should stop.

## Current implemented retail path

### Demo 1: Store A month-over-month diagnostic

Demo 1 analyzes Store A across February, March, and April 2026.

It shows why one metric alone is not enough. For example, April 2026 showed recovery in traffic and transaction scale, but order conversion and average order value declined. Refund pressure and invalid-order pressure improved at the same time.

The purpose is to preserve a careful operating profile, not to label a month as simply good or bad.

Main file:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

### Demo 2: Same-period cross-store diagnostic

Demo 2 extends the analysis to selected Stores B-F under the same March 2026 reporting window.

The point is not to rank the stores. The point is to place selected store-period fields under the same reporting window and field contract before any operating interpretation is made. Store type, activity involvement, refund pressure, invalid-order pressure, search-entry structure, top-SKU evidence, and weak region context can all limit what a cross-store comparison means.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

### Future work: Comparability gate

A pairwise comparability gate is planned as the next stage, but it is not presented as a finished demo in the current repository.

The future gate needs more store-pair evidence, repeated reporting windows, and stronger market-context fields before it should be treated as an implemented decision rule. A reliable gate should judge whether stores can be compared using transaction order volume, transaction amount, activity or promotion status, activity involvement and intensity based on existing activity fields, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

The current demo sample is still small. To avoid subjective regional classification, I do not currently classify store locations into market-area types. The existing `region_type` field is treated as weak context only, not as a hard market-area label or peer-store grouping rule.

A future 48-store version can revisit this gate after more stores and reporting windows are added.

## How the original memory layer connects

The original livestream memory layer is the technical base of this project.

It was first built for a livestream commerce setting where product facts such as price, promotion, stock, shipping policy, and product features can change over time. The system stores facts with type, entity, slot, active state, overwrite behavior, freshness handling, and traceable retrieval.

That same reliability problem appears in retail operations data. A store-period metric is also not useful unless the system remembers its time window, source field, definition, and limitation.

The retail extension therefore applies the same idea to Meituan backend metrics:

- preserve metric definitions;
- keep store-period evidence separate;
- avoid stale or unsupported retrieval;
- avoid using one metric as a full explanation;
- carry limitation notes into later answers.

## Field contract examples

The project keeps field names consistent with `retail_ops/data/DATA_DICTIONARY.md`.

Important examples:

| Field | Boundary |
|---|---|
| `order_conversion_rate_pct` | Follows the backend formula `order_users / entry_users * 100`; it must not be recomputed from `valid_orders / entry_users`. |
| `activity_cost_ratio_pct` | Means activity cost divided by activity original transaction amount; it should not be described as traditional ROI. |
| `estimated_income_proxy` | Treated as a platform-displayed estimated income proxy, not audited profit. |
| `refund_pressure_pct` | A selected-period refund-pressure signal, not a perfect original-order cohort refund rate. |
| `region_type` | Weak region or market-context evidence; not a mature market-area classification and not a hard peer-grouping rule. |
| `top3_sku_transaction_amount_share_pct` | Lightweight top-SKU concentration evidence, not full product-category sales share. |

## Relevance to target programmes

This project is relevant to Business Decision Analytics because it starts from a real operating decision problem: how to compare stores before copying, rejecting, or adjusting an operating action.

It is relevant to Data Science because the work depends on metric definitions, SQL-derived diagnostics, field lineage, validation, and scenario-based evaluation rather than informal interpretation.

It is relevant to Language and Technology because the memory/retrieval layer is designed to answer with evidence and limitation awareness, instead of producing fluent but unsupported business advice.

## Current scope

This is an ongoing prototype, not a finished 48-store operating platform.

Current scope:

- selected Meituan backend data has been manually structured into canonical CSV files;
- Demo 1 and Demo 2 are implemented as local staged diagnostics;
- the comparability gate is intentionally left as future work because the current sample is not sufficient for reliable store-pair comparability judgments;
- automated Meituan backend ingestion is not implemented;
- full 48-store automated decision support is not implemented;
- causal attribution of sales growth to search ranking, promotion, or conversion change is not claimed;
- full SKU category classification across the catalogue is not implemented.

At this stage, the value is in showing how messy but real backend data can be turned into a cautious, traceable, and testable decision-support prototype.

## Recommended reading path

| Order | File | Why read it |
|---:|---|---|
| 1 | `README.md` | Overall project structure and implementation boundary. |
| 2 | `retail_ops/README.md` | Retail operations extension overview. |
| 3 | `retail_ops/data/DATA_DICTIONARY.md` | Canonical metric definitions and field names. |
| 4 | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` | Single-store monthly diagnostic. |
| 5 | `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` | Same-period cross-store diagnostic. |
| 6 | `eval/` and `retail_ops/scripts/` | Validation and scenario-based evaluation checks. |

## Supporting validation references

The following files are part of the project evidence and consistency contract:

- `retail_ops/LINEAGE.md`
- `retail_ops/scripts/validate_retail_data_contract.py`
- `retail_ops/outputs/retail_data_contract_validation_result.txt`

## One-sentence summary

This project shows how Meituan backend metrics can be normalized, checked, compared cautiously, and connected to a memory layer so that retail decision support preserves evidence, definitions, and limitations before making any operational judgement.

The current repository implements two limited demos and documents their evidence boundaries; it does not present the work as a finished 48-store operating platform.
