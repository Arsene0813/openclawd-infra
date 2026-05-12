# Project Summary for Admissions Review

## Project title

**Lifecycle-Aware AI Memory Layer for Retail Decision Support**

Repository: `livestream-agent-memory-layer`

## One-minute summary

This project grew out of a real operating problem in my Meituan instant-retail stores.

The Meituan merchant backend gives detailed store-level data, but it is mainly designed for reviewing one store at a time. Once the operation expands across many stores, the harder problem is not collecting more backend metrics. The harder problem is deciding which stores can be compared, under what conditions they can be compared, and what kind of operating judgement the evidence can actually support.

I built a staged prototype that uses a metric dictionary, SQL diagnostics, generated memory facts, and retrieval/evaluation checks to keep retail decision support inside an evidence boundary. The current retail demos do not try to rank stores or produce automatic recommendations. They test a more basic question first: whether a comparison is valid enough to discuss.

## Business problem

In Meituan instant retail, store competition is not only about whether a store has products online. It is about whether the store can move through an operating chain:

| Operating step | Practical meaning |
|---|---|
| Being seen | The store or product can be discovered by consumers in a specific local context. |
| Being entered | Exposure turns into store visits. |
| Being ordered | Visits turn into submitted and paid orders. |
| Being selected again / maintaining share | The store can sustain demand, trust, and local visibility over time. |

Promotion, subsidy, price adjustment, SKU arrangement, ranking position, and fulfillment stability are tools inside this chain. They are not isolated goals.

This is why I do not treat activity cost as a simple ROI problem. A new store may need stronger activity support to gain first exposure and first orders. A store under local price pressure may need pricing or subsidy tools to defend visibility and market share. A store with high search exposure may still have weak results if entry, order conversion, refund pressure, invalid orders, or SKU concentration create friction.

The decision-support problem is therefore not: "Which store is best?"

The better question is:

> Which stores are comparable, for which question, under which metric definitions and limitations?

## Technical approach

The retail prototype has four layers.

| Layer | What it does |
|---|---|
| Metric dictionary | Preserves the original meaning of Meituan backend metrics and maps Chinese backend labels to canonical project field names. |
| SQL diagnostics | Turns selected backend exports into store-period and pairwise comparison outputs. |
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

The point is not to rank the stores. The point is to structure comparable fields before any operating interpretation is made. Store type, activity involvement, refund pressure, invalid-order pressure, search-entry structure, top-SKU evidence, and weak region context can all limit what a cross-store comparison means.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

### Demo 3: Pairwise comparability gate

Demo 3 turns the Demo 2 output into a pairwise comparability gate.

It compares store pairs under three narrow question types:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

The same two stores may be usable for one question and unsafe for another. For example, two stores can have similar search-entry structure but still be unsafe for activity-strategy transfer if activity-order share, activity-cost ratio, refund pressure, invalid-order pressure, or store type differ too much.

Main files:

- `retail_ops/demo/demo_3_pairwise_comparability_gate.md`
- `retail_ops/demo/demo_3_pairwise_answer_path.md`
- `retail_ops/demo/demo_3_pairwise_experiment_notes.md`

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
- Demo 1, Demo 2, and Demo 3 are implemented as local staged diagnostics;
- Demo 3 currently exists as SQL output, saved CSV output, validation, evaluation, and a narrow file-backed answer path;
- automated Meituan backend ingestion is not implemented;
- full 48-store automated decision support is not implemented;
- causal attribution of sales growth to search ranking, promotion, or conversion change is not claimed;
- full SKU category classification across the catalogue is not implemented.

The boundary is intentional. The current goal is to show how messy but real backend data can be turned into a cautious, traceable, and testable decision-support prototype.

## Recommended reading path

| Order | File | Why read it |
|---:|---|---|
| 1 | `README.md` | Overall project structure and implementation boundary. |
| 2 | `retail_ops/README.md` | Retail operations extension overview. |
| 3 | `retail_ops/data/DATA_DICTIONARY.md` | Canonical metric definitions and field names. |
| 4 | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` | Single-store monthly diagnostic. |
| 5 | `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` | Same-period cross-store diagnostic. |
| 6 | `retail_ops/demo/demo_3_pairwise_comparability_gate.md` | Pairwise comparability gate. |
| 7 | `retail_ops/demo/demo_3_pairwise_experiment_notes.md` | Concrete examples showing why Demo 3 matters. |
| 8 | `retail_ops/demo/demo_3_pairwise_answer_path.md` | Narrow file-backed answer path for Demo 3. |
| 9 | `eval/` and `retail_ops/scripts/` | Validation and scenario-based evaluation checks. |

## Supporting validation references

The following files are part of the project evidence and consistency contract:

- `retail_ops/LINEAGE.md`
- `retail_ops/scripts/validate_retail_data_contract.py`
- `retail_ops/outputs/retail_data_contract_validation_result.txt`

## One-sentence summary

This project shows how Meituan backend metrics can be normalized, checked, compared cautiously, and connected to a memory layer so that retail decision support preserves evidence, definitions, and limitations before making any operational judgement.
