# Project Summary for Admissions Review

## Project Title

**Lifecycle-Aware AI Memory Layer for Retail Decision Support**

Repository name: `livestream-agent-memory-layer`

## 1. Core Problem

This project grew out of a real operating problem in Meituan instant retail.

Meituan's merchant backend provides rich data for each individual store, including exposure, ranking, entry, order conversion, payment conversion, transaction amount, activity cost, refund pressure, invalid orders, and SKU-level evidence. The problem is not that data is missing. The problem is that the backend is mainly designed for reviewing one store at a time.

My actual business problem is cross-store comparison.

I operate many stores, and I need to understand which stores can be compared, under what conditions they can be compared, and which signals are useful enough to support future store expansion. A single-store dashboard can show whether one store improved or declined, but it does not directly answer whether one store's operating pattern can be copied to another store.

This project is my attempt to turn rich but single-store-oriented Meituan backend data into a more structured decision-support prototype. The goal is to use data science, SQL, and a memory layer to identify reusable operating signals, so that future store expansion can be supported by evidence rather than only by experience or intuition.

## 2. Business Context

The business operates in Meituan-style instant retail.

The key operating chain is:

```text
being seen -> being entered -> being ordered -> being selected again or maintaining market share
```

For this type of business, promotion, subsidy, price, SKU mix, ranking position, and fulfillment quality are not isolated goals. They are operating tools inside this chain.

A new store may use stronger activity or subsidy to gain exposure and first orders. A store under external price competition may need to defend visibility and market share even when short-term activity efficiency is not ideal. A store with high exposure but weak conversion needs a different interpretation from a store with order growth but refund pressure.

This is why the project does not try to label stores as simply good or bad. The more useful question is whether a store is comparable with another store, what evidence supports that comparison, and what limitations should prevent over-interpretation.

## 3. Why Single-Store Backend Data Is Not Enough

The Meituan backend is useful for operating one store. It can show detailed metrics for that store during a selected period.

However, my business problem involves many stores. Cross-store comparison is harder because stores may differ in:

- region context;
- store type;
- local competition;
- ranking condition;
- exposure structure;
- search-entry dependence;
- activity and subsidy intensity;
- refund pressure;
- invalid-order pressure;
- SKU mix;
- operating stage.

If these differences are ignored, the same metric can be misread.

For example, a store with higher transaction amount may also have heavier activity involvement. A store with strong exposure may still have weak entry or order conversion. A store with apparent order growth may have refund or invalid-order pressure. A store with concentrated top-SKU sales may not represent a broad category strategy.

The project therefore focuses on comparability before recommendation.

## 4. Technical Direction

The current technical direction is a staged workflow:

```text
Meituan backend metrics
-> data dictionary
-> SQL diagnostic output
-> lineage and interpretation limits
-> generated memory facts
-> retrieval / evaluation
-> cautious answer or refusal
```

SQL is used to organize selected backend metrics into comparable diagnostic outputs.

The data dictionary preserves canonical field names and metric definitions. This matters because the project should not invent new names for existing Meituan backend metrics.

The lineage layer records which source fields support each claim.

The memory layer records each store's observed state, supporting evidence, calculation notes, confidence level, and limitations.

The memory layer is not meant to make decisions directly. Its role is to preserve evidence and prevent unsafe reuse of isolated metrics across stores, periods, and operating contexts.

## 5. Connection to the Earlier Livestream Memory Prototype

The project started from a livestream commerce memory prototype.

In the earlier livestream setting, product facts such as price, promotion, stock status, shipping policy, and product features could change over time. The system therefore needed structured fact extraction, typed memory, overwrite control, freshness filtering, traceable retrieval, and fallback when reliable information was unavailable.

The retail extension applies the same reliability principle to store-operation data.

Store metrics also change by period and context. A store's March profile should not be mixed casually with another store's April profile. A promotion-heavy store should not be compared directly with a low-activity store without recording that limitation. A top-SKU signal should not be overstated as full product-category analysis.

The technical idea is therefore not simply to connect an LLM to business data. The purpose is to control how operating evidence is stored, retrieved, compared, and limited.

## 6. Completed Retail Demo 1

Demo 1 is:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

It analyzes Store A across February, March, and April 2026.

The demo uses normalized Meituan-style backend metrics, including:

- `exposure_users`
- `exposure_times`
- `store_average_rank`
- `entry_users`
- `search_entry_users`
- `order_users`
- `order_conversion_rate_pct`
- `payment_users`
- `payment_conversion_rate_pct`
- `transaction_amount`
- `transaction_orders`
- `average_order_value`
- `estimated_income_proxy`
- `activity_original_transaction_amount`
- `activity_orders`
- `activity_cost`
- `merchant_subsidy_amount`
- `platform_subsidy_amount`
- `refund_amount`
- `valid_orders`
- `invalid_orders`
- top-SKU evidence

Demo 1 shows that a single store's month-over-month change should not be explained from one metric alone. Exposure, entry, ranking, transaction scale, order conversion, average order value, activity cost, refund pressure, invalid-order pressure, and top-SKU evidence need to be read together.

## 7. Completed Retail Demo 2

Demo 2 is:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

It uses five anonymized stores, B-F, from the same March 2026 reporting window.

The purpose is not to choose the best store. The purpose is to test whether cross-store backend data can be structured into a comparable diagnostic form before any operating conclusion is made.

Demo 2 adds:

- March 2026 source tables for Stores B-F;
- top search term evidence with original Chinese backend values and conservative English helper translations;
- top SKU evidence with original Chinese SKU names and conservative English helper translations;
- carried-through canonical backend fields and backend-formula fields;
- SQL-derived comparability diagnostics;
- generated Demo 2 retail memory facts;
- validation scripts for source data, SQL output, and generated facts;
- offline Demo 2 facts evaluation;
- a separate file-backed endpoint, `/chat_retail_ops_demo2_kb`.

The Demo 2 output separates carried-through canonical fields from SQL-derived diagnostics.

Carried-through canonical or backend-formula fields include:

- `region_type`
- `store_type`
- `business_district_rank`
- `activity_cost_ratio_pct`

SQL-derived diagnostic signals include:

- `search_entry_rate_pct`
- `search_entry_share_pct`
- `activity_order_share_pct`
- `refund_pressure_pct`
- `invalid_order_pressure_pct`
- `top3_sku_transaction_amount_share_pct`
- `comparison_scope_flag`
- `comparison_limit_notes`

`activity_cost_ratio_pct` is not treated as a newly invented Demo 2 metric. In this project it follows the backend-formula definition documented in the data dictionary: activity cost divided by activity original transaction amount. The field is useful, but it should be read as activity-cost-ratio evidence rather than traditional ROI.

These fields are not store labels. They are evidence used to decide whether comparison is safe, limited, or unsupported.

## 8. Current Retail Memory Slots

The current generated retail memory facts use these canonical slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `order_quality_pressure_profile`
- `top3_sku_product_mix_note`
- `single_metric_attribution_guard`

Supporting SQL observations such as traffic recovery, transaction recovery, order-conversion decline, refund-pressure improvement, invalid-order-pressure improvement, or SKU concentration are not standalone store-stage labels. They are evidence inside broader memory profiles.

## 9. Data Contract and Lineage

The retail extension uses:

- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/LINEAGE.md`
- `retail_ops/scripts/validate_retail_data_contract.py`
- `retail_ops/outputs/retail_data_contract_validation_result.txt`
- `scripts/validate_project_consistency.py`

The data dictionary fixes canonical field names and backend metric definitions.

The lineage file maps claims to source fields, SQL-derived metrics, memory slots, and interpretation limits.

This matters because Meituan backend metrics should not be treated as generic business metrics.

For example, `order_conversion_rate_pct` follows the backend business definition:

```text
order_conversion_rate_pct = order_users / entry_users * 100
```

It should not be recomputed as:

```text
valid_orders / entry_users
```

because `valid_orders` is an order-status metric, while `order_users` is a user-level funnel metric.

## 10. What the Prototype Demonstrates

This project demonstrates abilities relevant to business analytics, data science, AI systems, and retail decision support:

- identifying a real multi-store operating problem from live business operations;
- recognizing that rich backend data can still be hard to use when it is designed around single-store review;
- converting selected Meituan backend exports into documented and comparable data structures;
- preserving canonical metric definitions instead of inventing inconsistent field names;
- using SQL to derive cautious diagnostic signals;
- separating observed evidence from operating interpretation;
- recording comparison scope and interpretation limits;
- using memory facts to preserve store state, source evidence, confidence, and limitations;
- evaluating whether the system preserves definitions and avoids unsupported conclusions;
- building toward a data-supported expansion workflow for a multi-store instant-retail model.

## 11. Current Limitations

This repository is an ongoing prototype, not a finished production system.

Current limitations include:

- Demo 1 is based on one Store A month-over-month case;
- Demo 2 uses five anonymized stores, B-F, from the same March 2026 reporting window;
- the current system does not yet perform full 48-store grouping;
- automated Meituan backend ingestion is not implemented;
- promotion cycle dates are unknown;
- estimated income is treated as a platform-displayed proxy, not audited profit;
- refund amount is treated as refund pressure because it is counted by refund-success date;
- top-SKU evidence is used as lightweight product-mix evidence, not full product-category sales share;
- the Demo 2 endpoint is file-backed and separate from the Store A retrieval endpoint;
- the system supports cautious diagnostic comparison, not final automated operating decisions.

These limitations are visible because the project is designed to avoid overstating what the data can prove.

## 12. Next Development Step

The next development step is to expand the current comparability-first path from a five-store sample toward broader multi-store coverage.

The planned next work is:

- include more stores while preserving the same metric contract;
- apply a clearer comparability gate before comparing stores;
- separate same-period, same-region, same-store-type, and different-region comparisons;
- improve peer-store selection logic;
- distinguish reusable operating signals from one-store-only observations;
- keep refusing or qualifying unsupported rankings, causal claims, subsidy decisions, or full-business recommendations.

The goal is to support store expansion with reliable cross-store signals, not to automate decision-making blindly.

## 13. Files to Review

Recommended files for admissions review:

1. `README.md` — project overview, implementation boundary, and validation snapshot.
2. `PROJECT_SUMMARY_FOR_ADMISSIONS.md` — admissions-oriented summary.
3. `retail_ops/README.md` — retail operations extension overview.
4. `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` — Store A month-over-month demo.
5. `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` — B-F same-period cross-store demo.
6. `retail_ops/data/DATA_DICTIONARY.md` — Meituan backend metric definitions and canonical fields.
7. `retail_ops/LINEAGE.md` — claim-to-field lineage and interpretation limits.
8. `retail_ops/outputs/demo2_cross_store_comparability_output.csv` — Demo 2 SQL output.
9. `retail_ops/outputs/generated_demo2_retail_memory_facts.json` — Demo 2 memory-facing facts.
10. `eval/results/eval_retail_demo2_facts_result.txt` — Demo 2 offline facts evaluation result.

SQL files, generated outputs, validation scripts, and result files are supporting evidence. They show that the project is not only a written idea, but a staged prototype with data contract, lineage, validation, and evaluation.

## 14. One-Sentence Summary

This project turns rich but single-store-oriented Meituan backend data into a staged cross-store decision-support prototype, using SQL, metric lineage, and memory facts to help identify reusable operating signals for future store expansion.
