# Project Summary for Admissions Review

## 1. Project Title

**Lifecycle-Aware AI Memory Layer for Retail Decision Support**

Repository name: `livestream-agent-memory-layer`

## 2. Short Summary

This project is a working local prototype built from a real Meituan-style instant retail operating problem.

Meituan's merchant backend provides detailed store-level metrics, but the backend is mainly designed for reviewing one store at a time. Once the business expands across many stores, the harder problem is not simply collecting more data. The harder problem is deciding which stores can be compared, under what conditions they can be compared, and which operating signals are strong enough to support a cautious decision.

The current retail prototype uses SQL to organize selected Meituan backend exports into consistent diagnostic outputs. It then connects those outputs to a memory / retrieval design that preserves evidence, metric definitions, limitations, and conservative operating profiles.

The goal is not to let an LLM make operating decisions directly. The goal is to reduce unsupported comparison, preserve the evidence boundary behind each answer, and make future multi-store decision support more reliable.

## 3. Business Problem

In Meituan-style instant retail, stores compete through an operating chain:

| Operating step | Meaning |
|---|---|
| Being seen | whether consumers can discover the store or product |
| Being entered | whether exposure turns into store visits |
| Being ordered | whether visits turn into orders |
| Being selected again or maintaining share | whether the store can sustain demand and customer trust |

Promotion, subsidy, price adjustment, SKU arrangement, ranking position, and fulfillment stability are not isolated goals. They are operating levers inside this chain.

Activity cost or subsidy level should not be judged only as a simple ROI question. A new store may need activity support to gain visibility. A store under strong local competition may need price pressure or subsidy to maintain share. A store with high search exposure may still fail if entry, order conversion, refund pressure, invalid orders, or SKU structure create friction.

This is why the project focuses on comparability before recommendation.

## 4. Supported Question Contract

The current retail prototype supports three levels of questions.

| Level | Implemented demo | Supported question |
|---|---|---|
| Store-period diagnosis | Demo 1 | What changed inside one store across observed months, and which signals should not be interpreted alone? |
| Same-period cross-store diagnosis | Demo 2 | How do selected Stores B-F compare under the same reporting window, and what limits apply before interpretation? |
| Pairwise comparability gate | Demo 3 | For one selected store pair and one narrow question type, is the comparison usable, partially usable, or not usable for strategy transfer? |

The expected output is not a simple label such as good store, bad store, or copy this strategy.

The expected output is:

- supported evidence;
- relevant metric definitions;
- comparable scope;
- limitation notes;
- a refusal or warning when the evidence is not strong enough.

## 5. Livestream Memory Layer

The original memory-layer prototype was built for an LLM-powered livestream commerce setting, where product information changes frequently.

The livestream layer supports:

- structured fact extraction from product-related input;
- typed memory for product price, promotion, stock status, shipping policy, and product features;
- product-level entity separation;
- overwrite control and soft deactivation;
- freshness-aware retrieval;
- active-state filtering;
- traceable retrieval output;
- fallback or refusal when reliable memory is not available;
- scenario-based evaluation.

The main endpoints are:

- `/chat_mem`
- `/chat_livestream_kb`

The value of this layer is not that it is a general chatbot. The value is that changing commercial facts are stored and retrieved with structure, source awareness, and lifecycle rules.

## 6. Retail Data Contract

The retail extension applies the same reliability principle to Meituan-style operating data.

The project includes a metric dictionary and lineage layer because Meituan backend metrics should not be treated as generic business metrics without checking their original platform meaning, denominator, reporting window, and data grain.

Key files:

| File | Purpose |
|---|---|
| `retail_ops/data/DATA_DICTIONARY.md` | canonical Meituan-style metric definitions and field names |
| `retail_ops/LINEAGE.md` | claim-to-field lineage and interpretation limits |
| `retail_ops/scripts/validate_retail_data_contract.py` | validation script for field contract |
| `retail_ops/outputs/retail_data_contract_validation_result.txt` | validation result |

Important examples:

| Field | Boundary |
|---|---|
| `order_conversion_rate_pct` | follows the backend formula `order_users / entry_users * 100`; it must not be recomputed as `valid_orders / entry_users` |
| `activity_cost_ratio_pct` | describes activity cost divided by activity original transaction amount; it should not be described as traditional ROI |
| `estimated_income_proxy` | treated as a platform-displayed estimated income proxy, not audited profit |
| `refund_amount` | counted by refund-success date, so SQL-derived refund pressure should not be treated as a perfect original-order cohort refund rate |
| `region_type` | weak region or market-context metadata; not a store-stage label, not a mature market-area classification, and not a hard peer-store grouping rule |

## 7. Retail Demo 1

Demo 1 analyzes Store A, a self-operated Qingdao store, across February, March, and April 2026.

It uses normalized Meituan-style backend metrics such as:

- exposure users;
- exposure times;
- average ranking;
- entry users;
- search entry users;
- order users;
- payment users;
- transaction amount;
- transaction orders;
- estimated income proxy;
- activity original transaction amount;
- activity orders;
- activity cost;
- merchant subsidy amount;
- refund amount;
- valid orders;
- invalid orders;
- top-SKU evidence.

The demo shows why operational performance should not be interpreted from one metric alone. April 2026 showed recovery in traffic and transaction scale, but order conversion and average order value declined. At the same time, refund pressure and invalid-order pressure improved.

The point is not to label April as simply good or bad. The point is to preserve a more careful operating profile.

Canonical retail memory slots currently used include:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `order_quality_pressure_profile`
- `single_metric_attribution_guard`
- `top3_sku_product_mix_note`

## 8. Retail Demo 2

Demo 2 extends the retail analysis from one store to selected Stores B-F under the same March 2026 reporting window.

This step is important because multi-store operations create a different problem from single-store review. A store can have different traffic structure, activity involvement, refund pressure, invalid-order pressure, store type, SKU concentration, and local market context.

Demo 2 creates a cautious cross-store diagnostic layer before any operating recommendation is made.

It does not rank stores as winners or losers. It organizes comparable backend fields and SQL-derived diagnostics so that later comparison can preserve scope and limitations.

## 9. Retail Demo 3

Demo 3 turns the Demo 2 output into a pairwise comparability gate.

It compares every store pair across three narrow question types:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

The key idea is that the same pair of stores may be comparable for one question but not for another.

For example, two stores may have similar search-entry structure, making a narrow traffic-structure comparison usable. The same two stores may still be unsafe for activity-strategy transfer if activity-order share, activity-cost ratio, refund pressure, invalid-order pressure, or store type differ too much.

Demo 3 currently exists as offline SQL output and evaluation. It is not yet exposed through a retrieval endpoint.

## 10. How SQL and Memory Work Together

The SQL layer is not used to create artificial store labels.

The SQL layer organizes backend metrics into a more comparable structure and derives limited diagnostic fields such as:

- `search_entry_share_pct`
- `activity_order_share_pct`
- `refund_pressure_pct`
- `invalid_order_pressure_pct`
- `top3_sku_transaction_amount_share_pct`
- `pairwise_comparison_decision`
- `pairwise_limit_notes`

The memory layer is used to preserve:

- the store entity;
- the reporting period;
- the metric source;
- the relevant evidence;
- the limitation;
- the answer boundary.

This matters because a growing multi-store dataset can easily create false confidence. The system should remember not only what a metric says, but also when that metric is not enough to support a conclusion.

## 11. Relevance to Target Programmes

This project is relevant to Business Decision Analytics because it starts from a real operating decision problem: how to compare stores before copying or rejecting an operating action.

It is relevant to Data Science because it uses documented metric definitions, SQL-derived diagnostics, field lineage, validation scripts, and scenario-based evaluation rather than relying on informal interpretation.

It is relevant to Language and Technology because the memory / retrieval layer is designed to answer with evidence and limitation awareness instead of producing fluent but unsupported business advice.

## 12. Current Boundary

This repository is an ongoing prototype, not a finished production system.

Current boundaries:

- automated Meituan backend ingestion is not implemented;
- full 48-store automated decision support is not implemented;
- Demo 3 is currently offline SQL / output / evaluation, not yet a retrieval endpoint;
- SKU-level category classification across the full catalog is not implemented;
- causal attribution of sales growth to search ranking, promotion, or conversion change is not claimed;
- `region_type` is not treated as a market-area classification;
- top-SKU evidence is used as lightweight product-mix evidence, not full product-category sales share.

These boundaries are part of the project design. The purpose is to avoid overstating what the current data can prove.

## 13. Recommended Reading Path

| Order | File | Why read it |
|---|---|---|
| 1 | `README.md` | technical overview and implementation boundary |
| 2 | `PROJECT_STATUS.md` | current implemented scope |
| 3 | `retail_ops/README.md` | retail operations extension overview |
| 4 | `retail_ops/data/DATA_DICTIONARY.md` | Meituan backend metric definitions and canonical field names |
| 5 | `retail_ops/LINEAGE.md` | claim-to-field lineage and interpretation limits |
| 6 | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` | Store A month-over-month diagnostic |
| 7 | `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` | B-F same-period diagnostic |
| 8 | `retail_ops/demo/demo_3_pairwise_comparability_gate.md` | pairwise comparability gate |
| 9 | `eval/` | scenario-based evaluation and boundary checks |

SQL files, generated outputs, validation scripts, and result files are supporting evidence. They should be read after the summary and demo documents.

## 14. One-Sentence Summary

This project shows how changing commercial information and Meituan-style backend metrics can be normalized, checked, traced, retrieved with evidence, and used cautiously for retail decision support.
