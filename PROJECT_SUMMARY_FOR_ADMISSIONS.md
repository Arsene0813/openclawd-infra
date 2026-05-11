# Project Summary for Admissions Review

## Project Title

**Lifecycle-Aware AI Memory Layer: From Livestream Commerce to Retail Decision Support**

Repository name: `livestream-agent-memory-layer`

## 1. Short Summary

This project is a working local prototype for making AI-assisted commerce interaction and retail decision support more reliable through structured memory, lifecycle-aware retrieval, traceable sources, and conservative use of changing operational data.

The project began as a lifecycle-aware memory layer for an LLM-powered livestream commerce system. In that setting, product information changes frequently: prices are updated, promotions expire, stock status changes, delivery rules vary, and product features need to be retrieved accurately.

The system addresses this reliability problem by extracting structured product facts, storing them as typed memory, applying overwrite and freshness rules, retrieving relevant facts with traceable sources, and falling back or refusing when reliable memory is not available.

The project now extends the same design principle to Meituan-style instant retail operations data. The first completed retail demo uses normalized Meituan backend metrics and SQL-derived diagnostics to examine one self-operated Qingdao store across February, March, and April 2026. It shows why operational performance should not be interpreted from a single metric such as exposure, ranking, transaction amount, conversion rate, or refund amount.

The broader purpose is not to build a generic chatbot. The purpose is to explore how changing commercial knowledge can be structured, updated, checked and reused carefully.

## 2. Why I Built This Project

AI agents can produce fluent responses while still relying on outdated, conflicting, or weakly matched information. This is especially risky in commerce and retail contexts.

If an AI assistant uses an outdated price, an expired promotion, an old stock status, or a weakly matched memory entry, the answer may sound plausible but be operationally wrong.

I wanted to build a system that does not only remember information, but manages memory more carefully:

- What type of fact is this?
- Which product or store does it refer to?
- Is it still active?
- Has it been overwritten by newer information?
- Is it fresh enough to reuse?
- What source supports the answer?
- Should the system refuse instead of answering?

These questions shaped the memory-layer design.

## 3. Current Livestream Memory Prototype

The current livestream prototype supports:

- structured fact extraction from product-related input;
- typed memory for product price, promotion, stock status, shipping policy, and product features;
- product-level entity separation;
- overwrite control and soft deactivation;
- freshness-aware retrieval;
- active-state filtering;
- traceable retrieval output;
- fallback or refusal when no reliable fact is available;
- scenario-based evaluation.

The main endpoints are:

- `/chat_mem` for structured fact ingestion;
- `/chat_livestream_kb` for retrieval from the structured livestream knowledge base.

The current scenario-based livestream checks pass for the implemented product-memory cases.

The evaluation checks behavior such as price retrieval, price overwrite, unsupported-query fallback, product-level separation, stock overwrite, promotion overwrite, shipping-policy retrieval, product-feature retrieval, and non-fact filtering.

## 4. Core Technical Idea

The core idea is that memory should not be treated as an unstructured pile of chat history. Instead, commercial facts should be represented with explicit structure.

For example, a raw message such as:

```text
A款价格是99元
```

can be converted into a structured fact:

```json
{
  "type": "product_price",
  "product_ref": "A款",
  "value": "99元",
  "slot": "price",
  "is_active": true
}
```

If a newer fact appears:

```text
A款价格是89元
```

the newer price becomes active, while the older price is softly deactivated rather than silently deleted.

This allows the system to preserve traceability while avoiding the use of outdated information as current knowledge.

## 5. Retail Operations Extension

The retail operations extension applies the same lifecycle-aware memory principle to Meituan-style instant retail operations data.

In store operations, knowledge also changes over time. Exposure, ranking, conversion, promotion cost, refund pressure, order quality, and product mix can all vary by month, store, activity condition, and local context.

The difficulty is not that Meituan backend data is unavailable. The difficulty is that the backend data is primarily designed for single-store operation, while my business problem involves 48 stores that cannot be compared naively. Stores differ by operating region and market context, but region alone is not a reliable comparison rule: two stores in the same city may still face different purchasing power, delivery radius, local competition, activity pressure, product mix, ranking condition, and order-quality pressure. Therefore, the retail extension is not just a SQL reporting exercise. It is a first step toward a comparability-first decision-support system: normalize the backend metrics, preserve their original definitions, derive limited diagnostic signals, and use the memory layer to remember evidence, limitations, and whether a comparison is valid.

A decision-support system should therefore avoid unsupported conclusions such as:

- “This store is good because transaction amount increased.”
- “This store should increase exposure because search traffic is high.”
- “This month performed better because ranking improved.”
- “This strategy can be copied to other stores.”

Instead, the system should first check the structure, scope, and limitations of the data.

## 6. Completed Retail Demo 1

The first completed retail demo is:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

This demo analyzes Store A, a self-operated Qingdao store, across February, March, and April 2026.

The demo starts from normalized Meituan-style backend metrics including:

- exposure users;
- exposure times;
- average ranking;
- entry users;
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
- invalid orders;
- top-selling SKU evidence.

The demo describes Store A's visibility and entry structure using exposure, ranking, `entry_users`, and `search_entry_users`. These metrics show whether the store was being seen and entered, but they do not assign a fixed operating label or prove causal growth.

It also shows that Store A had a high activity-order share in all three months. In this project, activity and subsidy are treated as operating levers inside the store's competitive context, not as standalone causal proof or a simple ROI judgment.

Therefore, the system should avoid attributing performance changes to search exposure, ranking, or promotion alone.

## 7. Retail Data Contract and Lineage

To avoid inconsistent metric usage, the retail extension includes a data-contract and lineage layer.

Key files:

- `retail_ops/data/DATA_DICTIONARY.md` — bilingual Meituan backend metric definitions and consistency rules.
- `retail_ops/LINEAGE.md` — claim-to-field lineage and interpretation limits.
- `retail_ops/scripts/validate_retail_data_contract.py` — automated consistency validation.
- `retail_ops/outputs/retail_data_contract_validation_result.txt` — saved validation evidence.
- `retail_ops/outputs/generated_retail_memory_facts.json` — SQL-derived retail memory facts.

The validation script checks:

- required canonical fields;
- forbidden alias fields;
- generated memory fact source fields;
- JSON validity for generated retail memory facts;
- consistency between source CSV, SQL output, metric dictionary, lineage, and generated facts.

Current validation result:
```text
Retail data contract validation PASSED.
Checked source CSV headers: 46
Checked top-SKU CSV headers: 9
Checked SQL output headers: 61
Checked generated retail memory facts: 6
Checked known store_id values: ['A']
Checked expected entity_id values: ['store_A']
No forbidden alias fields found.
Entity ID convention is documented and validated.
```

This is important because backend platform metrics should not be treated as generic business metrics without checking their exact definitions and calculation logic.

For example, backend-reported order conversion should not be recomputed from valid orders divided by entry users, because `valid_orders` is an order-status metric while `order_users` is a user-level funnel metric.

## 8. Main Finding from Demo 1

April 2026 showed strong recovery in traffic and transaction scale:

- exposure users increased from March to April;
- search exposure users increased from March to April;
- entry users increased from March to April;
- transaction orders increased from March to April;
- transaction amount increased from March to April;
- estimated income proxy increased from March to April.

However, April should not be interpreted as a simple improvement in conversion quality.

Order conversion declined from March to April, and average order value also declined. At the same time, refund pressure and invalid-order pressure improved.

This means the system should generate cautious interpretation flags rather than a simple store-performance label.

Canonical retail memory slots currently used by the retail memory facts are:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `order_quality_pressure_profile`
- `single_metric_attribution_guard`
- `top3_sku_product_mix_note`

`refund_pressure_improved` is not a canonical retail memory slot. It is a SQL-derived supporting observation when it appears as evidence inside the broader `order_quality_pressure_profile`.

Order-conversion decline, average-order-value decline, and traffic recovery are supporting observations, not standalone canonical flags in the current demo.

## 9. Why This Matters for Decision Support

The retail demo shows why direct SQL classification can be misleading.

A naive SQL classifier might label April as a high-exposure and high-performance month.

That would be incomplete because:

- activity order share stayed above 95% in all three observed months;
- order conversion declined;
- average order value declined;
- refund pressure improved;
- estimated income proxy improved, but the platform calculation is not fully transparent;
- top-selling SKU data only provides lightweight product-mix evidence, not full category-share analysis.

The better design is to convert SQL-derived observations into structured memory facts with:

- entity;
- time period;
- source;
- confidence level;
- limitation;
- decision warning.

This connects SQL analysis with lifecycle-aware memory.

## 10. What This Project Demonstrates

This project demonstrates abilities relevant to AI, data science, business analytics, and retail decision support:

- identifying a real reliability problem in AI-assisted commerce;
- converting unstructured product interaction into structured facts;
- designing typed memory policies for different fact types;
- managing updates, overwrites, freshness, and active-state filtering;
- using traceable retrieval rather than hidden memory use;
- evaluating system behavior through scenario-based cases;
- organizing Meituan-style backend metrics into a documented data dictionary;
- using SQL-derived metrics for cautious operational interpretation;
- documenting metric lineage and calculation boundaries;
- validating data-contract consistency across CSV, SQL, memory facts, and documentation;
- recognizing confounding conditions such as activity and subsidy profile;
- avoiding unsupported causal claims from incomplete business data;
- connecting customer-facing AI memory with internal retail decision support.

## 11. Current Limitations

This repository is an ongoing prototype, not a finished production system.

Current limitations include:

- the livestream memory system remains a local prototype rather than a deployed production service;
- Demo 1 is based on one Store A month-over-month case, so it should not be generalized across all stores;
- Demo 2 uses five anonymized stores, B-F, from the same March 2026 reporting window, not the full 48-store business;
- the Demo 2 endpoint is file-backed and separate from the Store A retrieval endpoint;
- Demo 2 supports cautious cross-store diagnostic comparison, not final operating recommendations;
- the current system does not yet perform full 48-store grouping, store-stage classification, or automated daily decision generation;
- automated Meituan backend ingestion is not implemented;
- promotion cycle dates are unknown;
- estimated income is treated as a platform-displayed proxy, not audited profit;
- refund amount is treated as refund pressure because refund date is based on refund-success date;
- top-selling SKU evidence is used as lightweight product-mix evidence, not as full product-category sales share;
- full SKU classification is deferred to a future automated classification step.

These limitations are intentional and visible in the project because the goal is to avoid overstating what the data can prove.

## 12. Current Retail Demo 2

Demo 2 has now been implemented as a limited same-period cross-store comparability diagnostic.

It uses five anonymized Meituan stores, B-F, all from the same March 2026 reporting window. The purpose is not to choose the best store or recommend subsidy allocation directly. The purpose is to test whether multi-store backend data can be structured into a comparable form before any operating conclusion is made.

Demo 2 adds:

- March 2026 source tables for Stores B-F;
- top search term evidence with original Chinese backend values and conservative English helper translations;
- top SKU evidence with original Chinese SKU names and conservative English helper translations;
- SQL-derived comparability diagnostics;
- generated Demo 2 retail memory facts;
- validation scripts for source data, SQL output, and generated facts;
- an offline Demo 2 facts evaluation;
- a separate file-backed endpoint, /chat_retail_ops_demo2_kb.

The Demo 2 SQL output derives diagnostic signals such as:

- search-entry rate;
- search-entry share;
- activity-order share;
- refund pressure;
- invalid-order pressure;
- top-3 SKU transaction amount share;
- comparison scope flags;
- comparison limitation notes.

These fields are not used to label stores as good or bad. They are used to preserve the conditions under which comparison is possible.

The key design point is that multi-store decision support should first ask:

- Are these stores from the same reporting period?
- Are the metrics available under the same data contract?
- Is search traffic unusually dominant?
- Is activity involvement high enough to limit transaction comparison?
- Is refund or invalid-order pressure high enough to constrain interpretation?
- Is top-SKU evidence only lightweight product-mix evidence rather than a full category structure?

This is the practical bridge from single-store SQL diagnostics to a multi-store memory layer.

## 13. Next Development Step

The next development step is not to claim that the full 48-store decision-support system is finished.

The next step is to expand the current Demo 2 path from a five-store sample toward broader multi-store coverage.

The planned next work is:

- include more stores while preserving the same metric contract;
- add a clearer comparability gate for same-period store groups;
- separate same-region, same-store-type, and different-region comparisons;
- improve cross-store explanation logic without turning it into store ranking;
- eventually connect Demo 2 facts to a broader retrieval and decision-support workflow;
- keep refusing or qualifying questions that ask for unsupported rankings, causal claims, or full-business recommendations.

This follows the same principle as the memory layer: the system should refuse or qualify conclusions when the evidence is weak, stale, incomplete, or not comparable.

## 14. Files to Review

Recommended files for admissions review:

1. README.md — project overview, implementation boundary, and validation snapshot.
2. PROJECT_SUMMARY_FOR_ADMISSIONS.md — admissions-oriented narrative and relevance.
3. retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md — completed Store A month-over-month demo.
4. retail_ops/data/DATA_DICTIONARY.md — Meituan backend metric definitions and canonical fields.
5. retail_ops/LINEAGE.md — claim-to-field lineage and interpretation limits for Demo 1 and Demo 2.
6. retail_ops/outputs/demo2_cross_store_comparability_output.csv — Demo 2 cross-store diagnostic output.
7. retail_ops/outputs/generated_demo2_retail_memory_facts.json — generated Demo 2 memory-facing facts.
8. eval/results/eval_retail_demo2_facts_result.txt — Demo 2 offline facts evaluation result.
9. eval/eval_retail_report.md — Store A retrieval and refusal evaluation.

SQL files, generated outputs, validation scripts, and result files remain supporting evidence for the demos, but they are not separate admissions entry points.

## 15. One-Sentence Summary

This project shows how changing commercial information can be normalized, checked, traced, validated, retrieved with evidence, and used cautiously for livestream product interaction and Meituan-style retail operations decision support, moving from single-store diagnostics toward controlled multi-store comparability.

