# Project Summary for Admissions Review

## Project Title

Meituan Instant Retail Cross-Store Decision Support Prototype

Repository name: `livestream-agent-memory-layer`

## Review Focus

This project is not a polished dashboard or a broad enterprise platform.

It is a staged prototype based on a real multi-store operating problem: Meituan's merchant backend provides detailed store-level data, but the data is mainly designed for single-store review rather than cross-store comparison.

The current repository shows how selected Meituan backend metrics can be organized into a controlled decision-support workflow:

1. preserve Meituan backend metric definitions;
2. organize selected store-period data with SQL;
3. record source fields, confidence, limitations, and interpretation boundaries;
4. test whether later answers preserve comparison limits instead of making unsupported recommendations.

The current retail extension supports:

- Store A month-over-month diagnosis;
- same-period Stores B-F cross-store comparability diagnosis;
- Stores B-F pairwise comparability gating for narrow operating questions.

It does not claim full 48-store automation.

## 1. Core Problem

This project grew out of a real operating problem in Meituan instant retail.

Meituan's merchant backend gives detailed data for each store, including exposure, ranking, entry, order conversion, payment conversion, transaction amount, activity cost, refund pressure, invalid orders, and SKU-level evidence.

The problem is not that data is missing. The problem is that the backend is mainly designed for reviewing one store at a time.

My actual business problem is cross-store comparison. I operate many stores, and I need to understand:

- which stores can be compared;
- under what conditions they can be compared;
- which operating signals are reusable;
- which signals are too context-dependent to transfer;
- when the system should refuse to make a recommendation.

A single-store dashboard can show whether one store improved or declined. It does not directly answer whether one store's operating pattern can be copied to another store.

This project is my attempt to turn rich but single-store-oriented Meituan backend data into a structured decision-support prototype.

## 2. Business Context

The business operates in Meituan-style instant retail.

The key operating chain is:

being seen -> being entered -> being ordered -> being selected again or maintaining market share

For this business, promotion, subsidy, price, SKU mix, ranking position, and fulfillment quality are not isolated goals. They are operating tools inside this chain.

A new store may use stronger activity or subsidy to gain exposure and first orders. A store under external price competition may need to defend visibility and market share even when short-term activity efficiency is not ideal. A store with high exposure but weak conversion needs a different interpretation from a store with order growth but refund pressure.

The useful question is not whether a store is simply good or bad. The useful question is whether a store is comparable with another store, what evidence supports that comparison, and what limits should prevent over-interpretation.

## 3. Why Single-Store Backend Data Is Not Enough

Cross-store comparison is difficult because stores can differ in:

- reporting period;
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

For example:

- a store with higher `transaction_amount` may also have heavier activity involvement;
- a store with strong `exposure_users` may still have weak `entry_users` or `order_conversion_rate_pct`;
- a store with apparent order growth may have refund or invalid-order pressure;
- a store with concentrated top-SKU evidence may not represent a broad product-category strategy.

The project therefore focuses on comparability before recommendation.

## 4. Technical Direction

The current technical direction is a staged workflow:

Meituan backend metrics -> DATA_DICTIONARY.md -> SQL diagnostic output -> LINEAGE.md and interpretation limits -> generated memory facts -> retrieval / evaluation -> cautious answer or refusal

SQL is used to organize selected backend metrics into comparable diagnostic outputs.

The data dictionary preserves canonical field names and metric definitions.

The current data-contract check is implemented in `retail_ops/scripts/validate_retail_data_contract.py`, with saved output in `retail_ops/outputs/retail_data_contract_validation_result.txt`.

The lineage layer records which source fields support each claim.

The memory layer is not meant to make operating decisions directly. Its role is to preserve store-period evidence, confidence, limitations, and prior judgments so that later answers do not compare the wrong stores or overstate weak evidence.

## 5. Implemented Retail Path

### Demo 1: Store A Month-over-Month Diagnostic

Demo 1 uses Store A across February, March, and April 2026.

It shows that a store's operating state cannot be interpreted from one metric alone. Traffic recovery, transaction recovery, conversion decline, average-order-value decline, refund-pressure improvement, invalid-order-pressure improvement, activity involvement, and top-SKU concentration can move in different directions.

The purpose of Demo 1 is to show how Meituan backend data can be normalized, checked, traced, and converted into cautious memory-facing facts.

### Demo 2: Same-Period B-F Cross-Store Diagnostic

Demo 2 uses Stores B-F in the same March 2026 reporting window.

It does not rank stores as simply better or worse. It structures comparable backend metrics and creates limited cross-store diagnostic evidence.

Demo 2 is useful because it moves beyond one-store analysis while still keeping the comparison scope small enough to verify. It includes source CSVs, SQL output, generated memory facts, validation scripts, and offline evaluation.

### Demo 3: Pairwise Comparability Gate

Demo 3 takes the Demo 2 B-F output and tests every store pair under three narrow question types:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

This is closer to the actual decision problem. A store pair may be usable for one question but not another.

Demo 3 does not classify stores by market area, does not treat `region_type` as a hard peer-grouping rule, does not rank stores, and does not generate final operating recommendations.

Its purpose is to make comparability testable before strategy transfer.

## 6. Evidence Boundary

The current repository is deliberately limited.

It can support:

- Store A month-over-month diagnostic interpretation;
- limited B-F same-period cross-store comparison;
- pairwise comparability checks for selected B-F store pairs and three question types;
- metric-boundary checks around activity-cost ratio, top-SKU concentration, search-entry evidence, refund pressure, invalid-order pressure, and comparison limits.

It cannot yet support:

- automated Meituan backend ingestion;
- full 48-store decision support;
- daily automated operating recommendations;
- clean causal claims about promotion effects;
- complete product-category sales share;
- store-stage classification across all stores;
- production deployment.

The limitation is intentional. The project is designed to show that I can convert a real operational problem into a verifiable data and retrieval workflow without pretending that the current evidence proves more than it does.

## 7. What This Demonstrates

This project demonstrates:

- ability to identify a real analytics problem inside an operating business;
- ability to preserve source metric definitions instead of treating backend metrics as generic business numbers;
- ability to use SQL to structure selected operating data;
- ability to design comparison boundaries before drawing conclusions;
- ability to connect data science reasoning with retrieval and memory-layer design;
- ability to build evaluation cases that check refusal, qualification, and metric-boundary preservation.

The main point is not that the prototype is large. The main point is that it is controlled: each claim should be tied to a field, each comparison should have a scope, and each recommendation should be limited by available evidence.

## One-Sentence Summary

This project turns detailed but single-store-oriented Meituan backend data into a staged SQL and memory-layer prototype for cautious cross-store comparison, pairwise comparability checks, and limitation-preserving retail decision support.
