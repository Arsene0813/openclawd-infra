# Retail Operations Architecture

This document maps the current retail decision-support prototype.

The purpose is not to present enterprise infrastructure. The purpose is to show how selected Meituan backend metrics move through a controlled data path before any answer is allowed to make a comparison.

The implemented retail scope stops at Demo 2. A pairwise comparability gate is planned as future work, but it is not currently implemented as a finished demo.

## Current implemented flow

```text
Meituan backend metrics
-> DATA_DICTIONARY.md
-> canonical CSV files
-> SQL diagnostics
-> output CSV files
-> generated memory facts
-> validation and evaluation
```

## Implemented demos

| Demo | Status | Purpose |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic. |
| Demo 2 | Implemented | Stores B-F same-period cross-store diagnostic structure. |
| Comparability gate | Future work | Not currently implemented as a finished demo. |

## Artifact contract

| Layer | Input | Output | Boundary |
|---|---|---|---|
| Metric dictionary | Meituan backend metric names and definitions | Canonical project field definitions | `DATA_DICTIONARY.md` is the naming source of truth. |
| Canonical CSV files | Manually structured backend evidence | Store-period, search-term, and SKU evidence tables | Current ingestion is manual, not automated Meituan API ingestion. |
| SQL diagnostics | Canonical CSV files | Diagnostic ratios, pressure indicators, scope flags, and limitation notes | SQL structures evidence; it does not make final operating decisions. |
| Generated memory facts | SQL outputs and source paths | Evidence records with observed values, source fields, confidence, and limitations | Facts preserve boundaries for later retrieval. |
| Evaluation | Generated facts and scenario checks | Pass/fail checks for evidence-boundary behavior | These are scenario checks, not broad model benchmarks. |

## Current implemented files

Current implemented SQL files:

- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

Current implemented output files:

- `retail_ops/outputs/store_a_demo1_sql_output.csv`
- `retail_ops/outputs/store_a_demo1_interpretation_summary.csv`
- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

Current implemented evaluation file:

- `eval/eval_retail_demo2_scope_boundary.py`

## Why the comparability gate is future work

The Meituan backend provides rich store-level data. The limitation is not that the backend has no data. The limitation is that the backend workflow is mainly designed for reviewing one store at a time.

For a 48-store operation, a later decision-support layer should help decide which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

A reliable comparability gate should depend on transaction order volume, transaction amount, activity involvement, activity intensity, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

The current demo sample is still small. To avoid subjective regional classification, the current project treats `region_type` as weak context only. It is not a hard market-area classification, store-stage label, or peer-store grouping rule.

## Boundary

The current architecture does not rank stores.

The current architecture does not classify market areas.

The current architecture does not claim to decide which operating action should be copied between stores.

The current architecture demonstrates how selected Meituan backend metrics can be converted into store-period diagnostic outputs with explicit field definitions and interpretation limits.
