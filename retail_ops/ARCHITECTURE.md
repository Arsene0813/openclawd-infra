# Retail Operations Architecture

This document describes the current implemented retail-operations architecture.

The implemented retail scope stops at Demo 2.

A comparability gate is planned as future work, but it is not currently implemented as a finished demo.

## Current Implemented Flow

The current retail flow is:

Meituan backend metrics
-> DATA_DICTIONARY.md
-> canonical CSV files
-> SQL diagnostics
-> output CSV files
-> generated memory facts
-> validation and evaluation

## Implemented Demos

| Demo | Status | Purpose |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic. |
| Demo 2 | Implemented | Stores B-F same-period cross-store diagnostic structure. |
| Comparability gate | Future work | Not currently implemented as a finished demo. |

## Current Files

Current implemented SQL files:

- retail_ops/sql/01_store_a_month_over_month_diagnostic.sql
- retail_ops/sql/02_demo2_cross_store_comparability.sql

Current implemented output files:

- retail_ops/outputs/store_a_demo1_sql_output.csv
- retail_ops/outputs/store_a_demo1_interpretation_summary.csv
- retail_ops/outputs/demo2_cross_store_comparability_output.csv

Current implemented evaluation files:

- eval/eval_retail_demo2_comparability_gate.py

## Why the Comparability Gate Is Future Work

The Meituan backend provides rich and usable store-level data. The limitation is not data quality. The limitation is that the backend workflow is mainly designed for reviewing one store at a time.

For a 48-store operation, a later decision-support layer should help decide which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

A reliable comparability gate should depend on transaction order volume, transaction amount, whether the store is under activity or promotion, activity intensity, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

The current demo sample is still small. To avoid subjective regional classification, the current project treats region_type as weak context only. It is not a hard market-area classification, store-stage label, or peer-store grouping rule.

## Boundary

The current architecture does not rank stores.

The current architecture does not classify market areas.

The current architecture does not claim to decide which operating action should be copied between stores.

The current architecture demonstrates how selected Meituan backend metrics can be converted into store-period diagnostic outputs with explicit field definitions and interpretation limits.
