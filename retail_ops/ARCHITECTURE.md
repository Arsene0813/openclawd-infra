# Retail Operations Architecture

This file gives a compact map of the current retail decision-support prototype.

The purpose is not to present enterprise infrastructure. The purpose is to show how messy Meituan-style backend evidence moves through a controlled data path before the memory layer answers questions.

```text
Meituan-style backend exports
        ↓
Canonical data dictionary
        ↓
Normalized store-period / search / SKU evidence tables
        ↓
Offline SQL diagnostics
        ↓
SQL output with comparison flags and limitation notes
        ↓
Generated retail memory facts
        ↓
Retrieval endpoint and offline evaluation
        ↓
Cautious answer, qualification, or refusal
```

## Layer 1: Backend Evidence

Current source files include:

- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`

These files are manually structured from Meituan merchant-backend evidence. They are not full automated ingestion.

## Layer 2: Metric Contract

Main files:

- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/LINEAGE.md`

This layer preserves field names, backend definitions, calculation rules, and interpretation limits. Existing Meituan backend metrics should not be silently renamed or redefined.

## Layer 3: SQL Diagnostics

Main files:

- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

SQL is used to prepare comparison-ready diagnostic evidence. It should not assign fixed store-stage labels or make final operating decisions.

## Layer 4: Generated Memory Facts

Main files:

- `retail_ops/outputs/generated_retail_memory_facts.json`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- `retail_ops/scripts/generate_demo2_retail_memory_facts.py`

The memory-facing facts record store-period observations, calculation notes, confidence, source fields, and limitations.

## Layer 5: Retrieval and Evaluation

Current retail evaluation checks whether the system can answer from supported evidence and qualify or refuse unsupported claims.

Current boundary:

- Demo 1: one Store A month-over-month diagnostic.
- Demo 2: five anonymized stores, B-F, in the same March 2026 reporting window.
- Full 48-store automated grouping is not implemented yet.
