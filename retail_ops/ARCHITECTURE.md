# Retail Operations Architecture

This file gives a compact map of the current retail decision-support prototype.

The purpose is not to present enterprise infrastructure. The purpose is to show how Meituan-style backend evidence moves through a controlled data path before any answer is allowed to make a comparison.

Data flow:

    Meituan-style backend evidence
        ↓
    Canonical metric dictionary
        ↓
    Normalized store-period / search / SKU tables
        ↓
    Offline SQL diagnostics
        ↓
    SQL outputs with diagnostic ratios and limitation notes
        ↓
    Generated retail memory facts
        ↓
    Offline retrieval / evaluation checks
        ↓
    Cautious answer, qualification, or refusal

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

This layer preserves field names, backend definitions, calculation rules, and interpretation limits.

Existing Meituan backend metrics should not be silently renamed or redefined. For example, `order_conversion_rate_pct` follows `order_users / entry_users * 100`; it must not be recomputed from `valid_orders / entry_users`.

## Layer 3: SQL Diagnostics

Main files:

- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

SQL is used to prepare comparison-ready diagnostic evidence. It creates ratios, shares, pressure indicators, and limitation notes.

The SQL layer should not assign fixed store-stage labels or make final operating decisions.

## Layer 4: Generated Memory Facts

Main files:

- `retail_ops/outputs/generated_retail_memory_facts.json`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- `retail_ops/scripts/generate_demo2_retail_memory_facts.py`

The memory-facing facts record:

- store identity;
- reporting period;
- observed values;
- source fields;
- calculation notes;
- confidence as evidence-trace confidence;
- limitations.

## Layer 5: Offline Evaluation and Boundary Checks

Current retail evaluation checks whether later answers can stay inside supported evidence.

Current boundary:

- Demo 1: one Store A month-over-month diagnostic.
- Demo 2: five anonymized stores, B-F, in the same March 2026 reporting window.
- Demo 2 generated facts are file-backed and evaluated offline.
- The future comparability gate is documented but not implemented as a finished demo.
- Full 48-store automated grouping is not implemented.
