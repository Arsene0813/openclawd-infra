# Retail Operations Architecture

This file explains the current retail decision-support prototype.

The purpose is not to present an enterprise system. The purpose is to show how selected Meituan merchant-backend evidence is turned into structured diagnostic evidence before any memory-layer answer is allowed to compare stores or discuss operating actions.

The current architecture is:

    Meituan merchant-backend screenshots / exports
        ↓
    canonical data dictionary and field contract
        ↓
    structured store-period, search, activity, refund, order-quality, and SKU evidence
        ↓
    offline SQL diagnostics
        ↓
    SQL outputs with derived comparison signals and limitation notes
        ↓
    generated retail memory facts or pairwise gate rows
        ↓
    retrieval / offline evaluation
        ↓
    cautious answer, qualified comparison, or refusal

## 1. Backend Evidence

Current source files include:

- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_source_notes.md`

These files are manually structured from Meituan merchant-backend evidence.

They are not automated Meituan ingestion, not crawler output, and not a full 48-store data warehouse. The current point is narrower: preserve selected backend metrics in a consistent format so that later comparison logic does not silently change metric meaning.

## 2. Metric Contract

Main files:

- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/LINEAGE.md`
- `retail_ops/FIELD_USAGE_REVIEW.md`

This layer protects the project from metric drift.

The data dictionary defines the implemented field names and metric meanings. Existing Meituan backend metrics should not be silently renamed or redefined. SQL-derived diagnostic fields must be documented before they are used in outputs, generated facts, or evaluation cases.

Important examples:

- `store_id` is the canonical store identifier in CSV and SQL outputs.
- `entity_id` is the retrieval-layer identifier generated from `store_id`.
- `region_type` is weak operating-context evidence, not a hard market-area classification.
- `order_conversion_rate_pct` follows the Meituan backend definition based on `order_users / entry_users`.
- `activity_cost_ratio_pct` is activity cost divided by activity original transaction amount, not traditional ROI.
- `top3_sku_transaction_amount_share_pct` is lightweight top-SKU concentration evidence, not full product-category share.

## 3. SQL Diagnostics

Main files:

- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`
- `retail_ops/sql/03_demo2_pairwise_comparability_gate.sql`

SQL is used to prepare comparison-ready diagnostic evidence. It should not assign fixed store-stage labels or make final operating decisions.

The current SQL path has three stages.

### 3.1 Demo 1: Store A Month-over-Month Diagnostic

Demo 1 organizes Store A across February, March, and April 2026.

It supports cautious interpretation of:

- visibility and entry movement;
- search-entry structure;
- transaction and order movement;
- order conversion and average order value;
- activity involvement;
- refund and invalid-order pressure;
- limited top-SKU evidence.

Demo 1 is useful for showing that one store cannot be understood through a single metric.

### 3.2 Demo 2: Same-Period Cross-Store Diagnostic

Demo 2 organizes Stores B-F in the same March 2026 reporting window.

It creates same-period diagnostic rows and comparison-limit notes before any cross-store interpretation is attempted.

Demo 2 is still limited. It does not rank stores, classify markets, or decide which store's strategy should be copied.

### 3.3 Demo 3: Pairwise Comparability Gate

Demo 3 takes the Demo 2 output and creates pairwise comparison rows.

It tests whether two store-period rows can be compared for one narrow question at a time:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

The output is:

- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`

Demo 3 does not create a final recommendation. It creates a gate: whether a pair is comparable, partially comparable, or not comparable for the selected question, with limitation notes preserved.

## 4. Memory Facts and Pairwise Evidence

Main files:

- `retail_ops/outputs/generated_retail_memory_facts.json`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`
- `retail_ops/scripts/generate_demo2_retail_memory_facts.py`

The memory-facing facts record store-period observations, calculation notes, confidence, source fields, and limitations.

Demo 1 and Demo 2 currently generate memory facts.

Demo 3 currently produces pairwise gate rows, not memory facts. It also has a narrow file-backed answer path that reads the saved pairwise output and returns the pairwise decision, relevant gap fields, and limitation notes.

## 5. Retrieval and Evaluation

Current retail evaluation checks whether the system can answer from supported evidence and qualify or refuse unsupported claims.

Current implemented scopes:

- Demo 1: Store A month-over-month diagnostic.
- Demo 2: five anonymized Stores B-F in the same March 2026 reporting window.
- Demo 3: pairwise comparability gate over the existing Demo 2 B-F output.

Current API boundary:

- `/chat_retail_ops_kb` supports Store A Demo 1.
- `/chat_retail_ops_demo2_kb` supports the Demo 2 file-backed facts path.
- Demo 3 is currently SQL output, saved CSV output, documentation, validation, offline evaluation, and a narrow file-backed answer script. It is not yet exposed through a retrieval endpoint or API endpoint.

## 6. Why This Architecture Fits the Business Problem

The operating problem is not simply whether one store has higher sales than another.

For Meituan instant retail, store competition depends on whether a store can be seen, entered, ordered from, and repeatedly selected within a specific local context. Promotion, subsidy, price, SKU mix, ranking position, and fulfillment quality are operating levers inside that chain.

The architecture therefore puts comparability before recommendation.

A later answer should first ask:

- Are the stores from the same reporting period?
- Are the required fields present?
- Is the comparison question narrow enough?
- Are activity, refund, invalid-order, store-type, and SKU differences too large?
- Does the evidence support comparison, partial comparison, or refusal?

Only after those checks should the system discuss possible operating interpretation.
