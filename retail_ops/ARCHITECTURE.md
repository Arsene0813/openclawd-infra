# Retail Operations Architecture

This file gives a compact map of the current retail decision-support prototype.

The project starts from a practical Meituan instant-retail problem: single-store backend reports are detailed, but they are mainly designed for reviewing one store at a time. Cross-store decisions need aligned reporting windows, comparable store context, and explicit interpretation limits.

## Current Demo Scope

| Demo | Status | Purpose |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic. |
| Demo 2 | Implemented | Stores B-F same-period diagnostic structure. |
| Pairwise comparability gate | Future work | Planned gate for judging whether selected store-period records can be compared for a specific operating question. |

## Current Data Path

The current file-based data path is:

    selected Meituan backend metrics
    -> canonical CSV tables
    -> DATA_DICTIONARY.md field contract
    -> SQL diagnostic output
    -> generated retail memory facts
    -> validation and scenario-based boundary checks

The design is intentionally file-based at this stage. The priority is to keep each diagnostic claim traceable to source fields and output files before expanding toward a larger 48-store workflow.

## Layer Contract

| Layer | Input | Output | Boundary |
|---|---|---|---|
| Backend evidence | Selected Meituan merchant-backend metrics and manually structured evidence tables. | Canonical CSV source files. | Not full automated ingestion. |
| Metric contract | Canonical CSV fields and backend definitions. | `retail_ops/data/DATA_DICTIONARY.md` and `retail_ops/LINEAGE.md`. | Existing Meituan backend metrics should not be silently renamed or redefined. |
| SQL diagnostics | Store-period, search, activity, refund, order-quality, and top-SKU evidence. | SQL output files with ratios, shares, pressure indicators, and limitation notes. | SQL should not assign fixed store-stage labels or final operating decisions. |
| Generated memory facts | SQL outputs and supporting source tables. | Retrieval-facing memory facts with observed values, source fields, calculation notes, confidence, and limitations. | Memory facts are summaries, not raw backend exports. |
| Offline evaluation | Generated facts, SQL outputs, and current-scope docs. | Eval result text files and consistency checks. | Evaluations check evidence boundaries; they are not causal business experiments. |

## Implemented Source Files

Current source files include:

- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`

## Implemented SQL Files

- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

The second file keeps its historical path name for reference stability, but its current implemented meaning is a same-period cross-store diagnostic, not an implemented pairwise gate.

## Implemented Memory Outputs

- `retail_ops/outputs/generated_retail_memory_facts.json`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

The memory-facing facts record:

- store identity
- reporting period
- observed values
- source fields
- calculation notes
- evidence-trace confidence
- limitations

## Current Boundary

The current implemented retail scope stops at Demo 2.

The future pairwise comparability gate is documented in `retail_ops/COMPARABILITY_GATE_V0.md`, but the current SQL output and generated facts should be read as diagnostic evidence rather than as a transfer rule or store-ranking system.
