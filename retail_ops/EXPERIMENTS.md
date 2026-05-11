# Retail Operations Experiment Map

This file records the current analytical experiments in the retail operations extension.

The purpose is to make the prototype easier to review as a data-science decision-support project. These are not production experiments and not causal A/B tests.

## Experiment 1: Store A Month-over-Month Diagnostic

Question:

Can a single store's monthly operating movement be interpreted without reducing the result to one metric?

Evidence:

- Store A February, March, and April 2026 store-period metrics.
- Store A top-SKU evidence.
- SQL-derived month-over-month diagnostics.
- Generated Store A retail memory facts.

Pass condition:

The system can describe exposure, entry, ranking, transaction, conversion, activity, refund, invalid-order, and top-SKU movement together. It should not claim that one metric alone caused the result.

## Experiment 2: Demo 2 Same-Period Cross-Store Comparability

Question:

Can multiple store-period records be structured before cross-store interpretation?

Evidence:

- Stores B-F, March 2026.
- Store-period metrics.
- Top search-term evidence.
- Top-SKU evidence.
- SQL output with `comparison_scope_flag` and `comparison_limit_notes`.

Pass condition:

The system can compare stores only within the documented scope and must preserve limits related to region context, store type, activity involvement, refund pressure, invalid-order pressure, product-mix evidence, and data completeness.

## Experiment 3: Retail Memory Fact Generation

Question:

Can SQL diagnostic outputs be converted into retrieval-facing memory facts without losing source fields and limitations?

Evidence:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- `retail_ops/scripts/generate_demo2_retail_memory_facts.py`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

Pass condition:

Each generated fact keeps the store entity, period, slot, observed values, calculation notes, source fields, confidence, source path, lineage path, and limitations.

## Experiment 4: Unsupported Claim Guard

Question:

Does the system avoid overclaiming when the current evidence does not support a conclusion?

Evidence:

- Retail evaluation cases.
- Demo 2 facts evaluation.
- Data dictionary and lineage rules.

Pass condition:

The system should qualify or refuse unsupported claims about causal attribution, audited profit, full 48-store generalization, final store ranking, promotion decisions, or full product-category share.
