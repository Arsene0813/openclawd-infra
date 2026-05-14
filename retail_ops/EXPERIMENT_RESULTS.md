# Retail Operations Experiment Results

These are implementation checks for a staged decision-support prototype. They are not causal business experiments and not broad LLM benchmarks.

Each record below follows the same structure: question, evidence path, expected behavior, and current result.

## Experiment 1: Store A Month-over-Month Diagnostic

Question: Can selected Meituan backend metrics for one store be organized into a month-over-month diagnostic without changing the backend metric meanings?

Evidence path:

- Source data: `retail_ops/data/store_a_monthly_metrics.csv`
- SQL: `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- Output: `retail_ops/outputs/store_a_demo1_sql_output.csv`
- Generated facts: `retail_ops/outputs/generated_retail_memory_facts.json`

Expected behavior:

The output may describe observed month-over-month movement, but it should not attribute performance change to one metric alone.

Current result:

Implemented. Included in:

    python3 retail_ops/scripts/validate_retail_data_contract.py

## Experiment 2: Demo 2 Same-Period Store Diagnostic

Question: Can selected B-F store-period rows be placed under one March 2026 reporting window and one field contract before any stronger comparison is attempted?

Evidence path:

- Source data: `retail_ops/data/demo2_store_period_metrics.csv`
- SQL: `retail_ops/sql/02_demo2_cross_store_comparability.sql`
- Output: `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

Expected behavior:

The SQL output should include `comparison_scope_flag` and `comparison_limit_notes`, while staying at row-level same-period diagnostic scope.

Current result:

Implemented. Checked by:

    python3 eval/eval_retail_demo2_scope_boundary.py

Result path:

    eval/results/eval_retail_demo2_scope_boundary_result.txt

## Experiment 3: Demo 2 Memory-Fact Generation

Question: Can the Demo 2 diagnostic output be converted into retrieval-facing memory facts without losing source fields, observed values, or limitation notes?

Evidence path:

- Generator: `retail_ops/scripts/generate_demo2_retail_memory_facts.py`
- Generated facts: `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- Eval: `eval/eval_retail_demo2_facts.py`

Expected behavior:

Generated facts should preserve canonical field names and expose the main evidence slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `order_quality_pressure_profile`
- `top3_sku_product_mix_note`
- `single_metric_attribution_guard`

Current result:

Implemented. Checked by:

    python3 eval/eval_retail_demo2_facts.py

Result path:

    eval/results/eval_retail_demo2_facts_result.txt

## Experiment 4: Answer-Boundary Behavior

Question: Can expected answer patterns preserve metric boundaries when Demo 2 evidence is used?

Evidence path:

- Eval: `eval/eval_retail_demo2_answer_behavior.py`
- SQL output: `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- Generated facts: `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

Expected behavior:

The answer checks should preserve these boundaries:

- `activity_cost_ratio_pct` is not traditional ROI or profit margin.
- `top3_sku_transaction_amount_share_pct` is not full product-category sales share.
- Search-entry evidence does not prove causal performance.
- Activity evidence describes operating-tool usage, not automatic promotion-transfer logic.
- `same_period_diagnostic_ready` is not a finished pairwise comparability decision.
- `region_type` is weak context only.

Current result:

Implemented as offline scenario checks. Checked by:

    python3 eval/eval_retail_demo2_answer_behavior.py

Result path:

    eval/results/eval_retail_demo2_answer_behavior_result.txt

## Experiment 5: Future Comparability-Gate Contract

Question: Can the project document a future pairwise comparability gate without accidentally exposing it as a finished current feature?

Evidence path:

- Design note: `retail_ops/COMPARABILITY_GATE_V0.md`
- Eval: `eval/eval_future_comparability_gate_contract.py`

Expected behavior:

The future gate may define planned factors such as transaction order volume, transaction amount, activity status, activity intensity, store type, region and market context, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows. It should not appear as a current implemented gate in Demo 2 outputs.

Current result:

Contract documented as future work. Checked by:

    python3 eval/eval_future_comparability_gate_contract.py

## Experiment 6: Whole-Project Consistency Check

Question: Do the current reviewer-facing documents, retail docs, scripts, and outputs remain consistent with the Demo 1 / Demo 2 scope?

Evidence path:

- Validator: `scripts/validate_project_consistency.py`
- Data-contract validator: `retail_ops/scripts/validate_retail_data_contract.py`

Expected behavior:

The project should keep Demo 2 as the current implemented retail scope, keep the pairwise comparability gate as future work, and avoid stale or overclaimed wording.

Current result:

Checked by:

    python3 retail_ops/scripts/validate_retail_data_contract.py
    python3 scripts/validate_project_consistency.py

## Demo 2 Derived-Metric Scope

Demo 2 is intentionally narrower than Demo 1. It is a same-period B-F diagnostic for field-contract consistency and comparison-boundary behavior, not a full multi-store diagnostic model. Some dictionary-defined derived metrics, including `refund_order_pressure_pct` and `search_exposure_share_pct`, are not expanded as required Demo 2 output columns at this stage.
