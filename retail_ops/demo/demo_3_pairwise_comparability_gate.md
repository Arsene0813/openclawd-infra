# Demo 3: Pairwise Comparability Gate

## 1. Purpose

Demo 3 turns the Demo 2 cross-store diagnostic output into a pairwise comparability gate.

The question is not which store is best.

The question is narrower:

Can two store-period rows be compared for a specific operating question?

This matters because two stores may be comparable for one question but not for another.

For example, two stores may have similar search-entry structure, so they can be discussed together for `search_entry_structure`. The same two stores may still differ too much in activity involvement, refund pressure, invalid-order pressure, or store type to support an `activity_transfer` comparison.

Demo 3 therefore checks comparability before interpretation.

## 2. Input

Demo 3 uses the existing Demo 2 output:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

That file contains same-period diagnostic evidence for Stores B-F in March 2026.

## 3. Output

Demo 3 generates:

- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`

With five stores, there are ten store pairs.

Each pair is tested against three supported question types, producing thirty pairwise question rows.

Current supported `comparison_question_type` values are:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

These names must stay consistent with:

- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/sql/03_demo2_pairwise_comparability_gate.sql`
- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`
- `eval/eval_retail_demo3_pairwise_gate.py`

## 4. Business Logic

Meituan instant-retail stores compete through a chain of operating conditions:

- being seen;
- being entered;
- being ordered;
- being selected again or maintaining local share.

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are operating levers inside that chain.

Those levers should not be interpreted in isolation.

A new store may need stronger activity or subsidy to gain first exposure and first orders. A store under external price pressure may use price or activity tools to defend visibility and local share. A store with high exposure but weak conversion should be read differently from a store with order growth and refund pressure.

Demo 3 keeps this logic narrow. It does not turn every metric difference into a recommendation. It first asks whether the two stores can be compared for the selected question.

## 5. Region-Type Boundary

Demo 3 keeps `region_type` as weak operating-context evidence.

`region_type` is not:

- a market-area classification;
- a store-stage label;
- a hard peer-grouping rule;
- a sufficient condition for strategy transfer.

The current sample is too small to classify stores reliably by market area.

Future market-area classification would require stronger evidence, such as purchasing power, delivery radius, local competition, price pressure, activity intensity, refund pressure, invalid-order pressure, SKU evidence, and repeated reporting windows.

Until those fields are defined and validated, Demo 3 uses `region_type_comparison_note` only as a limitation note.

## 6. Pairwise Decision Field

The main SQL-derived decision field is:

- `pairwise_comparison_decision`

Current implemented values are:

- `comparable_with_limits`
- `partially_comparable`
- `not_comparable_for_strategy_transfer`

`comparable_with_limits` means the pair can be compared for the selected narrow question, but limitation notes must still be preserved.

`partially_comparable` means some evidence can be compared, but gaps or context differences prevent a stronger conclusion.

`not_comparable_for_strategy_transfer` means the pair is too different or incomplete for transferring an operating strategy under the current evidence contract.

This field is a comparability gate. It is not a final operating recommendation.

## 7. Limitation Notes

The main limitation field is:

- `pairwise_limit_notes`

This field should be carried into any later answer.

It prevents the answer layer from hiding context differences such as:

- activity gap;
- activity-cost-ratio gap;
- refund-pressure gap;
- invalid-order-pressure gap;
- store-type difference;
- unresolved market classification;
- top-SKU concentration difference.

A later answer should not summarize the pair as simply comparable or not comparable. It should explain which evidence supports the comparison and which limits still apply.

## 8. Prototype Threshold Rationale

The current Demo 3 thresholds are prototype guardrails.

They make the comparison behavior explicit and testable on the current Stores B-F March 2026 sample.

They are not:

- statistical significance thresholds;
- universal Meituan operating rules;
- final business-policy thresholds.

Current threshold use:

- `search_entry_share_gap_pct` helps decide whether two stores can be discussed for `search_entry_structure`.
- `activity_order_share_gap_pct` helps limit `activity_transfer` when activity involvement differs too much.
- `activity_cost_ratio_gap_pct` helps limit `activity_transfer` when subsidy or activity-cost structure differs too much.
- `refund_pressure_gap_pct` helps limit `activity_transfer` and `order_quality_pressure` comparison.
- `invalid_order_pressure_gap_pct` helps limit `activity_transfer` and `order_quality_pressure` comparison.
- `top3_sku_concentration_gap_pct` adds a product-mix limitation note when top-SKU concentration differs.

These thresholds should be recalibrated before any broader 48-store version.

A later version would need broader distributional evidence, repeated reporting windows, store-type segmentation, and manually reviewed operating outcomes.

## 9. Current Gap Fields

Demo 3 uses SQL-derived gap fields to make comparison behavior explicit.

Current implemented gap fields include:

- `search_entry_share_gap_pct`
- `activity_order_share_gap_pct`
- `activity_cost_ratio_gap_pct`
- `refund_pressure_gap_pct`
- `invalid_order_pressure_gap_pct`
- `top3_sku_concentration_gap_pct`

These fields are diagnostic signals.

They are not causal proof.

For example, a large `activity_order_share_gap_pct` means the stores differ in activity involvement. It does not prove that activity caused the performance difference.

A large `refund_pressure_gap_pct` means refund pressure differs between the two stores. It does not identify the exact cause of refund behavior.

A large `top3_sku_concentration_gap_pct` means top-three SKU concentration differs. It does not represent full product-category sales share.

## 10. Directionality Boundary

`reference_store_id` and `candidate_store_id` identify the two stores in the pairwise output.

They do not mean that the reference store's strategy should be copied to the candidate store.

For example, a row with `reference_store_id = B`, `candidate_store_id = E`, and `comparison_question_type = activity_transfer` does not prove that Store B's activity strategy should transfer to Store E.

It only means the current gate has tested whether the B/E pair can be discussed for the narrow `activity_transfer` question under the implemented evidence contract.

Supported answer shape:

- state the selected pair;
- state the selected question type;
- state the pairwise decision;
- cite the relevant gap fields;
- carry the limitation notes;
- avoid final operating recommendations unless future evidence supports them.

Unsupported answer shape:

- Store B is better than Store E.
- Copy Store B's promotion strategy to Store E.
- These stores belong to the same market type.
- The activity strategy caused the result.
- Rank all stores by operating quality.

## 11. Current Boundary

Demo 3 currently supports:

- pairwise comparison over the existing Demo 2 output;
- three narrow question types;
- documented SQL-derived gap fields;
- saved CSV output;
- validation;
- offline evaluation.

Demo 3 currently does not support:

- full 48-store ranking;
- market-area classification;
- causal promotion-effect analysis;
- final operating recommendation;
- retrieval endpoint or API answer path.

The next narrow implementation step is to expose the Demo 3 pairwise output through a file-backed answer path. That answer path should return the pairwise decision, relevant gap fields, and limitation notes before discussing any possible operating interpretation.

## 12. Why This Matters

The value of Demo 3 is not complexity.

The value is that it makes comparison discipline explicit.

In the actual business problem, Meituan backend data is rich but mainly designed for single-store operation. With many stores, the difficult part is not only collecting more metrics. The difficult part is deciding which stores can be compared, under what question, and with what limitations.

Demo 3 is a small step toward that decision-support layer.
