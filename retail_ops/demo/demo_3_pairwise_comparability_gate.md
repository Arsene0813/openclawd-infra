# Demo 3: Pairwise Comparability Gate

## Purpose

Demo 3 turns the Demo 2 cross-store diagnostic into a small pairwise comparability gate.

The operating question is narrow:

> Can these two store-period rows be compared for this specific question?

This is closer to the real multi-store problem than a simple store ranking. A pair of stores may be usable for comparing search-entry structure, but not usable for transferring activity or subsidy strategy.

## Input

Demo 3 uses the existing Demo 2 output:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

That file carries normalized backend fields and SQL-derived diagnostics for Stores B-F in March 2026.

## Output

Demo 3 generates:

- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`

The output compares every pair of stores across three question types:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

With five stores, there are ten store pairs. Each pair is tested against three question types, producing thirty pairwise question rows.

## Why This Gate Is Useful

Meituan instant-retail stores compete through a chain of operating conditions:

- being seen;
- being entered;
- being ordered;
- being selected again or maintaining market share.

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are operating levers inside that chain.

The gate checks whether two store rows are similar enough for a specific comparison before any operating pattern is copied from one store to another.

A comparison can be valid for one question and invalid for another.

## Question Types

| `comparison_question_type` | What it tests | What it does not prove |
|---|---|---|
| `search_entry_structure` | Whether the pair can be compared for search-entry structure. | It does not prove that one store's ranking or title strategy should be copied. |
| `activity_transfer` | Whether activity-related evidence is similar enough to discuss transferability. | It does not mean direct promotion copying. |
| `order_quality_pressure` | Whether refund pressure and invalid-order pressure can be compared with limits. | It does not identify the exact cause of refund or cancellation pressure. |

## Region-Type Boundary

Demo 3 keeps `region_type` as weak market-context evidence.

`region_type` is not a market-area classification and is not a hard peer-store grouping rule. The current sample is too small to classify stores reliably by market area.

Future market-area classification should require stronger evidence such as purchasing power, delivery radius, competition, order structure, price pressure, activity intensity, refund pressure, invalid-order pressure, and SKU evidence.

Demo 3 therefore uses `region_type_comparison_note` only as a limitation note.

## Prototype Threshold Rationale

The current Demo 3 thresholds are prototype guardrails. They make comparison behavior explicit and testable on the current Stores B-F March 2026 sample.

They are not statistical significance thresholds, not universal Meituan operating rules, and not final business-policy thresholds.

| Threshold use | Current rule | Interpretation boundary |
|---|---:|---|
| Search-entry structure | `search_entry_share_gap_pct <= 15` can support `comparable_with_limits` for `search_entry_structure`. | This only supports a narrow traffic-structure comparison. |
| Activity-transfer guard | `activity_order_share_gap_pct > 15` constrains activity-transfer comparison. | Large activity-order-share gaps mean activity involvement differs too much for direct transfer. |
| Activity-cost guard | `activity_cost_ratio_gap_pct > 10` constrains activity-transfer comparison. | Activity-cost-ratio gaps describe different subsidy/cost structures, not ROI or profit margin. |
| Refund-pressure guard | `refund_pressure_gap_pct > 5` constrains activity-transfer and order-quality comparison. | Refund amount is dated by refund-success date and is not a perfect original-order cohort measure. |
| Invalid-order-pressure guard | `invalid_order_pressure_gap_pct > 4` constrains activity-transfer and order-quality comparison. | Cancelled-order pressure may reflect fulfillment, stock, customer, or operational differences not visible in the current sample. |

For a later 48-store version, these thresholds should be recalibrated with broader distributional evidence, store-type segmentation, repeated reporting windows, and manually reviewed operating outcomes.

## Pairwise Gate Decisions

The implemented SQL-derived decision field is `pairwise_comparison_decision`.

| `pairwise_comparison_decision` | Meaning |
|---|---|
| `comparable_with_limits` | The pair can be compared for the selected narrow question, but limits must be preserved. |
| `partially_comparable` | Some evidence is comparable, but gaps or context differences prevent a stronger conclusion. |
| `not_comparable_for_strategy_transfer` | The pair is too different or incomplete for transferring an operating strategy. |

These decisions are not store-stage labels and are not final operating recommendations.

## Concrete Examples from Current Output

### Example 1: Same pair, different question

Pair: Store B vs Store E
Period: March 2026

| Question type | Decision | Key evidence | How to read it |
|---|---|---|---|
| `search_entry_structure` | `comparable_with_limits` | `search_entry_share_gap_pct = 1.69` | The pair can be discussed for narrow search-entry structure, but this does not support copying activity strategy. |
| `activity_transfer` | `not_comparable_for_strategy_transfer` | `activity_order_share_gap_pct = 19.64`, `refund_pressure_gap_pct = 8.34`, `invalid_order_pressure_gap_pct = 6.75` | The pair should not be used for direct activity-strategy transfer. |
| `order_quality_pressure` | `partially_comparable` | refund and invalid-order pressure gaps are large | The pair can only support a limited order-quality discussion, not a broad operating recommendation. |

This example shows why Demo 3 is question-specific. The same store pair can be usable for one narrow comparison and unsafe for another.

### Example 2: Same `region_type` does not mean same market area

Pair: Store B vs Store C
Period: March 2026

Both stores have the same `region_type` value, but Demo 3 still uses:

- `region_type_value_matches_but_not_market_classification`

The pair is not automatically treated as a reliable peer group.

For this pair, `search_entry_share_gap_pct = 19.9`, `activity_order_share_gap_pct = 17.77`, and `activity_cost_ratio_gap_pct = 14.67`.

This is why Demo 3 does not use `region_type` alone as a comparability rule.

### Example 3: Similar search-entry structure can still fail activity transfer

Pair: Store B vs Store F
Period: March 2026

| Question type | Decision | Key evidence |
|---|---|---|
| `search_entry_structure` | `comparable_with_limits` | `search_entry_share_gap_pct = 0.21` |
| `activity_transfer` | `not_comparable_for_strategy_transfer` | `activity_cost_ratio_gap_pct = 11.96` |
| `order_quality_pressure` | `comparable_with_limits` | refund and invalid-order pressure gaps stay within current prototype guardrails |

This example shows the current gate logic: traffic-structure similarity is not enough to transfer activity or subsidy strategy.

## Current Boundary

Demo 3 is a small comparability prototype over the current B-F March 2026 sample.

It currently supports:

- pairwise comparison over the existing Demo 2 output;
- three narrow question types;
- documented SQL-derived gap fields;
- validation and offline evaluation.

It does not currently expose a retrieval endpoint.

Expansion toward all 48 stores should keep the same field contract, threshold documentation, and limitation-preserving answer behavior.
