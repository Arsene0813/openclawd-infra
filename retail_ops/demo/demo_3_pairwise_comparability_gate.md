# Demo 3: Pairwise Comparability Gate

## Purpose

Demo 3 turns the Demo 2 cross-store diagnostic into a small pairwise comparability gate.

The operating question is narrow: can these two store-period rows be compared for this specific question?

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

- being seen
- being entered
- being ordered
- being selected again or maintaining market share

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are operating levers inside that chain.

The gate checks whether two store rows are similar enough for a specific comparison before any operating pattern is copied from one store to another.

Example:

- Two stores may both have search-heavy entry structures, supporting a narrow search-entry comparison.
- The same pair may have very different activity-order share, activity-cost ratio, refund pressure, invalid-order pressure, or store operating format.
- In that case, the gate should preserve the comparison limit instead of turning the pair into a direct strategy-transfer case.

## Region-Type Boundary

Demo 3 keeps `region_type` as weak market-context evidence.

`region_type` is not a market-area classification and is not a hard peer-store grouping rule. The current sample is too small to classify stores reliably by market area.

Future market-area classification should require stronger evidence such as purchasing power, delivery radius, competition, order structure, price pressure, activity intensity, refund pressure, invalid-order pressure, and SKU evidence.

Demo 3 therefore uses `region_type_comparison_note` only as a limitation note.

## Prototype Threshold Rationale

The current Demo 3 thresholds are prototype guardrails. They make comparison behavior explicit and testable on the current Stores B-F March 2026 sample.

They are not statistical significance thresholds, not universal Meituan operating rules, and not final business-policy thresholds.

Current implemented examples:

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

Current implemented values:

| `pairwise_comparison_decision` | Meaning |
|---|---|
| `comparable_with_limits` | The pair can be compared for the selected narrow question, but limits must be preserved. |
| `partially_comparable` | Some evidence is comparable, but gaps or context differences prevent a stronger conclusion. |
| `not_comparable_for_strategy_transfer` | The pair is too different or incomplete for transferring an operating strategy. |

## Current Boundary

Demo 3 is a small comparability prototype over the current B-F March 2026 sample.

It currently supports:

- pairwise comparison over the existing Demo 2 output
- three narrow question types
- documented SQL-derived gap fields
- validation and offline evaluation

Expansion toward all 48 stores should keep the same field contract, threshold documentation, and limitation-preserving answer behavior.
