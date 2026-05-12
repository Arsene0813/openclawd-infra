# Demo 3: Pairwise Comparability Gate

## Purpose

Demo 3 turns the Demo 2 cross-store diagnostic into a small pairwise comparability gate.

The purpose is not to rank stores. The purpose is to test whether two store-period rows can be compared for a narrow operating question before any strategy transfer is considered.

## Input

Demo 3 uses the existing Demo 2 output:

- retail_ops/outputs/demo2_cross_store_comparability_output.csv

That file already carries normalized backend fields and SQL-derived diagnostics for Stores B-F in March 2026.

## Output

Demo 3 generates:

- retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv

The output compares every pair of stores across three question types:

- search_entry_structure
- activity_transfer
- order_quality_pressure

With five stores, there are ten store pairs. Each pair is tested against three question types, producing thirty pairwise question rows.

## Why this gate is useful

A pair of stores may be comparable for one question but not for another.

For example, two stores may both rely heavily on search entry, which can support a narrow comparison of search-entry structure. But the same pair may not be comparable for subsidy or activity strategy transfer if activity-order share, activity-cost ratio, refund pressure, invalid-order pressure, or store operating format differs too much.

This is closer to the real operating problem: the question is not which store is best. The question is what kind of comparison is safe.

## Region-Type Boundary

Demo 3 does not use region_type as a market-area classification.

region_type is preserved only as weak context. The current sample is too small to classify stores reliably by market area. Future classification should require more stores and stronger evidence such as purchasing power, delivery radius, competition, order structure, price pressure, activity intensity, refund pressure, invalid-order pressure, and SKU evidence.

Demo 3 therefore uses region_type_comparison_note only as a limitation note. It is not a hard peer-store grouping rule.

## Pairwise Gate Decisions

Current implemented values:

| Decision | Meaning |
|---|---|
| comparable_with_limits | The pair can be compared for the selected narrow question, but limits must be preserved. |
| partially_comparable | Some evidence is comparable, but gaps or context differences prevent a stronger conclusion. |
| not_comparable_for_strategy_transfer | The pair is too different or incomplete for transferring an operating strategy. |

## Current Boundary

Demo 3 is still a small prototype. It does not implement full 48-store grouping, automated Meituan ingestion, market-area classification, or final operating recommendations.

Its value is that it makes comparability testable.
