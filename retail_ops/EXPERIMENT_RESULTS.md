# Retail Operations Experiment Results

These are review and evaluation cases, not causal experiments.

They test whether the prototype preserves evidence boundaries and avoids unsupported operating conclusions.

## Current Implemented Results

| Area | Status | Result type |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic output. |
| Demo 2 | Implemented | Stores B-F same-period cross-store diagnostic output. |
| Comparability gate | Future work | Planned pairwise gate, not an implemented experiment result. |

## What Demo 1 Shows

Demo 1 shows that selected Meituan backend metrics can be organized into a month-over-month diagnostic for one store.

The purpose is not to attribute performance to a single metric. The purpose is to preserve field definitions, time windows, and interpretation limits.

## What Demo 2 Shows

Demo 2 shows that selected same-period store rows can be structured for cross-store diagnostic review.

The purpose is not to rank stores. The purpose is to make selected store-period rows easier to inspect under consistent metric definitions.

## Why the Comparability Gate Is Not Yet an Experiment Result

A reliable future pairwise comparability gate should first decide whether stores can be compared at all.

That decision should depend on transaction order volume, transaction amount, current activity involvement and intensity based on existing activity fields, explicit activity status or campaign-calendar evidence if available, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

The current sample is still limited. To avoid subjective regional classification, the project does not currently classify store locations into market-area types. The existing `region_type` field remains weak context only.
