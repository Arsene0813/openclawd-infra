# Project Status

This repository currently demonstrates two implemented retail-operations diagnostics on top of the lifecycle-aware memory-layer prototype.

## Implemented Retail Scope

| Area | Status | Role |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic. |
| Demo 2 | Implemented | Stores B-F same-period cross-store diagnostic structure. |
| Comparability gate | Future work | Not currently implemented as a finished demo. |

## Current Retail Narrative

The Meituan merchant backend provides rich and usable store-level data. The limitation is not data quality. The limitation is that the backend workflow is mainly designed for reviewing one store at a time.

Because the business has 48 stores, the longer-term technical problem is to build a decision-support layer that can help decide which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

The current repository does not claim to solve the full 48-store comparison problem. It shows the first two implemented steps:

1. convert selected Meituan backend metrics into a structured store-period diagnostic;
2. compare selected stores in the same reporting window with explicit field definitions and interpretation limits.

## Why the Comparability Gate Is Future Work

A reliable comparability gate should depend on factors such as:

- transaction order volume;
- transaction amount;
- whether the store is currently under activity or promotion;
- activity intensity;
- store type;
- region and market context;
- competition environment;
- SKU structure;
- refund pressure;
- invalid-order pressure;
- repeated reporting windows.

The current demo sample is still small. To avoid subjective regional classification, the project does not currently classify store locations into market-area types.

The existing `region_type` field remains weak context only. It is not a hard market-area classification, store-stage label, or peer-store grouping rule.

A future 48-store version can revisit the comparability gate after more store data, more reporting windows, and stronger market-context fields are available.

## Current Validation Focus

Current validation should focus on:

- data dictionary consistency;
- Demo 1 output consistency;
- Demo 2 output consistency;
- field-definition boundaries;
- refusal to overclaim from limited sample data.

## Current Commands

Run the current implemented checks with:

python3 scripts/validate_project_consistency.py
python3 retail_ops/scripts/validate_retail_data_contract.py
python3 eval/eval_retail_demo2_scope_boundary.py
