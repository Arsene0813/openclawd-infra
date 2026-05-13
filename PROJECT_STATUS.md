# Project Status

This repository currently demonstrates two implemented retail-operations diagnostics on top of the lifecycle-aware memory-layer prototype.

## Implemented Retail Scope

| Area | Status | Role |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic. |
| Demo 2 | Implemented | Stores B-F same-period cross-store diagnostic structure. |
| Comparability gate | Future work | Planned pairwise gate for judging whether selected stores can be compared for a selected operating question. |

## Current Retail Narrative

The Meituan merchant backend provides detailed store-level data. The harder technical problem is that the backend workflow is mainly designed for reviewing one store at a time.

Because the business has 48 stores, the longer-term decision-support problem is to decide:

- which stores can be compared;
- under what conditions they can be compared;
- which metric changes support a cautious operating interpretation;
- which comparisons should be refused or qualified.

The current repository does not claim to solve the full 48-store comparison problem. It shows the first two implemented steps:

1. convert selected Meituan backend metrics into structured store-period diagnostics;
2. compare selected same-period store rows with explicit field definitions and interpretation limits.

## Why the Comparability Gate Is Future Work

A reliable comparability gate should depend on factors such as:

- transaction order volume;
- transaction amount;
- activity status;
- activity intensity;
- store type;
- region and market context;
- competition environment;
- SKU structure;
- refund pressure;
- invalid-order pressure;
- repeated reporting windows.

The current demo sample is still small.

To avoid subjective regional classification, the project does not currently classify store locations into market-area types. The existing `region_type` field remains weak context only. It is not a hard market-area classification, store-stage label, or peer-store grouping rule.

A future 48-store version can revisit the comparability gate after more store data, more reporting windows, and stronger market-context fields are available.

## Current Validation Focus

Current validation focuses on:

- data dictionary consistency;
- Demo 1 output consistency;
- Demo 2 output consistency;
- field-definition boundaries;
- generated memory fact structure;
- offline answer-boundary checks;
- refusal to overclaim from limited sample data.

## Current Commands

Run the current implemented checks with:

    python3 retail_ops/scripts/validate_retail_data_contract.py
    python3 scripts/validate_demo2_retail_endpoint_boundary.py
    python3 scripts/validate_project_consistency.py
