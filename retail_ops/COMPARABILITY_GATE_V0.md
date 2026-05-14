# Future Work: Comparability Gate

This note records the next planned stage of the retail operations prototype.

The current implemented retail scope stops at Demo 2:

- Demo 1: Store A month-over-month diagnostic.
- Demo 2: Stores B-F same-period cross-store diagnostic structure.

The next stage is a pairwise comparability gate. It is documented here as future work, not as a finished demo.

## Why This Gate Matters

The Meituan merchant backend provides rich store-level data, but the workflow is mainly designed for reviewing one store at a time.

For a 48-store operation, the harder problem is deciding which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

This is not a ranking problem. The gate should not answer "which store is best." It should answer a narrower question:

> Can these two store-period records be compared for this specific operating question, and what limitations should constrain the answer?

## What Store Comparability Should Depend On

A reliable gate should consider at least:

  * transaction order volume;
  * transaction amount;
  * activity status;
  * activity intensity;
  * store type;
  * region and market context;
  * competition environment;
  * SKU structure;
  * refund pressure;
  * invalid-order pressure;
  * repeated reporting windows.

These factors affect whether two stores are actually comparable as operating cases.

The future gate should not create a store label from one threshold. It should compare selected store-period records for a specific operating question.

| Future factor | Current evidence available | Current limitation | Future evidence needed |
|---|---|---|---|
| Reporting-window alignment | `period_start`, `period_end`, `period_month` | Demo 2 uses one March 2026 window only | repeated windows across more stores |
| Order volume | `transaction_orders`, `valid_orders`, `invalid_orders` | one-period volume may be unstable | repeated order-volume bands |
| Transaction scale | `transaction_amount`, `average_order_value`, `estimated_income_proxy` | estimated income is only a platform-displayed proxy | clearer income / cost breakdown if available |
| Activity involvement | `activity_orders`, `activity_order_share_pct`, `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount` | promotion cycle dates and mechanism details are not included | activity calendar and promotion type evidence |
| Store type | `store_type` | store type alone does not prove comparability | broader sample by store type |
| Region / market context | `region_type` | current demo sample is too small for reliable regional classification | more store data, local consumption-level evidence, and competition-context evidence |
| Competition context | not currently structured | local competitor density and price pressure are not included | competitor / local market evidence |
| SKU structure | top-SKU transaction-amount and sales-volume evidence | top-SKU evidence is not full category-share analysis | broader SKU classification or category mapping |
| Refund pressure | `refund_amount`, `refund_pressure_pct`, `full_refund_orders`, `refund_orders_all_or_partial` | refund amount is counted by refund-success date | cohort-level refund or reason evidence if available |
| Invalid-order pressure | `valid_orders`, `invalid_orders`, `invalid_order_pressure_pct` | cancellation reasons are not included | invalid-order reason categories |
| Data completeness | `comparison_scope_flag`, `comparison_limit_notes` | current notes are diagnostic guardrails, not a finished gate | explicit pairwise decision output after broader data |
## Current `region_type` Boundary

The current demo sample is still small, so the project deliberately avoids classifying store locations by subjective experience, intuition, address impression, or habitual labels.

Taking `region_type` as an example, the current project does not use it to decide that a store belongs to a fixed market-area type. It is kept only as currently available region-context evidence. It should not be used as:

  * a hard market-area classification;
  * a store-stage label;
  * a peer-store grouping rule;
  * a substitute for local consumption-level evidence;
  * a substitute for competition-context evidence.

A more reliable market-area classification should wait until more store data is available and can be judged together with data comparability, actual local consumption level, competition environment, activity conditions, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

The current project should not classify stores into city-center, county, community, mature-market, or immature-market groups based on intuition.

## Future Gate Output Shape

The names below describe a proposed future gate output shape. They are not current implemented data-contract fields. Before any of them is used in CSV outputs, generated memory facts, or evaluation cases, the field must be documented in `retail_ops/data/DATA_DICTIONARY.md` and linked through `retail_ops/LINEAGE.md`.

A future comparability gate should take a narrow query shape:

- `reference_store_id`
- `candidate_store_id`
- `period_start`
- `period_end`
- `comparison_question_type`

It should return:

- `comparison_decision`: comparable, comparable with limits, not comparable, or insufficient evidence;
- `supporting_fields`: canonical source fields and SQL-derived diagnostics used;
- `blocking_or_limiting_factors`: period mismatch, activity intensity, store type, weak region context, refund pressure, invalid-order pressure, SKU evidence limits, missing competition context, or missing repeated reporting windows;
- `allowed_interpretation`: what can be discussed;
- `unsupported_interpretation`: what must not be claimed.

The gate should not globally rank all stores. It should decide whether a selected pair can be compared for a selected operating question.

## Future Market-Context Fields

A future 48-store version can revisit stronger market-context fields after broader store data and more reporting windows are added.

Possible future fields could include:

- `market_area_type`
- `market_area_type_source`
- `market_area_type_confidence`

These fields are not implemented in the current repository. They should only be added after they are documented in `retail_ops/data/DATA_DICTIONARY.md` and supported by broader evidence.

