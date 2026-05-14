# Future Work: Comparability Gate

This note records the next planned stage of the retail operations prototype.

The current implemented retail scope stops at Demo 2:

- Demo 1: Store A month-over-month diagnostic.
- Demo 2: Stores B-F same-period cross-store diagnostic structure.

The next stage is a pairwise comparability gate. It is documented here as future work, not as a finished demo.

## Why This Gate Matters

The Meituan merchant backend provides rich store-level data, but the workflow is mainly designed for reviewing one store at a time.

For a 48-store operation, the harder problem is deciding which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

The gate should not answer which store is best.

It should answer a narrower question: can these two store-period records be compared for this specific operating question, and what limitations should constrain the answer?

## Current Demo 2 Output vs Future Pairwise Gate

### Current Demo 2 Output

Current Demo 2 provides:

- row-level same-period diagnostic readiness;
- current implemented fields including `comparison_scope_flag` and `comparison_limit_notes`;
- evidence about whether each store-period row is inside the current Demo 2 scope;
- interpretation limits that should be preserved in later retrieval or analysis.

The current `comparison_limit_notes` field covers only a limited subset of future gate factors. It does not yet encode transaction-volume bands, transaction-scale bands, competition context, repeated-window stability, or data-supported market-area classification.

### Future Pairwise Comparability Gate

A future pairwise comparability gate should provide:

- pair-level decision for a specific comparison question;
- input that identifies a reference store-period, a candidate store-period, and the operating question being asked;
- output that explains whether the selected records can be compared;
- supporting evidence and limiting factors.

The current `comparison_scope_flag` should not be treated as a finished pairwise comparability decision.

## What Store Comparability Should Depend On

A reliable gate should consider at least:

- transaction order volume;
- transaction amount;
- activity status;
- activity involvement and possible activity status inferred from existing activity fields;
- activity intensity;
- store type;
- region and market context;
- competition environment;
- SKU structure;
- refund pressure;
- invalid-order pressure;
- repeated reporting windows.

These factors affect whether two stores are actually comparable as operating cases.

The future gate should not create a store label from one threshold. It should compare selected store-period records for a specific operating question.

## Candidate Gate Factors

| Future factor | Current evidence available | Current limitation | Future evidence needed |
|---|---|---|---|
| Reporting-window alignment | `period_start`, `period_end`, `period_month` | Demo 2 uses one March 2026 window only. | Repeated windows across more stores. |
| Order volume | `transaction_orders`, `valid_orders`, `invalid_orders` | One-period volume may be unstable. | Repeated order-volume bands. |
| Transaction scale | `transaction_amount`, `average_order_value`, `estimated_income_proxy` | `estimated_income_proxy` is only a platform-displayed proxy. | Clearer income and cost evidence if available. |
| Activity involvement | `activity_orders`, `activity_order_share_pct`, `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount` | Promotion cycle dates, mechanism details, and exact activity calendar are not included. | Activity calendar, promotion type, and activity mechanism evidence. |
| Activity intensity | `activity_order_share_pct`, `activity_cost_ratio_pct`, `activity_cost` | Current thresholds are diagnostic guardrails, not a finished transfer rule. | Repeated activity windows and stronger evidence on activity mechanism. |
| Store type | `store_type` | Store type alone does not prove comparability. | Broader sample by store type. |
| Region and market context | `region_type` | Current demo sample is too small for reliable regional classification. | More store data, local consumption-level evidence, and competition-context evidence. |
| Competition context | Not currently structured. | Local competitor density and price pressure are not included. | Competitor and local market evidence. |
| SKU structure | Top-SKU transaction-amount and sales-volume evidence. | Top-SKU evidence is not full category-share analysis. | Broader SKU classification or category mapping. |
| Refund pressure | `refund_amount`, `refund_pressure_pct`, `full_refund_orders`, `refund_orders_all_or_partial` | Refund amount is counted by refund-success date. | Cohort-level refund or refund-reason evidence if available. |
| Invalid-order pressure | `valid_orders`, `invalid_orders`, `invalid_order_pressure_pct` | Cancellation reasons are not included. | Invalid-order reason categories. |
| Data completeness | `comparison_scope_flag`, `comparison_limit_notes` | Current notes are diagnostic guardrails, not a finished gate. | Explicit pairwise decision output after broader data. |

## Current `region_type` Boundary

The current demo sample is still small. Because of that, the project deliberately avoids subjective regional classification.

The existing `region_type` field remains weak context only. It must not be used as:

- a hard market-area classification;
- a store-stage label;
- a peer-store grouping rule;
- a substitute for local consumption-level evidence;
- a substitute for competition-context evidence.

A more reliable market-area classification should wait until more store data is available and can be judged together with data comparability, actual local consumption level, competition environment, activity conditions, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

The current project should not classify stores into city-center, county, community, mature-market, or immature-market groups based on intuition.

## Future Gate Input Shape

A future comparability gate should take a narrow query shape:

- `reference_store_id`
- `candidate_store_id`
- `period_start`
- `period_end`
- `comparison_question_type`

These names describe a proposed future gate input shape. They are not current implemented data-contract fields.

Before any of them is used in CSV outputs, generated memory facts, or evaluation cases, the field must be documented in `retail_ops/data/DATA_DICTIONARY.md` and linked through `retail_ops/LINEAGE.md`.

## Future Gate Output Shape

A future comparability gate should return:

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
