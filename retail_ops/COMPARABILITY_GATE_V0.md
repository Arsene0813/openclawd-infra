# Future Work: Comparability Gate

This note records the next planned stage of the retail operations prototype.

The current implemented retail scope stops at Demo 2:

- Demo 1: Store A month-over-month diagnostic.
- Demo 2: Stores B-F same-period cross-store diagnostic structure.

A pairwise comparability gate is not currently implemented as a finished demo.

## Why the gate is future work

The Meituan merchant backend provides rich and usable store-level data. The limitation is not data quality. The limitation is that the backend workflow is mainly designed for reviewing one store at a time.

For a 48-store operation, a useful decision-support system should eventually help decide which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

However, this is not a simple ranking problem. It should not be reduced to asking which store is better.

## What store comparability should depend on

A reliable comparability gate should consider at least:

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

These factors affect whether two stores are actually comparable as operating cases.

## Current limitation

The current demo sample is still small.

Because of that, this project deliberately avoids subjective regional classification. The existing `region_type` field remains weak context only. It must not be used as a hard market-area classification, store-stage label, or peer-store grouping rule.

For example, the current project should not classify stores into city-center, county, community, mature-market, or immature-market groups based on intuition.

## Future direction

A future 48-store version can revisit the comparability gate after more store data and more reporting windows are added.

At that stage, the project may add stronger market-context fields, but only after they are documented and supported by broader evidence.

Possible future fields could include:

- `market_area_type`
- `market_area_type_source`
- `market_area_type_confidence`

Until then, the project should stop at Demo 2 and describe the comparability gate as planned future work rather than an implemented conclusion.
