# Cross-Store Meituan Metrics Comparison Report

## Purpose

Meituan merchant backend provides many detailed store-level indicators, but it is mainly designed for reviewing one store at a time.

For multi-store operations, the challenge is not the lack of data. The challenge is the difficulty of comparing many stores efficiently under the same metric structure.

This SQL-based extension reorganizes manually collected and anonymized Meituan backend metrics into a unified cross-store table. It calculates comparable indicators related to search traffic, conversion, refund pressure, promotion intensity, and visitor value.

## Data Scope

- Period: March 2026
- Stores: 5 anonymized representative stores
- Regions: Qingdao urban area and Yantai urban area
- Store types: self-operated and partner-operated
- Category: contact lenses and care-solution related instant retail
- Source: manually organized Meituan merchant backend metrics

## Derived Metrics

The SQL layer calculates:

1. Search visit share = search visitors / store visitors
2. Search exposure-to-visit rate = search visitors / search exposure
3. Refund revenue ratio = refund amount / revenue
4. Refund order ratio = refund orders / valid orders
5. Promotion GMV to revenue ratio = promotion GMV / revenue
6. Merchant subsidy to revenue ratio = merchant subsidy / revenue
7. Merchant subsidy to promotion GMV ratio = merchant subsidy / promotion GMV
8. Revenue per visitor = revenue / store visitors

## Cross-Store Observations

Most stores in this sample are strongly search-driven. 4 out of 5 stores have search visit share above 85%, suggesting that search visibility is a major traffic source in this category.

However, search traffic alone does not explain store performance. Store D has the highest revenue in this sample and can be reviewed as a reference store. It should not be copied directly without checking region, store type, product mix, promotion structure, refund pressure, fulfillment reliability, and customer profile.

Stores with visible refund pressure in the current sample: B, A, E.

Stores with high subsidy intensity but relatively low refund pressure: C.

These observations should not be treated as causal conclusions. They show why a cross-store SQL layer is useful before storing operational observations in an AI memory system.

## Generated Operational Memory Facts

The pipeline converts selected SQL observations into structured memory facts.

Current generated fact types include:

- `cross_store_pattern`
- `store_metric_signal`
- `risk_signal`
- `promotion_efficiency_signal`

Each fact includes scope, period, confidence, and source so that an AI assistant can reuse the observation conservatively.

## Decision-Support Principle

The memory layer should store operational observations with clear scope, time period, and confidence level.

It should avoid treating one store's pattern as a universal rule without checking region, store type, traffic source structure, refund pressure, promotion intensity, product mix, and visitor value.
