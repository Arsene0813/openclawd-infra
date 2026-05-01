# Evaluation Report

## Purpose

This evaluation is designed to test whether the memory layer behaves reliably in commerce-oriented scenarios.

The goal is not to claim broad benchmark performance. Instead, the evaluation checks whether the system can manage changing business facts, retrieve active knowledge, avoid stale or unsupported information, and provide traceable supporting memory.

## Evaluation Scope

The current evaluation covers the following behavior categories:

| Category | Purpose |
|---|---|
| Fact extraction | Extract structured facts from product-related messages |
| Price overwrite | Use newer prices instead of older active prices |
| Entity separation | Keep different product facts separate |
| Stock status update | Update inventory-related facts over time |
| Promotion memory | Store and retrieve promotional information |
| Freshness filtering | Avoid treating time-limited information as always valid |
| Shipping policy | Retrieve and update delivery-related facts |
| Non-fact filtering | Avoid storing greetings or general opinions as business facts |
| Fallback / refusal | Avoid inventing unsupported answers |
| Traceable retrieval | Return supporting memory items for answers |
| Retail decision-support extension | Explore how the same memory design can support operational analysis |

## Summary of Current Cases

| Category | Number of Cases | Status |
|---|---:|---|
| Fact extraction | 1 | To be tested |
| Price overwrite | 2 | To be tested |
| Entity separation | 2 | To be tested |
| Stock status update | 2 | To be tested |
| Promotion memory | 2 | To be tested |
| Freshness filtering | 1 | To be tested |
| Shipping policy | 2 | To be tested |
| Product feature | 1 | To be tested |
| Non-fact filtering | 2 | To be tested |
| Fallback / refusal | 2 | To be tested |
| Traceable retrieval | 1 | To be tested |
| Retail decision-support extension | 2 | Experimental |

Total cases: 20

## Example Case

### price_overwrite_001

Setup:

1. A款日抛隐形眼镜现在价格是99元。
2. A款日抛隐形眼镜价格更新为89元。

Query:

A款日抛隐形眼镜现在多少钱？

Expected behavior:

The system should answer with the newer active price, 89元, and should not use the old price, 99元, as the current answer.

## Interpretation

A successful result indicates that the memory layer is not simply retrieving semantically similar past text. Instead, it is applying lifecycle-aware memory behavior, including overwrite control, active-state filtering, entity separation, and traceable retrieval.

## Limitations

This evaluation is scenario-based and small-scale. It is intended to verify core system behavior rather than provide a general benchmark.

Some cases, especially retail decision-support cases, are exploratory. They are included to show how the same memory-layer architecture may extend from livestream commerce to broader retail operations and business decision-support scenarios.

## Next Steps

- Expand the evaluation set to 30–40 cases.
- Add more cases for ambiguous product names.
- Add more tests for expired promotions and time-sensitive knowledge.
- Record pass/fail results after each local run.
- Separate evaluation cases into core memory tests and retail decision-support extension tests.
