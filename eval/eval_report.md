# Evaluation Report

## Purpose

This evaluation checks whether the livestream knowledge and memory layer behaves reliably in commerce-oriented scenarios.

The goal is not to claim broad benchmark performance. Instead, the evaluation verifies whether the system can extract structured facts, update active knowledge, route queries to the correct fact type, retrieve traceable sources, and refuse unsupported questions.

## Current Result

| Result | Count |
|---|---:|
| Passed | 11 |
| Failed | 0 |
| Total | 11 |

| Metric | Value |
|---|---:|
| Pass rate | 100% |
| Behavior groups covered | 5 |
| Fact types covered | 5 |
| Unsupported-query cases | 1 |
| Non-fact filtering cases | 1 |
| Overwrite / lifecycle-update cases | 3 |

Machine-readable summary:

- `eval/results/eval_summary.csv`

## Covered Behaviors

| Category | Behavior Group | Purpose |
|---|---|---|
| Product price retrieval | Retrieval | Check whether stored product prices can be retrieved |
| Price overwrite | Lifecycle update | Check whether newer prices supersede older active prices |
| Entity separation | Entity control | Check whether different products remain separate |
| Stock status | Retrieval | Check whether stock facts can be stored and retrieved |
| Stock overwrite | Lifecycle update | Check whether newer stock status supersedes older stock status |
| Promotion retrieval | Retrieval | Check whether promotion facts can be retrieved |
| Promotion overwrite | Lifecycle update | Check whether newer promotion facts supersede older ones |
| Shipping policy | Retrieval | Check whether delivery-related facts can be retrieved |
| Product feature | Retrieval | Check whether product feature facts can be retrieved |
| Non-fact filtering | Noise filtering | Check whether greetings are not treated as factual knowledge |
| Fallback/refusal | Unsupported-query handling | Check whether unsupported queries are refused |

## Data-Science-Oriented Framing

This evaluation can be understood as a small scenario-based test set. Each case defines an input setup, a query, an expected behavior, and a pass/fail result.

This makes the evaluation more transparent, repeatable, and extensible than only reporting "11/11 passed."

## Limitations

This evaluation is scenario-based and small-scale. It verifies core system behavior rather than providing a general language-model benchmark.

The current evaluation does not yet measure large-scale product coverage, noisy real-world inputs, timestamp-controlled stale-fact rejection, ambiguous product-name disambiguation, or multi-store retail operations decision support.

## Next Steps

1. Add timestamp-controlled freshness tests.
2. Add stale promotion and stale stock cases.
3. Add ambiguous product-name cases.
4. Add similar-product separation cases.
5. Save machine-readable evaluation results after each run.
