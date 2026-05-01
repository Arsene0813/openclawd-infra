# Evaluation Report

## Purpose

This evaluation checks whether the livestream knowledge and memory layer behaves reliably in commerce-oriented scenarios.

The goal is not to claim broad benchmark performance. Instead, the evaluation verifies whether the system can extract structured facts, update active knowledge, route queries to the correct fact type, retrieve traceable sources, and refuse unsupported questions.

## Evaluation Flow

Each scenario follows a two-stage flow:

1. Setup messages are sent to `/chat_mem`.
   - This allows the system to extract structured facts and store them in the knowledge base.
2. The query is sent to `/chat_livestream_kb`.
   - This retrieves active livestream facts from the structured knowledge base and returns an answer with routing and source information.

## Current Result

The current evaluation contains 11 scenario-based cases.

| Result | Count |
|---|---:|
| Passed | 11 |
| Failed | 0 |
| Total | 11 |

The saved run output is available at:

- `eval/results/eval_result_11_pass.txt`

## Covered Behaviors

| Category | Purpose |
|---|---|
| Product price retrieval | Check whether stored product prices can be retrieved |
| Price overwrite | Check whether newer prices supersede older active prices |
| Entity separation | Check whether different products remain separate |
| Stock status | Check whether stock facts can be stored and retrieved |
| Stock overwrite | Check whether newer stock status supersedes older stock status |
| Promotion retrieval | Check whether promotion facts can be retrieved |
| Promotion overwrite | Check whether newer promotion facts supersede older ones |
| Shipping policy | Check whether delivery-related facts can be retrieved |
| Product feature | Check whether product feature facts can be retrieved |
| Non-fact filtering | Check whether greetings or non-factual messages are not treated as usable product knowledge |
| Fallback/refusal | Check whether unsupported queries are refused rather than answered from unrelated memory |

## Interpretation

Passing these cases suggests that the system is not simply retrieving semantically similar past text. It demonstrates behavior closer to a lifecycle-aware memory layer, including fact extraction, overwrite control, active knowledge retrieval, fact-type routing, source traceability, and fallback when reliable memory is unavailable.

## Limitations

This evaluation is scenario-based and small-scale. It is designed to verify core system behavior rather than provide a general language-model benchmark.

Freshness filtering is implemented through `source_ts` and `freshness_days`, but full timestamp-aging tests require a fixture that can insert facts with controlled historical timestamps. The current evaluation therefore focuses first on overwrite behavior, active-state filtering, routing, fallback, and traceable retrieval.

## Next Steps

- Add timestamp-controlled freshness tests.
- Add more ambiguous product-name cases.
- Add more non-fact filtering cases.
- Save machine-readable evaluation results after each run.
- Separate core livestream memory tests from future retail decision-support extension tests.
