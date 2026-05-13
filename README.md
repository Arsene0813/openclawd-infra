# Lifecycle-Aware AI Memory Layer for Retail Decision Support

Repository name: `livestream-agent-memory-layer`

This repository is a working local prototype built from a real operating problem in Meituan-style instant retail.

Meituan's merchant backend provides detailed store-level metrics, but the data is mainly designed for reviewing one store at a time. Once the business expands across many stores, the harder problem is deciding which stores can be compared, under what conditions they can be compared, and which operating signals are strong enough to support a cautious decision.

The current retail prototype uses SQL to organize selected Meituan backend exports into consistent diagnostic outputs, then uses a memory / retrieval layer to preserve evidence, limitations, and conservative operating profiles.

The goal is not to let an LLM make operating decisions directly. The goal is to reduce unsupported comparisons and preserve the evidence boundary behind each answer.

## How to Read This Repository

This repository has two connected layers.

| Layer | Where to look | What it shows |
|---|---|---|
| Memory-layer prototype | `api/`, `eval/`, `scripts/` | typed memory, overwrite control, retrieval, fallback, and evaluation |
| Retail operations extension | `retail_ops/` | Meituan-style metric definitions, SQL diagnostics, generated memory facts, same-period diagnostics, scope/limit checks, and evaluation |

Recommended first read:

1. `PROJECT_SUMMARY_FOR_ADMISSIONS.md`
2. `README.md`
3. `PROJECT_STATUS.md`
4. `retail_ops/README.md`
5. `retail_ops/data/DATA_DICTIONARY.md`

For deeper evidence, read `retail_ops/LINEAGE.md`, the SQL files, generated outputs, validation scripts, and evaluation files.

The repository should be read as a staged prototype with explicit evidence boundaries, not as a finished operating platform.

## Current Implementation Boundary

| Area | Current status | Evidence |
|---|---|---|
| Livestream memory API | Implemented local prototype | `/chat_mem`, `/chat`, `/health`, `api/main.py` |
| Structured memory lifecycle | Implemented for livestream product facts | overwrite control, soft deactivation, freshness filtering, active-state filtering |
| Livestream evaluation | Implemented | current implemented cases pass |
| Retail metric dictionary and lineage | Implemented | `retail_ops/data/DATA_DICTIONARY.md`, `retail_ops/LINEAGE.md` |
| Retail data-contract validation | Implemented as lightweight guardrail | `retail_ops/scripts/validate_retail_data_contract.py`, `retail_ops/outputs/retail_data_contract_validation_result.txt` |
| Retail memory facts generation | Implemented for Store A Demo 1 and Demo 2 B-F facts | `retail_ops/outputs/generated_retail_memory_facts.json`, `retail_ops/outputs/generated_demo2_retail_memory_facts.json` |
| Retail local answer endpoints | Implemented for Demo 1 and Demo 2 | `/chat_retail_ops_kb`, `/chat_retail_ops_demo2_kb`; file-backed/local retail evidence only |
| Automated Meituan backend ingestion | Not implemented yet | future work |
| Full 48-store decision-support system | Not implemented yet | future work |

The retail extension currently has two implemented retail stages and one planned next stage:

1. Demo 1: Store A month-over-month diagnostic.
2. Demo 2: same-period Stores B-F cross-store diagnostic with scope/limit checks.
3. Future work: a comparability gate for deciding which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

The comparability gate is not currently implemented as a finished demo. The current implemented retail scope stops at Demo 2.

## What Is Implemented

### Livestream Commerce Memory Layer

The livestream memory layer supports:

- structured fact extraction from product-related input;
- typed memory for product price, promotion, stock status, shipping policy, and product features;
- product-level entity separation;
- overwrite control and soft deactivation;
- freshness-aware retrieval and active-state filtering;
- traceable retrieval outputs;
- fallback or refusal when no reliable fact is available;
- scenario-based evaluation.

Current implemented API endpoints in `api/main.py` include:
- `/health`
- `/chat`
- `/chat_mem`

### Retail Operations Extension

The retail extension applies the same lifecycle-aware memory principle to Meituan-style instant retail operations data.

The retail extension currently has two implemented demos and one planned next stage:

- Demo 1: `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`
- Demo 2: `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

Demo 1 analyzes Store A, a self-operated Qingdao store, across February, March, and April 2026. It shows why traffic, transaction amount, conversion, activity cost, refund pressure, invalid orders, and top-SKU evidence should not be interpreted from one metric alone.

Demo 2 analyzes five anonymized stores, B-F, over the same March 2026 reporting window. It structures comparable backend metrics, derives cautious diagnostic signals, and preserves interpretation limits before any operating recommendation is made.

A future pairwise comparability gate should build on the Demo 2 B-F structure, but it is not currently implemented as a finished demo.


The future gate should avoid market-area classification until more store data and repeated reporting windows are available. The current project treats `region_type` as weak context only.

The retail extension includes:

- Meituan-style backend metric definitions;
- Store A monthly metrics;
- B-F same-period cross-store metrics;
- SQL-derived diagnostic metrics;
- claim-to-field lineage rules;
- generated retail memory facts;
- local data-contract checks;
- Store A retail retrieval evaluation;
- Demo 2 facts and answer-boundary evaluations;
- future comparability-gate limitation notes;
- refusal to overclaim from the current limited sample.

## Core Memory Behavior

Simple example:

Initial fact: A product price is 99.

Stored structured memory:

- type: product_price
- product_ref: product A
- value: 99
- slot: price
- is_active: true

Updated fact: the same product price is 89.

Expected behavior:

- the newer price becomes the active fact;
- the older price is preserved but softly deactivated;
- a later price query retrieves the current active price;
- unsupported or stale information triggers fallback or refusal instead of a fabricated answer.

## Architecture Overview

The basic architecture is:

User or operator input -> non-fact filtering -> structured fact extraction -> fact policy registry -> entity and slot assignment -> overwrite or soft deactivation -> typed knowledge store -> retrieval and fact-type routing -> freshness / active-state filtering -> traceable answer or fallback

## Key Design Ideas

| Design Choice | Problem It Addresses | Why It Matters |
|---|---|---|
| Structured fact extraction | Raw chat history is hard to update or verify | Converts interaction into explicit knowledge |
| Typed memory | Different information types behave differently | Price, promotion, stock, and product features need different rules |
| Entity and slot storage | Product facts can conflict across products | Prevents facts about different products from overwriting each other |
| Soft deactivation | Old facts should not disappear silently | Preserves traceability while keeping current knowledge active |
| Freshness filtering | Promotions and stock status become outdated quickly | Reduces the risk of using stale information |
| Retrieval gating | Similarity alone does not prove reliability | Prevents weakly matched memory from being used as verified knowledge |
| Traceable sources | Hidden memory use is hard to inspect | Makes answers easier to debug and evaluate |
| Data-contract validation | Backend metric names can drift across files | Keeps CSV, SQL, memory facts, lineage, and documentation consistent |
| Future comparability-gate design | Cross-store metrics can be misleading without scope control | Defines the future evidence boundary before strategy transfer |

## Retail Demo 1: Why It Matters

The Store A demo shows why operational performance should not be interpreted from one metric alone.

April 2026 showed traffic and transaction recovery, but order conversion and average order value declined. At the same time, refund pressure and invalid-order pressure improved.

The SQL layer therefore produces cautious retail memory slots such as:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `order_quality_pressure_profile`
- `single_metric_attribution_guard`
- `top3_sku_product_mix_note`

Traffic recovery, transaction recovery, conversion decline, average-order-value decline, refund-pressure improvement, and invalid-order-pressure improvement are supporting observations.

They are not standalone canonical memory slots in the current retail memory facts. `refund_pressure_improved` may appear as SQL-derived supporting evidence, but it is not a canonical retail memory slot. The canonical memory slot for refund and invalid-order pressure is `order_quality_pressure_profile`.

The purpose is not to label a store as simply good or bad. The purpose is to prevent unsupported conclusions from incomplete, non-comparable, or promotion-distorted data.

## Retail Demo 2: Why It Matters

Demo 2 moves from single-store month-over-month diagnosis to same-period cross-store diagnosis.

It uses Stores B-F in the March 2026 reporting window.

The purpose is not to rank stores. The purpose is to organize comparable fields, preserve metric definitions, and test whether cross-store interpretation remains inside the evidence boundary.

Demo 2 includes:

- same-period source CSVs for Stores B-F;
- top search term evidence with English helper translations;
- top SKU evidence with original Chinese SKU names and English helper translations;
- SQL-derived scope/limit diagnostics;
- generated Demo 2 retail memory facts;
- a file-backed local Demo 2 retail endpoint at `/chat_retail_ops_demo2_kb` using generated Demo 2 retail memory facts;
- offline facts evaluation;
- answer-boundary evaluation.


## Retail Data Contract

The retail extension includes a data-contract layer for field consistency and metric lineage.

Key files:

- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/LINEAGE.md`
- `retail_ops/scripts/validate_retail_data_contract.py`
- `retail_ops/outputs/retail_data_contract_validation_result.txt`

This is important because Meituan backend metrics should not be treated as generic business metrics without checking their exact definitions.

For example, backend-reported order conversion should not be recomputed from valid orders divided by entry users, because `valid_orders` is an order-status metric while `order_users` is a user-level funnel metric.

## Evaluation Snapshot

The evaluations are scenario-based behavior checks, not broad language-model benchmarks.

| Evaluation | Scope | Result |
|---|---|---:|
| Livestream memory evaluation | fact retrieval, overwrite behavior, entity separation, fallback/refusal, non-fact filtering | current implemented cases pass |
| Retail retrieval evaluation | Store A retail-memory retrieval, attribution-warning behavior, unsupported-scope refusal | 8/8 passed |
| Retail Demo 2 facts evaluation | Store B-F generated fact coverage for visibility, activity, transaction/conversion, order-quality, SKU, and attribution-guard slots | 6/6 passed |
| Retail Demo 2 comparison-boundary consistency evaluation | checks that Demo 2 remains a row-level same-period diagnostic and does not pretend to be a pairwise gate | 5/5 passed |
| Retail Demo 2 offline answer-boundary check | checks that comparison answers preserve metric definitions and limits | 4/4 passed |
| Retail data-contract validation | required file presence, dictionary boundary phrases, Demo 1 / Demo 2 output headers, forbidden aliases, and generated fact structure | passed |
| Project consistency validation | required current-scope files, Demo 2 boundary wording, stale future-work artifacts, and forbidden retail endpoint claims | passed |

The evaluation value is not that the model is generally correct. The value is that the project has explicit checks for supported answers, unsupported-scope refusal, metric-boundary preservation, and comparability limits.

## What This Demonstrates

This project demonstrates abilities relevant to AI, data science, business analytics, and language-technology-related study:

- identifying a real reliability problem in AI-assisted commerce and retail operations;
- representing changing commercial facts as structured data;
- designing typed memory policies for different fact types;
- managing current vs outdated knowledge through overwrite and active-state rules;
- using retrieval with confidence, freshness, and source checks;
- preparing Meituan-style backend metrics with SQL;
- documenting metric definitions and lineage;
- validating field consistency across data, SQL, memory facts, and documentation;
- converting SQL-derived observations into cautious memory facts;
- testing whether cross-store comparisons preserve their evidence boundary;
- refusing or qualifying unsupported operational claims.

## Recommended Review Path

1. `PROJECT_SUMMARY_FOR_ADMISSIONS.md` — admissions-facing project summary.
2. `README.md` — project overview, architecture, implementation boundary, and evaluation status.
3. `PROJECT_STATUS.md` — short current implementation boundary.
4. `retail_ops/README.md` — retail operations extension overview.
5. `retail_ops/data/DATA_DICTIONARY.md` — Meituan backend metric definitions and canonical fields.
6. `retail_ops/LINEAGE.md` — source-to-SQL-to-memory lineage and interpretation limits.
7. `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` — Store A month-over-month retail operations demo.
8. `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` — same-period B-F cross-store diagnostic.
9. `retail_ops/FIELD_USAGE_REVIEW.md` — field-name review before future comparability-gate expansion.
10. `retail_ops/COMPARABILITY_GATE_V0.md` — future pairwise comparability-gate design note.
11. `retail_ops/EXPERIMENT_RESULTS.md` — comparability and limitation-preserving review cases.
12. `eval/eval_report.md` and `eval/eval_retail_report.md` — scenario-based evaluation reports.

## Running the Project

This project is intended to run locally with Docker Compose.

Prerequisites:

- Docker and Docker Compose.
- Python 3.10+ for local scripts.

The local Docker Compose setup includes:

- FastAPI API service;
- Ollama for local model inference and embeddings;
- Qdrant for vector storage and typed memory retrieval.

Useful commands:

- Start services: `docker compose up -d`
- Pull Qwen model: `docker exec -it oc_ollama ollama pull qwen2.5:14b`
- Pull embedding model: `docker exec -it oc_ollama ollama pull bge-m3`
- Initialize Qdrant collections: `python3 scripts/init_qdrant_collections.py`
- Check API health: `curl http://127.0.0.1:8000/health`
- Rebuild API after code changes: `docker compose up -d --build api`

## Running Validation and Evaluation

Retail data-contract validation:

- `python3 retail_ops/scripts/validate_retail_data_contract.py`

Project consistency validation:

- `python3 scripts/validate_project_consistency.py`

Retail Demo 2 checks:

- `python3 scripts/validate_demo2_retail_endpoint_boundary.py`
- `python3 retail_ops/scripts/validate_demo2_staging_data.py`
- `python3 retail_ops/scripts/validate_demo2_comparability_output.py`
- `python3 retail_ops/scripts/validate_demo2_retail_memory_facts.py`
- `python3 eval/eval_retail_demo2_facts.py`
- `python3 eval/eval_retail_demo2_scope_boundary.py`
- `python3 eval/eval_retail_demo2_answer_behavior.py`


Retail retrieval evaluation:

- `python3 retail_ops/scripts/load_retail_facts_to_qdrant.py`
- `python3 eval/eval_retail.py`
- `cat eval/results/eval_retail_result.txt`

Livestream memory evaluation:

- `docker compose up -d --build`
- `curl http://127.0.0.1:8000/health`
- `docker compose exec api rm -rf /app/eval`
- `docker compose cp eval api:/app/eval`
- `docker compose exec api python /app/eval/eval_livestream.py`

Saved evaluation outputs include:

- `eval/results/eval_result_11_pass.txt`
- `eval/results/eval_retail_result.txt`
- `eval/results/eval_retail_demo2_facts_result.txt`
- `eval/results/eval_retail_demo2_scope_boundary_result.txt`
- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

## Repository Structure

- `api/`
  - `main.py`
  - `Dockerfile`
  - `requirements.txt`

- `scripts/`
  - `init_qdrant_collections.py`
  - `validate_demo2_retail_endpoint_boundary.py`
  - `validate_project_consistency.py`

- `eval/`
  - `eval_livestream.py`
  - `eval_livestream_cases.json`
  - `eval_retail.py`
  - `eval_retail_cases.json`
  - `eval_retail_demo2_facts.py`
  - `eval_retail_demo2_scope_boundary.py`
  - `eval_retail_demo2_answer_behavior.py`
  - `eval_report.md`
  - `eval_retail_report.md`
  - `results/`

- `retail_ops/`
  - `README.md`
  - `LINEAGE.md`
  - `FIELD_USAGE_REVIEW.md`
  - `COMPARABILITY_GATE_V0.md`
  - `EXPERIMENT_RESULTS.md`
  - `data/`
  - `sql/`
  - `outputs/`
  - `scripts/`
  - `demo/`

- `case_studies/`
  - `from_livestream_to_retail_decision_support.md`

- `PROJECT_STATUS.md`
- `PROJECT_SUMMARY_FOR_ADMISSIONS.md`
- `README.md`
- `docker-compose.yml`

## Limitations

This repository is a working prototype, not a finished production system.

Current limitations:

- Demo 1 supports Store A month-over-month retail retrieval.
- Demo 2 supports a limited same-period B-F cross-store comparability diagnostic.
- Demo 2 memory facts are currently file-backed and exposed only through a local prototype endpoint, not a production retail API endpoint.
- The comparability gate is planned as future work, not currently implemented as a finished demo.
- The current implemented retail scope stops at Demo 2.
- Automated Meituan backend ingestion is not implemented yet.
- Full 48-store comparison, peer selection, and automated daily operating recommendations are not implemented yet.
- Promotion cycle dates, competitor density, delivery conditions, rating/review signals, and stockout history are not yet included.
- Estimated income is treated as a platform-displayed proxy, not audited profit.
- Top-SKU evidence is used qualitatively, not as full product-category sales share.
- Full automated SKU classification is deferred to future work.

Short-term improvements:

- Revisit the comparability gate after broader 48-store data, repeated reporting windows, and stronger market-context evidence are available.
- Keep the answer boundary narrow: given a store pair and a question type, return comparison decision, evidence fields, and limit notes.
- Expand beyond the B-F sample only after the same field contract and evaluation checks are preserved.
- Add more answer-boundary cases around `activity_cost_ratio_pct`, `region_type`, top-SKU concentration, refund pressure, invalid-order pressure, and strategy-transfer limits.
- Update validators later when future comparability-gate fields are actually implemented.

## Retail Demo 2 Answer-Behavior Boundary Evaluation

Demo 2 includes an offline answer-boundary check:

- `eval/eval_retail_demo2_answer_behavior.py`
- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

This check focuses on whether comparison answers preserve the implemented metric contract:

- `activity_cost_ratio_pct` is treated as activity-cost-ratio evidence, not ROI or profit margin.
- `top3_sku_transaction_amount_share_pct` is treated as lightweight top-SKU concentration evidence, not full product-category sales share.
- search-entry comparison stays tied to `search_entry_rate_pct`, `search_entry_share_pct`, `search_entry_users`, and `entry_users`.
- promotion or subsidy strategy transfer is qualified unless activity, subsidy, refund, invalid-order, and comparison-limit evidence support it.


## Future Work: Comparability Gate

The current implemented retail scope stops at Demo 2.

A comparability gate is planned as future work. It should eventually help judge which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

This is not currently implemented as a finished demo because the sample is still limited. Store comparability should depend on transaction order volume, transaction amount, whether the store is under activity or promotion, activity intensity, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

To avoid subjective regional classification, the current project treats `region_type` as weak context only. It is not a hard market-area classification or peer-store grouping rule.
