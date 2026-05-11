# Lifecycle-Aware AI Memory Layer for Retail Decision Support

Repository name: `livestream-agent-memory-layer`

This repository is a working local prototype built from a real operating problem in Meituan-style instant retail.

Meituan's merchant backend provides detailed single-store metrics, but the data is mainly designed for reviewing one store at a time. Once the business expands across many stores, the harder problem is deciding which stores can be compared, under what conditions they can be compared, and which operating signals are strong enough to support a cautious decision.

The current retail prototype starts with one Store A diagnostic demo. It uses SQL to normalize selected Meituan backend exports into a consistent data structure, then uses a memory layer to store evidence, limitations, and conservative operating profiles.

The goal is not to let an LLM make operating decisions directly. The goal is to prevent unsupported comparisons and preserve the evidence boundary behind each answer.

The project began from an earlier lifecycle-aware memory layer for livestream commerce. That earlier system handled changing product facts such as price, promotion, stock status, shipping policy, and product features. The same design principle is now being applied to retail operations data: changing commercial information should be structured, checked, updated, retrieved with traceable sources, and refused when the available evidence is not enough.

## Current Implementation Boundary

| Layer | Current Status | Evidence |
|---|---|---|
| Livestream memory API | Implemented local prototype | `/chat_mem`, `/chat_livestream_kb`, `api/main.py` |
| Structured memory lifecycle | Implemented for livestream product facts | overwrite control, soft deactivation, freshness filtering, active-state filtering |
| Livestream evaluation | Implemented | Scenario checks pass for the implemented cases |
| Retail SQL diagnostic | Implemented for Store A Demo 1 and Demo 2 cross-store diagnostics | `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`, `retail_ops/sql/02_demo2_cross_store_comparability.sql` |
| Retail metric dictionary and lineage | Implemented | `retail_ops/data/DATA_DICTIONARY.md`, `retail_ops/LINEAGE.md` |
| Retail data-contract validation | Implemented | `retail_ops/scripts/validate_retail_data_contract.py`, `retail_ops/outputs/retail_data_contract_validation_result.txt` |
| Retail memory facts generation | Implemented as JSON exports | `retail_ops/outputs/generated_retail_memory_facts.json`, `retail_ops/outputs/generated_demo2_retail_memory_facts.json` |
| Retail facts Qdrant loading path | Implemented for Store A Demo 1 facts | `retail_ops/scripts/load_retail_facts_to_qdrant.py` |
| Retail retrieval endpoint | Implemented for Store A Demo 1 and file-backed Demo 2 facts | `/chat_retail_ops_kb`, `/chat_retail_ops_demo2_kb` |
| Retail retrieval evaluation | Implemented for Store A Demo 1; Demo 2 facts have offline eval | `eval/eval_retail.py`, `eval/eval_retail_demo2_facts.py` |
| Cross-store decision support | Implemented as Demo 2 same-period B-F comparability diagnostic | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` |
| Automated Meituan backend ingestion | Not implemented yet | Future work |
| Full 48-store decision-support system | Not implemented yet | Future work |

## What Is Implemented

### Livestream commerce memory layer

The livestream memory layer supports:

- structured fact extraction from product-related input;
- typed memory for product price, promotion, stock status, shipping policy, and product features;
- product-level entity separation;
- overwrite control and soft deactivation;
- freshness-aware retrieval and active-state filtering;
- traceable retrieval outputs;
- fallback or refusal when no reliable fact is available;
- scenario-based evaluation.

Current endpoints:

- `/chat_mem`
- `/chat_livestream_kb`

Current evaluation result:

- The implemented livestream memory cases pass.

### Retail operations extension

The retail extension applies the same lifecycle-aware memory principle to Meituan-style instant retail operations data.

The retail extension currently has two implemented demos:

- Demo 1: `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`
- Demo 2: same-period cross-store comparability diagnostic using Stores B-F in March 2026

Demo 1 analyzes Store A, a self-operated Qingdao store, across February, March, and April 2026.

Demo 2 analyzes five anonymized stores, B-F, over the same March 2026 reporting window. It does not rank stores as simply better or worse. It structures comparable backend metrics, derives cautious diagnostic signals, and preserves interpretation limits before any operating recommendation is made.

It includes:

- Meituan-style backend metric definitions;
- Store A monthly metrics;
- SQL-derived diagnostic metrics;
- claim-to-field lineage rules;
- generated retail memory facts;
- local data-contract checks;
- narrow Store A retail retrieval evaluation.

Demo 2 includes:

- cross-store March 2026 source CSVs for Stores B-F;
- top search term evidence with English helper translations;
- top SKU evidence with original Chinese SKU names and English helper translations;
- SQL-derived comparability diagnostics;
- generated Demo 2 retail memory facts;
- a file-backed Demo 2 retail endpoint at `/chat_retail_ops_demo2_kb`;
- offline facts evaluation for Demo 2.

Current retail evaluation result:

- The current Store A retail retrieval cases pass.
- Demo 2 offline retail facts evaluation passes for the implemented B-F cases.

## Core Memory Behavior

Simple example:

Initial fact:

```text
A款价格是99元
```

Stored structured memory:

```json
{
  "type": "product_price",
  "product_ref": "A款",
  "value": "99元",
  "slot": "price",
  "is_active": true
}
```

Updated fact:

```text
A款价格是89元
```

Expected behavior:

- the newer price becomes the active fact;
- the older price is preserved but softly deactivated;
- a later query such as `A款多少钱？` retrieves the current active price;
- unsupported or stale information triggers fallback or refusal instead of a fabricated answer.

## Architecture Overview

```text
User / Operator Input
-> Non-Fact Filtering
-> Structured Fact Extraction
-> Fact Policy Registry
-> Entity + Slot Assignment
-> Overwrite / Soft Deactivation
-> Qdrant Typed Knowledge Store
-> Retrieval + Fact-Type Routing
-> Freshness / Active-State Filtering
-> Traceable Answer or Fallback
```

## Key Design Ideas

| Design Choice | Problem It Addresses | Why It Matters |
|---|---|---|
| Structured fact extraction | Raw chat history is hard to update or verify | Converts interaction into explicit knowledge |
| Typed memory | Different information types behave differently | Price, promotion, stock, and product features need different rules |
| Entity + slot storage | Product facts can conflict across products | Prevents facts about different products from overwriting each other |
| Soft deactivation | Old facts should not disappear silently | Preserves traceability while keeping current knowledge active |
| Freshness filtering | Promotions and stock status become outdated quickly | Reduces the risk of using stale information |
| Retrieval gating | Similarity alone does not prove reliability | Prevents weakly matched memory from being used as verified knowledge |
| Traceable sources | Hidden memory use is hard to inspect | Makes answers easier to debug and evaluate |
| Data-contract validation | Backend metric names can drift across files | Keeps CSV, SQL, memory facts, lineage, and documentation consistent |

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

Traffic recovery, transaction recovery, conversion decline, average-order-value decline, refund-pressure improvement, and invalid-order-pressure improvement are supporting observations. They are not standalone canonical memory slots in the current retail memory facts.

`refund_pressure_improved` may appear as SQL-derived supporting evidence, but it is not a canonical retail memory slot. The canonical memory slot for refund and invalid-order pressure is `order_quality_pressure_profile`.

The purpose is not to label a store as simply good or bad. The purpose is to prevent unsupported conclusions from incomplete, non-comparable, or promotion-distorted data.

## Retail Data Contract

The retail extension includes a data-contract layer for field consistency and metric lineage.

Key files:

- `retail_ops/data/DATA_DICTIONARY.md` — bilingual Meituan backend metric definitions and consistency rules.
- `retail_ops/LINEAGE.md` — claim-to-field lineage and interpretation limits.
- `retail_ops/scripts/validate_retail_data_contract.py` — local validation for required canonical fields, forbidden aliases, source fields, and JSON fact structure.
- `retail_ops/outputs/retail_data_contract_validation_result.txt` — saved validation result.

Current validation result:
```text
Retail data contract validation PASSED.
Checked source CSV headers: 46
Checked top-SKU CSV headers: 9
Checked SQL output headers: 61
Checked generated retail memory facts: 6
Checked known store_id values: ['A']
Checked expected entity_id values: ['store_A']
No forbidden alias fields found.
Entity ID convention is documented and validated.
```

This is important because Meituan backend metrics should not be treated as generic business metrics without checking their exact definitions. For example, backend-reported order conversion should not be recomputed from valid orders divided by entry users, because `valid_orders` is an order-status metric while `order_users` is a user-level funnel metric.

## Evaluation Snapshot

The evaluations are scenario-based behavior checks, not broad language-model benchmarks.

| Evaluation | Scope | Result |
|---|---|---:|
| Livestream memory evaluation | fact retrieval, overwrite behavior, entity separation, fallback/refusal, non-fact filtering | Current implemented cases pass |
| Retail retrieval evaluation | Store A retail-memory retrieval, attribution-warning behavior, unsupported-scope refusal | Current Store A cases pass |
| Retail Demo 2 facts evaluation | Store B-F generated fact coverage for visibility, activity, order-quality, SKU, and attribution-guard slots | 5/5 passed |
| Retail data-contract validation | field naming, required metrics, source fields, forbidden aliases, JSON fact structure | passed |
| Project consistency validation | required files, documented endpoints, Demo 2 endpoint, Demo 2 artifacts, stale aliases | passed |

The evaluation value is not that the model is generally correct. The value is that the system has explicit checks for supported answers, overwrite behavior, unsupported-scope refusal, and data-contract consistency.

## What This Demonstrates

This project demonstrates abilities relevant to AI, data science, business analytics, and language-technology-related study:

- identifying a real reliability problem in AI-assisted commerce;
- representing changing commercial facts as structured data;
- designing typed memory policies for different fact types;
- managing current vs outdated knowledge through overwrite and active-state rules;
- using retrieval with confidence, freshness, and source checks;
- preparing Meituan-style backend metrics with SQL;
- documenting metric definitions and lineage;
- validating field consistency across data, SQL, memory facts, and documentation;
- converting SQL-derived observations into cautious memory facts;
- refusing or qualifying unsupported operational claims.

## Recommended Review Path

For a quick admissions review, read these files in order:

1. `README.md` — project overview, implementation boundary, and validation snapshot.
2. `PROJECT_SUMMARY_FOR_ADMISSIONS.md` — admissions-oriented narrative and relevance.
3. `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` — completed retail operations demo.
4. `retail_ops/data/DATA_DICTIONARY.md` — Meituan backend metric definitions and canonical fields.
5. `retail_ops/LINEAGE.md` — claim-to-field lineage and interpretation limits.
6. `eval/eval_retail_report.md` — Store A retail retrieval and refusal evaluation.
7. `retail_ops/outputs/demo2_cross_store_comparability_output.csv` — Demo 2 cross-store diagnostic output.
8. `eval/results/eval_retail_demo2_facts_result.txt` — Demo 2 facts evaluation result.

The SQL files, generated outputs, and validation scripts are supporting evidence for the demos rather than separate admissions entry points.

## Running the Project

This project is intended to run locally with Docker Compose.

### Prerequisites

- Docker and Docker Compose.
- Python 3.10+ for local scripts.

The local Docker Compose setup includes:

- FastAPI API service;
- Ollama for local model inference and embeddings;
- Qdrant for vector storage and typed memory retrieval.

Optional environment variables are documented in `.env.example`.

### 1. Start the services

```bash
docker compose up -d
```

### 2. Pull the local Ollama models

```bash
docker exec -it oc_ollama ollama pull qwen2.5:14b
docker exec -it oc_ollama ollama pull bge-m3
```

### 3. Initialize the Qdrant collections

```bash
python3 scripts/init_qdrant_collections.py
```

This creates the Qdrant collections used by the prototype if they do not already exist.

### 4. Check that the API is running

```bash
curl http://127.0.0.1:8000/health
```

### 5. Rebuild the API after code changes

```bash
docker compose up -d --build api
```

## Running Validation and Evaluation

### Retail data-contract validation

To regenerate the Store A Demo 1 SQL output and run the retail/project consistency checks together:

```bash
python retail_ops/scripts/regenerate_demo1_sql_and_validate.py
```

To run the retail data-contract validator directly:

```bash
python retail_ops/scripts/validate_retail_data_contract.py
```

### Retail retrieval evaluation

After starting the local services and loading the generated retail facts:

```bash
python3 retail_ops/scripts/load_retail_facts_to_qdrant.py
python3 eval/eval_retail.py
cat eval/results/eval_retail_result.txt
```

### Livestream memory evaluation

The evaluation runner is designed to run against the local API service.

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/health
docker compose exec api rm -rf /app/eval
docker compose cp eval api:/app/eval
docker compose exec api python /app/eval/eval_livestream.py
```

Saved evaluation outputs:

- `eval/results/eval_result_11_pass.txt`
- `eval/results/eval_retail_result.txt`

## Repository Structure

```text
api/
  main.py
  Dockerfile
  requirements.txt

scripts/
  init_qdrant_collections.py

eval/
  eval_livestream.py
  eval_livestream_cases.json
  eval_retail.py
  eval_retail_cases.json
  eval_report.md
  eval_retail_report.md
  results/

retail_ops/
  README.md
  LINEAGE.md
  data/
    DATA_DICTIONARY.md
    store_a_monthly_metrics.csv
    store_a_top_skus.csv
  sql/
    01_store_a_month_over_month_diagnostic.sql
  outputs/
    store_a_demo1_sql_output.csv
    store_a_demo1_interpretation_summary.csv
    generated_retail_memory_facts.json
    retail_data_contract_validation_result.txt
  scripts/
    load_retail_facts_to_qdrant.py
    validate_retail_data_contract.py
  demo/
    demo_1_store_a_month_over_month_diagnostic.md

case_studies/
  from_livestream_to_retail_decision_support.md

PROJECT_STATUS.md
PROJECT_SUMMARY_FOR_ADMISSIONS.md
README.md
docker-compose.yml
```

## Limitations

This repository is a working prototype, not a finished production system.

Current limitations:

- the retail retrieval path is narrow and limited to Store A Demo 1;
- cross-store decision support is not implemented yet;
- automated Meituan backend ingestion is not implemented yet;
- promotion cycle dates are unknown in Demo 1;
- estimated income is treated as a platform-displayed proxy, not audited profit;
- top-SKU evidence is used qualitatively, not as full product-category sales share;
- full automated SKU classification is deferred to future work.

These limitations are intentionally visible because the project is designed to avoid overstating what the data can prove.

## Next Steps

Short-term improvements:

- add Demo 2: a cross-store comparability gate for sampled Meituan stores;
- check whether stores are comparable by period alignment, coarse market context, store type, activity and subsidy profile, order volume, visibility/ranking signals, conversion profile, refund/invalid-order pressure, SKU evidence, and data completeness;
- add Demo 2 retail evaluation cases for `comparable`, `partially_comparable`, `not_comparable`, and `insufficient_data` outcomes;
- add a dedicated metric-lineage retrieval eval case after adding the corresponding retrievable retail memory fact;
- update the evaluation runner so it automatically regenerates retail and livestream summary files after each run.

Medium-term improvements:

- expand retail memory retrieval beyond Store A Demo 1;
- add timestamp-controlled freshness tests for stale promotion, outdated stock facts, and outdated retail diagnostic facts;
- separate extraction, routing, storage, lifecycle policy, and evaluation logic into clearer modules;
- make the fact-policy registry easier to extend through configuration;
- explore automated SKU classification for care solutions, daily lenses, color lenses, period lenses, accessories, and other product groups;
- add a safer profit-margin-aware SKU recommendation layer only after margin data is explicitly available.
