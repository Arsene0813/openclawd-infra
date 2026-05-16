# Meituan Instant-Retail Decision Support Prototype

Repository: `livestream-agent-memory-layer`

Evidence-grounded retail decision-support prototype with lifecycle-aware retrieval.

## Core Research Question

This is a local working prototype built from a real Meituan instant-retail operating problem. The retail evidence is manually structured from merchant-backend metrics used in multi-store operations. While the Meituan backend provides detailed single-store metrics, it is not designed to support cross-store operational reasoning at scale.

As the number of stores increased, the central problem became not data availability, but comparability.

This project investigates:
- which store-period records are meaningfully comparable;
- under what operational conditions comparisons remain valid;
- what kinds of operational conclusions the available evidence can support;
- and where the system should stop before generating unsupported judgments.

The goal is not fully automated decision-making, but building a more reliable evidence-based framework for multi-store operational analysis and future business replication.

## Current Prototype Workflow

The current prototype implements a staged decision-support workflow:

1. preserve backend metric definitions;
2. use SQL to structure selected store-period data;
3. convert diagnostic evidence into memory facts with source fields, observed values, source paths, and limitations;
4. test whether later answers stay inside the evidence boundary.

The single source of truth for retail field names and metric meanings is:

- `retail_ops/data/DATA_DICTIONARY.md`

## Key Design Principles

This prototype emphasizes:

- preserving exact backend metric semantics instead of flattening them into generic business metrics;
- structuring store-period observations before making stronger comparability claims;
- converting diagnostics into retrieval-facing evidence records rather than unsupported summaries;
- preserving source fields, observed values, source paths, confidence labels, and limitations;
- evaluating whether generated answers stay inside the available evidence boundary;
- refusing unsupported operational conclusions instead of overextending inference.

## Fast Reading Path

For an admissions or project review, read in this order:

1. `PROJECT_SUMMARY_FOR_ADMISSIONS.md`  
   Admissions-facing explanation of the real business problem and prototype scope.

2. `retail_ops/data/DATA_DICTIONARY.md`  
   Canonical Meituan-style backend metric definitions and field naming rules.

3. `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`  
   Current retail Demo 2: same-period B-F diagnostic.

4. `retail_ops/LINEAGE.md`  
   Source-to-SQL-to-memory lineage and interpretation limits.

5. `retail_ops/EXPERIMENTS.md`  
   Current analytical experiment map and failure modes.

### Document Responsibility Note

This README is the repository landing page. It gives the project purpose, implemented scope, fast reading path, and current boundary.

Detailed metric definitions are maintained in `retail_ops/data/DATA_DICTIONARY.md`.

Claim-to-data lineage is maintained in `retail_ops/LINEAGE.md`.

Experiment and validation logic is maintained in `retail_ops/EXPERIMENTS.md`.

The future pairwise comparability-gate design is maintained in `retail_ops/COMPARABILITY_GATE_V0.md`.

This avoids repeating the same explanation across multiple documents while keeping the full evidence and implementation notes available in their owner documents.

For implementation details, see:

- `retail_ops/README.md`
- `PROJECT_STATUS.md`
- SQL files
- scripts
- generated outputs
- evaluation files

## Current Implemented Scope

Retail Demo 2 is the current same-period diagnostic endpoint. A pairwise comparability gate is future work.

The current retail path does not assume that two stores are directly comparable for pricing, subsidy, SKU, ranking, or fulfillment decisions. Instead, it first structures selected store-period observations under a consistent reporting window and shared metric definitions before performing diagnostic comparison.

| Area | Implemented now | Current boundary |
|---|---|---|
| Livestream memory layer | Typed product facts, overwrite control, soft deactivation, active-state retrieval, fallback/refusal, scenario evaluation | Not a general-purpose production agent memory platform |
| Retail metric dictionary | Meituan-style backend metric definitions and naming boundaries | Not automatic ingestion from Meituan backend |
| Retail Demo 1 | Store A month-over-month diagnostic across February, March, and April 2026 | Not causal explanation of monthly performance |
| Retail Demo 2 | Same-period B-F diagnostic with scope and limitation checks | Not a pairwise store-period comparability gate |
| Retail memory facts | Generated facts with observed values, source fields, source paths, confidence, and limitations | Not a full 48-store automated decision system |
| Evaluation | Scenario-based checks for supported answers, unsupported-scope refusal, and metric-boundary preservation | Not a broad LLM benchmark |

## Architecture

The prototype has two connected layers.

| Layer | Purpose | Main files |
|---|---|---|
| Memory-layer prototype | Store and retrieve typed facts while handling updates, stale knowledge, and unsupported questions | `api/`, `scripts/`, `eval/` |
| Retail operations extension | Structure Meituan-style backend metrics and preserve diagnostic evidence for cautious comparison | `retail_ops/` |

Basic flow:

```text
backend metrics / operator input
-> metric dictionary and data contract
-> SQL diagnostic output
-> generated memory facts
-> retrieval with source fields and limitations
-> qualified answer or refusal
```

The important design choice is that memory facts are not just summaries. They carry source fields, observed values, calculation notes, source paths, supporting source paths, confidence labels, and limitations.

## Implemented API and Retrieval Scope

The local FastAPI prototype includes general chat, memory, and retrieval endpoints.

The key implemented paths are:

- `/health`
- `/chat`
- `/chat_mem`
- `/chat_livestream_kb`
- `/chat_retail_ops_kb`
- `/chat_retail_ops_demo2_kb`

The retail endpoints are local prototype endpoints. Demo 2 currently uses file-backed generated retail memory facts; it is not a production Meituan API integration.

## Retail Demo 1: Store A Month-over-Month Diagnostic

Demo 1 analyzes one self-operated Qingdao store across February, March, and April 2026.

It shows why operational performance should not be interpreted from one metric alone. Traffic, ranking, transaction amount, order conversion, activity cost, refund pressure, invalid orders, and top-SKU evidence can move in different directions.

Main file:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

The purpose is not to label the store as simply good or bad. The purpose is to keep month-over-month interpretation inside a documented evidence boundary.

## Retail Demo 2: Same-Period B-F Diagnostic

Demo 2 analyzes five anonymized stores, B-F, over the same March 2026 reporting window.

In this repository, Demo 2 means same-period diagnostic evidence and guardrails. File paths that include `cross_store_comparability` are retained for reference stability and should be read through:

- `retail_ops/data/DATA_DICTIONARY.md`

Demo 2 includes:

- same-period store-period metrics;
- top search-term evidence;
- top-SKU transaction-amount evidence;
- SQL-derived diagnostic fields;
- `comparison_scope_flag`;
- `comparison_limit_notes`;
- generated Demo 2 retail memory facts;
- facts evaluation and answer-boundary evaluation.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

Future pairwise comparability gate design:

- `retail_ops/COMPARABILITY_GATE_V0.md`

## Evaluation Snapshot

Saved outputs under `eval/results/` and `retail_ops/outputs/` should be treated as the source of truth if this summary table is later updated.

The evaluations are scenario-based behavior checks. Their value is not proving that the model is generally correct. Their value is checking whether answers preserve metric definitions, source boundaries, and comparison limits.

| Check | Scope | Result |
|---|---|---:|
| Livestream memory evaluation | Fact retrieval, overwrite behavior, entity separation, fallback/refusal, non-fact filtering | Current implemented cases pass |
| Retail retrieval evaluation | Store A retail-memory retrieval and unsupported-scope refusal | 8/8 passed |
| Retail Demo 2 facts evaluation | Store B-F generated fact coverage across diagnostic slots | 6/6 passed |
| Retail Demo 2 scope-boundary evaluation | Demo 2 remains a row-level same-period diagnostic and does not expose future pairwise-gate schema | 5/5 passed |
| Retail Demo 2 answer-boundary evaluation | Activity-cost ratio, top-SKU share, search-entry comparison, promotion-transfer limits | 4/4 passed |
| Retail data-contract validation | Dictionary phrases, source/output headers, forbidden aliases, generated fact structure | Passed |
| Project consistency validation | Current-scope files, Demo 2 boundary wording, stale future-work artifacts, endpoint claims | Passed |

## Running the Project Locally

This repository can be run as a local prototype with Docker Compose.

Minimum local run path:

1. Start services with `docker compose up -d`.
2. Pull the chat model with `docker exec -it oc_ollama ollama pull qwen2.5:14b`.
3. Pull the embedding model with `docker exec -it oc_ollama ollama pull bge-m3`.
4. Initialize Qdrant collections with `python3 scripts/init_qdrant_collections.py`.
5. Check the API with `curl http://127.0.0.1:8000/health`.
6. Rebuild the API after code changes with `docker compose up -d --build api`.

The local setup uses FastAPI, Ollama, and Qdrant. The retail evidence files, SQL outputs, and evaluation scripts can also be reviewed without running the API.

## Reproduce Key Checks

Retail data-contract validation:

```bash
python3 retail_ops/scripts/validate_retail_data_contract.py
```

Demo 2 fact coverage evaluation:

```bash
python3 eval/eval_retail_demo2_facts.py
```

Demo 2 answer-boundary evaluation:

```bash
python3 eval/eval_retail_demo2_answer_behavior.py
```

Project consistency validation:

```bash
python3 scripts/validate_project_consistency.py
```

For the full local API setup with Docker Compose, Qdrant, Ollama, and FastAPI, see:

- `PROJECT_STATUS.md`
- `docker-compose.yml`

## Key Evidence Files

| File | Why it matters |
|---|---|
| `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Admissions-facing summary of the real business problem and prototype scope |
| `retail_ops/data/DATA_DICTIONARY.md` | Canonical backend metric definitions and naming boundaries |
| `retail_ops/LINEAGE.md` | How source fields support SQL diagnostics and memory facts |
| `retail_ops/FIELD_USAGE_REVIEW.md` | Field-name review before future comparability-gate expansion |
| `retail_ops/EXPERIMENTS.md` | Experiment questions, inputs, transformations, pass conditions, and failure modes |
| `retail_ops/COMPARABILITY_GATE_V0.md` | Future pairwise comparability-gate design note |
| `retail_ops/sql/` | SQL transformations for Demo 1 and Demo 2 |
| `retail_ops/outputs/` | Generated diagnostic outputs and generated memory facts |
| `eval/` | Scenario-based evaluation scripts and reports |

## What This Demonstrates

This project demonstrates:

- turning a real retail operating problem into a structured data problem;
- preserving exact backend metric definitions instead of treating them as generic business metrics;
- using SQL to place selected store-period records under a shared diagnostic structure before stronger comparability claims are made;
- converting diagnostic outputs into retrieval-facing memory facts;
- preserving source fields, observed values, source paths, supporting source paths, confidence, and limitations;
- testing whether answers qualify or refuse unsupported operating claims;
- defining future comparability requirements before strategy transfer.

The strongest point of the project is not that it is a finished operating platform. The strongest point is that it makes the evidence boundary explicit before comparing stores or suggesting operating actions.

## Current Limitations

Current limitations:

- Demo 1 covers Store A month-over-month analysis.
- Demo 2 covers a limited same-period B-F diagnostic.
- Automated Meituan backend ingestion is not implemented.
- Full 48-store peer selection and daily automated recommendations are not implemented.
- Promotion cycle dates, competitor density, delivery conditions, rating/review signals, and stockout history are not yet included.
- Estimated income is treated as a platform-displayed proxy, not audited profit.
- Top-SKU evidence is used as lightweight product-mix evidence, not full product-category sales share.
- `region_type` is weak region or market-context evidence only. It is not a mature market-area classification, store-stage label.

## Future Work: Comparability Gate

Full future-gate design is maintained in `retail_ops/COMPARABILITY_GATE_V0.md`. This section only summarizes the next planned direction.

The next planned stage is a pairwise comparability gate.

The gate should answer a narrow question:

Can these two store-period records be compared for this specific operating question?

Future work should:

- build a pairwise gate around a reference store, a candidate store, and a specific comparison question;
- decide whether the selected records are comparable, comparable with limits, not comparable, or insufficiently supported;
- consider transaction order volume, transaction amount, activity involvement, activity intensity, explicit activity status or campaign-calendar evidence if available, store type, market context, competition, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows;
- avoid subjective regional classification based on intuition, address impression, or habitual labels;
- keep `region_type` as weak context unless broader store data and stronger market-context evidence support new documented fields;
- add any future market-area classification as new documented fields, such as `market_area_type`, rather than changing the meaning of `region_type`;
- preserve the same field contract and evaluation checks when expanding beyond the current sample.

The gate should not produce a global store ranking or a universal comparability score. A store pair may be comparable for search-entry structure but not comparable for promotion transfer, pricing pressure, SKU strategy, or fulfillment interpretation.
