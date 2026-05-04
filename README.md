# Livestream Agent Memory Layer

A working prototype for making AI-assisted retail interaction more reliable through structured memory, lifecycle-aware retrieval, and traceable knowledge use.

## 30-Second Summary

AI agents can generate fluent responses while still using outdated, conflicting, or weakly matched information. This project explores how to reduce that risk in a livestream/retail commerce setting.

The system extracts structured product facts such as price, promotion, stock status, shipping policy, and product features; stores them as typed memory; applies freshness and overwrite rules; retrieves relevant facts with traceable sources; and falls back or refuses when reliable memory is not available.

This project began from an earlier LLM-powered livestream system and later informed my interest in adapting structured retrieval from customer-facing product interaction to internal retail decision support.

## Current Status

- Working local prototype with Docker Compose
- FastAPI service + Ollama + Qdrant
- Structured fact extraction and typed memory
- Product-level entity separation
- Overwrite control and soft deactivation
- Freshness-aware retrieval and active-state filtering
- Traceable retrieval outputs
- Scenario-based evaluation: 11 / 11 current cases passed
- Main endpoints: `/chat_mem` for fact ingestion and `/chat_livestream_kb` for structured retrieval

## Why This Matters

In commerce settings, information changes quickly. Product prices may change, promotions may expire, stock status may become outdated, and shipping policies may vary. A simple chatbot or vector-memory system may retrieve semantically similar but outdated information.

This project treats memory as structured, updateable, and inspectable knowledge rather than raw chat history. The goal is not only to generate answers, but to make the system’s use of past information more reliable and easier to evaluate.

## Quick Review Path

For admissions or non-specialist reviewers:

1. Read this README for the project overview.
2. See `PROJECT_SUMMARY_FOR_ADMISSIONS.md` for a concise application-oriented summary.
3. See `eval/eval_report.md` for the current behavior-based evaluation.
4. See `case_studies/from_livestream_to_retail_decision_support.md` for how this memory-layer idea extends toward retail operations decision support.

## Current Capabilities

Currently, the system can:

- extract structured facts from interaction instead of relying only on raw conversational history
- store facts through a centralized fact-policy registry rather than scattered type-specific rules
- attach each fact to `scope`, `slot`, and `entity_id` for more explicit storage semantics
- apply overwrite at the `entity_id + slot` level for single-value facts such as product price
- preserve multi-value facts such as product features without collapsing them into a single active value
- preserve older superseded records through soft deactivation instead of hard deletion
- attach lifecycle metadata such as timestamps, freshness windows, active-state flags, and last-seen signals
- separate product memory across entities through lightweight `product_ref` extraction
- filter non-fact messages such as greetings from entering memory
- return traceable supporting memory items instead of relying on hidden recall
- refuse or fall back safely when no sufficiently reliable structured memory is available
- hard refusal mainly applies to KB-oriented endpoints; in `/chat_mem`, the system may still generate a reply without memory support when no reliable stored memory is available
- support a small evaluation setup covering routing and fallback behavior in the current livestream knowledge layer

## Example Workflows

### 1. Structured fact extraction from interaction

A livestream or product-related input can be converted into a structured fact instead of remaining only as raw chat history. For example, a message such as `A款价格是99元` can be stored as a typed fact like `product_price = 99元`, making it available for later retrieval and update handling.

### 2. Traceable typed retrieval

When the system answers a follow-up query, it can retrieve the relevant stored fact and surface the supporting memory entry rather than relying on hidden recall. For example, after storing `A款价格是99元`, a later query such as `A款多少钱` can be answered with traceable evidence showing which fact was used.

### 3. Overwrite and soft deactivation

If newer information of the same type appears, the system can update the active fact without discarding the older record entirely. For example, if `A款价格是99元` is later followed by `A款价格是89元`, the newer price can become active while the older one is preserved through soft deactivation for traceability.

### 4. Product-level separation

The system can separate facts across products instead of forcing all livestream knowledge into a single default product context. For example, `A款价格是99元` and `B款价格是199元` can be stored under different product entities, so that later retrieval and overwrite behavior remains product-specific.

### 5. Livestream commerce query routing

For livestream commerce queries, the system does not rely on a separate intent classifier. Instead, it performs semantic retrieval over eligible knowledge candidates, applies fact-type-specific validity checks and routing thresholds, and uses the highest-scoring eligible fact type among the top-k retrieval candidates as the routed type.

In other words, livestream fact-type routing is implemented through retrieval-time score competition rather than a standalone classification step.

Current examples include:

- `这款多少钱` → product price
- `今天有什么优惠` → promotions
- `现在有货吗` → stock status
- `多久能发货` → shipping policy
- `这款有什么特点` → product features

### 6. Lifecycle-aware fallback and refusal

The system can also fall back or refuse when stored knowledge should no longer be used. This includes cases where a fact is stale, inactive, unsupported, or otherwise not reliable enough to support an answer. For example, an outdated promotion may still remain in storage for history tracking, but it can be filtered out or refused at answer time rather than being treated as current knowledge.

## Evaluation

This repository includes a scenario-based evaluation setup for the livestream knowledge and memory layer.

The evaluation is intentionally small-scale and behavior-focused. It is not intended as a broad language-model benchmark. Instead, it checks whether the system can handle commerce-oriented memory behavior such as:

- structured fact extraction
- livestream fact-type routing
- product-level entity separation
- overwrite behavior for updated facts
- active-state filtering
- non-fact filtering
- fallback or refusal when no reliable memory is available
- traceable retrieval through returned sources

Current evaluation result: **11 / 11 scenario-based cases passed**.

The current cases cover product price retrieval, price overwrite, unsupported-query fallback, entity separation, stock status retrieval, stock overwrite, promotion retrieval, promotion overwrite, shipping policy retrieval, product feature retrieval, and non-fact filtering.

Evaluation files:

- `eval/eval_livestream_cases.json` — scenario cases and expected outcomes
- `eval/eval_livestream.py` — lightweight evaluation runner
- `eval/eval_report.md` — evaluation scope, current results, limitations, and next steps
- `eval/results/eval_result_11_pass.txt` — saved run output

The main evaluation flow uses `/chat_mem` to ingest structured facts and `/chat_livestream_kb` to retrieve and answer from the structured livestream knowledge base.

## Running the Project

This project is intended to run locally with Docker Compose.

Start the services with:

```bash
docker compose up -d
```

The local setup includes the API service, Ollama for local model inference and embeddings, and Qdrant for vector storage and typed memory retrieval.

You can check that the API is running with:

```bash
curl http://127.0.0.1:8000/health
```

If you update the API code, rebuild the API service with:

```bash
docker compose up -d --build api
```

## Running the Evaluation

The evaluation runner is designed to run against the local API service. It uses `/chat_mem` to ingest setup messages into the structured memory layer, and then uses `/chat_livestream_kb` to retrieve livestream knowledge and answer the final query.

Start the local services first:

```bash
docker compose up -d --build
```

Check that the API is running:

```bash
curl http://127.0.0.1:8000/health
```

Copy the evaluation folder into the API container and run the evaluation:

```bash
docker compose exec api rm -rf /app/eval
docker compose cp eval api:/app/eval
docker compose exec api python /app/eval/eval_livestream.py
```

The latest saved evaluation output is available at `eval/results/eval_result_11_pass.txt`.

## Repository Structure

- `api/main.py` — main FastAPI application, including memory logic, retrieval control, routing, overwrite behavior, and debug endpoints
- `api/Dockerfile` — Docker image definition for the API service
- `api/requirements.txt` — Python dependencies for the API service
- `docker-compose.yml` — local multi-service setup for the API, Ollama, and Qdrant
- `docs/` — project diagrams and alignment notes
- `eval/eval_livestream_cases.json` — scenario-based evaluation cases and expected outcomes
- `eval/eval_livestream.py` — lightweight evaluation runner for the livestream memory cases
- `eval/eval_report.md` — evaluation scope, current result, limitations, and next steps
- `eval/results/eval_result_11_pass.txt` — saved output from the 11/11 evaluation run
- `PROJECT_SUMMARY_FOR_ADMISSIONS.md` — concise project summary for admissions review
- `README.md` — project overview, capabilities, evaluation notes, and usage instructions

## Notes

This repository reflects an ongoing iteration rather than a finished product. The project is still intentionally compact, and a number of design choices remain visible at the application level for ease of inspection and experimentation.

At the current stage, the implementation still keeps much of the logic in a single main service file. This makes iteration easier during development and keeps policy, overwrite, routing, and retrieval behavior easy to inspect, even though future cleanup would likely separate routing, memory policy, evaluation, and storage logic more clearly as the project grows.

## Next Steps

Short-term improvements:

- add timestamp-controlled freshness tests for stale promotion and outdated stock facts
- add more ambiguous product-reference cases, such as similar product names or incomplete product mentions
- save machine-readable evaluation results after each run, in addition to the current text log
- improve non-fact filtering cases beyond simple greetings

Medium-term improvements:

- separate extraction, routing, storage, lifecycle policy, and evaluation logic into clearer modules
- make the fact-policy registry easier to extend through configuration
- add a retail operations decision-support extension using anonymized or synthetic store metrics
- explore how structured operational memory can support cross-store comparison and data-driven business decision-making
