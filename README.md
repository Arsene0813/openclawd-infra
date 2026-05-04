# From an LLM-Powered Livestream System to a Policy-Driven, Lifecycle-Aware Agent Memory Layer
A prototype memory and retrieval layer for improving AI response reliability in retail product interaction.

## Note for Admissions Reviewers

This repository has been continuously improved after my supplementary application material was prepared. The core project remains the same: it is a lifecycle-aware memory layer evolved from an earlier LLM-powered livestream system.

The original focus was on structured fact extraction, typed memory, overwrite control, freshness filtering, and traceable retrieval for livestream commerce knowledge such as product price, promotion, stock status, shipping policy, and product features.

Recent updates are intended to make the project easier to inspect and evaluate. They add clearer documentation, more evaluation cases, and an extension toward retail operations decision support. These additions do not replace the original project; they show how the same memory-layer design can be applied to broader commerce and data-driven decision-making scenarios.

## Current Status

- Core API: working locally with Docker Compose
- Memory layer: structured fact extraction, typed memory, overwrite control, lifecycle metadata, and traceable retrieval
- Livestream knowledge types: product price, promotion, stock status, shipping policy, and product features
- Evaluation: 11 / 11 scenario-based cases passed
- Main endpoints: `/chat_mem` for fact ingestion and `/chat_livestream_kb` for structured retrieval

## Introduction

An iterative project that extends an earlier LLM-powered livestream system into a policy-driven memory and retrieval layer for a more agent-like architecture.

This project began as the next iteration of an earlier livestream system. Instead of focusing only on fluent generation, it now moves toward structured fact extraction, policy-driven typed memory, entity-aware overwrite control, lifecycle-aware retrieval, and traceable decision behavior.

At its current stage, the system can handle livestream commerce knowledge such as product price, promotions, stock status, shipping policy, and product features. It supports typed fact storage, product-level entity separation, slot-based overwrite, non-fact filtering, and traceable retrieval behavior rather than relying on ungrounded generation.

## Current Scope

Currently, the project includes:

- structured fact extraction from interaction
- policy-driven typed memory through a centralized fact-policy registry
- slot-based overwrite control with soft deactivation
- entity-aware storage based on `entity_id + slot` instead of type-only overwrite
- product-level memory separation through lightweight `product_ref` extraction
- lifecycle metadata such as timestamps, freshness windows, active-state flags, and reuse metadata
- non-fact filtering so greetings and lightweight chat do not enter memory
- a hybrid memory path in which chat-vector recall is still used for conversational retrieval, while structured facts are extracted and written into the typed knowledge layer when applicable
- automatic routing across five livestream fact types:
  - product price
  - promotions
  - stock status
  - shipping policy
  - product features
- a broader policy registry that also includes additional session-scoped fact types used by extraction beyond the livestream commerce categories emphasized in this README
- traceable retrieval outputs and inspectable fallback behavior
- a scenario-based evaluation setup covering product price retrieval, overwrite behavior, entity separation, stock status, promotion retrieval, shipping policy, product features, non-fact filtering, and fallback/refusal behavior
- A legacy strict-threshold chat-memory endpoint is still kept for comparison and debugging, but it is not the primary interaction path of the current memory layer.

## Why This Project

My earlier livestream system could already support product explanation and customer-facing dialogue, but I gradually realized that fluent generation alone was not enough to make such a system reliable in practice. The main limitation was not language generation itself, but the lack of a clear way to manage memory over time: newer information had no principled mechanism for replacing older information, outdated content could still remain retrievable, and it was often difficult to explain why a particular past item of information had been used in a response.

This project emerged from that limitation. Its purpose is to make an LLM-based interaction layer more dependable and easier to inspect by introducing structured memory, retrieval gating, overwrite logic, and lifecycle-aware updates.

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
