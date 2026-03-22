# From an LLM-Powered Livestream System to a Lifecycle-Aware Agent Memory Layer

An iterative project that extends an earlier LLM-powered livestream system into a lifecycle-aware memory and retrieval layer for a more agent-like architecture.

This project started as the next iteration of my earlier livestream system. Instead of focusing only on fluent generation, it moves toward structured memory, typed knowledge storage, overwrite control, lifecycle-aware retrieval, and traceable decision behavior.

At its current stage, the system can also handle livestream commerce knowledge such as product price, promotions, stock status, shipping policy, and product features. It can route a query to the most relevant fact type, answer from stored knowledge with traceable support, and refuse stale or unsupported cases instead of relying on ungrounded generation.

## Current Scope

At the moment, the project includes:

- structured fact extraction from interaction
- typed knowledge storage
- overwrite control and soft deactivation
- lifecycle-aware retrieval
- automatic routing across five livestream fact types
- refusal for stale or unsupported cases
- a small evaluation setup for routing and refusal behavior

## Why This Project

My earlier livestream system could already support product explanation and customer-facing dialogue, but I gradually realized that fluent generation alone was not enough to make such a system reliable in practice. The main limitation was not language generation itself, but the lack of a clear way to manage memory over time: newer information had no principled mechanism for replacing older information, outdated content could still remain retrievable, and it was often difficult to explain why a particular past item of information had been used in a response.

This project grew out of that limitation. Its purpose is to make an LLM-based interaction layer more dependable and easier to inspect by introducing structured memory, retrieval gating, overwrite logic, and lifecycle-aware updates.

## System Evolution

- **Local LLM backend**  
  The project began as a simple local model-serving setup for conversational use in a livestream-oriented setting. At that stage, the main goal was to make local generation work reliably enough to support product explanation and customer-facing interaction.

- **Raw vector chat memory**  
  The next step was to add vector-based recall so that past exchanges could be retrieved instead of being discarded entirely. This made the system more context-aware, but it also exposed an important limitation: retrieval alone did not make memory reliable. Past content could be surfaced based on similarity without any clear standard for whether it should still matter.

- **Retrieval gating and traceability**  
  To address that, I introduced retrieval gating and refusal conditions so that memory would only be used when there was enough evidence that it was genuinely relevant. I also added traceable evidence in retrieval outputs, so that memory usage could be inspected afterward rather than remaining hidden inside model behavior.

- **Structured facts and typed memory**  
  Once retrieval became more controlled, the next issue became clearer: storing past text was still not the same as representing knowledge in a stable form. The system therefore moved from unstructured recall toward structured fact extraction and typed memory, allowing different categories of information to be stored and handled more consistently.

- **Update policies and overwrite control**  
  After typed memory was in place, I added overwrite logic so that newer same-type facts could replace older active knowledge in a controlled way, rather than letting conflicting information accumulate indefinitely. Older facts were preserved through soft deactivation instead of being deleted without trace, which made memory updates easier to inspect and reason about.

- **Lifecycle-aware retrieval**  
  From there, the project evolved into a lifecycle-aware memory layer. Stored facts were no longer treated as timeless entries, but as knowledge objects with timestamps, active-state flags, freshness windows, and reuse metadata. This allowed retrieval to prefer currently valid knowledge while filtering out inactive or stale facts when appropriate.

- **Livestream commerce knowledge routing**  
  The next iteration extended the same memory architecture into livestream commerce knowledge. Structured facts such as product price, promotions, stock status, shipping policy, and product features could be written into typed memory and retrieved later through the same lifecycle-aware logic. On top of that, I added automatic fact-type routing so that the system could handle livestream questions without requiring the caller to manually specify which category of knowledge should be searched.

- **Explainable refusal and small-scale evaluation**  
  At the current stage, the system can not only answer from stored knowledge, but also refuse when available knowledge is stale, inactive, or insufficiently reliable. Retrieval outputs now expose routing decisions and filtered reasons, making failure cases easier to inspect rather than leaving them ambiguous. To make the current behavior easier to verify, I also added a small evaluation setup covering successful routing cases as well as refusal cases such as stale promotions and unsupported queries.

## Current Capabilities

At its current stage, the system can:

- extract structured facts from interactions instead of relying only on raw conversational history
- store knowledge as typed memory so that different categories of information can be handled more consistently
- apply overwrite control when newer same-type facts supersede older active facts
- preserve older records through soft deactivation rather than deleting them without trace
- attach lifecycle metadata such as timestamps, freshness windows, and active-state signals to stored knowledge
- filter inactive or stale facts during retrieval instead of treating all stored knowledge as equally usable
- surface supporting memory items, routing decisions, and filtered reasons for inspection
- refuse when no sufficiently reliable knowledge is available instead of fabricating an answer
- route livestream commerce queries automatically across product price, promotions, stock status, shipping policy, and product features
- run a small evaluation set for livestream routing and refusal behavior

## Example Workflows

### 1. Structured fact extraction from interaction

A conversational input can be converted into a structured fact instead of remaining only as raw chat history. For example, a message such as `我住在北京` can be stored as a typed fact like `location = 北京`, making it available for later retrieval and update handling.

### 2. Traceable typed retrieval

When the system answers a follow-up query, it can retrieve the relevant stored fact and surface the supporting memory item rather than relying on hidden recall. For example, after storing `我住在北京`, a later query such as `我住在哪` can be answered with traceable evidence showing which fact was used.

### 3. Overwrite and soft deactivation

If newer information of the same type appears, the system can update the active fact without discarding the older record entirely. For example, if `我住在上海` is later followed by `我住在北京`, the newer location can become active while the older one is preserved through soft deactivation for traceability.

### 4. Livestream commerce question routing

For livestream commerce queries, the system can route the question to the most relevant fact type instead of requiring the caller to specify it manually.

Current examples include:

- `这款多少钱` → product price
- `今天有什么优惠` → promotions
- `现在有货吗` → stock status
- `多久能发货` → shipping policy
- `这款有什么特点` → product features

### 5. Lifecycle-aware refusal

The system can also refuse when stored knowledge should no longer be used. This includes cases where a fact is stale, inactive, or otherwise not reliable enough to support an answer. For example, an outdated promotion may still remain in storage for history tracking, but it can be filtered out during retrieval and refused at answer time rather than being treated as current knowledge.

## Evaluation

The repository includes a small evaluation setup for the current livestream knowledge layer.

At the moment, the evaluation covers seven scenarios in total:

- five successful routing cases:
  - product price
  - promotions
  - stock status
  - shipping policy
  - product features
- two refusal cases:
  - stale promotion knowledge
  - unsupported or no-match queries

The evaluation files are:

- `eval_livestream_cases.json` – evaluation cases and expected outcomes
- `eval_livestream.py` – a simple script that runs the cases against the local API and reports pass/fail results

A typical run checks whether the system:

- routes a query to the expected fact type
- produces a non-refusal answer when reliable knowledge is available
- refuses when knowledge is stale or when no sufficiently reliable knowledge can be found
- returns responses that contain the expected key information

This evaluation is intentionally small. Its purpose is not to claim broad benchmark coverage, but to verify that the current routing, retrieval, and refusal behavior matches the intended design for the livestream scenarios implemented so far.

## Running the Evaluation

Start the local services first:

```bash
docker compose up -d
```

If you have changed the API code, rebuild the API service:

```bash
docker compose up -d --build api
```

The evaluation script is designed to run against the local API at `http://127.0.0.1:8000`.

Because the script depends on the API container environment, a simple way to run it is to copy the evaluation files into the running API container and execute the script there:

```bash
docker cp eval_livestream.py oc_api:/app/eval_livestream.py
docker cp eval_livestream_cases.json oc_api:/app/eval_livestream_cases.json
docker exec -it oc_api python /app/eval_livestream.py
```

A typical successful run reports pass/fail results for each case and prints a short summary at the end.

## Repository Structure

- `api/main.py` - main FastAPI application, including memory logic, retrieval control, routing, and debug endpoints
- `api/Dockerfile` - Docker image definition for the API service
- `api/requirements.txt` - Python dependencies for the API service
- `docker-compose.yml` - local multi-service setup for the API and supporting services
- `eval_livestream_cases.json` - small evaluation set for livestream routing and refusal cases
- `eval_livestream.py` - simple evaluation runner for the livestream cases
- `README.md` - project overview, current capabilities, evaluation notes, and usage instructions

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

This project also includes a small evaluation setup for the livestream knowledge layer. After the services are running, the evaluation script can be copied into the API container and executed there:

```bash
docker cp eval_livestream.py oc_api:/app/eval_livestream.py
docker cp eval_livestream_cases.json oc_api:/app/eval_livestream_cases.json
docker exec -it oc_api python /app/eval_livestream.py
```

## Notes

This repository reflects an ongoing iteration rather than a finished product. The project is still intentionally compact, and a number of design choices remain visible at the application level for ease of inspection and experimentation.

At the current stage, the implementation still keeps much of the logic in a single main service file. This makes iteration easier during development, but future cleanup would likely separate routing, memory policy, evaluation, and storage logic more clearly as the project grows.

## Next Steps

- Expand the evaluation set beyond the current small livestream cases.
- Make routing and retrieval policies easier to configure and maintain.
- Explore richer profile-level memory on top of the current typed fact layer.
- Continue separating core memory logic from higher-level application behavior.
