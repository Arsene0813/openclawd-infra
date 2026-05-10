## System Evolution

- **Local LLM backend**  
  The project began as a simple local model-serving setup for conversational use in a livestream-oriented setting. The initial goal was to make local generation work reliably enough to support product explanation and customer-facing interaction.

- **Raw vector chat memory**  
  The next step was to add vector-based recall so that past exchanges could be retrieved instead of being discarded entirely. This made the system more context-aware, but it also exposed an important limitation: retrieval alone did not make memory reliable. Past content could be surfaced based on similarity without any clear standard for whether it should still matter.

- **Retrieval gating and traceability**  
  To address that limitation, I introduced retrieval gating and fallback or refusal conditions so that memory would only be used when there was enough evidence that it was genuinely relevant. I also added traceable evidence in retrieval outputs so that memory usage could be inspected afterward rather than remaining hidden inside model behavior.

- **Structured facts and typed memory**  
  Once retrieval became more controlled, the next issue became clearer: storing past text was still not the same as representing knowledge in a stable form. The system therefore moved from unstructured recall toward structured fact extraction and typed memory, allowing different categories of information to be stored and handled more consistently.

- **Update policies and overwrite control**  
  After typed memory was in place, I added overwrite logic so that newer facts could replace older active knowledge in a controlled way rather than letting conflicting information accumulate indefinitely. Older facts were preserved through soft deactivation instead of being deleted without trace, which made memory updates easier to inspect and reason about.

- **Lifecycle-aware retrieval**  
  The project then evolved into a lifecycle-aware memory layer. Stored facts were no longer treated as timeless entries, but as knowledge objects with timestamps, active-state flags, freshness windows, and reuse metadata. This allowed retrieval to prefer currently valid knowledge while filtering out inactive or stale facts when appropriate.

- **Livestream commerce knowledge routing**  
  The next iteration extended the same memory architecture into livestream commerce knowledge. Structured facts such as product price, promotions, stock status, shipping policy, and product features could be written into typed memory and retrieved later through the same lifecycle-aware logic. On top of that, I added automatic fact-type routing so that the system could handle livestream questions without requiring the caller to manually specify which category of knowledge should be searched.

- **Policy-driven memory behavior**  
  As the number of fact types increased, hard-coded rules for overwrite, freshness, routing thresholds, scope, and storage semantics became difficult to maintain. To address this, the system moved toward a centralized fact-policy registry, making storage and retrieval behavior easier to extend and reason about.

- **Product-level entity separation**  
  Once livestream facts became more structured, it was no longer sufficient to store all product knowledge under a single default product context. The system therefore introduced lightweight product reference extraction so that facts could be attached to specific product entities when possible, while still falling back to a default product scope when no explicit reference was available.

- **Single-call extraction and non-fact filtering**  
  The earlier pipeline separated memory gating from fact extraction, which increased latency and could produce inconsistent behavior. The current version consolidates this flow so that extraction itself determines whether a message yields a storable fact, while obvious non-fact messages such as greetings are filtered out before memory write.

- **Explainable fallback and small-scale evaluation**  
  At the current stage, the system can not only answer from stored knowledge, but also fall back safely when available knowledge is stale, inactive, unsupported, or insufficiently reliable. Retrieval outputs expose routing decisions and filtered reasons, making failure cases easier to inspect rather than leaving them ambiguous. To make the current behavior easier to verify, I added a scenario-based evaluation setup covering successful retrieval, fact-type routing, overwrite behavior, product-level separation, non-fact filtering, and unsupported-query fallback. Freshness filtering is implemented in the retrieval layer, while timestamp-controlled freshness tests remain a planned next step.
