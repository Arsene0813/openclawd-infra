# Project Summary for University Application

## 1. Motivation

This project started from my earlier LLM-powered livestream system. Although the system could generate fluent product explanations and customer-facing responses, I found that fluent generation alone was not enough for reliable commercial use.

The key problem was memory management: newer information needed to replace older information, outdated information should not remain blindly retrievable, and the system should be able to explain which stored fact supported an answer.

## 2. Problem

In livestream commerce, product information changes frequently:
- product prices may be updated
- promotions may expire
- stock status may change
- shipping policy may vary
- product features need to remain entity-specific

A simple chatbot or vector memory system may retrieve semantically similar but outdated or conflicting information.

## 3. My Solution

I built a lifecycle-aware memory layer that:
- extracts structured facts from interaction
- stores facts as typed memory
- separates product-level entities
- applies overwrite rules to newer same-type facts
- preserves older facts through soft deactivation
- filters stale or inactive knowledge during retrieval
- returns traceable supporting memory items

## 4. Example

If the system first stores:

A款价格是99元

and later stores:

A款价格是89元

then a later query:

A款多少钱？

should use the newer active price, while preserving the older price as an inactive historical record.

## 5. Relevance to My Target Programmes

This project is relevant to my intended studies in AI, computing, business analytics, and data science because it involves:
- AI system design
- information extraction
- structured data representation
- retrieval-augmented generation
- memory lifecycle control
- evaluation of system behavior
- decision-support in commerce scenarios

## 6. Current Extension

I am extending the same design toward retail operations decision support. In instant-retail scenarios, store-level data such as search exposure, store visits, conversion rate, campaign cost, refund behavior, and product mix can be represented as structured operational memory for more reliable AI-assisted analysis.
