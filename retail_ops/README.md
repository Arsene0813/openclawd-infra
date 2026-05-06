# Retail Operations SQL Extension

This folder extends the original livestream-agent memory layer into a retail operations decision-support scenario.

The original project focused on lifecycle-aware memory for livestream commerce facts such as price, stock status, promotion, shipping policy, and product features. This extension applies the same design principle to multi-store Meituan instant-retail operations: operational observations should not be treated as timeless memory. They should be structured, scoped, updated, checked, and reused only when the supporting data is still reliable.

## Motivation

Meituan merchant backend provides many useful store-level metrics, but the interface is mainly store-centric. For multi-store operations, the challenge is not simply whether data exists. The challenge is how to compare stores under the same metric structure before making decisions.

This extension demonstrates a small SQL-based workflow for reorganizing manually collected and anonymized Meituan backend metrics into a cross-store comparison layer.

The goal is not to claim that the system can fully automate business decisions. The goal is to show how fragmented store-level metrics can be converted into traceable operational memory for conservative AI-assisted decision support.

## Data Scope

Public demo scope:

- Period: March 2026
- Stores: 5 anonymized representative stores
- Regions: Qingdao urban area and Yantai urban area
- Store types: self-operated and partner-operated
- Category: contact lenses and care-solution related instant retail
- Source: manually organized Meituan merchant backend metrics

Privacy note:

The public repository uses a small anonymized representative sample for readability and privacy protection. It should not be interpreted as the full operational scale of the underlying business context. The broader application supplement provides additional real-world operational background.

## Core Workflow

```text
Meituan backend metrics
        ↓
Anonymized store-level CSV
        ↓
SQLite / SQL derived metrics
        ↓
Cross-store ranking and conservative tags
        ↓
Structured operational memory facts
        ↓
Decision-support demo with evidence and refusal rules
```

## What the SQL Layer Calculates

The SQL workflow calculates comparable indicators across stores, including:

- Search visit share
- Search exposure-to-visit rate
- Refund revenue ratio
- Refund order ratio
- Promotion GMV to revenue ratio
- Merchant subsidy to revenue ratio
- Merchant subsidy to promotion GMV ratio
- Revenue per visitor

These indicators are not treated as causal proof. They are used as structured signals for conservative operational review.

## Decision-Support Principle

The memory layer should store operational observations with clear scope, time period, source, and confidence level.

For example, if one store performs well, the system should not automatically recommend copying that store's strategy across all stores. It should first check whether the comparison is supported by similar region, store type, product mix, refund pressure, promotion intensity, traffic structure, and visitor value.

## Files

```text
retail_ops/
├── README.md
├── data/
│   └── store_monthly_metrics_sample.csv
├── sql/
│   ├── 01_derived_metrics.sql
│   ├── 02_cross_store_ranking.sql
│   └── 03_conservative_store_tags.sql
├── outputs/
│   ├── derived_metrics_output.csv
│   ├── cross_store_ranking_output.csv
│   ├── store_tags_output.csv
│   ├── cross_store_comparison_report.md
│   └── generated_memory_facts.json
├── run_retail_pipeline.py
└── demo_retail_decision_support.md
```

## How to Run

From the repository root:

```bash
python retail_ops/run_retail_pipeline.py
```

Expected outputs:

```text
retail_ops/outputs/derived_metrics_output.csv
retail_ops/outputs/cross_store_ranking_output.csv
retail_ops/outputs/store_tags_output.csv
retail_ops/outputs/cross_store_comparison_report.md
retail_ops/outputs/generated_memory_facts.json
```

## How This Connects to the Memory Layer

The SQL outputs are converted into structured operational memory facts in:

```text
retail_ops/outputs/generated_memory_facts.json
```

Each memory fact includes:

- fact type
- period
- store scope
- observation value
- decision-support use
- confidence level
- source SQL file or source metric

This prevents the AI assistant from overgeneralizing one store's pattern as a universal rule.

## Current Limitation

This is a data-readiness and decision-memory prototype, not a fully automated Meituan business optimization system.

The purpose is to demonstrate how fragmented store-level metrics can be standardized, compared, and converted into traceable memory facts before being reused by an AI assistant.
