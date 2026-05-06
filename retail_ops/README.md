# Retail Operations SQL Extension

This folder extends the original livestream-agent memory layer into a retail operations decision-support scenario.

The original project focused on lifecycle-aware memory for livestream commerce facts such as price, stock status, promotion, shipping policy, and product features. This extension applies the same idea to multi-store Meituan instant-retail operations.

## Motivation

Meituan merchant backend provides many detailed metrics, but the interface is mainly store-centric. It is difficult to compare multiple stores efficiently under the same metric structure.

This extension demonstrates a small SQL-based workflow for reorganizing manually collected store-level backend metrics into a cross-store comparison layer.

## Data Scope

- Period: March 2026
- Stores: 5 anonymized stores
- Regions: Qingdao urban area and Yantai urban area
- Store types: self-operated and partner-operated
- Category: contact lenses and care-solution related instant retail

## SQL Workflow

The SQL files calculate:

1. Derived operational metrics
2. Cross-store rankings
3. Conservative store-level operational tags

Key derived metrics include:

- Search visit share
- Search exposure-to-visit rate
- Refund revenue ratio
- Refund order ratio
- Promotion GMV to revenue ratio
- Merchant subsidy to revenue ratio
- Revenue per visitor

## Connection to Memory Layer

The output is converted into structured memory facts in:

`outputs/generated_memory_facts.json`

These facts are not treated as universal business conclusions. Each memory fact includes:

- type
- period
- store scope
- value
- decision use
- confidence
- source

This prevents the AI assistant from overgeneralizing one store's pattern as a universal rule.

## Files

- `sql/01_derived_metrics.sql`: calculates comparable cross-store metrics
- `sql/02_cross_store_ranking.sql`: ranks stores across selected metrics
- `sql/03_conservative_store_tags.sql`: converts metrics into conservative operational tags
- `outputs/derived_metrics_output.csv`: SQL output for derived metrics
- `outputs/cross_store_ranking_output.csv`: SQL output for cross-store rankings
- `outputs/store_tags_output.csv`: SQL output for operational tags
- `outputs/cross_store_comparison_report.md`: short analysis report
- `outputs/generated_memory_facts.json`: structured facts for the memory layer
