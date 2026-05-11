import json
from pathlib import Path

FACTS_PATH = Path("retail_ops/outputs/generated_demo2_retail_memory_facts.json")

EXPECTED_ENTITIES = {"store_B", "store_C", "store_D", "store_E", "store_F"}

EXPECTED_SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "order_quality_pressure_profile",
    "top3_sku_product_mix_note",
    "single_metric_attribution_guard",
}

REQUIRED_KEYS = [
    "kind",
    "type",
    "entity_id",
    "slot",
    "period_label",
    "period_start",
    "period_end",
    "value",
    "observed_values",
    "calculation",
    "source_fields",
    "confidence",
    "source_path",
    "lineage_path",
    "limitations",
    "is_active",
    "period_granularity",
]

data = json.loads(FACTS_PATH.read_text(encoding="utf-8"))

if len(data) != 25:
    raise SystemExit(f"Expected 25 Demo 2 facts, got {len(data)}")

entities = {fact["entity_id"] for fact in data}
slots = {fact["slot"] for fact in data}

if entities != EXPECTED_ENTITIES:
    raise SystemExit(f"Unexpected entities: {sorted(entities)}")

if slots != EXPECTED_SLOTS:
    raise SystemExit(f"Unexpected slots: {sorted(slots)}")

seen_pairs = set()

for fact in data:
    missing_keys = [key for key in REQUIRED_KEYS if key not in fact]

    if missing_keys:
        raise SystemExit(f"Missing keys in fact: {missing_keys}")

    pair = (fact["entity_id"], fact["slot"])

    if pair in seen_pairs:
        raise SystemExit(f"Duplicate entity-slot pair: {pair}")

    seen_pairs.add(pair)

    if fact["kind"] != "retail_memory_fact":
        raise SystemExit(f"{pair}: bad kind")

    if fact["type"] != "retail_metric_profile":
        raise SystemExit(f"{pair}: bad type")

    if fact["period_label"] != "2026-03":
        raise SystemExit(f"{pair}: bad period_label")

    if fact["period_start"] != "2026-03-01":
        raise SystemExit(f"{pair}: bad period_start")

    if fact["period_end"] != "2026-03-31":
        raise SystemExit(f"{pair}: bad period_end")

    if fact["period_granularity"] != "month":
        raise SystemExit(f"{pair}: bad period_granularity")

    if fact["is_active"] is not True:
        raise SystemExit(f"{pair}: fact should be active")

    if fact["confidence"] not in {"high", "medium"}:
        raise SystemExit(f"{pair}: unexpected confidence")

    if not isinstance(fact["observed_values"], dict):
        raise SystemExit(f"{pair}: observed_values must be an object")

    if not isinstance(fact["source_fields"], list) or not fact["source_fields"]:
        raise SystemExit(f"{pair}: source_fields must be a non-empty list")

    if not isinstance(fact["limitations"], list) or not fact["limitations"]:
        raise SystemExit(f"{pair}: limitations must be a non-empty list")

print("Demo 2 retail memory facts validation PASSED.")
print("Checked facts: 25")
print("Checked entities: store_B, store_C, store_D, store_E, store_F")
print("Checked slots:", ", ".join(sorted(EXPECTED_SLOTS)))
print("Checked schema, period, source fields, limitations, and active status.")
