from pathlib import Path

API_PATH = Path("api/main.py")
text = API_PATH.read_text(encoding="utf-8")

checks = {
    "old_retail_endpoint": '@app.post("/chat_retail_ops_kb")',
    "demo2_retail_endpoint": '@app.post("/chat_retail_ops_demo2_kb")',
    "demo2_request_model": "class RetailOpsDemo2KbReq(BaseModel):",
    "demo2_entity_normalizer": "def normalize_demo2_retail_entity_id",
    "demo2_cross_store_detector": "def is_demo2_cross_store_query",
    "demo2_scope_guard": "def is_unsupported_demo2_retail_scope",
    "demo2_fact_loader": "def load_demo2_retail_facts",
    "demo2_endpoint_function": "async def chat_retail_ops_demo2_kb",
}

failures = []

for name, needle in checks.items():
    count = text.count(needle)

    if count != 1:
        failures.append(f"{name}: expected 1 occurrence of {needle!r}, found {count}")

required_phrases = [
    "Demo 2 currently supports only the anonymized B-F same-period diagnostic sample",
    "Demo 2 supports cautious diagnostic comparison, not best-store ranking",
    "generated_demo2_retail_memory_facts.json",
    "demo2_cross_store",
    "file_backed_retail_memory_facts",
]

for phrase in required_phrases:
    if phrase not in text:
        failures.append(f"Missing required Demo 2 phrase: {phrase}")

for forbidden in [
    "git status",
    "shortail",
    "short/scripts",
    "DATA_DICTIONARY.mdp",
    "competitor quantity",
]:
    if forbidden in text:
        failures.append(f"Found likely paste pollution in api/main.py: {forbidden}")

if failures:
    print("Demo 2 API endpoint validation FAILED.")
    for failure in failures:
        print("-", failure)
    raise SystemExit(1)

print("Demo 2 API endpoint validation PASSED.")
print("Checked old retail endpoint remains present.")
print("Checked new Demo 2 endpoint and helper functions.")
print("Checked Demo 2 file-backed facts path and refusal boundaries.")
