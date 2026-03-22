import json
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8000"
CASES_FILE = Path("eval_livestream_cases.json")


def check_case(case: dict, result: dict) -> tuple[bool, list[str]]:
    errors = []
    expect = case.get("expect", {})

    expected_refusal = expect.get("refusal")
    if expected_refusal is not None:
        actual_refusal = result.get("refusal")
        if actual_refusal != expected_refusal:
            errors.append(
                f"refusal expected {expected_refusal}, got {actual_refusal}"
            )

    expected_fact_type = expect.get("routed_fact_type")
    if expected_fact_type is not None:
        actual_fact_type = result.get("routed_fact_type")
        if actual_fact_type != expected_fact_type:
            errors.append(
                f"routed_fact_type expected {expected_fact_type}, got {actual_fact_type}"
            )

    expected_reply_contains = expect.get("reply_contains")
    if expected_reply_contains:
        actual_reply = result.get("reply", "")
        if expected_reply_contains not in actual_reply:
            errors.append(
                f"reply does not contain expected text: {expected_reply_contains}"
            )

    return (len(errors) == 0, errors)


def main():
    if not CASES_FILE.exists():
        print(f"[ERROR] Missing file: {CASES_FILE}")
        return

    with CASES_FILE.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    passed = 0
    failed = 0

    with httpx.Client(timeout=60) as client:
        for i, case in enumerate(cases, start=1):
            name = case.get("name", f"case_{i}")
            endpoint = case.get("endpoint")
            payload = case.get("payload", {})

            if not endpoint:
                print(f"[FAIL] {name}")
                print("  - missing endpoint")
                failed += 1
                continue

            url = f"{BASE_URL}{endpoint}"

            try:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                print(f"[FAIL] {name}")
                print(f"  - request error: {e}")
                failed += 1
                continue

            ok, errors = check_case(case, result)

            if ok:
                print(f"[PASS] {name}")
                passed += 1
            else:
                print(f"[FAIL] {name}")
                for err in errors:
                    print(f"  - {err}")
                print(f"  - reply: {result.get('reply')}")
                print(f"  - routed_fact_type: {result.get('routed_fact_type')}")
                failed += 1

    total = passed + failed
    print()
    print("=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {total}")


if __name__ == "__main__":
    main()
