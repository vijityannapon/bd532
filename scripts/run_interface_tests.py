"""Run the Postman collection's scenarios against a live server, without Postman.

Usage:  python scripts/run_interface_tests.py [--base-url http://localhost:8000]
Writes reports/interface_test_results.md (evidence for Section 4.2).
"""
import argparse
import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "postman" / "credit_risk_api.postman_collection.json"


def expected_status(item):
    for ev in item.get("event", []):
        for line in ev["script"]["exec"]:
            m = re.search(r"to\.have\.status\((\d+)\)", line)
            if m:
                return int(m.group(1))
    return None


def expected_band(item):
    for ev in item.get("event", []):
        for line in ev["script"]["exec"]:
            m = re.search(r'decision_band\)\.to\.eql\("(\w+)"\)', line)
            if m:
                return m.group(1)
    return None


def main(base_url):
    coll = json.loads(COLLECTION.read_text())
    variables = {"baseUrl": base_url, "lastRequestId": "", "pdS2": ""}
    rows, passed = [], 0
    with httpx.Client(timeout=30) as client:
        for folder in coll["item"]:
            for item in folder["item"]:
                r = item["request"]
                body = r.get("body", {}).get("raw")
                if body == "{{bigBatch}}":
                    base = json.loads(re.search(r"const a = (\{.*\});", item["event"][0]["script"]["exec"][0]).group(1))
                    body = json.dumps({"applications": [{"applicant": base, "explain": False}] * 101})
                for k, v in variables.items():
                    body = body.replace("{{%s}}" % k, str(v)) if body else body
                url = r["url"]["raw"].replace("{{baseUrl}}", base_url)
                t0 = time.perf_counter()
                resp = client.request(r["method"], url, content=body,
                                      headers={"Content-Type": "application/json"})
                ms = (time.perf_counter() - t0) * 1000
                exp = expected_status(item)
                ok = resp.status_code == exp
                try:
                    j = resp.json()
                except ValueError:
                    j = {}
                detail = ""
                if "prediction" in j:
                    p = j["prediction"]
                    detail = f"PD={p['probability_of_default']:.1%}, band={p['decision_band']}"
                    want = expected_band(item)
                    if want and p["decision_band"] != want:
                        ok = False
                        detail += f" (expected {want})"
                    variables["lastRequestId"] = j["request_id"]
                    if item["name"].startswith("S2"):
                        variables["pdS2"] = p["probability_of_default"]
                    if item["name"].startswith("S4"):
                        same = p["probability_of_default"] == variables["pdS2"]
                        ok = ok and same
                        detail += ", same PD as S2" if same else ", PD DIFFERS from S2"
                elif "error" in j:
                    detail = j["error"]["code"]
                elif "count" in j:
                    detail = f"{j['count']} results"
                elif "feedback_id" in j:
                    detail = "stored"
                elif "model_loaded" in j:
                    detail = f"model_loaded={j['model_loaded']}"
                elif "features" in j:
                    detail = f"{len(j['features'])} features, gender used: {'gender' in j['features']}"
                passed += ok
                rows.append((folder["name"], item["name"], r["method"], url.replace(base_url, ""),
                             exp, resp.status_code, f"{ms:.0f}", detail, "PASS" if ok else "FAIL"))
    lines = ["# UI-Model interface test results", "",
             f"Server: `{base_url}`  |  {passed}/{len(rows)} passed  |  {time.strftime('%Y-%m-%d %H:%M')}", "",
             "| Group | Scenario | Method | Path | Expected | Actual | ms | Output | Result |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    out = ROOT / "reports" / "interface_test_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return passed == len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    raise SystemExit(0 if main(ap.parse_args().base_url) else 1)
