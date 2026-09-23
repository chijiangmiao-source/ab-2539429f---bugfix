#!/usr/bin/env python3
"""Live API smoke checks executed by the one-shot `verify` service.

Two required scenarios are exercised over real HTTP:

1. the four-tensor ring: co-optimal root-cut classification must report
   0 mandatory / 6 optional / 1 absent cuts, with the diagonal cut
   (A,C)|(B,D) absent because forcing it builds a size-16 intermediate;
2. an index occurring three times must be rejected (HTTP 422) with an
   ``index_occurrence`` error carrying a precise JSON path.

Exits 0 only when every assertion holds.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def _request(base: str, method: str, path: str, payload: object = None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def ring4() -> dict:
    return {
        "tensors": [
            {"name": "A", "indices": [
                {"name": "i", "dimension": 2}, {"name": "j", "dimension": 2}]},
            {"name": "B", "indices": [
                {"name": "j", "dimension": 2}, {"name": "k", "dimension": 2}]},
            {"name": "C", "indices": [
                {"name": "k", "dimension": 2}, {"name": "l", "dimension": 2}]},
            {"name": "D", "indices": [
                {"name": "l", "dimension": 2}, {"name": "i", "dimension": 2}]},
        ]
    }


def bad_three_occurrences() -> dict:
    return {
        "tensors": [
            {"name": "A", "indices": [{"name": "i", "dimension": 2}]},
            {"name": "B", "indices": [
                {"name": "i", "dimension": 2}, {"name": "j", "dimension": 2}]},
            {"name": "C", "indices": [
                {"name": "i", "dimension": 2}, {"name": "j", "dimension": 2}]},
        ]
    }


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail and not ok else ""))
    return ok


def main(base: str) -> int:
    failures = 0

    print(f"Smoke testing API at {base}")
    status, body = _request(base, "GET", "/healthz")
    failures += not check("GET /healthz -> 200 ok", status == 200 and body.get("status") == "ok",
                          f"got {status} {body}")

    # scenario 1: four-tensor ring co-optimal classification
    status, body = _request(base, "POST", "/api/plan", ring4())
    ok = status == 200
    failures += not check("POST /api/plan ring4 -> 200", ok, f"got {status} {body}")
    if ok:
        summary = body["summary"]
        failures += not check(
            "ring4 optimum peak=4 total=20",
            summary["peak_memory"] == 4 and summary["total_multiplications"] == 20,
            f"got {summary}",
        )
        counts = {"mandatory": 0, "optional": 0, "absent": 0}
        absent_cut = None
        for cut in body["cuts"]:
            counts[cut["classification"]] += 1
            if cut["classification"] == "absent":
                absent_cut = cut
        failures += not check(
            "ring4 classes: 0 mandatory / 6 optional / 1 absent",
            counts == {"mandatory": 0, "optional": 6, "absent": 1},
            f"got {counts}",
        )
        diagonal = absent_cut is not None and (
            {tuple(absent_cut["left"]), tuple(absent_cut["right"])}
            == {("A", "C"), ("B", "D")}
        )
        failures += not check("the unique absent cut is (A,C)|(B,D)", diagonal,
                              f"got {absent_cut}")
        evidenced = absent_cut is not None and "16" in absent_cut.get("evidence", "")
        failures += not check(
            "absent-cut evidence cites forced intermediate size 16",
            evidenced,
            f"evidence={absent_cut.get('evidence') if absent_cut else None}",
        )

    # scenario 2: index occurring three times is a located 422
    status, body = _request(base, "POST", "/api/plan", bad_three_occurrences())
    ok = status == 422 and any(e.get("code") == "index_occurrence" for e in body.get("errors", []))
    failures += not check("triple index -> HTTP 422 with index_occurrence", ok,
                          f"got {status} {body}")
    err = next((e for e in body.get("errors", []) if e.get("code") == "index_occurrence"), None)
    located = bool(err) and err.get("path", [None])[0] == "tensors"
    failures += not check("error path locates tensors[*].indices[*].name",
                          located and err["path"][-1] == "name", f"got {err}")

    print()
    if failures:
        print(f"SMOKE FAILED: {failures} check(s) failed")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://api:8000"
    sys.exit(main(base))
