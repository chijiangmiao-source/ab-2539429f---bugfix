#!/usr/bin/env python3
"""Live API smoke checks executed by the one-shot `verify` service.

Three required scenarios are exercised over real HTTP:

1. the four-tensor ring: co-optimal root-cut classification must report
   0 mandatory / 6 optional / 1 absent cuts, with the diagonal cut
   (A,C)|(B,D) absent because forcing it builds a size-16 intermediate;
2. a four-tensor network whose unique optimal root cut is reachable only
   through a non-minimal-peak frontier record of one side (the local peak
   is masked by an input on the other side): the cut
   {T0,T1,T2} | {T3} must be the single mandatory cut, identical to the
   canonical tree root, with the other six cuts absent;
3. an index occurring three times must be rejected (HTTP 422) with an
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


def masked_peak_network() -> dict:
    return {
        "tensors": [
            {"name": "T0", "indices": [
                {"name": "e0", "dimension": 3},
                {"name": "e3", "dimension": 4},
                {"name": "e4", "dimension": 5}]},
            {"name": "T1", "indices": [
                {"name": "e0", "dimension": 3},
                {"name": "e1", "dimension": 5},
                {"name": "e5", "dimension": 4}]},
            {"name": "T2", "indices": [
                {"name": "e1", "dimension": 5},
                {"name": "e2", "dimension": 3},
                {"name": "e3", "dimension": 4},
                {"name": "d6", "dimension": 4}]},
            {"name": "T3", "indices": [
                {"name": "e2", "dimension": 3},
                {"name": "e4", "dimension": 5},
                {"name": "e5", "dimension": 4},
                {"name": "d7", "dimension": 4},
                {"name": "d8", "dimension": 3}]},
        ]
    }


def _tree_leaves(node: dict) -> frozenset:
    if node.get("type") == "leaf":
        return frozenset({node["name"]})
    return _tree_leaves(node["left"]) | _tree_leaves(node["right"])


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

    # scenario 2: unique mandatory cut reached via a masked higher local peak
    status, body = _request(base, "POST", "/api/plan", masked_peak_network())
    ok = status == 200
    failures += not check("POST /api/plan masked-peak -> 200", ok, f"got {status} {body}")
    if ok:
        summary = body["summary"]
        failures += not check(
            "masked-peak optimum peak=720 total=8640 cotrees=1",
            (summary["peak_memory"] == 720
             and summary["total_multiplications"] == 8640
             and summary["num_cotrees"] == 1),
            f"got {summary}",
        )
        failures += not check(
            "canonical tree is ((((T1)(T2))(T0))(T3))",
            summary["canonical"] == "((((T1)(T2))(T0))(T3))",
            f"got {summary['canonical']}",
        )

        root = body["tree"]
        root_partition = {
            frozenset(_tree_leaves(root["left"])),
            frozenset(_tree_leaves(root["right"])),
        }
        expected = {frozenset({"T0", "T1", "T2"}), frozenset({"T3"})}
        failures += not check(
            "canonical root sides = {T0,T1,T2} | {T3}",
            root.get("type") == "node" and root_partition == expected,
            f"got {[sorted(s) for s in root_partition]}",
        )

        counts = {"mandatory": 0, "optional": 0, "absent": 0}
        mandatory = None
        absent_contradicts_root = False
        for cut in body["cuts"]:
            counts[cut["classification"]] += 1
            partition = {frozenset(cut["left"]), frozenset(cut["right"])}
            if cut["classification"] == "mandatory":
                mandatory = cut
            if cut["classification"] == "absent" and partition == root_partition:
                absent_contradicts_root = True
        failures += not check(
            "masked-peak classes: exactly 1 mandatory / 0 optional / 6 absent",
            counts == {"mandatory": 1, "optional": 0, "absent": 6},
            f"got {counts}",
        )
        mandatory_ok = bool(mandatory) and (
            {frozenset(mandatory["left"]), frozenset(mandatory["right"])}
            == root_partition == expected
        )
        failures += not check(
            "the mandatory cut is {T0,T1,T2} | {T3} and matches the tree root",
            mandatory_ok, f"got {mandatory}",
        )
        evidence_ok = bool(mandatory) and "720" in mandatory.get("evidence", "") \
            and "8640" in mandatory.get("evidence", "")
        failures += not check(
            "mandatory-cut evidence cites peak 720 and total 8640",
            evidence_ok, f"evidence={mandatory.get('evidence') if mandatory else None}",
        )
        failures += not check(
            "no absent-cut evidence negates the canonical tree root",
            not absent_contradicts_root,
        )
        last = body["steps"][-1]
        last_partition = {frozenset(last["left"]), frozenset(last["right"])}
        failures += not check(
            "final contraction step merges {T0,T1,T2} with T3",
            last_partition == expected, f"got {last}",
        )

    # scenario 3: index occurring three times is a located 422
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
