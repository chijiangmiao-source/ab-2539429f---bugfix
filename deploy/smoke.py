#!/usr/bin/env python3
"""Live API smoke checks executed by the one-shot `verify` service.

Three required scenarios are exercised over real HTTP:

1. the four-tensor ring: co-optimal root-cut classification must report
   0 mandatory / 6 optional / 1 absent cuts, with the diagonal cut
   (A,C)|(B,D) absent because forcing it builds a size-16 intermediate;
2. a four-tensor network with a unique mandatory root cut whose optimal
   side subtree has a Pareto-masked peak: must report peak 720, total
   8640, exactly one mandatory cut {T0,T1,T2}|{T3} and six absent cuts,
   and no absent evidence may contradict the response's own canonical
   tree;
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


def masked_peak_network() -> dict:
    return {
        "tensors": [
            {"name": "T0", "indices": [
                {"name": "e0", "dimension": 3}, {"name": "e3", "dimension": 4},
                {"name": "e4", "dimension": 5}]},
            {"name": "T1", "indices": [
                {"name": "e0", "dimension": 3}, {"name": "e1", "dimension": 5},
                {"name": "e5", "dimension": 4}]},
            {"name": "T2", "indices": [
                {"name": "e1", "dimension": 5}, {"name": "e2", "dimension": 3},
                {"name": "e3", "dimension": 4}, {"name": "d6", "dimension": 4}]},
            {"name": "T3", "indices": [
                {"name": "e2", "dimension": 3}, {"name": "e4", "dimension": 5},
                {"name": "e5", "dimension": 4}, {"name": "d7", "dimension": 4},
                {"name": "d8", "dimension": 3}]},
        ]
    }


def _leaf_names(node: object) -> set:
    if not isinstance(node, dict):
        return set()
    if node.get("type") == "leaf":
        return {node["name"]}
    return _leaf_names(node.get("left")) | _leaf_names(node.get("right"))


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

    # scenario 2: unique mandatory root cut with a Pareto-masked side peak
    status, body = _request(base, "POST", "/api/plan", masked_peak_network())
    ok = status == 200
    failures += not check("POST /api/plan masked-peak -> 200", ok, f"got {status} {body}")
    if ok:
        summary = body["summary"]
        failures += not check(
            "masked-peak exact optimum peak=720 total=8640, cotrees=1",
            (summary["peak_memory"] == 720
             and summary["total_multiplications"] == 8640
             and summary["num_cotrees"] == 1),
            f"got {summary}",
        )
        failures += not check(
            "canonical tree is ((((T1)(T2))(T0))(T3))",
            summary["canonical"] == "((((T1)(T2))(T0))(T3))",
            f"got {summary.get('canonical')}",
        )

        tree = body.get("tree", {})
        root_left = _leaf_names(tree.get("left"))
        root_right = _leaf_names(tree.get("right"))
        failures += not check(
            "canonical tree root leaf sets {T0,T1,T2} | {T3}",
            root_left == {"T0", "T1", "T2"} and root_right == {"T3"},
            f"got {root_left} | {root_right}",
        )

        steps = body.get("steps", [])
        final = steps[-1] if steps else {}
        final_split = {tuple(sorted(final.get("left", []))),
                       tuple(sorted(final.get("right", [])))}
        failures += not check(
            "final contraction step merges {T0,T1,T2} with {T3}",
            final_split == {("T0", "T1", "T2"), ("T3",)},
            f"got {final}",
        )

        counts = {"mandatory": 0, "optional": 0, "absent": 0}
        mandatory_cuts, absent_cuts = [], []
        for cut in body["cuts"]:
            counts[cut["classification"]] += 1
            if cut["classification"] == "mandatory":
                mandatory_cuts.append(cut)
            elif cut["classification"] == "absent":
                absent_cuts.append(cut)
        failures += not check(
            "masked-peak classes: 1 mandatory / 0 optional / 6 absent",
            counts == {"mandatory": 1, "optional": 0, "absent": 6},
            f"got {counts}",
        )

        the_cut = mandatory_cuts[0] if mandatory_cuts else None
        cut_ok = bool(the_cut) and (
            {tuple(the_cut["left"]), tuple(the_cut["right"])}
            == {("T0", "T1", "T2"), ("T3",)}
        )
        failures += not check(
            "the unique mandatory cut is {T0,T1,T2}|{T3}", cut_ok,
            f"got {the_cut}",
        )
        evidence_ok = (
            bool(the_cut)
            and "720" in the_cut.get("evidence", "")
            and "8640" in the_cut.get("evidence", "")
        )
        failures += not check(
            "mandatory evidence cites peak 720 and total 8640",
            evidence_ok, f"evidence={the_cut.get('evidence') if the_cut else None}",
        )

        # no absent evidence may contradict the response's own canonical tree
        absent_pairs = [
            (tuple(c["left"]), tuple(c["right"])) for c in absent_cuts
        ]
        contradicts = (
            (("T0", "T1", "T2"), ("T3",)) in absent_pairs
            or (("T3",), ("T0", "T1", "T2")) in absent_pairs
        )
        failures += not check(
            "canonical root cut is never labelled absent",
            not contradicts and all(c.get("evidence") for c in absent_cuts),
            f"absent cuts={absent_pairs}",
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
