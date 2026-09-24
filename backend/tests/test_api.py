"""End-to-end API smoke tests using FastAPI's in-process ASGI transport.

These exercise the real HTTP stack (routing, status codes, JSON shape):

* a four-tensor ring: the co-optimal root-cut classification;
* the error boundary for an index occurring three times: it must be
  rejected with HTTP 422 and a locatable error path.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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
    # unique optimal root cut {T0,T1,T2}|{T3} needs a side subtree whose
    # own peak is Pareto-masked by the root join (see test_planner.py)
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


def _leaf_names(node) -> set[str]:
    if node["type"] == "leaf":
        return {node["name"]}
    return _leaf_names(node["left"]) | _leaf_names(node["right"])


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_proxied_healthz():
    # same probe exposed under /api so the web reverse proxy can reach it
    r = client.get("/api/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_smoke_four_tensor_cotree_classification():
    r = client.post("/api/plan", json=ring4())
    assert r.status_code == 200, r.text
    body = r.json()

    # optimal objective for the d=2 ring
    assert body["summary"]["peak_memory"] == 4
    assert body["summary"]["total_multiplications"] == 20
    assert body["summary"]["canonical"]

    by_class = {"mandatory": [], "optional": [], "absent": []}
    for cut in body["cuts"]:
        by_class[cut["classification"]].append(cut)

    # ring symmetry: no mandatory root cut; exactly one absent cut (A,C)|(B,D)
    assert len(by_class["mandatory"]) == 0
    assert len(by_class["optional"]) == 6
    assert len(by_class["absent"]) == 1

    absent = by_class["absent"][0]
    assert {tuple(absent["left"]), tuple(absent["right"])} == {
        ("A", "C"),
        ("B", "D"),
    }
    assert absent["evidence"]
    # forcing the diagonal cut creates a 16-element intermediate vs peak 4
    assert "16" in absent["evidence"]

    # every cut and step carries evidence / metadata for the UI
    for cut in body["cuts"]:
        assert cut["label"] in ("必现", "可选", "不出现")
        assert cut["evidence"]
    assert len(body["steps"]) == 3
    assert body["tree"]["type"] == "node"


def test_smoke_unique_mandatory_root_cut_with_masked_side_peak():
    r = client.post("/api/plan", json=masked_peak_network())
    assert r.status_code == 200, r.text
    body = r.json()

    # exact summary metrics and canonical tree
    summary = body["summary"]
    assert summary["peak_memory"] == 720
    assert summary["total_multiplications"] == 8640
    assert summary["num_cotrees"] == 1
    assert summary["canonical"] == "((((T1)(T2))(T0))(T3))"

    # canonical tree root splits the leaves into exactly {T0,T1,T2} | {T3}
    assert body["tree"]["type"] == "node"
    assert _leaf_names(body["tree"]["left"]) == {"T0", "T1", "T2"}
    assert _leaf_names(body["tree"]["right"]) == {"T3"}

    # last contraction step merges that subtree with T3
    assert len(body["steps"]) == 3
    final = body["steps"][-1]
    assert {tuple(sorted(final["left"])), tuple(sorted(final["right"]))} == {
        ("T0", "T1", "T2"),
        ("T3",),
    }

    by_class = {"mandatory": [], "optional": [], "absent": []}
    for cut in body["cuts"]:
        by_class[cut["classification"]].append(cut)

    # exactly one mandatory cut — the canonical root cut — and six absent
    assert len(by_class["mandatory"]) == 1
    assert len(by_class["optional"]) == 0
    assert len(by_class["absent"]) == 6

    mandatory = by_class["mandatory"][0]
    assert {tuple(mandatory["left"]), tuple(mandatory["right"])} == {
        ("T0", "T1", "T2"),
        ("T3",),
    }
    # evidence states it reaches peak 720 and total 8640, and is unique
    assert "720" in mandatory["evidence"]
    assert "8640" in mandatory["evidence"]
    assert mandatory["optimal_root_cuts"] == 1

    # no absent evidence may contradict the response's own canonical tree:
    # its root cut is not among the absent cuts
    absent_pairs = [
        (tuple(c["left"]), tuple(c["right"])) for c in by_class["absent"]
    ]
    assert (("T0", "T1", "T2"), ("T3",)) not in absent_pairs
    assert (("T3",), ("T0", "T1", "T2")) not in absent_pairs
    for cut in by_class["absent"]:
        assert cut["evidence"]
        assert cut["optimal_root_cuts"] == 1


def test_smoke_index_occurring_three_times_is_422_and_located():
    payload = {
        "tensors": [
            {"name": "A", "indices": [{"name": "i", "dimension": 2}]},
            {"name": "B", "indices": [
                {"name": "i", "dimension": 2}, {"name": "j", "dimension": 2}]},
            {"name": "C", "indices": [
                {"name": "i", "dimension": 2}, {"name": "j", "dimension": 2}]},
        ]
    }
    r = client.post("/api/plan", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert "errors" in body and len(body["errors"]) >= 1
    err = next(e for e in body["errors"] if e["code"] == "index_occurrence")
    assert "3 次" in err["message"]
    # path locates the offending input: tensors[<n>].indices[<m>].name
    assert err["path"][0] == "tensors"
    assert err["path"][2] == "indices"
    assert err["path"][4] == "name"
    # the first occurrence of i is at tensors[0].indices[0]
    assert err["path"][1] == 0 and err["path"][3] == 0


def test_malformed_json_is_400():
    r = client.post("/api/plan", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["errors"][0]["code"] == "invalid_json"
