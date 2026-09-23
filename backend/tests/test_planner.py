"""Tests for the contraction planner: validation, cost model, canonical
tie-break and root-cut classification (cross-checked by brute force)."""

from __future__ import annotations

import random

import pytest

from app.planner import (
    Planner,
    Tensor,
    ValidationError,
    network_from_payload,
    plan,
    validate_network,
)

from tests.brute import BruteForce


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def ring4(d: int = 2) -> dict:
    """A(ij)-B(jk)-C(kl)-D(li), every index dimension d."""
    return {
        "tensors": [
            {"name": "A", "indices": [{"name": "i", "dimension": d}, {"name": "j", "dimension": d}]},
            {"name": "B", "indices": [{"name": "j", "dimension": d}, {"name": "k", "dimension": d}]},
            {"name": "C", "indices": [{"name": "k", "dimension": d}, {"name": "l", "dimension": d}]},
            {"name": "D", "indices": [{"name": "l", "dimension": d}, {"name": "i", "dimension": d}]},
        ]
    }


def random_network(rng: random.Random, n: int) -> tuple[list[Tensor], dict[str, int]]:
    """Connected random network: spanning tree of binary edges + extra edges
    + dangling indices; 1-6 distinct indices per tensor, each index occurs
    at most twice."""
    dims: dict[str, int] = {}
    idx_of: list[list[str]] = [[] for _ in range(n)]
    used = set()

    def fresh(prefix: str) -> str:
        k = f"{prefix}{len(used)}"
        used.add(k)
        return k

    # path backbone guarantees connectivity while respecting the 6-index cap
    for i in range(1, n):
        k = fresh("e")
        dims[k] = rng.randint(2, 5)
        idx_of[i].append(k)
        idx_of[i - 1].append(k)

    # extra edges
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.25 and len(idx_of[i]) < 6 and len(idx_of[j]) < 6:
                k = fresh("e")
                dims[k] = rng.randint(2, 5)
                idx_of[i].append(k)
                idx_of[j].append(k)

    # dangling indices
    for i in range(n):
        while len(idx_of[i]) < 1 or (len(idx_of[i]) < 6 and rng.random() < 0.4):
            k = fresh("d")
            dims[k] = rng.randint(2, 5)
            idx_of[i].append(k)
            if len(idx_of[i]) >= 6:
                break

    tensors = [Tensor(name=f"T{i}", indices=tuple(idx_of[i])) for i in range(n)]
    return tensors, dims


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_index_occurring_three_times_is_rejected_with_location(self):
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": "i", "dimension": 2}]},
                {"name": "B", "indices": [
                    {"name": "i", "dimension": 2},
                    {"name": "j", "dimension": 2},
                ]},
                {"name": "C", "indices": [
                    {"name": "i", "dimension": 2},
                    {"name": "j", "dimension": 2},
                ]},
            ]
        }
        errors = validate_network(payload)
        codes = [e["code"] for e in errors]
        assert "index_occurrence" in codes
        err = next(e for e in errors if e["code"] == "index_occurrence")
        assert "出现 3 次" in err["message"]
        # points at the first occurrence of the offending index
        assert err["path"][0] == "tensors"
        assert err["path"][-1] == "name"

    def test_index_occurring_four_times_reports_count(self):
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": "i", "dimension": 2}]},
                {"name": "B", "indices": [{"name": "i", "dimension": 2}]},
                {"name": "C", "indices": [{"name": "i", "dimension": 2}]},
                {"name": "D", "indices": [
                    {"name": "i", "dimension": 2},
                    {"name": "x", "dimension": 2},
                ]},
            ]
        }
        errors = validate_network(payload)
        assert any(e["code"] == "index_occurrence" and "4 次" in e["message"] for e in errors)

    def test_tensor_count_bounds(self):
        one = {"tensors": [{"name": "A", "indices": [{"name": "i", "dimension": 2}]}]}
        assert any(e["code"] == "tensor_count" for e in validate_network(one))

        twelve = {
            "tensors": [
                {"name": f"T{i}", "indices": [{"name": f"x{i}", "dimension": 2}]}
                for i in range(12)
            ]
        }
        # 12 lone tensors are also disconnected; at least the count error fires
        assert any(e["code"] == "tensor_count" for e in validate_network(twelve))

    def test_index_count_and_uniqueness(self):
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": "i", "dimension": 2}]},
                {
                    "name": "B",
                    "indices": [
                        {"name": k, "dimension": 2} for k in ["i", "a", "b", "c", "d", "e", "f"]
                    ],
                },
            ]
        }
        errors = validate_network(payload)
        assert any(e["code"] == "too_many_indices" for e in errors)

        payload["tensors"][1]["indices"] = [
            {"name": "i", "dimension": 2},
            {"name": "i", "dimension": 2},
        ]
        errors = validate_network(payload)
        assert any(e["code"] == "duplicate_index_in_tensor" for e in errors)

    def test_dimension_range_and_type(self):
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": "i", "dimension": 1}]},
                {"name": "B", "indices": [{"name": "i", "dimension": 1001}]},
            ]
        }
        errors = validate_network(payload)
        assert sum(1 for e in errors if e["code"] == "dimension_range") == 2

        payload["tensors"][1]["indices"][0]["dimension"] = "8"
        errors = validate_network(payload)
        assert any(e["code"] == "invalid_dimension" for e in errors)

    def test_inconsistent_dimension(self):
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": "i", "dimension": 2}]},
                {"name": "B", "indices": [{"name": "i", "dimension": 3}]},
            ]
        }
        errors = validate_network(payload)
        assert any(e["code"] == "inconsistent_dimension" for e in errors)

    def test_duplicate_tensor_name(self):
        payload = {
            "tensors": [
                {"name": "X", "indices": [{"name": "i", "dimension": 2}]},
                {"name": "X", "indices": [{"name": "i", "dimension": 2}]},
            ]
        }
        assert any(e["code"] == "duplicate_name" for e in validate_network(payload))

    def test_disconnected(self):
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": "i", "dimension": 2}]},
                {"name": "B", "indices": [{"name": "i", "dimension": 2}]},
                {"name": "C", "indices": [{"name": "z", "dimension": 2}]},
            ]
        }
        errors = validate_network(payload)
        assert any(e["code"] == "disconnected" for e in errors)
        assert "C" in next(e for e in errors if e["code"] == "disconnected")["message"]

    def test_non_ascii_index(self):
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": "甲", "dimension": 2}]},
                {"name": "B", "indices": [{"name": "甲", "dimension": 2}]},
            ]
        }
        assert any(e["code"] == "non_ascii_index" for e in validate_network(payload))

    def test_boundary_values_accepted(self):
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": "i", "dimension": 1000}]},
                {"name": "B", "indices": [{"name": "i", "dimension": 1000}]},
            ]
        }
        assert validate_network(payload) == []

    def test_invalid_payload_raises_with_error_list(self):
        with pytest.raises(ValidationError) as exc:
            network_from_payload({"tensors": "nope"})
        assert exc.value.errors


# ---------------------------------------------------------------------------
# Cost model on hand-computed examples
# ---------------------------------------------------------------------------


class TestCostModel:
    def test_two_tensors(self):
        # A(ij) dims i=2,j=3 ; B(ik) dims i=2,k=4 -> contract i
        result = plan(
            {
                "tensors": [
                    {"name": "A", "indices": [
                        {"name": "i", "dimension": 2}, {"name": "j", "dimension": 3}]},
                    {"name": "B", "indices": [
                        {"name": "i", "dimension": 2}, {"name": "k", "dimension": 4}]},
                ]
            }
        )
        # mult = |i j k| = 2*3*4 = 24 ; result jk size 12 ; inputs 6, 8
        assert result["summary"]["total_multiplications"] == 24
        assert result["summary"]["peak_memory"] == 12
        assert result["steps"][0]["contracted_indices"] == ["i"]
        cut = result["cuts"][0]
        assert cut["classification"] == "mandatory"

    def test_ring4_costs_and_classes(self):
        result = plan(ring4(2))
        assert result["summary"]["peak_memory"] == 4
        assert result["summary"]["total_multiplications"] == 20

        classes = {c["classification"] for c in result["cuts"]}
        counts = {k: sum(1 for c in result["cuts"] if c["classification"] == k)
                  for k in ("mandatory", "optional", "absent")}
        # 7 unordered bipartitions of 4 leaves:
        # 4 singleton cuts + AB|CD + AD|BC are co-optimal (optional);
        # AC|BD forces the giant 2^4 intermediate (absent)
        assert counts == {"mandatory": 0, "optional": 6, "absent": 1}
        assert classes <= {"optional", "absent"}

        absent = next(c for c in result["cuts"] if c["classification"] == "absent")
        assert (absent["left"], absent["right"]) in (
            (["A", "C"], ["B", "D"]),
            (["B", "D"], ["A", "C"]),
        )
        # forced cut creates the 16-element intermediate (i,j,k,l all live)
        assert "16" in absent["evidence"]

        for cut in result["cuts"]:
            assert cut["evidence"]

    def test_peak_includes_inputs(self):
        # A with a huge dangling dimension: input size itself dominates peak
        result = plan(
            {
                "tensors": [
                    {"name": "A", "indices": [
                        {"name": "i", "dimension": 2}, {"name": "h", "dimension": 1000}]},
                    {"name": "B", "indices": [{"name": "i", "dimension": 2}]},
                ]
            }
        )
        # A is 2000; result h is 1000; peak must be 2000
        assert result["summary"]["peak_memory"] == 2000

    def test_large_dimensions_use_exact_integers(self):
        # dims up to 1000 with 6-way contractions exceed 64-bit range
        payload = {
            "tensors": [
                {"name": "A", "indices": [{"name": k, "dimension": 1000} for k in ("a", "b", "i")]},
                {"name": "B", "indices": [{"name": k, "dimension": 1000} for k in ("i", "c", "d")]},
            ]
        }
        result = plan(payload)
        assert result["summary"]["total_multiplications"] == 1000 ** 5
        assert isinstance(result["summary"]["total_multiplications"], int)


# ---------------------------------------------------------------------------
# Brute-force cross-checks (random networks)
# ---------------------------------------------------------------------------


def _network(tensors, dims):
    from app.planner import Network

    return Network(tensors=tuple(tensors), dims=dims)


@pytest.mark.parametrize("seed", range(40))
def test_dp_matches_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 8)
    tensors, dims = random_network(rng, n)

    planner = Planner(_network(tensors, dims))
    root = planner.solve()
    brute = BruteForce(tensors, dims).solve()

    bp, bt, bc = brute.optimum()
    assert (root.peak, root.total, root.canon) == (bp, bt, bc)

    # exact number of co-optimal trees (bounded counting DP)
    brute_total_counts = brute.cotree_counts_all((1 << n) - 1)
    assert planner.count_optimal_trees(bp, bt) == brute_total_counts[(bp, bt)]

    # the Pareto table is only allowed to keep non-dominated pairs
    for mask in range(1, 1 << n):
        assert set(planner.table[mask]) == set(brute.pareto_counts(mask))

    # cap-propagation counting DP: every retained entry must be a real
    # brute-force pair within the global bounds ...
    bounded = planner.bounded_count_table(bp, bt)
    for mask in range(1, 1 << n):
        brute_pairs = brute.cotree_counts_all(mask)
        for pair, cnt in bounded[mask].items():
            assert pair in brute_pairs
            assert pair[0] <= bp and pair[1] <= bt
            assert cnt == brute_pairs[pair]

    # ... and, critically, the exact optimal root count must be reproduced
    # (an over-aggressive cap would make this too small)
    assert bounded[(1 << n) - 1][(bp, bt)] == brute_total_counts[(bp, bt)]

    # root-cut classification matches brute realizability
    brute_cuts = set(brute.root_cut_counts())
    result_cuts = planner.classify(root)
    got_optional_or_mandatory = {
        _mask_of(c["left"], planner.names) for c in result_cuts
        if c["classification"] in ("mandatory", "optional")
    }
    assert got_optional_or_mandatory == brute_cuts

    mandatory = [c for c in result_cuts if c["classification"] == "mandatory"]
    if len(brute_cuts) == 1:
        assert len(mandatory) == 1
    else:
        assert mandatory == []

    # every absent cut really is suboptimal
    for c in result_cuts:
        if c["classification"] == "absent":
            assert _mask_of(c["left"], planner.names) not in brute_cuts


def _mask_of(names, all_names):
    m = 0
    for i, nm in enumerate(all_names):
        if nm in names:
            m |= 1 << i
    return m


def test_canonical_serialization_format():
    result = plan(ring4(2))
    canon = result["summary"]["canonical"]
    assert canon.startswith("(") and canon.endswith(")")
    for name in "ABCD":
        assert f"({name})" in canon


def test_plan_response_shape():
    result = plan(ring4(3))
    assert set(result) == {"summary", "tree", "steps", "cuts", "indices", "inputs"}
    assert result["summary"]["num_tensors"] == 4
    assert len(result["inputs"]) == 4
    assert len(result["indices"]) == 4
    for idx in result["indices"]:
        assert idx["occurrences"] == 2
        assert len(idx["on_tensors"]) == 2
    # 3 internal contraction steps for 4 leaves
    assert len(result["steps"]) == 3
    peak_steps = max(s["result_size"] for s in result["steps"])
    assert result["summary"]["peak_memory"] == max(
        peak_steps, *(i["size"] for i in result["inputs"])
    )
