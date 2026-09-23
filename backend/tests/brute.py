"""Brute-force reference solver used to cross-check the Pareto DP."""

from __future__ import annotations

from itertools import combinations

from app.planner import Tensor


def _bipartitions(mask: int):
    low = mask & -mask
    rest = mask ^ low
    sub = (rest - 1) & rest
    while True:
        yield low | sub, mask ^ (low | sub)
        if sub == 0:
            break
        sub = (sub - 1) & rest


class BruteForce:
    """Enumerates *every* contraction tree (no pruning) for every subset."""

    def __init__(self, tensors: list[Tensor], dims: dict[str, int]):
        self.names = [t.name for t in tensors]
        self.tidx = [frozenset(t.indices) for t in tensors]
        self.dims = dims
        self.n = len(tensors)
        self.unique: dict[int, frozenset[str]] = {}
        for mask in range(1, 1 << self.n):
            low = mask & -mask
            i = low.bit_length() - 1
            rest = mask ^ low
            self.unique[mask] = (
                self.tidx[i] if rest == 0 else self.unique[rest] ^ self.tidx[i]
            )
        # tables[mask] : list of dicts {peak,total,canon,cut}
        self.tables: dict[int, list[dict]] = {}

    def size(self, mask: int) -> int:
        p = 1
        for k in self.unique[mask]:
            p *= self.dims[k]
        return p

    def solve(self):
        for i in range(self.n):
            bit = 1 << i
            s = self.size(bit)
            self.tables[bit] = [
                {"peak": s, "total": 0, "canon": f"({self.names[i]})", "cut": None}
            ]
        for r in range(2, self.n + 1):
            for combo in combinations(range(self.n), r):
                mask = 0
                for i in combo:
                    mask |= 1 << i
                out: list[dict] = []
                for a, b in _bipartitions(mask):
                    mult = 1
                    for k in self.unique[a] | self.unique[b]:
                        mult *= self.dims[k]
                    res = self.size(mask)
                    for x in self.tables[a]:
                        for y in self.tables[b]:
                            c1, c2 = sorted((x["canon"], y["canon"]))
                            out.append(
                                {
                                    "peak": max(x["peak"], y["peak"], res),
                                    "total": x["total"] + y["total"] + mult,
                                    "canon": f"({c1}{c2})",
                                    "cut": (a, b),
                                }
                            )
                self.tables[mask] = out
        return self

    def optimum(self, mask: int | None = None) -> tuple[int, int, str]:
        mask = mask if mask is not None else (1 << self.n) - 1
        best = min(
            ((t["peak"], t["total"], t["canon"]) for t in self.tables[mask]),
            key=lambda x: (x[0], x[1], x[2]),
        )
        return best

    def cotree_counts(self, mask: int) -> dict[tuple[int, int], int]:
        p, t, _ = self.optimum(mask)
        counts: dict[tuple[int, int], int] = {}
        for row in self.tables[mask]:
            key = (row["peak"], row["total"])
            counts[key] = counts.get(key, 0) + 1
        return counts

    def pareto_counts(self, mask: int) -> dict[tuple[int, int], int]:
        """Count of trees for each pair on the brute Pareto frontier."""
        counts = self.cotree_counts_all(mask)
        items = sorted(counts)
        kept = {}
        best_total = None
        for pair in items:
            if best_total is None or pair[1] < best_total:
                kept[pair] = counts[pair]
                best_total = pair[1]
        return kept

    def cotree_counts_all(self, mask: int) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for row in self.tables[mask]:
            key = (row["peak"], row["total"])
            counts[key] = counts.get(key, 0) + 1
        return counts

    def root_cut_counts(self) -> dict[int, int]:
        """For the full network, count co-optimal (peak,total) trees per
        unordered root cut, keyed by the side containing leaf 0."""
        full = (1 << self.n) - 1
        p, t, _ = self.optimum(full)
        cut_counts: dict[int, int] = {}
        for row in self.tables[full]:
            if row["peak"] == p and row["total"] == t:
                a, b = row["cut"]
                low_side = a if a & 1 else b
                cut_counts[low_side] = cut_counts.get(low_side, 0) + 1
        return cut_counts
