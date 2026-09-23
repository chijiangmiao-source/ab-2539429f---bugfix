"""Tensor-network contraction planning.

Given a small connected tensor network (2-11 tensors) this module finds
contraction trees that

1. minimize the *peak memory* (the largest tensor that must ever exist,
   including inputs and every intermediate result), and then, as a
   secondary objective,
2. minimize the *total number of scalar multiplications*.

All arithmetic uses Python's arbitrary precision integers, so dimensions
up to 1000 never overflow.

Cost model (per the project specification)
------------------------------------------
For contracting groups ``A`` and ``B``:

* ``boundary(A)`` = indices occurring exactly once inside A (dangling or
  connecting to a tensor outside A);
* multiplication count = product of dimensions of ``boundary(A) |
  boundary(B)``;
* result size = product of dimensions of the new boundary
  ``boundary(A) ^ boundary(B)`` (the symmetric difference).

Tie-breaking: leaves carry unique tensor names. A tree is serialized as a
parenthesized string with the two child serializations recursively
sorted; the lexicographically smallest string among the trees tied on
(peak, total) is the *canonical solution*.

Root-cut classification
-----------------------
Every non-trivial unordered bipartition of the leaves is a candidate root
cut. Considering all contraction trees achieving the optimal (peak,
total):

* mandatory (必现): every co-optimal tree has this cut at its root —
  possible only when it is the *unique* realizable optimal root cut;
* optional (可选): some co-optimal tree uses it, but another co-optimal
  tree uses a different cut;
* absent (不出现): no co-optimal tree uses it.

A second DP keeps the full Pareto frontier of achievable (peak, total)
pairs for every leaf subset (a dominated pair can never participate in a
global optimum — replacing it by the dominating record cannot increase
either outer peak or outer total). Each frontier entry also carries the
exact number of distinct canonical tree shapes and the lexicographically
smallest serialization, which makes the classification and the
canonical-tree selection exact even when a large outer result masks a
locally suboptimal subtree peak.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants & validation
# ---------------------------------------------------------------------------

MAX_TENSORS = 11
MIN_TENSORS = 2
MAX_INDICES_PER_TENSOR = 6
MIN_DIM = 2
MAX_DIM = 1000


class ValidationError(Exception):
    """Raised for an invalid network.

    ``errors`` is a list of ``{"code", "message", "path"}`` mappings; path
    is the JSON-pointer-like location of the problem (empty for global).
    """

    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


def _err(code: str, message: str, path: list[str | int]) -> dict:
    return {"code": code, "message": message, "path": list(path)}


def validate_network(payload: object) -> list[dict]:
    """Validate the raw request payload; return error dicts (empty = ok)."""
    errors: list[dict] = []

    if not isinstance(payload, dict):
        return [_err("invalid_request", "请求体必须是 JSON 对象", [])]

    tensors = payload.get("tensors")
    if not isinstance(tensors, list):
        errors.append(_err("missing_tensors", "字段 tensors 必须是数组", []))
        return errors

    if not (MIN_TENSORS <= len(tensors) <= MAX_TENSORS):
        errors.append(
            _err(
                "tensor_count",
                f"张量数量必须在 {MIN_TENSORS} 至 {MAX_TENSORS} 之间，当前为 {len(tensors)}",
                ["tensors"],
            )
        )

    seen_names: set[str] = set()
    names: list[str] = []
    # index name -> list of (tensor position, index position)
    index_locations: dict[str, list[tuple[int, int]]] = {}
    index_dims: dict[str, int] = {}

    for i, t in enumerate(tensors):
        tpath = ["tensors", i]
        if not isinstance(t, dict):
            errors.append(_err("invalid_tensor", "每个张量必须是对象", tpath))
            continue

        name = t.get("name")
        if not isinstance(name, str) or not name:
            errors.append(
                _err("invalid_name", "张量 name 必须是非空字符串", tpath + ["name"])
            )
        else:
            if name in seen_names:
                errors.append(
                    _err(
                        "duplicate_name",
                        f"张量名称 {name!r} 重复，名称必须唯一",
                        tpath + ["name"],
                    )
                )
            seen_names.add(name)
            names.append(name)

        indices = t.get("indices")
        if not isinstance(indices, list) or not indices:
            errors.append(
                _err(
                    "invalid_indices",
                    "indices 必须是含 1 至 6 个互异索引的非空数组",
                    tpath + ["indices"],
                )
            )
            continue

        if len(indices) > MAX_INDICES_PER_TENSOR:
            errors.append(
                _err(
                    "too_many_indices",
                    f"张量 {name if isinstance(name, str) else i} 含 {len(indices)} 个索引，"
                    f"最多 {MAX_INDICES_PER_TENSOR} 个",
                    tpath + ["indices"],
                )
            )

        local_seen: set[str] = set()
        for j, idx in enumerate(indices):
            ipath = tpath + ["indices", j]
            if not isinstance(idx, dict):
                errors.append(_err("invalid_index", "每个索引必须是对象", ipath))
                continue
            kname = idx.get("name")
            dim = idx.get("dimension")
            if not isinstance(kname, str) or not kname:
                errors.append(
                    _err("invalid_index_name", "索引 name 必须是非空字符串", ipath + ["name"])
                )
                continue
            if not kname.isascii():
                errors.append(
                    _err(
                        "non_ascii_index",
                        f"索引名 {kname!r} 必须全部为 ASCII 字符",
                        ipath + ["name"],
                    )
                )
            if kname in local_seen:
                errors.append(
                    _err(
                        "duplicate_index_in_tensor",
                        f"张量 {name if isinstance(name, str) else i} 中索引 {kname!r} 重复，"
                        "同一张量内索引必须互异",
                        ipath + ["name"],
                    )
                )
            local_seen.add(kname)
            index_locations.setdefault(kname, []).append((i, j))

            if isinstance(dim, bool) or not isinstance(dim, int):
                errors.append(
                    _err(
                        "invalid_dimension",
                        f"索引 {kname!r} 的 dimension 必须是整数",
                        ipath + ["dimension"],
                    )
                )
            elif not (MIN_DIM <= dim <= MAX_DIM):
                errors.append(
                    _err(
                        "dimension_range",
                        f"索引 {kname!r} 的维数 {dim} 必须在 {MIN_DIM} 至 {MAX_DIM} 之间",
                        ipath + ["dimension"],
                    )
                )
            else:
                prev = index_dims.get(kname)
                if prev is not None and prev != dim:
                    errors.append(
                        _err(
                            "inconsistent_dimension",
                            f"索引 {kname!r} 在不同出现位置维数不一致：{prev} 与 {dim}",
                            ipath + ["dimension"],
                        )
                    )
                else:
                    index_dims[kname] = dim

    if errors:
        return errors

    for kname, locs in index_locations.items():
        if len(locs) not in (1, 2):
            first = locs[0]
            errors.append(
                _err(
                    "index_occurrence",
                    f"索引 {kname!r} 全网出现 {len(locs)} 次，必须恰好出现 1 或 2 次",
                    ["tensors", first[0], "indices", first[1], "name"],
                )
            )

    # connectivity: tensors are vertices, indices occurring twice are edges
    n = len(tensors)
    adj: list[set[int]] = [set() for _ in range(n)]
    for locs in index_locations.values():
        if len(locs) == 2:
            a, b = locs[0][0], locs[1][0]
            if a != b:  # repeated-within-tensor already rejected above
                adj[a].add(b)
                adj[b].add(a)
    reached = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in reached:
                reached.add(w)
                stack.append(w)
    if len(reached) != n:
        missing = [names[i] for i in range(n) if i not in reached]
        errors.append(
            _err(
                "disconnected",
                f"网络不连通：张量 {', '.join(missing)} 与其余张量之间没有共享索引",
                ["tensors"],
            )
        )

    return errors


# ---------------------------------------------------------------------------
# Problem model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tensor:
    name: str
    indices: tuple[str, ...]


@dataclass(frozen=True)
class Network:
    tensors: tuple[Tensor, ...]
    dims: dict[str, int]

    @property
    def n(self) -> int:
        return len(self.tensors)


def network_from_payload(payload: object) -> Network:
    errors = validate_network(payload)
    if errors:
        raise ValidationError(errors)
    assert isinstance(payload, dict)
    tensors = tuple(
        Tensor(name=t["name"], indices=tuple(i["name"] for i in t["indices"]))
        for t in payload["tensors"]
    )
    dims = {
        i["name"]: int(i["dimension"])
        for t in payload["tensors"]
        for i in t["indices"]
    }
    return Network(tensors=tensors, dims=dims)


# ---------------------------------------------------------------------------
# Bit-mask helpers
# ---------------------------------------------------------------------------


def _bipartitions(mask: int) -> Iterable[tuple[int, int]]:
    """All unordered bipartitions of ``mask`` into two non-empty sets.

    The side containing the lowest set bit is yielded first, so each
    unordered split appears exactly once. ``mask`` must contain at least
    two bits."""
    low = mask & -mask
    rest = mask ^ low
    # enumerate submasks of `rest` except `rest` itself (which would make
    # the complementary side empty), starting from the largest proper one
    sub = (rest - 1) & rest
    while True:
        a = low | sub
        yield a, mask ^ a
        if sub == 0:
            break
        sub = (sub - 1) & rest


@dataclass(frozen=True)
class Record:
    """One non-dominated achievable (peak, total) result for a leaf subset.

    ``canon`` is the lexicographically smallest tree serialization among
    trees achieving this pair and ``tree`` its nested structure. Tree
    *counts* are intentionally not stored here: Pareto-pruned child tables
    can undercount when a large parent result masks a child peak, so exact
    counting is done separately by ``count_optimal_trees`` with top-down
    allowance propagation.
    """

    peak: int
    total: int
    canon: str
    tree: tuple


@dataclass(frozen=True)
class CutOutcome:
    peak: int
    total: int


@dataclass
class Planner:
    network: Network

    def __post_init__(self) -> None:
        t = self.network
        self.names: tuple[str, ...] = tuple(x.name for x in t.tensors)
        self.tidx: tuple[frozenset[str], ...] = tuple(
            frozenset(x.indices) for x in t.tensors
        )
        self.dims = t.dims
        # indices occurring exactly once inside each subset
        self._unique: dict[int, frozenset[str]] = {}
        for mask in range(1, 1 << t.n):
            low = mask & -mask
            i = low.bit_length() - 1
            rest = mask ^ low
            self._unique[mask] = (
                self.tidx[i] if rest == 0 else self._unique[rest] ^ self.tidx[i]
            )
        # table[mask] : dict[(peak, total)] -> Record
        self.table: dict[int, dict[tuple[int, int], Record]] = {}
        # cached step metadata per (mask, a, b)
        self._steps: dict[tuple[int, int, int], dict] = {}

    # -- costs --------------------------------------------------------------

    def size(self, mask: int) -> int:
        """Size of the tensor produced by contracting ``mask`` (1 for a
        scalar): product of dimensions of indices occurring once inside."""
        prod = 1
        for k in self._unique[mask]:
            prod *= self.dims[k]
        return prod

    def split_cost(self, a: int, b: int) -> tuple[int, int]:
        """(multiplication count, result size) for joining groups a, b."""
        mult = 1
        for k in self._unique[a] | self._unique[b]:
            mult *= self.dims[k]
        return mult, self.size(a | b)

    def names_of(self, mask: int) -> list[str]:
        return [self.names[i] for i in range(self.network.n) if mask & (1 << i)]

    def step_info(self, mask: int, a: int, b: int) -> dict:
        key = (mask, a, b)
        step = self._steps.get(key)
        if step is None:
            mult, result_size = self.split_cost(a, b)
            step = {
                "left": sorted(self.names_of(a)),
                "right": sorted(self.names_of(b)),
                "multiplications": mult,
                "result_size": result_size,
                "result_indices": sorted(self._unique[mask]),
                "contracted_indices": sorted(
                    (self._unique[a] | self._unique[b]) - self._unique[mask]
                ),
                "union_boundary": sorted(self._unique[a] | self._unique[b]),
            }
            self._steps[key] = step
        return step

    # -- Pareto DP ----------------------------------------------------------

    def solve(self) -> Record:
        n = self.network.n
        for i in range(n):
            bit = 1 << i
            name = self.names[i]
            size = self.size(bit)
            self.table[bit] = {
                (size, 0): Record(
                    peak=size, total=0, canon=f"({name})", tree=("leaf", name)
                )
            }

        for r in range(2, n + 1):
            for combo in combinations(range(n), r):
                mask = 0
                for i in combo:
                    mask |= 1 << i
                # gather the best canonical tree for every achievable pair
                agg: dict[tuple[int, int], dict] = {}
                for a, b in _bipartitions(mask):
                    ta, tb = self.table[a], self.table[b]
                    mult, result_size = self.split_cost(a, b)
                    for (pa, pta), ra in ta.items():
                        for (pb, ptb), rb in tb.items():
                            peak = max(pa, pb, result_size)
                            total = pta + ptb + mult
                            key = (peak, total)
                            # orient children to match canonical ordering so
                            # that the recorded step aligns with the tree
                            if ra.canon <= rb.canon:
                                lm, rm, rl, rr = a, b, ra, rb
                            else:
                                lm, rm, rl, rr = b, a, rb, ra
                            step = self.step_info(mask, lm, rm)
                            canon = f"({rl.canon}{rr.canon})"
                            tree = ("node", rl.tree, rr.tree, step)
                            cur = agg.get(key)
                            if cur is None or canon < cur["canon"]:
                                agg[key] = {"canon": canon, "tree": tree}
                pruned = self._prune(agg)
                self.table[mask] = {
                    (peak, total): Record(
                        peak=peak,
                        total=total,
                        canon=info["canon"],
                        tree=info["tree"],
                    )
                    for (peak, total), info in pruned.items()
                }

        full = (1 << n) - 1
        best_pair = min(self.table[full])
        return self.table[full][best_pair]

    # -- exact tree counting (bounded) --------------------------------------

    def bounded_count_table(
        self, peak_cap: int, total_cap: int
    ) -> dict[int, dict[tuple[int, int], int]]:
        """Per-subset pair counts that can participate in a global tree with
        peak <= peak_cap and total <= total_cap.

        A plain rectangular bound (peak <= p*, total <= t*) is weak when the
        optimal peak is large, so total allowances are propagated top-down.
        For a split a | b with join cost ``mult`` a record of a with total
        ``t_a`` can only be part of a feasible tree if
        ``t_a <= T - mult - minTotal(b)``, where minTotal(b) is the cheapest
        feasible record of b. Propagating the *maximum* such allowance over
        every parent split yields, for each subset, the loosest total budget
        it may spend on any feasible completion — allowances from different
        parent splits never dominate one another (they assume different
        sibling subtrees), hence the max rather than the min."""
        n = self.network.n
        full = (1 << n) - 1

        def min_total(mask: int, cap_p: int) -> int | None:
            """Cheapest total for mask among Pareto records with peak <= P
            (the optimum under a peak cap always lies on the frontier)."""
            best = None
            for p, t in self.table[mask]:
                if p <= cap_p and (best is None or t < best):
                    best = t
            return best

        # ---- top-down allowance propagation -------------------------------
        allowance: dict[int, int] = {full: total_cap}
        for r in range(n, 1, -1):
            for combo in combinations(range(n), r):
                mask = 0
                for i in combo:
                    mask |= 1 << i
                budget = allowance.get(mask)
                if budget is None:
                    continue
                for a, b in _bipartitions(mask):
                    mult, result_size = self.split_cost(a, b)
                    if result_size > peak_cap:
                        continue
                    mtb = min_total(b, peak_cap)
                    mta = min_total(a, peak_cap)
                    if mtb is not None:
                        cand = budget - mult - mtb
                        if cand >= 0 and cand > allowance.get(a, -1):
                            allowance[a] = cand
                    if mta is not None:
                        cand = budget - mult - mta
                        if cand >= 0 and cand > allowance.get(b, -1):
                            allowance[b] = cand

        # ---- bottom-up counting -------------------------------------------
        counts: dict[int, dict[tuple[int, int], int]] = {}
        for i in range(n):
            bit = 1 << i
            size = self.size(bit)
            counts[bit] = (
                {(size, 0): 1}
                if size <= peak_cap and allowance.get(bit, -1) >= 0
                else {}
            )
        for r in range(2, n + 1):
            for combo in combinations(range(n), r):
                mask = 0
                for i in combo:
                    mask |= 1 << i
                budget = allowance.get(mask)
                if budget is None:
                    counts[mask] = {}
                    continue
                agg: dict[tuple[int, int], int] = {}
                for a, b in _bipartitions(mask):
                    ca, cb = counts[a], counts[b]
                    if not ca or not cb:
                        continue
                    mult, result_size = self.split_cost(a, b)
                    if result_size > peak_cap:
                        continue
                    for (pa, pta), na in ca.items():
                        for (pb, ptb), nb in cb.items():
                            peak = max(pa, pb, result_size)
                            total = pta + ptb + mult
                            if peak <= peak_cap and total <= budget:
                                key = (peak, total)
                                agg[key] = agg.get(key, 0) + na * nb
                counts[mask] = agg
        return counts

    def count_optimal_trees(self, peak_cap: int, total_cap: int) -> int:
        """Number of distinct leaf-labelled contraction trees for the full
        network achieving exactly (peak_cap, total_cap).

        A subtree inside a globally optimal tree must itself have
        ``peak <= peak_cap`` and ``total <= total_cap`` (otherwise the
        containing tree would exceed the global optimum), so every record
        outside those bounds can be discarded. The relevant tree count is
        the double factorial (2n-3)!! — *not* the Catalan number, which
        only counts bracketings for a fixed leaf order; retaining all cost
        pairs without the bounds would therefore explode at n=11."""
        counts = self.bounded_count_table(peak_cap, total_cap)
        return counts[(1 << self.network.n) - 1].get((peak_cap, total_cap), 0)

    @staticmethod
    def _prune(agg: dict[tuple[int, int], dict]) -> dict[tuple[int, int], dict]:
        """Keep the Pareto frontier: as peak grows, total must strictly
        improve (decrease) to survive."""
        items = sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        kept: dict[tuple[int, int], dict] = {}
        best_total = None
        for pair, info in items:
            peak, total = pair
            if best_total is None or total < best_total:
                kept[pair] = info
                best_total = total
        return kept

    # -- classification -----------------------------------------------------

    def _cut_outcome(self, a: int, b: int) -> CutOutcome:
        left_peak, left_total = min(self.table[a])
        right_peak, right_total = min(self.table[b])
        multiplications, result_size = self.split_cost(a, b)
        return CutOutcome(
            peak=max(left_peak, right_peak, result_size),
            total=left_total + right_total + multiplications,
        )

    def classify(self, root: Record) -> list[dict]:
        n = self.network.n
        full = (1 << n) - 1
        p_star, t_star = root.peak, root.total

        realized: list[int] = []  # root cuts (side containing leaf 0)
        # that can participate in a globally (p_star, t_star)-optimal tree
        for a, b in _bipartitions(full):
            outcome = self._cut_outcome(a, b)
            if outcome.peak == p_star and outcome.total == t_star:
                realized.append(a)

        unique_cut = len(realized) == 1
        num_optimal_cuts = len(realized)
        realized_set = set(realized)
        result: list[dict] = []

        def other_cuts_text(skip_a: int) -> str:
            others = [
                f"{sorted(self.names_of(x))} | {sorted(self.names_of(full ^ x))}"
                for x in realized
                if x != skip_a
            ]
            return "；".join(others)

        for a, b in _bipartitions(full):
            names_a = sorted(self.names_of(a))
            names_b = sorted(self.names_of(b))
            if a in realized_set:
                if unique_cut:
                    cls, zh = "mandatory", "必现"
                    evidence = (
                        f"在峰值 {p_star}、总乘法 {t_star} 的最优目标下，可达最优的根切分"
                        f"仅此一个，全部同优树的根节点都必现切分 {names_a} | {names_b}"
                    )
                else:
                    cls, zh = "optional", "可选"
                    evidence = (
                        f"此切分可达最优 (峰值 {p_star}, 总乘法 {t_star})；但共有 "
                        f"{num_optimal_cuts} 个不同的根切分可达最优，同优树亦可选择："
                        f"{other_cuts_text(a)}"
                    )
            else:
                cls, zh = "absent", "不出现"
                evidence = self._absent_evidence(a, b, p_star, t_star)
            result.append(
                {
                    "left": names_a,
                    "right": names_b,
                    "classification": cls,
                    "label": zh,
                    "evidence": evidence,
                    "optimal_root_cuts": num_optimal_cuts,
                }
            )

        order = {"mandatory": 0, "optional": 1, "absent": 2}
        result.sort(key=lambda c: (order[c["classification"]], c["left"], c["right"]))
        return result

    def _absent_evidence(
        self, a: int, b: int, p_star: int, t_star: int
    ) -> str:
        """Explain why forcing cut a | b cannot reach the optimum."""
        names_a = sorted(self.names_of(a))
        names_b = sorted(self.names_of(b))
        # lexicographically best achievable pair when forcing this cut
        outcome = self._cut_outcome(a, b)
        best_peak = outcome.peak
        best_total_at_peak = outcome.total
        min_total_under_star = None  # minimum total with combined peak <= p_star
        if best_peak <= p_star:
            min_total_under_star = best_total_at_peak

        cut = f"{names_a} | {names_b}"
        if best_peak > p_star:
            return (
                f"若强制根切分 {cut}，峰值至少为 {best_peak}，"
                f"高于最优峰值 {p_star}，故无同优树采用"
            )
        if min_total_under_star is not None and min_total_under_star > t_star:
            return (
                f"若强制根切分 {cut}，在峰值不超过 {p_star} 时总乘法数至少为 "
                f"{min_total_under_star}，高于最优总乘法数 {t_star}，故无同优树采用"
            )
        return f"根切分 {cut} 无法达到最优 (峰值 {p_star}, 总乘法 {t_star})"


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _collect_steps(tree: tuple, out: list[dict]) -> None:
    if tree[0] == "leaf":
        return
    _collect_steps(tree[1], out)
    _collect_steps(tree[2], out)
    step = dict(tree[3])
    step["order"] = len(out) + 1
    out.append(step)


def _tree_to_json(tree: tuple) -> dict:
    if tree[0] == "leaf":
        return {"type": "leaf", "name": tree[1]}
    step = tree[3]
    return {
        "type": "node",
        "left": _tree_to_json(tree[1]),
        "right": _tree_to_json(tree[2]),
        "contraction": {
            "multiplications": step["multiplications"],
            "result_size": step["result_size"],
            "result_indices": step["result_indices"],
            "contracted_indices": step["contracted_indices"],
        },
    }


def plan(payload: object) -> dict:
    network = network_from_payload(payload)
    planner = Planner(network)
    root = planner.solve()
    cuts = planner.classify(root)
    # exact count requires the bounded counting DP (a locally Pareto-
    # dominated subtree can still occur inside a global optimum whose outer
    # result masks its peak; bounding by (p*, t*) is provably lossless)
    num_cotrees = planner.count_optimal_trees(root.peak, root.total)

    steps: list[dict] = []
    _collect_steps(root.tree, steps)

    return {
        "summary": {
            "peak_memory": root.peak,
            "total_multiplications": root.total,
            "canonical": root.canon,
            "num_cotrees": num_cotrees,
            "num_tensors": network.n,
        },
        "tree": _tree_to_json(root.tree),
        "steps": steps,
        "cuts": cuts,
        "indices": [
            {
                "name": k,
                "dimension": planner.dims[k],
                "occurrences": sum(1 for idx in planner.tidx if k in idx),
                "on_tensors": [
                    planner.names[i]
                    for i in range(network.n)
                    if k in planner.tidx[i]
                ],
            }
            for k in sorted(planner.dims)
        ],
        "inputs": [
            {
                "name": network.tensors[i].name,
                "size": planner.size(1 << i),
                "indices": sorted(network.tensors[i].indices),
            }
            for i in range(network.n)
        ],
    }
