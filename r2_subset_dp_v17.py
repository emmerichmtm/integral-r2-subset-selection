#!/usr/bin/env python3
"""
Version V17.

Bellman-style left-to-right dynamic programming for fixed-cardinality
subset selection under the exact integral R2 indicator in the bi-objective case.

Assumptions:
- minimization objectives;
- the utopian point has been shifted to (0, 0);
- candidate points are positive and nondominated;
- after sorting, x increases and y decreases.

For a selected subsequence i_1 < ... < i_k, the exact integral R2 value is

    x[i_1]/2 + sum_r x[i_{r+1}] y[i_r] / (2(x[i_{r+1}] + y[i_r]))
    + y[i_k]/2 - sum_r x[i_r] y[i_r] / (2(x[i_r] + y[i_r])).

The implementation includes:
- the direct O(k n^2) Bellman DP (called LRDP in the report);
- a divide-and-conquer optimized DP that exploits the Monge transition matrix;
- a matrix-search / lower-envelope DP that computes each layer in linear time;
- brute-force verification for the demonstration instances; and
- a small CPU runtime comparison for exhaustive enumeration, LRDP, D&C DP, and selected matrix-search checks.

Only the Python standard library is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, groupby
from math import comb, inf, sin, sqrt
from statistics import mean
from time import perf_counter
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
VERSION = "V17"
TOLERANCE = 1e-9


@dataclass(frozen=True)
class SelectionResult:
    """Result object returned by the dynamic program or brute-force verifier."""

    indices: Tuple[int, ...]      # zero-based indices in the prepared front
    value: float
    points: Tuple[Point, ...]

    @property
    def one_based_indices(self) -> Tuple[int, ...]:
        return tuple(i + 1 for i in self.indices)


@dataclass(frozen=True)
class BenchmarkRecord:
    """One timing record for a fixed pair (n, k)."""

    n: int
    k: int
    subsets: int
    exhaustive_ms: Optional[float]
    lrdp_ms: float
    dcdp_ms: float
    exhaustive_status: str


def prepare_front(points: Iterable[Point]) -> List[Point]:
    """Sort points and retain the strict nondominated minimization front.

    Duplicate x-coordinates are first merged by keeping the smallest y-value.
    Dominated points are then removed while scanning from left to right. The
    returned front satisfies x_1 < ... < x_n and y_1 > ... > y_n.
    """
    pts = sorted((float(x), float(y)) for x, y in points)
    if not pts:
        raise ValueError("At least one candidate point is required.")
    for x, y in pts:
        if x <= 0 or y <= 0:
            raise ValueError("All coordinates must be positive after shifting the utopian point to (0, 0).")

    # For equal x, only the point with the smallest y can be nondominated.
    merged: List[Point] = []
    for x, group in groupby(pts, key=lambda p: p[0]):
        min_y = min(y for _, y in group)
        merged.append((x, min_y))

    front: List[Point] = []
    best_y = inf
    for x, y in merged:
        if y < best_y:
            front.append((x, y))
            best_y = y

    if not front:
        raise ValueError("No nondominated points remain.")
    for (x1, y1), (x2, y2) in zip(front, front[1:]):
        if not (x1 < x2 and y1 > y2):
            raise ValueError("Prepared front must satisfy x_1 < ... < x_n and y_1 > ... > y_n.")
    return front


def corner_cost(left: Point, right: Point) -> float:
    """Adjacent selected-point transition cost A_ij = x_j y_i/(2(x_j+y_i))."""
    _, y_left = left
    x_right, _ = right
    return (x_right * y_left) / (2.0 * (x_right + y_left))


def self_corner_cost(p: Point) -> float:
    """Diagonal correction A_ii = x_i y_i/(2(x_i+y_i))."""
    x, y = p
    return (x * y) / (2.0 * (x + y))


def exact_r2_for_indices(front: Sequence[Point], indices: Sequence[int]) -> float:
    """Exact integral R2 value of a selected subsequence.

    Lower values are better. The selected indices must be strictly increasing.
    """
    if len(indices) == 0:
        raise ValueError("The empty selected set is not handled by this normalized formula.")
    if any(i < 0 or i >= len(front) for i in indices):
        raise IndexError("Selected index out of range.")
    if any(i >= j for i, j in zip(indices, indices[1:])):
        raise ValueError("Selected indices must be strictly increasing.")

    value = front[indices[0]][0] / 2.0 + front[indices[-1]][1] / 2.0
    for i, j in zip(indices, indices[1:]):
        value += corner_cost(front[i], front[j])
    for i in indices:
        value -= self_corner_cost(front[i])
    return value


def _make_result(front: Sequence[Point], k: int, dp_last: Sequence[float], parent: List[List[int | None]]) -> SelectionResult:
    """Backtrack a solution from a DP table represented by final layer and parents."""
    best_total = inf
    best_last = None
    for j in range(k - 1, len(front)):
        total = dp_last[j] + front[j][1] / 2.0
        if total < best_total:
            best_total = total
            best_last = j

    assert best_last is not None
    indices = [best_last]
    j = best_last
    for r in range(k - 1, 0, -1):
        p = parent[r][j]
        assert p is not None
        indices.append(p)
        j = p
    indices.reverse()

    value_check = exact_r2_for_indices(front, indices)
    if abs(value_check - best_total) > TOLERANCE:
        raise RuntimeError("Internal consistency check failed.")

    return SelectionResult(indices=tuple(indices), value=best_total, points=tuple(front[i] for i in indices))


def dp_select_exact_r2(points: Sequence[Point], k: int) -> SelectionResult:
    """Solve fixed-cardinality exact R2 subset selection by direct O(k n^2) DP."""
    front = prepare_front(points)
    n = len(front)
    if not (1 <= k <= n):
        raise ValueError(f"k must satisfy 1 <= k <= n; got k={k}, n={n}.")

    diag = [self_corner_cost(p) for p in front]

    # dp[r][j] is the best chain value ending at j with exactly r+1 points,
    # not including the final y_j/2 boundary term.
    dp = [[inf for _ in range(n)] for _ in range(k)]
    parent: List[List[int | None]] = [[None for _ in range(n)] for _ in range(k)]

    for j, (xj, _) in enumerate(front):
        dp[0][j] = xj / 2.0 - diag[j]

    for r in range(1, k):
        for j in range(r, n):
            best_value = inf
            best_i = None
            for i in range(r - 1, j):
                candidate = dp[r - 1][i] + corner_cost(front[i], front[j]) - diag[j]
                if candidate < best_value:
                    best_value = candidate
                    best_i = i
            dp[r][j] = best_value
            parent[r][j] = best_i

    return _make_result(front, k, dp[k - 1], parent)


def dp_select_exact_r2_divide_conquer(points: Sequence[Point], k: int) -> SelectionResult:
    """Solve fixed-cardinality exact R2 subset selection by Monge D&C DP.

    The recurrence is identical to dp_select_exact_r2, but each layer is computed
    by divide-and-conquer using the monotonicity of the minimizing predecessor.
    Tie-breaking is by the smallest predecessor index, matching the direct DP.
    """
    front = prepare_front(points)
    n = len(front)
    if not (1 <= k <= n):
        raise ValueError(f"k must satisfy 1 <= k <= n; got k={k}, n={n}.")

    diag = [self_corner_cost(p) for p in front]
    dp_prev = [front[j][0] / 2.0 - diag[j] for j in range(n)]
    parent: List[List[int | None]] = [[None for _ in range(n)] for _ in range(k)]

    def compute_layer(layer: int, dp_before: Sequence[float]) -> Tuple[List[float], List[int | None]]:
        dp_curr = [inf for _ in range(n)]
        parent_row: List[int | None] = [None for _ in range(n)]

        def solve(j_left: int, j_right: int, opt_left: int, opt_right: int) -> None:
            if j_left > j_right:
                return
            mid = (j_left + j_right) // 2
            upper = min(opt_right, mid - 1)
            lower = opt_left
            best_value = inf
            best_i = None
            for i in range(lower, upper + 1):
                candidate = dp_before[i] + corner_cost(front[i], front[mid]) - diag[mid]
                if candidate < best_value:
                    best_value = candidate
                    best_i = i
            assert best_i is not None
            dp_curr[mid] = best_value
            parent_row[mid] = best_i

            solve(j_left, mid - 1, opt_left, best_i)
            solve(mid + 1, j_right, best_i, opt_right)

        # In layer r (zero-based), j must be at least r and predecessor i at least r-1.
        solve(layer, n - 1, layer - 1, n - 2)
        return dp_curr, parent_row

    for r in range(1, k):
        dp_prev, parent[r] = compute_layer(r, dp_prev)

    return _make_result(front, k, dp_prev, parent)



def _transition_value_for_row(front: Sequence[Point], dp_before: Sequence[float], row: int, x: float) -> float:
    """Implicit matrix entry D(row) + x*y_row/(2*(x+y_row))."""
    y = front[row][1]
    return dp_before[row] + (x * y) / (2.0 * (x + y))


def _strict_crossing_x(front: Sequence[Point], dp_before: Sequence[float], old: int, new: int) -> float:
    """Return the x-threshold after which the new row is strictly better.

    The function assumes old < new, hence y_old > y_new.  It returns t such
    that the new row is strictly better for x > t.  At x == t, the older row is
    kept, matching leftmost predecessor tie-breaking.  If the new row never
    becomes strictly better for finite x, return infinity.
    """
    y_old = front[old][1]
    y_new = front[new][1]
    if not (y_old > y_new):
        raise ValueError("Prepared front order requires y_old > y_new for old < new.")

    delta = dp_before[new] - dp_before[old]
    if delta <= 0.0:
        return 0.0

    span = (y_old - y_new) / 2.0
    if delta >= span:
        return inf

    # Solve x^2*(a-b)/(2*(x+a)*(x+b)) = delta for the positive root,
    # where a=y_old, b=y_new, and 0 < delta < (a-b)/2.
    a = y_old
    b = y_new
    d = a - b
    denom = 2.0 * (d - 2.0 * delta)
    B = 2.0 * delta * (a + b)
    disc = B * B + 8.0 * delta * a * b * (d - 2.0 * delta)
    return (B + sqrt(max(0.0, disc))) / denom


def _compute_layer_by_matrix_search(front: Sequence[Point], diag: Sequence[float], layer: int, dp_before: Sequence[float]) -> Tuple[List[float], List[int | None]]:
    """Compute one DP layer by a lower-envelope matrix-search sweep.

    This is an O(n)-per-layer implementation of the same recurrence as the
    direct Bellman DP.  It exploits the fact that the implicit transition rows
    are single-crossing functions of x.  Floating-point comparisons use a small
    tolerance and a local neighbor check to keep tie behavior aligned with the
    direct leftmost-predecessor DP.
    """
    n = len(front)
    dp_curr = [inf for _ in range(n)]
    parent_row: List[int | None] = [None for _ in range(n)]

    rows: List[int] = []
    starts: List[float] = []  # row q may win only for x > starts[q_index]
    pointer = 0

    def add_row(row: int) -> None:
        nonlocal pointer
        start = 0.0
        while rows:
            last = rows[-1]
            start = _strict_crossing_x(front, dp_before, last, row)
            if start == inf:
                return
            if start <= starts[-1] + 1e-15:
                rows.pop()
                starts.pop()
                if pointer > len(rows) - 1:
                    pointer = max(0, len(rows) - 1)
                continue
            break
        rows.append(row)
        starts.append(start if len(rows) > 1 else 0.0)
        if len(rows) == 1:
            pointer = 0

    for j in range(layer, n):
        # Endpoint j may use predecessors layer-1,...,j-1.  Insert the newly
        # feasible predecessor j-1 just before querying x_j.
        add_row(j - 1)
        if not rows:
            raise RuntimeError("Matrix-search envelope is unexpectedly empty.")

        xj = front[j][0]
        if pointer >= len(rows):
            pointer = len(rows) - 1

        # Move forward while the next envelope row is strictly better.  Direct
        # value comparison is used to make boundary ties behave like the direct
        # DP, which keeps the smallest predecessor index.
        while pointer + 1 < len(rows):
            current = rows[pointer]
            nxt = rows[pointer + 1]
            cur_val = _transition_value_for_row(front, dp_before, current, xj)
            next_val = _transition_value_for_row(front, dp_before, nxt, xj)
            if next_val < cur_val - 1e-12:
                pointer += 1
            else:
                break

        best_i = rows[pointer]

        # Defensive local correction for floating-point near-ties.
        candidates = [best_i]
        if pointer > 0:
            candidates.append(rows[pointer - 1])
        if pointer + 1 < len(rows):
            candidates.append(rows[pointer + 1])
        best_value = inf
        best_row = None
        for row in sorted(set(candidates)):
            value = _transition_value_for_row(front, dp_before, row, xj)
            if value < best_value - 1e-12 or (abs(value - best_value) <= 1e-12 and (best_row is None or row < best_row)):
                best_value = value
                best_row = row
        assert best_row is not None
        dp_curr[j] = best_value - diag[j]
        parent_row[j] = best_row

    return dp_curr, parent_row


def dp_select_exact_r2_matrix_search(points: Sequence[Point], k: int) -> SelectionResult:
    """Solve fixed-cardinality exact R2 subset selection by matrix search.

    This function computes each DP layer in O(n) arithmetic operations by a
    lower-envelope sweep over the staircase Monge transition matrix.  It returns
    the same leftmost-tie solution as the direct DP, up to floating-point
    tolerance in the implementation.
    """
    front = prepare_front(points)
    n = len(front)
    if not (1 <= k <= n):
        raise ValueError(f"k must satisfy 1 <= k <= n; got k={k}, n={n}.")

    diag = [self_corner_cost(p) for p in front]
    dp_prev = [front[j][0] / 2.0 - diag[j] for j in range(n)]
    parent: List[List[int | None]] = [[None for _ in range(n)] for _ in range(k)]

    for layer in range(1, k):
        dp_prev, parent[layer] = _compute_layer_by_matrix_search(front, diag, layer, dp_prev)

    return _make_result(front, k, dp_prev, parent)

def brute_force_select_exact_r2(points: Sequence[Point], k: int) -> SelectionResult:
    """Brute-force verification for small n."""
    front = prepare_front(points)
    best_value = inf
    best_indices: Tuple[int, ...] | None = None
    for indices in combinations(range(len(front)), k):
        value = exact_r2_for_indices(front, indices)
        if value < best_value:
            best_value = value
            best_indices = indices
    assert best_indices is not None
    return SelectionResult(indices=best_indices, value=best_value, points=tuple(front[i] for i in best_indices))


def brute_force_select_exact_r2_time_limited(
    points: Sequence[Point], k: int, time_limit_seconds: float
) -> Tuple[Optional[SelectionResult], float, str]:
    """Run exhaustive enumeration with a wall-clock time limit.

    Returns (result, elapsed_ms, status). If the time limit is exceeded,
    result is None and status is "TIME LIMIT". The loop checks the
    deadline periodically, so the elapsed time may be slightly above the limit.
    """
    front = prepare_front(points)
    deadline = perf_counter() + time_limit_seconds
    start = perf_counter()
    best_value = inf
    best_indices: Tuple[int, ...] | None = None

    for checked, indices in enumerate(combinations(range(len(front)), k), start=1):
        if checked % 4096 == 0 and perf_counter() >= deadline:
            return None, (perf_counter() - start) * 1000.0, "TIME LIMIT"
        value = exact_r2_for_indices(front, indices)
        if value < best_value:
            best_value = value
            best_indices = indices

    assert best_indices is not None
    result = SelectionResult(indices=best_indices, value=best_value, points=tuple(front[i] for i in best_indices))
    return result, (perf_counter() - start) * 1000.0, "PASS"


def check_monge_transition_matrix(points: Sequence[Point], tolerance: float = 1e-12) -> bool:
    """Numerically check the Monge inequality for A_ij.

    For rows and columns ordered by front index, the inequality is

        A[i,j] + A[ip,jp] <= A[i,jp] + A[ip,j]

    for i < ip and j < jp, because y_i > y_ip while x_j < x_jp.
    """
    front = prepare_front(points)
    n = len(front)
    for i in range(n):
        for ip in range(i + 1, n):
            for j in range(n):
                for jp in range(j + 1, n):
                    lhs = corner_cost(front[i], front[j]) + corner_cost(front[ip], front[jp])
                    rhs = corner_cost(front[i], front[jp]) + corner_cost(front[ip], front[j])
                    if lhs > rhs + tolerance:
                        return False
    return True


def deterministic_large_front(n: int) -> List[Point]:
    """Create a smooth deterministic nondominated front for checks and benchmarks."""
    points: List[Point] = []
    for i in range(n):
        x = 1.0 + 0.8 * i + 0.03 * sin(0.7 * i)
        # Strictly decreasing positive y with mild curvature.
        y = 60.0 / (1.0 + 0.055 * i) + 0.35 * (n - i) / n
        points.append((x, y))
    return prepare_front(points)


def _measure_mean_runtime_ms(function, points: Sequence[Point], k: int, repeats: int) -> float:
    """Measure mean runtime in milliseconds using time.perf_counter()."""
    # Warm start to avoid one-off setup artifacts.
    function(points, k)
    samples: List[float] = []
    for _ in range(repeats):
        start = perf_counter()
        function(points, k)
        samples.append((perf_counter() - start) * 1000.0)
    return mean(samples)


def runtime_benchmark_cases() -> List[BenchmarkRecord]:
    """Return the benchmark table used in the report."""
    cases = [(8, 4), (10, 5), (12, 6), (14, 7), (16, 8), (18, 9), (20, 10)]
    return _runtime_benchmark_for_cases(cases)


def runtime_benchmark_fixed_k6_cases() -> List[BenchmarkRecord]:
    """Return the fixed-k benchmark table used in the report."""
    cases = [(10, 6), (14, 6), (18, 6), (22, 6), (26, 6), (30, 6), (40, 6), (60, 6), (80, 6), (100, 6)]
    return _runtime_benchmark_for_cases(cases)


def _runtime_benchmark_for_cases(cases: Sequence[Tuple[int, int]]) -> List[BenchmarkRecord]:
    """Measure benchmark records for a list of (n, k) cases.

    LRDP and D&C DP are compared for every case. Exhaustive enumeration is
    attempted with a one-second wall-clock time limit. If it completes, its
    result is also checked against the DP solution; otherwise the record is
    marked as TIME LIMIT.
    """
    records: List[BenchmarkRecord] = []
    exhaustive_time_limit_seconds = 1.0
    for n, k in cases:
        points = deterministic_large_front(n)

        # Correctness sanity check before timing: this is always performed.
        direct = dp_select_exact_r2(points, k)
        fast = dp_select_exact_r2_divide_conquer(points, k)
        matrix = dp_select_exact_r2_matrix_search(points, k)
        if direct.indices != fast.indices or abs(direct.value - fast.value) > TOLERANCE:
            raise RuntimeError(f"Mismatch between LRDP and D&C DP for n={n}, k={k}.")
        if direct.indices != matrix.indices or abs(direct.value - matrix.value) > TOLERANCE:
            raise RuntimeError(f"Mismatch between LRDP and matrix-search DP for n={n}, k={k}.")

        brute_result, exhaustive_ms_single, exhaustive_status = brute_force_select_exact_r2_time_limited(
            points, k, exhaustive_time_limit_seconds
        )
        if brute_result is not None:
            if direct.indices != brute_result.indices or abs(direct.value - brute_result.value) > TOLERANCE:
                raise RuntimeError(f"Mismatch between LRDP and exhaustive enumeration for n={n}, k={k}.")
            exhaustive_ms: Optional[float] = exhaustive_ms_single
        else:
            exhaustive_ms = None

        dp_repeats = 10
        lrdp_ms = _measure_mean_runtime_ms(dp_select_exact_r2, points, k, dp_repeats)
        dcdp_ms = _measure_mean_runtime_ms(dp_select_exact_r2_divide_conquer, points, k, dp_repeats)

        records.append(
            BenchmarkRecord(
                n=n,
                k=k,
                subsets=comb(n, k),
                exhaustive_ms=exhaustive_ms,
                lrdp_ms=lrdp_ms,
                dcdp_ms=dcdp_ms,
                exhaustive_status=exhaustive_status,
            )
        )
    return records

def run_demo() -> None:
    datasets = {
        "seven-point staircase": {
            "points": [
                (2, 20), (4, 18), (6, 16), (9, 12), (11, 8), (14, 5), (17, 3)
            ],
            "ks": range(2, 6),
        },
        "six-point mildly irregular": {
            "points": [
                (1, 13), (2.5, 10), (4, 8.2), (6.5, 5.7), (9.5, 3.8), (13, 2.2)
            ],
            "ks": range(2, 6),
        },
        "five-point convex": {
            "points": [
                (1, 16), (2, 11), (4, 7), (8, 4), (15, 1.5)
            ],
            "ks": range(2, 6),
        },
        "twelve-point larger-k front": {
            "points": [
                (1, 30), (2, 22), (3.5, 17), (5, 13),
                (7, 10), (9, 7.8), (12, 6), (15, 4.7),
                (19, 3.6), (24, 2.7), (30, 1.9), (38, 1.2),
            ],
            "ks": range(6, 10),
        },
        "twenty-point graphical front": {
            "points": [
                (1, 40), (2, 32), (3, 28), (4, 24), (5.5, 20.5),
                (7, 17.5), (9, 15), (11.5, 12.8), (14, 10.8), (17, 9.1),
                (20, 7.6), (24, 6.2), (29, 5.1), (35, 4.2), (42, 3.4),
                (50, 2.7), (59, 2.15), (69, 1.7), (80, 1.35), (92, 1.05),
            ],
            "ks": [8, 10, 12],
        },
    }

    print(f"Fixed-cardinality exact integral R2 subset selection ({VERSION})")
    print("Direct Bellman DP, Monge divide-and-conquer DP, matrix-search DP,")
    print("brute-force verification, and a small CPU runtime comparison")
    print()

    for name, spec in datasets.items():
        points = spec["points"]
        ks = spec["ks"]
        front = prepare_front(points)
        print(f"Dataset: {name}")
        print(f"Prepared front: {front}")
        print(f"Monge transition check: {'PASS' if check_monge_transition_matrix(front) else 'FAIL'}")
        for k in ks:
            direct = dp_select_exact_r2(front, k)
            fast = dp_select_exact_r2_divide_conquer(front, k)
            matrix = dp_select_exact_r2_matrix_search(front, k)
            brute = brute_force_select_exact_r2(front, k)
            same_fast = direct.indices == fast.indices and abs(direct.value - fast.value) <= TOLERANCE
            same_matrix = direct.indices == matrix.indices and abs(direct.value - matrix.value) <= TOLERANCE
            same_brute = direct.indices == brute.indices and abs(direct.value - brute.value) <= TOLERANCE
            status = "PASS" if same_fast and same_matrix and same_brute else "FAIL"
            print(
                f"  k={k}: direct={direct.one_based_indices}, "
                f"D&C={fast.one_based_indices}, matrix={matrix.one_based_indices}, "
                f"brute={brute.one_based_indices}, R2={direct.value:.9f}, {status}"
            )
        print()

    large = deterministic_large_front(80)
    print("Dataset: eighty-point non-brute-force acceleration check")
    print(f"Prepared front length: {len(large)}")
    for k in (10, 20, 30):
        direct = dp_select_exact_r2(large, k)
        fast = dp_select_exact_r2_divide_conquer(large, k)
        matrix = dp_select_exact_r2_matrix_search(large, k)
        same = (direct.indices == fast.indices == matrix.indices and
                abs(direct.value - fast.value) <= TOLERANCE and
                abs(direct.value - matrix.value) <= TOLERANCE)
        status = "PASS" if same else "FAIL"
        print(
            f"  k={k}: direct={direct.one_based_indices}, "
            f"D&C={fast.one_based_indices}, matrix={matrix.one_based_indices}, "
            f"R2={direct.value:.9f}, {status}"
        )

    print()
    print("CPU runtime comparison on deterministic fronts")
    print("Times are mean wall-clock milliseconds from time.perf_counter().")
    print("Exhaustive enumeration has a one-second wall-clock time limit per case; DP variants use 10 repeats.")
    print()

    print("Balanced cases k = n/2")
    print(f"{'n':>3} {'k':>3} {'binom(n,k)':>10} {'exhaustive':>12} {'LRDP':>10} {'DC DP':>10}")
    for record in runtime_benchmark_cases():
        print(
            f"{record.n:>3d} {record.k:>3d} {record.subsets:>10d} "
            f"{(format(record.exhaustive_ms, '.3f') if record.exhaustive_ms is not None else '>1000'):>12} {record.lrdp_ms:>10.3f} {record.dcdp_ms:>10.3f}"
        )

    print()
    print("Fixed small cardinality cases k = 6")
    print(f"{'n':>3} {'k':>3} {'binom(n,k)':>10} {'exhaustive':>12} {'LRDP':>10} {'DC DP':>10}")
    for record in runtime_benchmark_fixed_k6_cases():
        print(
            f"{record.n:>3d} {record.k:>3d} {record.subsets:>10d} "
            f"{(format(record.exhaustive_ms, '.3f') if record.exhaustive_ms is not None else '>1000'):>12} {record.lrdp_ms:>10.3f} {record.dcdp_ms:>10.3f}"
        )


if __name__ == "__main__":
    run_demo()
