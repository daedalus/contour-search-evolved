#!/usr/bin/env python3
"""
Comprehensive benchmark comparing contour_search (champion) against
all other single-source shortest-path algorithms in the codebase.
"""

import math
import time
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from search.algorithms import (
    contour_search,
    astar,
    dijkstra,
    bellman_ford,
    spfa,
    bidirectional_dijkstra,
    greedy_best_first_search,
    bfs,
    dfs,
)
from alphaevolve.benchmarks import BENCHMARKS, WARMUP

RUNS = 30


ALGORITHMS: Dict[str, Callable] = {
    "contour_search": contour_search,
    "astar": astar,
    "dijkstra": dijkstra,
    "bidirectional_dijkstra": bidirectional_dijkstra,
    "greedy_best_first_search": greedy_best_first_search,
    "bellman_ford": bellman_ford,
    "spfa": spfa,
    "bfs": bfs,
    "dfs": dfs,
}

# Algorithms that accept (graph, start, goal, heuristic) signature
HEURISTIC_ALGOS = {"contour_search", "astar", "greedy_best_first_search"}


def run_bench(
    algo_fn: Callable, name: str, graph, start, goal, heuristic
) -> Tuple[float, float, float]:
    for _ in range(WARMUP):
        if name in HEURISTIC_ALGOS:
            algo_fn(graph, start, goal, heuristic)
        else:
            algo_fn(graph, start, goal)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        if name in HEURISTIC_ALGOS:
            result = algo_fn(graph, start, goal, heuristic)
        else:
            result = algo_fn(graph, start, goal)
        t1 = time.perf_counter()
        if result is None:
            raise RuntimeError("returned None (no path found)")
        times.append((t1 - t0) * 1000)
    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return median, mean, stdev


def verify_correct(algo_fn, name, graph, start, goal, heuristic, ref_fn) -> bool:
    """Verify algorithm returns same path cost as reference (contour_search)."""
    try:
        if name in HEURISTIC_ALGOS:
            result = algo_fn(graph, start, goal, heuristic)
        else:
            result = algo_fn(graph, start, goal)

        ref = ref_fn(graph, start, goal, heuristic)

        if result is None or ref is None:
            return result == ref

        # Compare path cost (sum of edge weights)
        def path_cost(g, path):
            if not path or len(path) < 2:
                return 0.0
            cost = 0.0
            for i in range(len(path) - 1):
                for e in g.neighbors(path[i]):
                    if e.target == path[i + 1]:
                        cost += e.weight
                        break
            return cost

        return abs(path_cost(graph, result) - path_cost(graph, ref)) < 1e-9
    except Exception:
        return False


def _path_cost(g, path):
    if not path or len(path) < 2:
        return 0.0
    cost = 0.0
    for i in range(len(path) - 1):
        for e in g.neighbors(path[i]):
            if e.target == path[i + 1]:
                cost += e.weight
                break
    return cost


def _run_algo(algo_fn, algo_name, graph, start, goal, heuristic):
    if algo_name in HEURISTIC_ALGOS:
        return algo_fn(graph, start, goal, heuristic)
    return algo_fn(graph, start, goal)


def _verify_algo(
    bench_name, algo_name, algo_fn, graph, start, goal, heuristic, ref, ref_cost
):
    try:
        r = _run_algo(algo_fn, algo_name, graph, start, goal, heuristic)
        if r:
            cost = _path_cost(graph, r)
            ok = abs(cost - ref_cost) < 1e-9
        else:
            ok = ref is None
        if not ok:
            print(f"    FAIL: {algo_name} on {bench_name} (cost mismatch)")
        return ok
    except Exception as e:
        print(f"    FAIL: {algo_name} on {bench_name}: {e}")
        return False


def _verify_all():
    print("  Verifying correctness against contour_search...")
    all_ok = True
    for bench_name, (graph, start, goal), heuristic in BENCHMARKS:
        ref = contour_search(graph, start, goal, heuristic)
        ref_cost = _path_cost(graph, ref) if ref else None
        for algo_name, algo_fn in ALGORITHMS.items():
            if algo_name == "contour_search":
                continue
            ok = _verify_algo(
                bench_name,
                algo_name,
                algo_fn,
                graph,
                start,
                goal,
                heuristic,
                ref,
                ref_cost,
            )
            if not ok:
                all_ok = False
    if all_ok:
        print("  All algorithms pass correctness check!")
    else:
        print("  WARNING: Some algorithms failed correctness!")
    print()
    return all_ok


def _benchmark_algo(algo_name, algo_fn):
    bench_times = {}
    success = True
    for bench_name, (graph, start, goal), heuristic in BENCHMARKS:
        try:
            median, mean, stdev = run_bench(
                algo_fn, algo_name, graph, start, goal, heuristic
            )
            bench_times[bench_name] = (median, mean, stdev)
        except Exception as e:
            print(f"    ERROR: {algo_name} on {bench_name}: {e}")
            bench_times[bench_name] = (float("inf"), float("inf"), 0.0)
            success = False
    return bench_times, success


def _print_results(results):
    print(
        f"  {'Algorithm':<28} {'Geo-mean':>10} {'Chain_1k':>10} {'Chain_5k':>10} {'Star':>10} {'Grid':>10} {'Dense':>10} {'Score':>10}"
    )
    print("  " + "-" * 88)
    for algo_name, geo, bt, score in results:
        chain1k = bt.get("chain_1k", (0, 0, 0))[0]
        chain5k = bt.get("chain_5k", (0, 0, 0))[0]
        star = bt.get("star_500", (0, 0, 0))[0]
        grid = bt.get("grid_2500", (0, 0, 0))[0]
        dense = bt.get("dense_80", (0, 0, 0))[0]
        print(
            f"  {algo_name:<28} {geo:>8.4f}ms {chain1k:>8.3f}ms {chain5k:>8.3f}ms {star:>8.3f}ms {grid:>8.3f}ms {dense:>8.3f}ms {score:>8.0f}"
        )


def _print_relative(results):
    ref_geo = next(r[1] for r in results if r[0] == "contour_search")
    print()
    print("  Relative to contour_search:")
    print(f"  {'Algorithm':<28} {'× slower':>10}")
    print("  " + "-" * 40)
    for algo_name, geo, bt, score in results:
        ratio = geo / ref_geo if ref_geo > 0 else float("inf")
        if algo_name == "contour_search":
            print(f"  {algo_name:<28} {ratio:>8.2f}x  (reference)")
        else:
            pct = (ratio - 1) * 100
            print(f"  {algo_name:<28} {ratio:>8.2f}x  (+{pct:.0f}%)")


def main():
    print("=" * 90)
    print("  Algorithm Benchmark: All Single-Source Shortest Path Algorithms")
    print("=" * 90)
    print()
    print(f"  Warmup: {WARMUP} runs  |  Timed: {RUNS} runs")
    print()

    _verify_all()

    results: List[Tuple[str, float, Dict[str, Tuple[float, float, float]]]] = []
    for algo_name, algo_fn in ALGORITHMS.items():
        bench_times, success = _benchmark_algo(algo_name, algo_fn)
        if success:
            medians = [bench_times[b][0] for b in [b[0] for b in BENCHMARKS]]
            geo = math.exp(statistics.fmean(math.log(m) for m in medians))
            score = 1.0 / (geo / 1000)
            results.append((algo_name, geo, bench_times, score))

    results.sort(key=lambda x: x[1])

    _print_results(results)
    print()
    print(f"  Best: {results[0][0]} ({results[0][1]:.4f}ms geo-mean)")
    _print_relative(results)


if __name__ == "__main__":
    main()
