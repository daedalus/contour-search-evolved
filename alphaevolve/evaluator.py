#!/usr/bin/env python3
"""
AlphaEvolve evaluator for contour_search algorithm.
Uses geometric mean of medians across benchmarks for stable scoring.
Higher score = faster algorithm.
"""

import math
import time
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from search.graph import Graph


def make_undirected_chain(n: int) -> Tuple[Graph, str, str]:
    g = Graph()
    for i in range(n - 1):
        g.add_edge(str(i), str(i + 1), weight=1, bidirectional=True)
    return g, "0", str(n - 1)


def make_undirected_star(n: int) -> Tuple[Graph, str, str]:
    g = Graph()
    for i in range(1, n):
        g.add_edge("center", str(i), weight=1, bidirectional=True)
    return g, "center", str(n - 1)


def make_undirected_grid(rows: int, cols: int) -> Tuple[Graph, str, str]:
    g = Graph()
    for r in range(rows):
        for c in range(cols):
            node = f"{r},{c}"
            if c + 1 < cols:
                g.add_edge(node, f"{r},{c+1}", weight=1, bidirectional=True)
            if r + 1 < rows:
                g.add_edge(node, f"{r+1},{c}", weight=1, bidirectional=True)
    return g, "0,0", f"{rows - 1},{cols - 1}"


def make_undirected_dense(n: int) -> Tuple[Graph, str, str]:
    g = Graph()
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(str(i), str(j), weight=abs(i - j), bidirectional=True)
    return g, "0", str(n - 1)


def h_manhattan(a: str, b: str) -> float:
    try:
        ar, ac = a.split(",")
        br, bc = b.split(",")
        return abs(int(ar) - int(br)) + abs(int(ac) - int(bc))
    except Exception:
        return 0.0


def h_id(a: str, b: str) -> float:
    try:
        return abs(int(a) - int(b))
    except Exception:
        return 0.0


def h_zero(a: str, b: str) -> float:
    return 0.0


BENCHMARKS = [
    ("chain_1k", make_undirected_chain(1000), h_id),
    ("chain_5k", make_undirected_chain(5000), h_id),
    ("star_500", make_undirected_star(500), h_zero),
    ("grid_2500", make_undirected_grid(50, 50), h_manhattan),
    ("dense_80", make_undirected_dense(80), h_id),
]

WARMUP = 5
RUNS = 15


def run_benchmark(
    fn: Callable, graph: Graph, start: str, goal: str, heuristic
) -> List[float]:
    for _ in range(WARMUP):
        fn(graph, start, goal, heuristic)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        result = fn(graph, start, goal, heuristic)
        if result is None:
            raise RuntimeError("Algorithm returned None (no path found)")
        times.append(time.perf_counter() - t0)
    return times


def evaluate_candidate(algorithm_fn: Callable) -> float:
    """
    Score = geometric mean of medians across benchmark topologies.
    Score = 1.0 / geometric_mean(median_times). Higher is better.
    """
    medians = []

    for name, (graph, start, goal), heuristic_fn in BENCHMARKS:
        try:
            times = run_benchmark(algorithm_fn, graph, start, goal, heuristic_fn)
            median_time = statistics.median(times)
            if median_time <= 0:
                return 0.0
            medians.append(median_time)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            return 0.0

    if not medians:
        return 0.0

    geo_mean = math.exp(statistics.fmean(math.log(m) for m in medians))
    return 1.0 / geo_mean


def evaluate_candidate_detailed(algorithm_fn: Callable) -> Dict:
    """Return per-benchmark stats dictionary."""
    details = {}
    totals = []

    for name, (graph, start, goal), heuristic_fn in BENCHMARKS:
        try:
            times = run_benchmark(algorithm_fn, graph, start, goal, heuristic_fn)
            median_ms = statistics.median(times) * 1000
            mean_ms = statistics.mean(times) * 1000
            stdev_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
            details[name] = {
                "median_ms": median_ms,
                "mean_ms": mean_ms,
                "stdev_ms": stdev_ms,
                "cv": stdev_ms / mean_ms if mean_ms > 0 else 0,
            }
            totals.append(mean_ms)
        except Exception as e:
            details[name] = {"error": str(e)}

    if totals:
        details["total_mean_ms"] = statistics.mean(totals)
        details["total_median_ms"] = statistics.median(totals)
    return details


def extract_algorithm(code: str) -> Callable:
    namespace = {}
    exec(code, namespace)
    fn = namespace.get('contour_search')
    if fn is None:
        raise ValueError("No 'contour_search' function found in code")
    return fn


def load_candidate_from_file(filepath: str) -> Callable:
    code = Path(filepath).read_text()
    return extract_algorithm(code)


if __name__ == "__main__":
    from search.algorithms import contour_search as baseline

    print("Evaluating baseline contour_search...")
    score = evaluate_candidate(baseline)
    details = evaluate_candidate_detailed(baseline)
    print(f"Baseline score: {score:.6f}")
    print(f"Baseline geo-mean runtime: {1.0/score*1000:.4f}ms")
    print()
    for name, d in details.items():
        if "error" in d:
            print(f"  {name}: ERROR {d['error']}")
        else:
            print(f"  {name}: median={d['median_ms']:.3f}ms  mean={d['mean_ms']:.3f}ms  "
                  f"stddev={d['stdev_ms']:.3f}ms  cv={d['cv']:.3f}")
    if "total_mean_ms" in details:
        print(f"\n  Overall mean: {details['total_mean_ms']:.3f}ms")
