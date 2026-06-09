#!/usr/bin/env python3
"""
AlphaEvolve evaluator for contour_search algorithm.
Uses geometric mean of medians across benchmarks for stable scoring.
Higher score = faster algorithm.
"""

import math
import time
import statistics
from pathlib import Path
from typing import Callable, Dict, List

from contour_search.graph import Graph
from alphaevolve.benchmarks import BENCHMARKS, WARMUP

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
    details: dict[str, object] = {}
    totals: list[float] = []

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
    namespace: dict[str, object] = {}
    exec(code, namespace)
    fn = namespace.get("contour_search")
    if fn is None:
        raise ValueError("No 'contour_search' function found in code")
    return fn  # type: ignore[return-value]


def load_candidate_from_file(filepath: str) -> Callable:
    code = Path(filepath).read_text()
    return extract_algorithm(code)


def compare_with_baseline(candidate_path: str):
    from contour_search.algorithms import contour_search as baseline

    candidate_fn = load_candidate_from_file(candidate_path)

    print("Baseline contour_search:")
    base_score = evaluate_candidate(baseline)
    print(
        f"  Score: {base_score:.6f}  Geo-mean: {1.0 / base_score * 1000:.4f}ms"
        if base_score > 0
        else "  FAILED"
    )

    print()
    print("Candidate M115 (parent-pruning):")
    cand_score = evaluate_candidate(candidate_fn)
    print(
        f"  Score: {cand_score:.6f}  Geo-mean: {1.0 / cand_score * 1000:.4f}ms"
        if cand_score > 0
        else "  FAILED"
    )

    if base_score > 0 and cand_score > 0:
        diff = (cand_score - base_score) / base_score * 100
        print()
        print(f"Change: {diff:+.2f}%")
        if diff > 1.0:
            print("PROMOTED! (+>1%)")
        elif diff < -1.0:
            print("REJECTED (<-1%)")
        else:
            print("WASH (within ±1%)")


def print_benchmark_details(details: Dict):
    bench_names_list = [b[0] for b in BENCHMARKS]
    for name in bench_names_list:
        d = details.get(name, {})
        if "error" in d:
            print(f"  {name}: ERROR {d['error']}")
        elif isinstance(d, dict):
            print(
                f"  {name}: median={d['median_ms']:.3f}ms  mean={d['mean_ms']:.3f}ms  "
                f"stddev={d['stdev_ms']:.3f}ms  cv={d['cv']:.3f}"
            )
    if "total_mean_ms" in details:
        print(f"  Overall mean: {details['total_mean_ms']:.3f}ms")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        compare_with_baseline(sys.argv[1])
    else:
        from contour_search.algorithms import contour_search as baseline

        print("Evaluating baseline contour_search...")
        score = evaluate_candidate(baseline)
        details = evaluate_candidate_detailed(baseline)
        print(f"Baseline score: {score:.6f}")
        print(f"Baseline geo-mean runtime: {1.0 / score * 1000:.4f}ms")
        print()
        print_benchmark_details(details)
