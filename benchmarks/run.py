#!/usr/bin/env python3
"""Benchmark all graph search algorithms with full statistical output."""

import math
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search.graph import Graph
from search import algorithms as A


def make_chain(n: int) -> tuple[Graph, str, str]:
    g = Graph()
    for i in range(n - 1):
        g.add_edge(str(i), str(i + 1), weight=1, bidirectional=True)
    return g, "0", str(n - 1)


def make_star(n: int) -> tuple[Graph, str, str]:
    g = Graph()
    for i in range(1, n):
        g.add_edge("center", str(i), weight=1, bidirectional=True)
    return g, "center", str(n - 1)


def make_grid(rows: int, cols: int) -> tuple[Graph, str, str]:
    g = Graph()
    for r in range(rows):
        for c in range(cols):
            node = f"{r},{c}"
            if c + 1 < cols:
                g.add_edge(node, f"{r},{c+1}", weight=1, bidirectional=True)
            if r + 1 < rows:
                g.add_edge(node, f"{r+1},{c}", weight=1, bidirectional=True)
    return g, "0,0", f"{rows - 1},{cols - 1}"


def make_dense(n: int) -> tuple[Graph, str, str]:
    g = Graph()
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(str(i), str(j), weight=abs(i - j), bidirectional=True)
    return g, "0", str(n - 1)


def h_id(a: str, b: str) -> float:
    try:
        return abs(int(a) - int(b))
    except Exception:
        return 0.0


def h_manhattan(a: str, b: str) -> float:
    try:
        ar, ac = a.split(",")
        br, bc = b.split(",")
        return abs(int(ar) - int(br)) + abs(int(ac) - int(bc))
    except Exception:
        return 0.0


def h_zero(a: str, b: str) -> float:
    return 0.0


BENCHMARKS: list[tuple[str, tuple[Graph, str, str]]] = [
    ("chain_1k", make_chain(1000)),
    ("chain_5k", make_chain(5000)),
    ("star_500", make_star(500)),
    ("grid_2500", make_grid(50, 50)),
    ("dense_80", make_dense(80)),
]

HEURISTICS: dict[str, object] = {
    "chain_1k": h_id,
    "chain_5k": h_id,
    "star_500": h_zero,
    "grid_2500": h_manhattan,
    "dense_80": h_id,
}

WARMUP = 5
RUNS = 15

ALGORITHMS: list[tuple[str, object, str]] = [
    ("bfs", A.bfs, "path"),
    ("dfs", A.dfs, "path"),
    ("dijkstra", A.dijkstra, "path"),
    ("bellman_ford", A.bellman_ford, "path"),
    ("spfa", A.spfa, "path"),
    ("dag_shortest_path", A.dag_shortest_path, "path"),
    ("astar", A.astar, "heuristic"),
    ("greedy_best_first", A.greedy_best_first_search, "heuristic"),
    ("contour_search", A.contour_search, "heuristic"),
    ("bidir_bfs", A.bidirectional_bfs, "path"),
    ("bidir_dijkstra", A.bidirectional_dijkstra, "path"),
    ("topo_sort", A.topological_sort, "graph"),
    ("prim_mst", A.prim_mst, "graph"),
    ("kruskal_mst", A.kruskal_mst, "graph"),
    ("floyd_warshall", A.floyd_warshall, "all_pairs"),
    ("johnson", A.johnson, "all_pairs"),
]


def gather_times(fn, g, s, t, h) -> list[float]:
    if h is None:
        for _ in range(WARMUP):
            fn(g, s, t)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            fn(g, s, t)
            times.append(time.perf_counter() - t0)
    else:
        for _ in range(WARMUP):
            fn(g, s, t, h)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            fn(g, s, t, h)
            times.append(time.perf_counter() - t0)
    return times


def gather_times_graph(fn, g) -> list[float]:
    for _ in range(WARMUP):
        fn(g)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn(g)
        times.append(time.perf_counter() - t0)
    return times


def stats_from(times: list[float]) -> dict:
    n = len(times)
    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if n > 1 else 0.0
    try:
        mode = statistics.mode(times)
    except statistics.StatisticsError:
        mode = None
    if n > 2 and stdev > 0:
        skew = (sum((t - mean) ** 3 for t in times) / n) / (stdev ** 3)
    else:
        skew = None
    return dict(mean=mean, median=median, mode=mode, stdev=stdev, skew=skew)


def run_benchmarks(with_stats=False) -> dict[str, dict[str, float | dict]]:
    results: dict[str, dict] = {}

    for name, fn, kind in ALGORITHMS:
        row = {}
        for bname, (g, s, t) in BENCHMARKS:
            try:
                if kind == "all_pairs":
                    if len(g.nodes()) >= 500:
                        continue
                    times = gather_times_graph(fn, g)
                elif kind == "heuristic":
                    times = gather_times(fn, g, s, t, HEURISTICS[bname])
                elif kind == "graph":
                    times = gather_times_graph(fn, g)
                else:
                    times = gather_times(fn, g, s, t, None)
                if with_stats:
                    row[bname] = stats_from(times)
                else:
                    row[bname] = statistics.median(times)
            except Exception:
                pass
        results[name] = row

    return results


def print_markdown(results: dict[str, dict]) -> None:
    bnames = [b[0] for b in BENCHMARKS]
    has_stats = any(isinstance(v, dict) for v in results.get(list(results)[0], {}).values())

    if not has_stats:
        header = f"| {'algorithm':<20} |"
        sep = f"| {'-'*20} |"
        for bn in bnames:
            header += f" {bn:>13} |"
            sep += f" {'-'*13} |"
        header += f" {'mean':>9} |"
        sep += f" {'-'*9} |"

        print(header)
        print(sep)

        for name, row in results.items():
            times = [row.get(bn) for bn in bnames]
            valid = [t for t in times if t is not None]
            mean = statistics.mean(valid) if valid else 0
            line = f"| {name:<20} |"
            for t in times:
                if t is not None:
                    line += f" {t*1000:>10.3f}  |"
                else:
                    line += f" {'—':>10}  |"
            line += f" {mean*1000:>8.3f} |"
            print(line)
    else:
        stat_labels = ["mean", "median", "mode", "stdev", "skew"]
        for stat in stat_labels:
            if stat == "mode":
                header = f"| {'algo':<20} |"
                sep = f"| {'-'*20} |"
                for bn in bnames:
                    header += f" {bn+'_'+stat:>13} |"
                    sep += f" {'-'*13} |"
                header += f" {'mean_'+stat:>9} |"
                sep += f" {'-'*9} |"
                print(f"\n## {stat}")
                print(header)
                print(sep)
                for name, row in results.items():
                    vals = [row.get(bn) for bn in bnames]
                    mode_vals = []
                    mean_val = 0.0
                    count = 0
                    for v in vals:
                        if v is not None and v.get(stat) is not None:
                            mode_vals.append(v[stat] * 1000)
                            mean_val += v[stat] * 1000
                            count += 1
                    mode_mean = mean_val / count if count else 0
                    seen = set()
                    line = f"| {name:<20} |"
                    for v in vals:
                        if v is not None and v.get(stat) is not None:
                            m = v[stat] * 1000
                            line += f" {m:>10.3f}  |"
                        else:
                            line += f" {'—':>10}  |"
                    line += f" {mode_mean:>8.3f} |"
                    print(line)
                continue
            if stat == "skew":
                header = f"| {'algo':<20} |"
                sep = f"| {'-'*20} |"
                for bn in bnames:
                    header += f" {bn+'_'+stat:>13} |"
                    sep += f" {'-'*13} |"
                print(f"\n## {stat}")
                print(header)
                print(sep)
                for name, row in results.items():
                    vals = [row.get(bn) for bn in bnames]
                    line = f"| {name:<20} |"
                    for v in vals:
                        if v is not None and v.get(stat) is not None:
                            line += f" {v[stat]:>10.4f}  |"
                        else:
                            line += f" {'—':>10}  |"
                    print(line)
                continue
            header = f"| {'algo':<20} |"
            sep = f"| {'-'*20} |"
            for bn in bnames:
                header += f" {bn+'_'+stat:>11} |"
                sep += f" {'-'*11} |"
            header += f" {'mean_'+stat:>8} |"
            sep += f" {'-'*8} |"
            print(f"\n## {stat}")
            print(header)
            print(sep)
            for name, row in results.items():
                vals = [row.get(bn) for bn in bnames]
                valid = []
                for v in vals:
                    if v is not None and v.get(stat) is not None:
                        valid.append(v[stat] * 1000)
                row_mean = statistics.mean(valid) if valid else 0
                line = f"| {name:<20} |"
                for v in vals:
                    if v is not None and v.get(stat) is not None:
                        line += f" {v[stat]*1000:>10.3f}  |"
                    else:
                        line += f" {'—':>10}  |"
                line += f" {row_mean:>8.3f} |"
                print(line)


def print_csv(results: dict[str, dict]) -> None:
    bnames = [b[0] for b in BENCHMARKS]
    print("algorithm," + ",".join(bnames) + ",mean")
    for name, row in results.items():
        times = [row.get(bn) for bn in bnames]
        valid = [t for t in times if t is not None]
        mean = statistics.mean(valid) if valid else 0
        ms = [f"{t*1000:.3f}" if t is not None else "" for t in times]
        print(f"{name},{','.join(ms)},{mean*1000:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["md", "csv"], default="md")
    parser.add_argument("--stats", action="store_true", help="Print full stats table")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=15)
    args = parser.parse_args()

    WARMUP = args.warmup
    RUNS = args.runs

    results = run_benchmarks(with_stats=args.stats)

    if args.format == "csv":
        print_csv(results)
    else:
        print_markdown(results)
