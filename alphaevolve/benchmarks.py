"""
Shared benchmark graph definitions and heuristic functions.
Used by evaluator.py and benchmark_all.py.
"""

from typing import Callable, List, Tuple

from contour_search.graph import Graph


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
                g.add_edge(node, f"{r},{c + 1}", weight=1, bidirectional=True)
            if r + 1 < rows:
                g.add_edge(node, f"{r + 1},{c}", weight=1, bidirectional=True)
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


BENCHMARKS: List[Tuple[str, Tuple[Graph, str, str], Callable]] = [
    ("chain_1k", make_undirected_chain(1000), h_id),
    ("chain_5k", make_undirected_chain(5000), h_id),
    ("star_500", make_undirected_star(500), h_zero),
    ("grid_2500", make_undirected_grid(50, 50), h_manhattan),
    ("dense_80", make_undirected_dense(80), h_id),
]

WARMUP = 5
