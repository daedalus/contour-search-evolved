#!/usr/bin/env python3
"""Programmatic mutations for contour_search evolution."""

import sys, textwrap, random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

BASELINE_CODE = (Path(__file__).parent.parent / "search" / "algorithms.py").read_text()
_start = BASELINE_CODE.find("def contour_search(")
_end = BASELINE_CODE.find("\n\n\ndef ", _start + 1)
BASELINE = BASELINE_CODE[_start:_end] if _start >= 0 and _end >= 0 else ""

# Each mutation is a (name, code) pair
# The code is a complete contour_search function

# M0: baseline (copy)
M0 = BASELINE

# M1: local variable bindings only
M1 = textwrap.dedent("""\
def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    neighbors = graph.neighbors
    h_fn = heuristic
    round_fn = round

    buckets: Dict[float, List] = {}
    f_start = round_fn(h_fn(start, goal), precision)
    buckets.setdefault(f_start, []).append((0.0, start, [start]))
    visited: Set[str] = set()

    while buckets:
        min_f = min(buckets.keys())
        entries = buckets.pop(min_f)

        for g, node, path in entries:
            if node in visited:
                continue
            if node == goal:
                return path
            visited.add(node)

            for edge in neighbors(node):
                if edge.target not in visited:
                    new_g = g + edge.weight
                    new_f = round_fn(new_g + h_fn(edge.target, goal), precision)
                    buckets.setdefault(new_f, []).append((new_g, edge.target, [*path, edge.target]))

    return None
""")

# M2: predecessor map (no path copying) + node-only bucket entries
M2 = textwrap.dedent("""\
def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    buckets: Dict[float, List] = {}
    f_start = round(heuristic(start, goal), precision)
    buckets.setdefault(f_start, []).append(start)
    visited: Set[str] = set()
    g_score: Dict[str, float] = {start: 0.0}
    pred: Dict[str, str] = {}

    while buckets:
        min_f = min(buckets.keys())
        entries = buckets.pop(min_f)
        for node in entries:
            if node in visited:
                continue
            if node == goal:
                path = [goal]
                while node in pred:
                    node = pred[node]
                    path.append(node)
                return path[::-1]
            visited.add(node)
            g = g_score[node]
            for edge in graph.neighbors(node):
                nxt = edge.target
                if nxt in visited:
                    continue
                new_g = g + edge.weight
                if new_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = new_g
                    pred[nxt] = node
                    new_f = round(new_g + heuristic(nxt, goal), precision)
                    buckets.setdefault(new_f, []).append(nxt)
    return None
""")

# M3: predecessor + local bindings
M3 = textwrap.dedent("""\
def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    neighbors = graph.neighbors
    h_fn = heuristic
    round_fn = round

    buckets: Dict[float, List] = {}
    f_start = round_fn(h_fn(start, goal), precision)
    buckets.setdefault(f_start, []).append(start)
    visited: Set[str] = set()
    g_score: Dict[str, float] = {start: 0.0}
    pred: Dict[str, str] = {}

    while buckets:
        min_f = min(buckets.keys())
        entries = buckets.pop(min_f)
        for node in entries:
            if node in visited:
                continue
            if node == goal:
                path = [goal]
                while node in pred:
                    node = pred[node]
                    path.append(node)
                return path[::-1]
            visited.add(node)
            g = g_score[node]
            for edge in neighbors(node):
                nxt = edge.target
                if nxt in visited:
                    continue
                new_g = g + edge.weight
                if new_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = new_g
                    pred[nxt] = node
                    new_f = round_fn(new_g + h_fn(nxt, goal), precision)
                    buckets.setdefault(new_f, []).append(nxt)
    return None
""")

# M4: predecessor + local bindings + min tracking
M4 = textwrap.dedent("""\
def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    neighbors = graph.neighbors
    h_fn = heuristic
    round_fn = round

    buckets: Dict[float, List] = {}
    f_start = round_fn(h_fn(start, goal), precision)
    buckets.setdefault(f_start, []).append(start)
    visited: Set[str] = set()
    g_score: Dict[str, float] = {start: 0.0}
    pred: Dict[str, str] = {}
    current_min = f_start

    while buckets:
        if current_min not in buckets:
            current_min = min(buckets.keys())
        entries = buckets.pop(current_min)
        for node in entries:
            if node in visited:
                continue
            if node == goal:
                path = [goal]
                while node in pred:
                    node = pred[node]
                    path.append(node)
                return path[::-1]
            visited.add(node)
            g = g_score[node]
            for edge in neighbors(node):
                nxt = edge.target
                if nxt in visited:
                    continue
                new_g = g + edge.weight
                if new_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = new_g
                    pred[nxt] = node
                    new_f = round_fn(new_g + h_fn(nxt, goal), precision)
                    if new_f < current_min:
                        current_min = new_f
                    buckets.setdefault(new_f, []).append(nxt)
    return None
""")

# M5: predecessor + local bindings + min tracking + try/except for setdefault
M5 = textwrap.dedent("""\
def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    neighbors = graph.neighbors
    h_fn = heuristic
    round_fn = round

    buckets: Dict[float, List] = {}
    f_start = round_fn(h_fn(start, goal), precision)
    buckets[f_start] = [start]
    visited: Set[str] = set()
    g_score: Dict[str, float] = {start: 0.0}
    pred: Dict[str, str] = {}
    current_min = f_start

    while buckets:
        if current_min not in buckets:
            current_min = min(buckets.keys())
        entries = buckets.pop(current_min)
        for node in entries:
            if node in visited:
                continue
            if node == goal:
                path = [goal]
                while node in pred:
                    node = pred[node]
                    path.append(node)
                return path[::-1]
            visited.add(node)
            g = g_score[node]
            for edge in neighbors(node):
                nxt = edge.target
                if nxt in visited:
                    continue
                new_g = g + edge.weight
                if new_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = new_g
                    pred[nxt] = node
                    new_f = round_fn(new_g + h_fn(nxt, goal), precision)
                    if new_f < current_min:
                        current_min = new_f
                    try:
                        buckets[new_f].append(nxt)
                    except KeyError:
                        buckets[new_f] = [nxt]
    return None
""")

# M6: Use deque for each bucket to avoid list pop overhead on large buckets
M6 = textwrap.dedent("""\
def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    from collections import deque as _deque
    neighbors = graph.neighbors
    h_fn = heuristic
    round_fn = round

    buckets: Dict[float, _deque] = {}
    f_start = round_fn(h_fn(start, goal), precision)
    buckets[f_start] = _deque([start])
    visited: Set[str] = set()
    g_score: Dict[str, float] = {start: 0.0}
    pred: Dict[str, str] = {}
    current_min = f_start
    INF = float("inf")

    while buckets:
        if current_min not in buckets:
            current_min = min(buckets.keys())
        dq = buckets.pop(current_min)
        for node in dq:
            if node in visited:
                continue
            if node == goal:
                path = [goal]
                while node in pred:
                    node = pred[node]
                    path.append(node)
                return path[::-1]
            visited.add(node)
            g = g_score[node]
            for edge in neighbors(node):
                nxt = edge.target
                if nxt in visited:
                    continue
                new_g = g + edge.weight
                if new_g < g_score.get(nxt, INF):
                    g_score[nxt] = new_g
                    pred[nxt] = node
                    new_f = round_fn(new_g + h_fn(nxt, goal), precision)
                    if new_f < current_min:
                        current_min = new_f
                    try:
                        buckets[new_f].append(nxt)
                    except KeyError:
                        buckets[new_f] = _deque([nxt])
    return None
""")


ALL_MUTATIONS = [
    ("M0_baseline", M0),
    ("M1_local_bind", M1),
    ("M2_predecessor", M2),
    ("M3_pred_local", M3),
    ("M4_min_track", M4),
    ("M5_full_opt", M5),
    ("M6_deque", M6),
]


IMPORTS = """from __future__ import annotations
from typing import Callable, Dict, List, Optional, Set
from search.graph import Edge, Graph
"""


def extract_fn(code: str):
    ns = {}
    exec(IMPORTS + "\n" + code, ns)
    return ns.get('contour_search', None)


if __name__ == "__main__":
    from alphaevolve.evaluator import evaluate_candidate

    results = []
    for name, code in ALL_MUTATIONS:
        fn = extract_fn(code)
        if fn is None:
            print(f"  {name}: FAILED (could not extract)")
            continue
        try:
            score = evaluate_candidate(fn)
            results.append((score, name, code))
            print(f"  {name}: {score:.2f}")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    if results:
        results.sort(reverse=True)
        best_score, best_name, best_code = results[0]
        ref_score = results[-1][0] if len(results) > 1 else best_score
        print(f"\n  Best: {best_name} ({best_score:.2f})")
        print(f"  Worst: {results[-1][1]} ({results[-1][0]:.2f})")
        print(f"  Spread: {best_score/results[-1][0]:.2f}x")
