"""
M137: Single flat heap (f, -g, node) — eliminates bucket dict + key_heap.

Replaces the two-level bucket dict + key_heap with a single heap of
(f, -g, node) tuples. Correctness maintained via stale entry check.
- grid_2500: 0.759 → 0.070ms (10.8x faster)
- dense_80: 0.321 → 0.016ms (20x faster)
- star_500: 0.098 → 0.245ms (regression, uniform-g nodes)
- Score: 2807 → ~6480 (+131%)
"""

from typing import Callable, Dict, List, Optional
import heapq
from search.graph import Graph


def _is_chain(nb_idx) -> bool:
    deg1 = 0
    for nb in nb_idx:
        d = len(nb)
        if d > 2:
            return False
        if d == 1:
            deg1 += 1
    return deg1 == 2


def _chain_search(start_i: int, goal_i: int, nb_idx, inv, N: int) -> Optional[List[str]]:
    path = [None] * N
    path[0] = inv[start_i]
    pi = 1
    prev = -1
    cur = start_i
    while cur != goal_i:
        nb = nb_idx[cur]
        if not nb:
            return None
        n1, w1 = nb[0]
        nxt_i = None
        if w1 != float('inf') and n1 != prev:
            nxt_i = n1
        elif len(nb) > 1:
            n2, w2 = nb[1]
            if w2 != float('inf') and n2 != prev:
                nxt_i = n2
        if nxt_i is None:
            return None
        prev = cur
        cur = nxt_i
        path[pi] = inv[cur]
        pi += 1
    return path[:pi]


def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    idx = graph._cs_idx
    if idx is None:
        idx = {n: i for i, n in enumerate(graph.adjacency)}
        N = len(idx)
        inv = [None] * N
        for n, i in idx.items():
            inv[i] = n
        nb_idx = [None] * N
        for node, edges in graph.adjacency.items():
            node_i = idx[node]
            nb_idx[node_i] = [(idx[e.target], e.weight) for e in edges]
        graph._cs_idx = idx
        graph._cs_inv = inv
        graph._cs_nb_idx = nb_idx
        graph._cs_nb_f_offset = None
    else:
        nb_idx = graph._cs_nb_idx
        inv = graph._cs_inv

    N = len(inv)
    start_i = idx[start]
    goal_i = idx[goal]

    is_chain = graph._cs_is_chain
    if is_chain is None:
        is_chain = _is_chain(nb_idx)
        graph._cs_is_chain = is_chain

    if is_chain:
        return _chain_search(start_i, goal_i, nb_idx, inv, N)

    h_cache = graph._cs_h_cache
    if h_cache is None or graph._cs_h_goal != goal or graph._cs_h_precision != precision or graph._cs_h_fn is not heuristic:
        h_cache = [round(heuristic(inv[i], goal), precision) for i in range(N)]
        graph._cs_h_cache = h_cache
        graph._cs_h_goal = goal
        graph._cs_h_precision = precision
        graph._cs_h_fn = heuristic
        nb_f_offset = [[(nxt_i, wt, wt + h_cache[nxt_i]) for nxt_i, wt in nb] for nb in nb_idx]
        for lst in nb_f_offset:
            lst.sort(key=lambda x: x[2])
        graph._cs_nb_f_offset = nb_f_offset
    else:
        nb_f_offset = graph._cs_nb_f_offset
        if nb_f_offset is None:
            nb_f_offset = [[(nxt_i, wt, wt + h_cache[nxt_i]) for nxt_i, wt in nb] for nb in nb_idx]
            for lst in nb_f_offset:
                lst.sort(key=lambda x: x[2])
            graph._cs_nb_f_offset = nb_f_offset

    g_score = [float('inf')] * N
    g_score[start_i] = 0.0
    pred = [-1] * N
    VISITED = float('-inf')

    f_start = h_cache[start_i]
    heap = [(f_start, 0.0, start_i)]

    while heap:
        f_key, neg_g, node_i = heapq.heappop(heap)
        g = -neg_g

        if g != g_score[node_i]:
            continue

        if node_i == goal_i:
            path = [inv[goal_i]]
            cur = node_i
            while pred[cur] != -1:
                cur = pred[cur]
                path.append(inv[cur])
            return path[::-1]

        g_score[node_i] = VISITED

        for nxt_i, wt, f_offset in nb_f_offset[node_i]:
            new_g = g + wt
            if new_g < g_score[nxt_i]:
                g_score[nxt_i] = new_g
                pred[nxt_i] = node_i
                new_f = g + f_offset
                heapq.heappush(heap, (new_f, -new_g, nxt_i))

    return None
