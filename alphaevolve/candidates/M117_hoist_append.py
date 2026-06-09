"""
M117: Hoist _last_list.append method reference.

Every _last_list.append(nxt_i) in the hot loop calls LOAD_ATTR, which
allocates a new method-wrapper object.  Hoisting _append = _last_list.append
outside the loop saves 1 bytecode + 1 PyObject allocation per call.

For chain_5k (5000 neighbor expansions): ~5000 fewer allocations.
"""

from typing import Callable, Dict, List, Optional
import heapq
from search.graph import Graph


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

    h_cache = graph._cs_h_cache
    if h_cache is None or graph._cs_h_goal != goal or graph._cs_h_precision != precision or graph._cs_h_fn is not heuristic:
        h_cache = [round(heuristic(inv[i], goal), precision) for i in range(N)]
        graph._cs_h_cache = h_cache
        graph._cs_h_goal = goal
        graph._cs_h_precision = precision
        graph._cs_h_fn = heuristic
        nb_f_offset = [[(nxt_i, wt, wt + h_cache[nxt_i]) for nxt_i, wt in nb] for nb in nb_idx]
        graph._cs_nb_f_offset = nb_f_offset
    else:
        nb_f_offset = graph._cs_nb_f_offset
        if nb_f_offset is None:
            nb_f_offset = [[(nxt_i, wt, wt + h_cache[nxt_i]) for nxt_i, wt in nb] for nb in nb_idx]
            graph._cs_nb_f_offset = nb_f_offset

    buckets: Dict[float, List[int]] = {}
    key_heap: List[float] = []
    f_start = h_cache[start_i]
    buckets[f_start] = [start_i]
    heapq.heappush(key_heap, f_start)

    g_score = [float('inf')] * N
    g_score[start_i] = 0.0
    pred = [-1] * N
    VISITED = float('-inf')

    _last_f = f_start
    _last_list = buckets[f_start]
    _append = _last_list.append

    while key_heap:
        f_key = heapq.heappop(key_heap)
        entries = buckets.pop(f_key, None)
        if entries is None:
            continue
        _last_f = f_key
        _last_list = entries
        _append = _last_list.append

        for node_i in entries:
            g = g_score[node_i]
            if g == VISITED:
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
                    if new_f == _last_f:
                        _append(nxt_i)
                    else:
                        try:
                            _last_list = buckets[new_f]
                            _append = _last_list.append
                            _append(nxt_i)
                        except KeyError:
                            _last_list = [nxt_i]
                            _append = _last_list.append
                            buckets[new_f] = _last_list
                            heapq.heappush(key_heap, new_f)
                        _last_f = new_f

    return None
