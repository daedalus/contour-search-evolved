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
        graph._cs_h_cache = None
        graph._cs_h_goal = ""
        graph._cs_h_precision = 0
        graph._cs_h_fn = None
    else:
        nb_idx = graph._cs_nb_idx
        inv = graph._cs_inv

    N = len(inv)
    start_i = idx[start]
    goal_i = idx[goal]

    # Integer-only path: scale everything by 10**precision
    SCALE = 10 ** precision

    h_cache = graph._cs_h_cache
    if h_cache is None or graph._cs_h_goal != goal or graph._cs_h_precision != precision or graph._cs_h_fn is not heuristic:
        h_cache_float = [heuristic(inv[i], goal) for i in range(N)]
        # Store rounded scaled ints
        h_cache = [int(h_cache_float[i] * SCALE + (0.5 if h_cache_float[i] >= 0 else -0.5)) for i in range(N)]
        graph._cs_h_cache = h_cache
        graph._cs_h_goal = goal
        graph._cs_h_precision = precision
        graph._cs_h_fn = heuristic
        # Pre-compute scaled neighbor data: (nxt_i, wt_scaled, f_offset_scaled)
        nb_data = [None] * N
        for node_i in range(N):
            nb = nb_idx[node_i]
            nb_data[node_i] = [
                (nxt_i,
                 int(wt * SCALE + (0.5 if wt >= 0 else -0.5)),
                 int(wt * SCALE + (0.5 if wt >= 0 else -0.5)) + h_cache[nxt_i])
                for nxt_i, wt in nb
            ]
        graph._cs_nb_data = nb_data
    else:
        nb_data = graph._cs_nb_data
        if nb_data is None:
            h_cache_float = [heuristic(inv[i], goal) for i in range(N)]
            nb_data = [None] * N
            for node_i in range(N):
                nb = nb_idx[node_i]
                nb_data[node_i] = [
                    (nxt_i,
                     int(wt * SCALE + (0.5 if wt >= 0 else -0.5)),
                     int(wt * SCALE + (0.5 if wt >= 0 else -0.5)) + h_cache[nxt_i])
                    for nxt_i, wt in nb
                ]
            graph._cs_nb_data = nb_data

    BIG = 10 ** 12
    g_score = [BIG] * N
    g_score[start_i] = 0
    pred = [-1] * N
    VISITED = -1

    buckets: Dict[int, List[int]] = {}
    key_heap: List[int] = []
    f_start = h_cache[start_i]
    buckets[f_start] = [start_i]
    heapq.heappush(key_heap, f_start)

    _last_f = f_start
    _last_list = buckets[f_start]

    while key_heap:
        f_key = heapq.heappop(key_heap)
        entries = buckets.pop(f_key, None)
        if entries is None:
            continue
        _last_f = f_key
        _last_list = entries

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

            for nxt_i, wt_scaled, f_offset in nb_data[node_i]:
                new_g = g + wt_scaled
                if new_g < g_score[nxt_i]:
                    g_score[nxt_i] = new_g
                    pred[nxt_i] = node_i
                    new_f = g + f_offset
                    if new_f == _last_f:
                        _last_list.append(nxt_i)
                    else:
                        try:
                            _last_list = buckets[new_f]
                            _last_list.append(nxt_i)
                        except KeyError:
                            _last_list = [nxt_i]
                            buckets[new_f] = _last_list
                            heapq.heappush(key_heap, new_f)
                        _last_f = new_f

    return None
