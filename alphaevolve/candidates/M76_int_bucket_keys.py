from typing import Callable, Dict, List, Optional
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

    try:
        nb_idx = graph._cs_nb_idx
        idx = graph._cs_idx
        inv = graph._cs_inv
    except AttributeError:
        for attr in list(graph.__dict__):
            if attr.startswith('_cs_'):
                delattr(graph, attr)
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

    N = len(inv)
    start_i = idx[start]
    goal_i = idx[goal]

    MULT = 10 ** precision
    h_cache_int = [int(round(heuristic(inv[i], goal), precision) * MULT + 0.5) for i in range(N)]

    buckets: Dict[int, List[int]] = {}
    f_start_int = h_cache_int[start_i]
    buckets[f_start_int] = [start_i]

    visited = bytearray(N)
    g_score = [float('inf')] * N
    g_score[start_i] = 0.0
    pred = [-1] * N

    current_min = f_start_int
    _last_f_int = f_start_int
    _last_list = buckets[f_start_int]

    while buckets:
        if current_min not in buckets:
            current_min = min(buckets.keys())
        entries = buckets.pop(current_min)
        _last_f_int = current_min
        _last_list = entries
        for node_i in entries:
            if visited[node_i]:
                continue
            if node_i == goal_i:
                path = [inv[goal_i]]
                cur = node_i
                while pred[cur] != -1:
                    cur = pred[cur]
                    path.append(inv[cur])
                return path[::-1]
            visited[node_i] = 1
            g = g_score[node_i]
            for nxt_i, wt in nb_idx[node_i]:
                if visited[nxt_i]:
                    continue
                new_g = g + wt
                if new_g < g_score[nxt_i]:
                    g_score[nxt_i] = new_g
                    pred[nxt_i] = node_i
                    new_f_int = int((new_g + 0.0) * MULT + 0.5) + h_cache_int[nxt_i]
                    if new_f_int < current_min:
                        current_min = new_f_int
                    if new_f_int == _last_f_int:
                        _last_list.append(nxt_i)
                    else:
                        lst = buckets.get(new_f_int)
                        if lst is None:
                            lst = [nxt_i]
                            buckets[new_f_int] = lst
                        else:
                            lst.append(nxt_i)
                        _last_f_int = new_f_int
                        _last_list = lst
    return None
