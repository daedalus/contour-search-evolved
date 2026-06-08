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
        nb_idx = {}
        for node, edges in graph.adjacency.items():
            node_i = idx[node]
            nb_idx[node_i] = [(idx[e.target], e.weight) for e in edges]
        graph._cs_idx = idx
        graph._cs_inv = inv
        graph._cs_nb_idx = nb_idx

    N = len(inv)
    start_i = idx[start]
    goal_i = idx[goal]

    h_cache = [round(heuristic(inv[i], goal), precision) for i in range(N)]

    buckets: Dict[float, List[int]] = {}
    f_start = h_cache[start_i]
    buckets[f_start] = [start_i]

    g_score = [float('inf')] * N
    g_score[start_i] = 0.0
    pred = [-1] * N
    VISITED = -1.0

    current_min = f_start
    _last_f = f_start
    _last_list = buckets[f_start]

    while buckets:
        if current_min not in buckets:
            current_min = min(buckets.keys())
        entries = buckets.pop(current_min)
        _last_f = current_min
        _last_list = entries
        for node_i in entries:
            if g_score[node_i] < 0:
                continue
            if node_i == goal_i:
                path = [inv[goal_i]]
                cur = node_i
                while pred[cur] != -1:
                    cur = pred[cur]
                    path.append(inv[cur])
                return path[::-1]
            g = g_score[node_i]
            g_score[node_i] = VISITED
            for nxt_i, wt in nb_idx[node_i]:
                if g_score[nxt_i] < 0:
                    continue
                new_g = g + wt
                if new_g < g_score[nxt_i]:
                    g_score[nxt_i] = new_g
                    pred[nxt_i] = node_i
                    new_f = new_g + h_cache[nxt_i]
                    if new_f < current_min:
                        current_min = new_f
                    if new_f == _last_f:
                        _last_list.append(nxt_i)
                    else:
                        lst = buckets.get(new_f)
                        if lst is None:
                            lst = [nxt_i]
                            buckets[new_f] = lst
                        else:
                            lst.append(nxt_i)
                        _last_f = new_f
                        _last_list = lst
    return None
