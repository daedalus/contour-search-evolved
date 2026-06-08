from typing import Callable, Deque, Dict, List, Optional
from collections import deque
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

    h_cache = [round(heuristic(inv[i], goal), precision) for i in range(N)]

    try:
        _cs_buckets = graph._cs_buckets
        _cs_key_order = graph._cs_key_order
        _cs_visited = graph._cs_visited
        _cs_g_score = graph._cs_g_score
        _cs_pred = graph._cs_pred
        _cs_N = graph._cs_N
        gen = graph._cs_gen
        gen += 1
        graph._cs_gen = gen
        if gen <= 0:
            _cs_visited = bytearray(N)
            _cs_g_score = [float('inf')] * N
            _cs_pred = [-1] * N
            graph._cs_visited = _cs_visited
            graph._cs_g_score = _cs_g_score
            graph._cs_pred = _cs_pred
            graph._cs_gen = 1
            gen = 1
    except AttributeError:
        _cs_buckets = {}
        _cs_key_order = deque()
        _cs_visited = bytearray(N)
        _cs_g_score = [float('inf')] * N
        _cs_pred = [-1] * N
        _cs_N = N
        graph._cs_buckets = _cs_buckets
        graph._cs_key_order = _cs_key_order
        graph._cs_visited = _cs_visited
        graph._cs_g_score = _cs_g_score
        graph._cs_pred = _cs_pred
        graph._cs_N = _cs_N
        graph._cs_gen = 0
        gen = 0

    if N != _cs_N:
        _cs_visited = bytearray(N)
        _cs_g_score = [float('inf')] * N
        _cs_pred = [-1] * N
        _cs_N = N
        graph._cs_visited = _cs_visited
        graph._cs_g_score = _cs_g_score
        graph._cs_pred = _cs_pred
        graph._cs_N = _cs_N

    _cs_key_order.clear()
    _cs_buckets.clear()
    _cs_visited[:] = b'\x00' * N

    f_start = h_cache[start_i]
    _cs_key_order.append(f_start)
    _cs_buckets[f_start] = [start_i]
    _cs_g_score[start_i] = 0.0

    g_score = _cs_g_score
    visited = _cs_visited
    pred = _cs_pred
    buckets = _cs_buckets
    key_order = _cs_key_order

    _last_f = f_start
    _last_list = buckets[f_start]

    if gen > 0:
        for i in range(N):
            g_score[i] = float('inf')
            pred[i] = -1
        g_score[start_i] = 0.0

    while key_order:
        f_key = key_order.popleft()
        entries = buckets.pop(f_key)
        _last_f = f_key
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
                    new_f = new_g + h_cache[nxt_i]
                    if new_f == _last_f:
                        _last_list.append(nxt_i)
                    else:
                        lst = buckets.get(new_f)
                        if lst is None:
                            lst = [nxt_i]
                            buckets[new_f] = lst
                            key_order.append(new_f)
                        else:
                            lst.append(nxt_i)
                        _last_f = new_f
                        _last_list = lst

    return None
