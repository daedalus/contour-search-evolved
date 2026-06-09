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


def _bf_reconstruct(pred, inv, node_i, goal_i):
    path = [inv[goal_i]]
    cur = node_i
    while pred[cur] != -1:
        cur = pred[cur]
        path.append(inv[cur])
    return path[::-1]


def _bf_init_idx(graph):
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
    return idx, inv, nb_idx


def _bf_get_hcache(graph, N, inv, nb_idx, goal, heuristic, precision):
    h_cache = graph._cs_h_cache
    needs_build = (
        h_cache is None
        or graph._cs_h_goal != goal
        or graph._cs_h_precision != precision
        or graph._cs_h_fn is not heuristic
    )
    if needs_build or graph._cs_nb_f_offset is None:
        if needs_build:
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
    return h_cache, nb_f_offset


def _bf_prepare(graph, start, goal, heuristic, precision):
    idx, inv, nb_idx = _bf_init_idx(graph)

    N = len(inv)
    start_i = idx[start]
    goal_i = idx[goal]

    is_chain = graph._cs_is_chain
    if is_chain is None:
        is_chain = _is_chain(nb_idx)
        graph._cs_is_chain = is_chain

    if is_chain:
        return start_i, goal_i, nb_idx, inv, N, True, None, None

    h_cache, nb_f_offset = _bf_get_hcache(graph, N, inv, nb_idx, goal, heuristic, precision)
    return start_i, goal_i, nb_idx, inv, N, False, h_cache, nb_f_offset


def _bf_expand_node(node_i, g, nb_f_offset, buckets, key_heap, g_score, pred, goal_i, inv, _last_f, _last_list):
    if node_i == goal_i:
        return _bf_reconstruct(pred, inv, node_i, goal_i), _last_f, _last_list

    g_score[node_i] = float('-inf')

    for nxt_i, wt, f_offset in nb_f_offset[node_i]:
        new_g = g + wt
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

    return None, _last_f, _last_list


def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    result = _bf_prepare(graph, start, goal, heuristic, precision)
    start_i, goal_i, nb_idx, inv, N, is_chain, h_cache, nb_f_offset = result

    if is_chain:
        return _chain_search(start_i, goal_i, nb_idx, inv, N)

    buckets: Dict[float, List[int]] = {}
    key_heap: List[float] = []
    f_start = h_cache[start_i]
    buckets[f_start] = [start_i]
    heapq.heappush(key_heap, f_start)

    g_score = [float('inf')] * N
    g_score[start_i] = 0.0
    pred = [-1] * N

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
            if g == float('-inf'):
                continue

            expanded = _bf_expand_node(node_i, g, nb_f_offset, buckets, key_heap, g_score, pred, goal_i, inv, _last_f, _last_list)
            found, _last_f, _last_list = expanded
            if found is not None:
                return found

    return None
