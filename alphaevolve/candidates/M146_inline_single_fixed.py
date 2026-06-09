from typing import Callable, List, Optional
import heapq
from contour_search.graph import Graph
from contour_search.algorithms import _cs_prepare, _chain_search


def _cs_reconstruct(pred, inv, node_i, goal_i):
    path = [inv[goal_i]]
    cur = node_i
    while pred[cur] != -1:
        cur = pred[cur]
        path.append(inv[cur])
    return path[::-1]


def _cs_batch_expand(node_i, g, nb_f_offset, heap, g_score, pred, goal_i, inv):
    if node_i == goal_i:
        return _cs_reconstruct(pred, inv, node_i, goal_i)
    g_score[node_i] = float("-inf")
    nb_entries = nb_f_offset[node_i]
    if len(nb_entries) > 4:
        _batch_f = None
        _batch_g = None
        _batch_list = None
        _goal_pos = -1
        for nxt_i, wt, f_offset in nb_entries:
            new_g = g + wt
            if new_g < g_score[nxt_i]:
                g_score[nxt_i] = new_g
                pred[nxt_i] = node_i
                new_f = g + f_offset
                if _batch_f == new_f and _batch_g == new_g:
                    _batch_list.append(nxt_i)
                    if nxt_i == goal_i:
                        _goal_pos = len(_batch_list) - 1
                else:
                    if _batch_list is not None:
                        if _goal_pos > 0:
                            _batch_list[0], _batch_list[_goal_pos] = (
                                _batch_list[_goal_pos],
                                _batch_list[0],
                            )
                        heapq.heappush(heap, (_batch_f, -_batch_g, _batch_list))
                    _batch_f = new_f
                    _batch_g = new_g
                    _batch_list = [nxt_i]
                    _goal_pos = 0 if nxt_i == goal_i else -1
        if _batch_list is not None:
            if _goal_pos > 0:
                _batch_list[0], _batch_list[_goal_pos] = (
                    _batch_list[_goal_pos],
                    _batch_list[0],
                )
            heapq.heappush(heap, (_batch_f, -_batch_g, _batch_list))
    else:
        for nxt_i, wt, f_offset in nb_entries:
            new_g = g + wt
            if new_g < g_score[nxt_i]:
                g_score[nxt_i] = new_g
                pred[nxt_i] = node_i
                heapq.heappush(heap, (g + f_offset, -new_g, nxt_i))
    return None


def contour_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
    precision: int = 3,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    result = _cs_prepare(graph, start, goal, heuristic, precision)
    start_i, goal_i, nb_idx, inv, N, is_chain, h_cache, nb_f_offset = result

    if is_chain:
        return _chain_search(start_i, goal_i, nb_idx, inv, N)

    g_score = [float("inf")] * N
    g_score[start_i] = 0.0
    pred = [-1] * N

    _hp_push = heapq.heappush
    _hp_pop = heapq.heappop
    heap = [(h_cache[start_i], 0.0, start_i)]

    while heap:
        f_key, neg_g, entry = _hp_pop(heap)
        g = -neg_g

        if isinstance(entry, list):
            for node_i in entry:
                if g != g_score[node_i]:
                    continue
                found = _cs_batch_expand(
                    node_i, g, nb_f_offset, heap, g_score, pred, goal_i, inv
                )
                if found is not None:
                    return found
        else:
            if g != g_score[entry]:
                continue
            node_i = entry
            if node_i == goal_i:
                path = [inv[goal_i]]
                cur = node_i
                while pred[cur] != -1:
                    cur = pred[cur]
                    path.append(inv[cur])
                return path[::-1]
            g_score[node_i] = float("-inf")
            nb_e = nb_f_offset[node_i]
            if len(nb_e) > 4:
                found = _cs_batch_expand(
                    node_i, g, nb_f_offset, heap, g_score, pred, goal_i, inv
                )
                if found is not None:
                    return found
            else:
                for nxt_i, wt, f_offset in nb_e:
                    new_g = g + wt
                    if new_g < g_score[nxt_i]:
                        g_score[nxt_i] = new_g
                        pred[nxt_i] = node_i
                        _hp_push(heap, (g + f_offset, -new_g, nxt_i))

    return None
