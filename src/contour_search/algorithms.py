from __future__ import annotations

import heapq
from collections import deque
from typing import Callable, Dict, List, Optional, Set

from contour_search.graph import Edge, Graph


def _bellman_relax(nodes, dist, pred, graph):
    updated = False
    for u in nodes:
        if dist[u] == float("inf"):
            continue
        for edge in graph.neighbors(u):
            nd = dist[u] + edge.weight
            if nd < dist[edge.target]:
                dist[edge.target] = nd
                pred[edge.target] = u
                updated = True
    return updated


def _bellman_has_neg_cycle(nodes, dist, graph):
    for u in nodes:
        if dist[u] == float("inf"):
            continue
        for edge in graph.neighbors(u):
            if dist[u] + edge.weight < dist[edge.target]:
                return True
    return False


def _reconstruct_path_dict(pred, goal):
    path: List[str] = []
    curr: Optional[str] = goal
    while curr is not None:
        path.append(curr)
        curr = pred[curr]
    return path[::-1]


def bellman_ford(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    nodes = graph.nodes()
    dist: Dict[str, float] = {n: float("inf") for n in nodes}
    pred: Dict[str, Optional[str]] = {n: None for n in nodes}
    dist[start] = 0.0

    for _ in range(len(nodes) - 1):
        if not _bellman_relax(nodes, dist, pred, graph):
            break

    if _bellman_has_neg_cycle(nodes, dist, graph):
        return None
    if dist[goal] == float("inf"):
        return None

    return _reconstruct_path_dict(pred, goal)


def bidirectional_bfs(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None
    if start == goal:
        return [start]

    forward_q: deque = deque([start])
    backward_q: deque = deque([goal])
    forward_visited: Dict[str, Optional[str]] = {start: None}
    backward_visited: Dict[str, Optional[str]] = {goal: None}

    def _reconstruct(meet: str) -> List[str]:
        path: List[str] = []
        curr: Optional[str] = meet
        while curr is not None:
            path.append(curr)
            curr = forward_visited[curr]
        path.reverse()
        curr = backward_visited[meet]
        while curr is not None:
            path.append(curr)
            curr = backward_visited[curr]
        return path

    while forward_q and backward_q:
        for _ in range(len(forward_q)):
            node = forward_q.popleft()
            for edge in graph.neighbors(node):
                if edge.target not in forward_visited:
                    forward_visited[edge.target] = node
                    if edge.target in backward_visited:
                        return _reconstruct(edge.target)
                    forward_q.append(edge.target)

        for _ in range(len(backward_q)):
            node = backward_q.popleft()
            for edge in graph.neighbors(node):
                if edge.target not in backward_visited:
                    backward_visited[edge.target] = node
                    if edge.target in forward_visited:
                        return _reconstruct(edge.target)
                    backward_q.append(edge.target)

    return None


def _bd_reconstruct(meet, forward_pred, backward_pred):
    path: List[str] = []
    curr: Optional[str] = meet
    while curr is not None:
        path.append(curr)
        curr = forward_pred[curr]
    path.reverse()
    curr = backward_pred[meet]
    while curr is not None:
        path.append(curr)
        curr = backward_pred[curr]
    return path


def _bd_forward_expand(
    f_node,
    f_dist,
    graph,
    forward_pq,
    forward_dist,
    forward_pred,
    backward_visited,
    backward_dist,
    backward_pred,
    best_cost,
    best_path,
):
    for edge in graph.neighbors(f_node):
        nd = f_dist + edge.weight
        if nd < forward_dist.get(edge.target, float("inf")):
            forward_dist[edge.target] = nd
            forward_pred[edge.target] = f_node
            heapq.heappush(forward_pq, (nd, edge.target))
            if edge.target in backward_visited or edge.target in backward_dist:
                total = nd + backward_dist.get(edge.target, float("inf"))
                if total < best_cost:
                    best_cost = total
                    best_path = _bd_reconstruct(
                        edge.target, forward_pred, backward_pred
                    )
    return best_cost, best_path


def _bd_backward_expand(
    b_node,
    b_dist,
    graph,
    backward_pq,
    backward_dist,
    backward_pred,
    forward_visited,
    forward_dist,
    forward_pred,
    best_cost,
    best_path,
):
    for edge in graph.neighbors(b_node):
        nd = b_dist + edge.weight
        if nd < backward_dist.get(edge.target, float("inf")):
            backward_dist[edge.target] = nd
            backward_pred[edge.target] = b_node
            heapq.heappush(backward_pq, (nd, edge.target))
            if edge.target in forward_visited or edge.target in forward_dist:
                total = nd + forward_dist.get(edge.target, float("inf"))
                if total < best_cost:
                    best_cost = total
                    best_path = _bd_reconstruct(
                        edge.target, forward_pred, backward_pred
                    )
    return best_cost, best_path


def _bd_try_meet(node, dist, opp_dist, opp_pred, pred, best_cost, best_path):
    if node in opp_pred:
        total = dist + opp_dist[node]
        if total < best_cost:
            return total, _bd_reconstruct(node, pred, opp_pred)
    return best_cost, best_path


def bidirectional_dijkstra(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None
    if start == goal:
        return [start]

    forward_pq: List[tuple] = [(0.0, start)]
    backward_pq: List[tuple] = [(0.0, goal)]
    forward_dist: Dict[str, float] = {start: 0.0}
    backward_dist: Dict[str, float] = {goal: 0.0}
    forward_pred: Dict[str, Optional[str]] = {start: None}
    backward_pred: Dict[str, Optional[str]] = {goal: None}
    forward_visited: Set[str] = set()
    backward_visited: Set[str] = set()
    best_path: Optional[List[str]] = None
    best_cost: float = float("inf")

    while forward_pq and backward_pq:
        if forward_pq[0][0] + backward_pq[0][0] >= best_cost:
            break

        f_dist, f_node = heapq.heappop(forward_pq)
        forward_visited.add(f_node)
        best_cost, best_path = _bd_try_meet(
            f_node,
            f_dist,
            backward_dist,
            backward_pred,
            forward_pred,
            best_cost,
            best_path,
        )
        if f_dist + backward_dist.get(f_node, float("inf")) <= best_cost:
            best_cost, best_path = _bd_forward_expand(
                f_node,
                f_dist,
                graph,
                forward_pq,
                forward_dist,
                forward_pred,
                backward_visited,
                backward_dist,
                backward_pred,
                best_cost,
                best_path,
            )

        b_dist, b_node = heapq.heappop(backward_pq)
        backward_visited.add(b_node)
        best_cost, best_path = _bd_try_meet(
            b_node,
            b_dist,
            forward_dist,
            forward_pred,
            backward_pred,
            best_cost,
            best_path,
        )
        if b_dist + forward_dist.get(b_node, float("inf")) <= best_cost:
            best_cost, best_path = _bd_backward_expand(
                b_node,
                b_dist,
                graph,
                backward_pq,
                backward_dist,
                backward_pred,
                forward_visited,
                forward_dist,
                forward_pred,
                best_cost,
                best_path,
            )

    return best_path


def greedy_best_first_search(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    pq: List[tuple] = [(heuristic(start, goal), start, [start])]
    visited: Set[str] = set()

    while pq:
        _, node, path = heapq.heappop(pq)

        if node == goal:
            return path

        if node in visited:
            continue
        visited.add(node)

        for edge in graph.neighbors(node):
            if edge.target not in visited:
                heapq.heappush(
                    pq,
                    (heuristic(edge.target, goal), edge.target, [*path, edge.target]),
                )

    return None


def spfa(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    nodes = graph.nodes()
    dist: Dict[str, float] = {n: float("inf") for n in nodes}
    pred: Dict[str, Optional[str]] = {n: None for n in nodes}
    in_queue: Dict[str, bool] = {n: False for n in nodes}
    count: Dict[str, int] = {n: 0 for n in nodes}
    dist[start] = 0.0

    q: deque = deque([start])
    in_queue[start] = True
    count[start] = 1

    while q:
        u = q.popleft()
        in_queue[u] = False

        for edge in graph.neighbors(u):
            nd = dist[u] + edge.weight
            if nd < dist[edge.target]:
                dist[edge.target] = nd
                pred[edge.target] = u
                if not in_queue[edge.target]:
                    q.append(edge.target)
                    in_queue[edge.target] = True
                    count[edge.target] += 1
                    if count[edge.target] >= len(nodes):
                        return None

    if dist[goal] == float("inf"):
        return None

    path: List[str] = []
    curr: Optional[str] = goal
    while curr is not None:
        path.append(curr)
        curr = pred[curr]
    return path[::-1]


def topological_sort(graph: Graph) -> Optional[List[str]]:
    nodes = graph.nodes()
    indegree: Dict[str, int] = {n: 0 for n in nodes}

    for u in nodes:
        for edge in graph.neighbors(u):
            indegree[edge.target] = indegree.get(edge.target, 0) + 1

    q: deque = deque([n for n in nodes if indegree[n] == 0])
    result: List[str] = []

    while q:
        u = q.popleft()
        result.append(u)
        for edge in graph.neighbors(u):
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                q.append(edge.target)

    return result if len(result) == len(nodes) else None


def dag_shortest_path(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    order = topological_sort(graph)
    if order is None:
        return None

    dist: Dict[str, float] = {n: float("inf") for n in order}
    pred: Dict[str, Optional[str]] = {n: None for n in order}
    dist[start] = 0.0

    for u in order:
        if dist[u] == float("inf"):
            continue
        for edge in graph.neighbors(u):
            nd = dist[u] + edge.weight
            if nd < dist[edge.target]:
                dist[edge.target] = nd
                pred[edge.target] = u

    if dist[goal] == float("inf"):
        return None

    path: List[str] = []
    curr: Optional[str] = goal
    while curr is not None:
        path.append(curr)
        curr = pred[curr]
    return path[::-1]


def prim_mst(graph: Graph, start: str = "") -> Optional[List[tuple[str, str, float]]]:
    nodes = graph.nodes()
    if not nodes:
        return None

    if not start:
        start = nodes[0]
    if start not in graph.adjacency:
        return None

    visited: Set[str] = set()
    pq: List[tuple] = [(0.0, start, start)]
    mst: List[tuple[str, str, float]] = []

    while pq and len(visited) < len(nodes):
        weight, u, v = heapq.heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        if u != v:
            mst.append((u, v, weight))
        for edge in graph.neighbors(v):
            if edge.target not in visited:
                heapq.heappush(pq, (edge.weight, v, edge.target))

    return mst if len(visited) == len(nodes) else None


def _find(parent: Dict[str, str], x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: Dict[str, str], rank: Dict[str, int], x: str, y: str) -> bool:
    rx, ry = _find(parent, x), _find(parent, y)
    if rx == ry:
        return False
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1
    return True


def kruskal_mst(graph: Graph) -> Optional[List[tuple[str, str, float]]]:
    nodes = graph.nodes()
    if not nodes:
        return None

    edges: List[tuple] = []
    seen: Set[tuple[str, str]] = set()
    for u in nodes:
        for edge in graph.neighbors(u):
            key = (u, edge.target) if u <= edge.target else (edge.target, u)
            if key not in seen:
                seen.add(key)
                edges.append((edge.weight, u, edge.target))

    edges.sort(key=lambda x: x[0])
    parent: Dict[str, str] = {n: n for n in nodes}
    rank: Dict[str, int] = {n: 0 for n in nodes}
    mst: List[tuple[str, str, float]] = []

    for weight, u, v in edges:
        if _union(parent, rank, u, v):
            mst.append((u, v, weight))

    root = _find(parent, nodes[0])
    return mst if all(_find(parent, n) == root for n in nodes) else None


def floyd_warshall(graph: Graph) -> Optional[Dict[str, Dict[str, float]]]:
    nodes = graph.nodes()
    if not nodes:
        return None

    dist: Dict[str, Dict[str, float]] = {
        u: {v: float("inf") for v in nodes} for u in nodes
    }
    for n in nodes:
        dist[n][n] = 0.0
    for u in nodes:
        for edge in graph.neighbors(u):
            if edge.weight < dist[u][edge.target]:
                dist[u][edge.target] = edge.weight

    for k in nodes:
        dk = dist[k]
        for i in nodes:
            dik = dist[i][k]
            if dik == float("inf"):
                continue
            di = dist[i]
            for j in nodes:
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd

    for i in nodes:
        if dist[i][i] < 0:
            return None

    return dist


def _johnson_potentials(nodes, graph):
    extra = "__johnson_source__"
    g = Graph()
    g.adjacency = {n: list(graph.neighbors(n)) for n in nodes}
    g.adjacency[extra] = [Edge(n, 0.0) for n in nodes]
    for n in nodes:
        g.adjacency.setdefault(n, [])

    h: Dict[str, float] = {n: float("inf") for n in list(nodes) + [extra]}
    h[extra] = 0.0
    all_nodes = [extra] + nodes
    for _ in range(len(all_nodes) - 1):
        updated = False
        for u in all_nodes:
            if h[u] == float("inf"):
                continue
            for edge in g.neighbors(u):
                nd = h[u] + edge.weight
                if nd < h[edge.target]:
                    h[edge.target] = nd
                    updated = True
        if not updated:
            break

    for u in all_nodes:
        if h[u] == float("inf"):
            continue
        for edge in g.neighbors(u):
            if h[u] + edge.weight < h[edge.target]:
                return None
    return h


def _johnson_dijkstra(src, nodes, reweighted, h):
    pq: List[tuple] = [(0.0, src)]
    d: Dict[str, float] = {n: float("inf") for n in nodes}
    d[src] = 0.0
    visited: Set[str] = set()
    while pq:
        cd, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for edge in reweighted.neighbors(u):
            nd = cd + edge.weight
            if nd < d[edge.target]:
                d[edge.target] = nd
                heapq.heappush(pq, (nd, edge.target))
    return {n: d[n] + h[n] - h[src] for n in nodes}


def johnson(graph: Graph) -> Optional[Dict[str, Dict[str, float]]]:
    nodes = graph.nodes()
    if not nodes:
        return None

    h = _johnson_potentials(nodes, graph)
    if h is None:
        return None

    reweighted = Graph()
    for u in nodes:
        for edge in graph.neighbors(u):
            rw = edge.weight + h[u] - h[edge.target]
            reweighted.add_edge(u, edge.target, weight=rw, bidirectional=False)

    return {src: _johnson_dijkstra(src, nodes, reweighted, h) for src in nodes}


def bfs(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    queue: deque = deque([[start]])
    visited: Set[str] = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        for edge in graph.neighbors(node):
            if edge.target not in visited:
                visited.add(edge.target)
                queue.append([*path, edge.target])

    return None


def dfs(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    stack: List[List[str]] = [[start]]
    visited: Set[str] = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node in visited:
            continue
        visited.add(node)

        for edge in graph.neighbors(node):
            if edge.target not in visited:
                stack.append([*path, edge.target])

    return None


def dijkstra(graph: Graph, start: str, goal: str) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    pq: List[tuple] = [(0.0, start, [start])]
    visited: Set[str] = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node == goal:
            return path

        if node in visited:
            continue
        visited.add(node)

        for edge in graph.neighbors(node):
            if edge.target not in visited:
                heapq.heappush(
                    pq, (cost + edge.weight, edge.target, [*path, edge.target])
                )

    return None


def _is_chain(nb_idx) -> bool:
    deg1 = 0
    for nb in nb_idx:
        d = len(nb)
        if d > 2:
            return False
        if d == 1:
            deg1 += 1
    return deg1 == 2


def _chain_search(
    start_i: int, goal_i: int, nb_idx, inv, N: int
) -> Optional[List[str]]:
    if start_i == goal_i:
        return [inv[start_i]]
    for init_dir in range(len(nb_idx[start_i])):
        nxt_i, w = nb_idx[start_i][init_dir]
        if w == float("inf"):
            continue
        path = [None] * N
        path[0] = inv[start_i]
        pi = 1
        prev, cur = start_i, nxt_i
        path[pi] = inv[cur]
        pi += 1
        while cur != goal_i:
            nb = nb_idx[cur]
            nxt_i = None
            for n, w in nb:
                if w != float("inf") and n != prev:
                    nxt_i = n
                    break
            if nxt_i is None:
                break
            prev, cur = cur, nxt_i
            path[pi] = inv[cur]
            pi += 1
        else:
            return path[:pi]  # type: ignore[return-value]
    return None


def _cs_reconstruct(pred, inv, node_i, goal_i):
    path = [inv[goal_i]]
    cur = node_i
    while pred[cur] != -1:
        cur = pred[cur]
        path.append(inv[cur])
    return path[::-1]


def _cs_process_single(
    node_i, g, nb_f_offset, heap, g_score, pred, goal_i, inv, _hp_push
):
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
            node_i, g, nb_f_offset, heap, g_score, pred, goal_i, inv, _hp_push
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


def _cs_batch_expand(
    node_i, g, nb_f_offset, heap, g_score, pred, goal_i, inv, _hp_push
):
    if node_i == goal_i:
        return _cs_reconstruct(pred, inv, node_i, goal_i)
    g_score[node_i] = float("-inf")
    _batch_f = None
    _batch_g = None
    _batch_list = None
    _goal_pos = -1
    for nxt_i, wt, f_offset in nb_f_offset[node_i]:
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
                    _hp_push(heap, (_batch_f, -_batch_g, _batch_list))
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
        _hp_push(heap, (_batch_f, -_batch_g, _batch_list))
    return None


def _cs_init_idx(graph):
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


def _cs_get_hcache(graph, N, inv, nb_idx, goal, heuristic, precision):
    h_cache = graph._cs_h_cache
    # Identity check (is not) avoids calling __eq__ on every call but will
    # spuriously rebuild if the user passes a freshly-instantiated callable
    # each time (e.g., a lambda in a loop or a per-call bound method).
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
        nb_f_offset = [
            [(nxt_i, wt, wt + h_cache[nxt_i]) for nxt_i, wt in nb] for nb in nb_idx
        ]
        for lst in nb_f_offset:
            lst.sort(key=lambda x: x[2])
        graph._cs_nb_f_offset = nb_f_offset
    else:
        nb_f_offset = graph._cs_nb_f_offset
    return h_cache, nb_f_offset


def _cs_prepare(graph, start, goal, heuristic, precision):
    idx, inv, nb_idx = _cs_init_idx(graph)

    N = len(inv)
    start_i = idx[start]
    goal_i = idx[goal]

    is_chain = graph._cs_is_chain
    if is_chain is None:
        is_chain = _is_chain(nb_idx)
        graph._cs_is_chain = is_chain

    if is_chain:
        return start_i, goal_i, nb_idx, inv, N, True, None, None

    h_cache, nb_f_offset = _cs_get_hcache(
        graph, N, inv, nb_idx, goal, heuristic, precision
    )
    return start_i, goal_i, nb_idx, inv, N, False, h_cache, nb_f_offset


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
                    node_i, g, nb_f_offset, heap, g_score, pred, goal_i, inv, _hp_push
                )
                if found is not None:
                    return found
        else:
            if g != g_score[entry]:
                continue
            found = _cs_process_single(
                entry, g, nb_f_offset, heap, g_score, pred, goal_i, inv, _hp_push
            )
            if found is not None:
                return found

    return None


def astar(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
) -> Optional[List[str]]:
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None

    pq: List[tuple] = [(heuristic(start, goal), 0.0, start, [start])]
    visited: Set[str] = set()

    while pq:
        f_score, g_score, node, path = heapq.heappop(pq)

        if node == goal:
            return path

        if node in visited:
            continue
        visited.add(node)

        for edge in graph.neighbors(node):
            if edge.target not in visited:
                new_g = g_score + edge.weight
                new_f = new_g + heuristic(edge.target, goal)
                heapq.heappush(pq, (new_f, new_g, edge.target, [*path, edge.target]))

    return None
