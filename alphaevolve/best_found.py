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
        nb = graph._cs_nb
    except AttributeError:
        nb = {}
        for node, edges in graph.adjacency.items():
            nb[node] = [(e.target, e.weight) for e in edges]
        graph._cs_nb = nb

    h_cache = {node: round(heuristic(node, goal), precision) for node in graph.adjacency}
    buckets: Dict[float, List] = {}
    f_start = h_cache[start]
    buckets[f_start] = [start]
    visited: Set[str] = set()
    visited_add = visited.add
    g_score: Dict[str, float] = {start: 0.0}
    pred: Dict[str, str] = {}
    current_min = f_start
    g_get = g_score.get
    INF = float("inf")

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
            visited_add(node)
            g = g_score[node]
            for nxt, wt in nb[node]:
                if nxt in visited:
                    continue
                new_g = g + wt
                if new_g < g_get(nxt, INF):
                    g_score[nxt] = new_g
                    pred[nxt] = node
                    new_f = new_g + h_cache[nxt]
                    if new_f < current_min:
                        current_min = new_f
                    bucket = buckets.get(new_f)
                    if bucket is None:
                        buckets[new_f] = [nxt]
                    else:
                        bucket.append(nxt)
    return None
