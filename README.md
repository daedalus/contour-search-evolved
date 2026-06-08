# hybrid-graph-search

17 graph search and shortest-path algorithms with a common `Optional[List[str]]` interface, plus evolutionary optimization of `contour_search` via AlphaEvolve.

## Algorithms

| Category | Algorithms |
|----------|------------|
| **Classic BFS/DFS** | `bfs`, `dfs` |
| **Shortest path** | `dijkstra`, `bellman_ford`, `spfa`, `dag_shortest_path` |
| **Heuristic** | `astar`, `greedy_best_first_search`, `contour_search` |
| **Bidirectional** | `bidirectional_bfs`, `bidirectional_dijkstra` |
| **All-pairs** | `floyd_warshall`, `johnson` |
| **MST** | `prim_mst`, `kruskal_mst` |
| **Ordering** | `topological_sort` |

## Usage

```python
from search import Graph, dijkstra

g = Graph()
g.add_edge("A", "B", weight=4)
g.add_edge("A", "C", weight=2)
g.add_edge("B", "D", weight=5)
g.add_edge("C", "D", weight=8)

path = dijkstra(g, "A", "D")
print(path)  # ['A', 'B', 'D']
```

Every algorithm returns `Optional[List[str]]` — a path from `start` to `goal`, or `None` if unreachable.

## Tests

```
pytest tests/ -q    # 262 tests (203 original + 59 contour-search stress tests)
```

## Contour Search & AlphaEvolve

`contour_search` is a bucket-based A* variant that groups nodes by quantized f-values, expanding all nodes in the cheapest bucket before moving to the next. It was evolved through 5 rounds of AlphaEvolve optimization:

| Mutation | Change | Runtime | vs baseline |
|----------|--------|---------|-------------|
| Baseline | path-copy, `min(buckets.keys())` | 8.90 ms | 1.00× |
| M2 | predecessor map (no path copies) | 2.90 ms | 3.07× |
| M3 | + local variable bindings | 2.71 ms | 3.28× |
| M4 | + current_min tracking | 2.40 ms | 3.71× |
| M7 | + `.get`/`INF`/method-ref micro-ops | 2.39 ms | 3.73× |
| **M61** | + cached tuple-based neighbors (strip `__dict__` access) | **1.80 ms** | **4.95×** |

M61 converts `Edge` objects to `(target, weight)` tuples once (lazily cached on the graph object via `graph._cs_nb`), replacing `edge.target`/`edge.weight` `__dict__` lookups with C-level tuple unpacking in the hot loop. The cache is invalidated on graph mutation (`add_edge` deletes `_cs_nb`, triggering a rebuild on the next call).

Evolution framework in `alphaevolve/`:
- `evaluator.py` — benchmark harness (5 graph topologies, median timing)
- `mutations.py` — programmatic variants M0–M61
- `evolve.py` — LLM-driven evolutionary loop
- `best_found.py` — current champion
