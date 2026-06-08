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

## Complexity

| Algorithm | Time | Space | Optimal? | Notes |
|-----------|------|-------|----------|-------|
| `bfs` | O(V + E) | O(V) | unweighted | Shortest path in unweighted |
| `dfs` | O(V + E) | O(V) | no | Finds any path, not shortest |
| `dijkstra` | O((V + E) log V) | O(V) | yes | Non-negative weights |
| `bellman_ford` | O(VE) | O(V) | yes | Handles negative; detects neg cycles |
| `spfa` | O(VE) worst / O(E) typical | O(V) | yes | Queue-based; fast on sparse |
| `dag_shortest_path` | O(V + E) | O(V) | yes | DAG only; fastest general SSSP |
| `astar` | O(E) typical / O(b^d) worst | O(V) | with admissible h | Heuristic-guided |
| `greedy_best_first` | O(E) typical | O(V) | no | Pure heuristic, no g-cost |
| `contour_search` | O(V + E) typical | O(V) | with admissible h | Bucket-based A* |
| `bidirectional_bfs` | O(b^(d/2)) | O(b^(d/2)) | unweighted | Meet-in-the-middle BFS |
| `bidirectional_dijkstra` | O((V + E) log V) | O(V) | yes | Meet-in-the-middle Dijkstra |
| `floyd_warshall` | O(V³) | O(V²) | yes | All-pairs; detect neg cycles |
| `johnson` | O(VE + V² log V) | O(V²) | yes | All-pairs on sparse graphs |
| `topological_sort` | O(V + E) | O(V) | — | Kahn's algorithm |
| `prim_mst` | O((V + E) log V) | O(V) | yes | Minimum spanning tree |
| `kruskal_mst` | O(E log E) | O(V + E) | yes | Sort-based MST |

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
| M61 | + cached tuple-based neighbors (strip `__dict__` access) | 1.80 ms | 4.95× |
| M63 | + precomputed heuristic cache (eliminate `round` + `h_fn` from hot loop) | 1.56 ms | 5.71× |
| **M68** | + int-indexed node storage (list/bytearray replace dict/set in hot loop) | **1.22 ms** | **7.30×** |

M61 converts `Edge` objects to `(target, weight)` tuples once (lazily cached on the graph object via `graph._cs_nb`), replacing `edge.target`/`edge.weight` `__dict__` lookups with C-level tuple unpacking in the hot loop. The cache is invalidated on graph mutation (`add_edge` deletes `_cs_*` attributes, triggering a rebuild on the next call).

M68 discards string-based dict/set operations in the inner loop entirely. Node names are mapped to integer indices (cached on the graph as `_cs_idx`/`_cs_inv`/`_cs_nb_idx`), and the hot loop uses `bytearray` for visited checks, `List[float]` for g-scores and heuristic cache, and `List[int]` for predecessors — replacing 4–5 dict operations per edge with C-level array accesses. This breaks through the micro-optimisation plateau that M63 hit, delivering 1.19×–1.42× per-benchmark over M63 (1.22 ms mean, 7.30× vs baseline).

Evolution framework in `alphaevolve/`:
- `evaluator.py` — benchmark harness (5 graph topologies, median timing)
- `mutations.py` — programmatic variants M0–M7
- `evolve.py` — LLM-driven evolutionary loop
- `best_found.py` — current champion (M68)
