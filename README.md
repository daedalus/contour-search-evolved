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

`contour_search` is a bucket-based A* variant that groups nodes by quantized f-values, expanding all nodes in the cheapest bucket before moving to the next. It was evolved through 6 rounds of AlphaEvolve optimization:

| Mutation | Change | Runtime | vs baseline |
|----------|--------|---------|-------------|
| Baseline | path-copy, `min(buckets.keys())` | 8.90 ms | 1.00× |
| M2 | predecessor map (no path copies) | 2.90 ms | 3.07× |
| M3 | + local variable bindings | 2.71 ms | 3.28× |
| M4 | + current_min tracking | 2.40 ms | 3.71× |
| M7 | + `.get`/`INF`/method-ref micro-ops | 2.39 ms | 3.73× |
| M61 | + cached tuple-based neighbors (strip `__dict__` access) | 1.80 ms | 4.95× |
| M63 | + precomputed heuristic cache (eliminate `round` + `h_fn` from hot loop) | 1.56 ms | 5.71× |
| **M68** | + int-indexed node storage (list/bytearray replace dict/set in hot loop) | 1.22 ms | 7.30× |
| **M72** | + cached bucket-list reference (eliminate `dict.get` from hot loop) | 0.974 ms | 9.14× |
| **M73** | + list-backed nb_idx (`List` replaces `Dict` for adjacency cache) | ~0.89 ms | ~10.0× |
| **M79** | + heap-ordered bucket keys (eliminates `current_min` + `min(buckets.keys())`) | **~0.63 ms** | **~14.1×** |

M61 converts `Edge` objects to `(target, weight)` tuples once (lazily cached on the graph object via `graph._cs_nb`), replacing `edge.target`/`edge.weight` `__dict__` lookups with C-level tuple unpacking in the hot loop. The cache is invalidated on graph mutation (`add_edge` deletes `_cs_*` attributes, triggering a rebuild on the next call).

M68 discards string-based dict/set operations in the inner loop entirely. Node names are mapped to integer indices (cached on the graph as `_cs_idx`/`_cs_inv`/`_cs_nb_idx`), and the hot loop uses `bytearray` for visited checks, `List[float]` for g-scores and heuristic cache, and `List[int]` for predecessors — replacing 4–5 dict operations per edge with C-level array accesses.

M72 caches the last-accessed bucket list reference (`_last_f`/`_last_list`), eliminating `dict.get` from the hot loop's common path. When consecutive node expansions produce the same f-value (the dominant case for well-informed heuristics), nodes are appended directly to the cached list — `dict.get`, the `if lst is None` check, and the `dict.__setitem__` path are all skipped. This kills the last dict operation in the hot loop: the profile shows `dict.pop` at 200 calls (from 1M) and `dict.get` at 0 calls across 200 iterations of chain_5k. Delivers 1.22× over M68 (0.974 ms mean, 9.14× vs baseline).

M73 stores the cached neighbor index (`_cs_nb_idx`) as a `List[Optional[List[Tuple[int, float]]]]` instead of `Dict[int, List[Tuple[int, float]]]`. After M68, node indices are already dense integers 0..N-1, so dict lookup by int key was already fast — but list subscript via `BINARY_SUBSCR` is still ~50ns faster per access. On grid_2500 (where every node expansion looks up its ~4 neighbors), this saves ~0.2ms per call (~10% improvement). On chains (2 neighbors per expansion), the effect is within noise.

### Signal vs noise

Every successful mutation removed a specific Python overhead from the per-edge hot loop. Mutations that reshuffled the same operations (try/except KeyError, defaultdict, int-quantized keys, heap min-tracking) all landed within noise or regressed:

| Mutation | Signal (removed from hot loop) | Gain |
|----------|--------------------------------|------|
| M2 | O(n) path copies → O(1) predecessor map | 3.07× |
| M3 | global lookups → local variable bindings | 1.07× |
| M4 | O(K) `min(buckets.keys())` → `current_min` tracking | 1.13× |
| M7 | per-call list allocation (`setdefault`) → `.get`+`if` | ~1.01× |
| M61 | per-edge `edge.target`/`edge.weight` `__dict__` probes → tuple unpacking | 1.25× |
| M63 | per-edge `round()` + `heuristic()` calls → `h_cache[nxt]` list read | 1.15× |
| M68 | string-keyed dict ops → int-indexed `bytearray`/array accesses | 1.28× |
| M72 | `dict.get` per edge → cached `_last_list` append | 1.25× |
| M73 | per-node `nb_idx[node_i]` dict → list subscript | ~1.10× |

The common pattern: every ≥1.15× win removed either a **hash-table operation** (dict get/set, `__dict__` probe) or a **Python-level allocation** (path copy, list alloc) from the per-edge path. The remaining hot loop is pure C-level array reads/writes and float math.

Evolution framework in `alphaevolve/`:
- `evaluator.py` — benchmark harness (5 graph topologies, median timing)
- `mutations.py` — programmatic variants M0–M7
- `evolve.py` — LLM-driven evolutionary loop
- `best_found.py` — current champion (M79)
- `candidates/` — historical candidates including failed mutations (M74–M78)
