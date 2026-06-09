# contour-search-evolved

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
| `contour_search` | O(V + E) typical | O(V) | with admissible h | Single-heap A* with batch-push |
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

In A* terminology, nodes sharing the same `f = g + h` value form a **contour** — like elevation lines on a topographic map (see Nilsson, *Problem-Solving Methods in Artificial Intelligence*, 1971; Russell & Norvig, *Artificial Intelligence: A Modern Approach*). `contour_search` exploits this by batch-pushing all neighbors with identical `(f, -g)` as a single heap entry and expanding them together on pop, processing the search space contour-by-contour rather than node-by-node.

It is a single-heap A\* variant that combines stale-entry checking (avoids re-expanding nodes discovered with worse g) with degree-gated batch-push (neighbors sharing identical `(f, -g)` are pushed as a single heap entry). It was evolved through multiple rounds of AlphaEvolve optimization (using the `alphaevolve` opencode skill):

| Mutation | Change | Geo-mean | vs baseline |
|----------|--------|----------|-------------|
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
| **M79** | + heap-ordered bucket keys (eliminates `current_min` + `min(buckets.keys())`) | ~0.63 ms | ~14.1× |
| **M89** | + -inf visited sentinel (replaces visited set with g-score check) | ~0.60 ms | ~14.9× |
| **M95** | + cached h_cache on graph object (avoid rebuild across calls) | ~0.57 ms | ~15.6× |
| **M104** | + try/except KeyError buckets + slotted Graph | ~0.50 ms | ~17.8× |
| **M107** | + f_offset cache + `pop(f_key, None)` optimization | ~0.45 ms | ~19.8× |
| **M118** | + chain topology fast-path (precomputed chain detection) | ~0.38 ms | ~23.4× |
| **M119** | + neighbor sort by f-offset + chain fast-path | 0.356 ms | ~25× |
| **M137** | + single flat heap + stale-entry check (replaces two-level bucket dict) | 0.157 ms | ~57× |
| **M138** | + degree-gated batch-push (fix star regression, push equal-priority neighbors as list) | 0.147 ms | ~61× |
| **M140** | + goal-first batch ordering (O(1) goal detection in batch) | ~0.127 ms | ~70× |
| **M146** | + inline single expansion + degree-bounded batch dispatch | ~0.143 ms | ~62× |

Current champion (`M146`): geo-mean ~0.14ms (chain_1k=0.29, chain_5k=1.59, star_500=0.057, grid_2500=0.091, dense_80=0.025), score ~7000, 262 tests passing. Further mutations are at the noise floor (~5–8% measurement stdev), making optimizations below ~10% unreliable.

M61 converts `Edge` objects to `(target, weight)` tuples once (lazily cached on the graph object via `graph._cs_nb`), replacing `edge.target`/`edge.weight` `__dict__` lookups with C-level tuple unpacking in the hot loop. The cache is invalidated on graph mutation (`add_edge` deletes `_cs_*` attributes, triggering a rebuild on the next call).

M68 discards string-based dict/set operations in the inner loop entirely. Node names are mapped to integer indices (cached on the graph as `_cs_idx`/`_cs_inv`/`_cs_nb_idx`), and the hot loop uses `bytearray` for visited checks, `List[float]` for g-scores and heuristic cache, and `List[int]` for predecessors — replacing 4–5 dict operations per edge with C-level array accesses.

M72 caches the last-accessed bucket list reference (`_last_f`/`_last_list`), eliminating `dict.get` from the hot loop's common path. When consecutive node expansions produce the same f-value (the dominant case for well-informed heuristics), nodes are appended directly to the cached list — `dict.get`, the `if lst is None` check, and the `dict.__setitem__` path are all skipped.

M73 stores the cached neighbor index (`_cs_nb_idx`) as a `List[Optional[List[Tuple[int, float]]]]` instead of `Dict[int, List[Tuple[int, float]]]`. After M68, node indices are already dense integers 0..N-1, so list subscript via `BINARY_SUBSCR` is still ~50ns faster per access.

M137 replaces the two-level bucket dict + key_heap + `_last_f`/`_last_list` with a single heap `(f, -g, node_i)` and a stale-entry check `if g != g_score[node_i]: continue`. This eliminates bucket management overhead (dict pop, the `_last_f` caching machinery) and ensures each node is expanded exactly once. Grid_2500 drops from 0.895ms to 0.070ms (−92%). However, on star topologies (uniform g, all nodes equal priority), the per-node heap operations regress ~2× vs the bucket approach.

M138 fixes the star regression: when expanding a high-fanout node (degree > 4), neighbors with identical `(f, -g)` are collected into a local list and pushed as a single heap entry. On pop, batch entries are detected via `isinstance(entry, list)` and iterated directly. star_500 recovers from 0.243ms to 0.106ms, grid_2500 stays at 0.076ms (within noise of M137).

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

## Benchmarks (median, ms)

```
                     chain_1k  chain_5k  star_500  grid_2500  dense_80
contour_search          0.293     1.586     0.057     0.091     0.025
dag_shortest_path       0.198     1.093     0.095     0.852     0.341
bidir_bfs               0.283     1.638     0.036     0.746     0.006
spfa                    0.600     3.277     0.267     2.018     0.491
```

Evolution framework in `alphaevolve/`:
- `benchmark_all.py` — full comparison of 9 single-source algorithms (30-run timed)
- `evaluator.py` — benchmark harness (5 graph topologies, geo-mean scoring)
- `evolve.py` — LLM-driven evolutionary loop
- `run_evolution.py` — load/verify/benchmark candidate mutations
- `candidates/` — M140 (previous champion) and M146 (current champion)
