./AGENTS.md

# contour-search-evolved

17 graph search algorithms (BFS/DFS/Dijkstra/A*/contour/etc.), plus evolutionary optimization of `contour_search` via the `alphaevolve` skill.

## Commands

```bash
# run all tests (262 total, ~10s)
pytest tests/ -q --timeout=10

# fast smoke (skip slow benchmarks)
pytest tests/ -q -m "not slow" --timeout=5

# benchmark champion vs all algorithms
python alphaevolve/benchmark_all.py

# evaluate a candidate mutation
python alphaevolve/run_evolution.py --candidate alphaevolve/candidates/M146_inline_single_fixed.py

# run evolution loop (needs LLM)
python alphaevolve/evolve.py --llm-cmd "..."
```

## Code Structure
- `src/contour_search/algorithms.py` — all 17 algorithms + private helpers (`_cs_*` for contour internals)

  - `src/contour_search/graph.py` — `Edge` dataclass, `Graph` dataclass (adjacency list)
- `alphaevolve/` — evolution framework: `evaluator.py` (scoring), `evolve.py` (LLM loop), `run_evolution.py` (loader/validator), `benchmark_all.py` (full comparison)
- `alphaevolve/candidates/` — M140 (prev champion), M146 (current champion)
- `alphaevolve/best_found.py` — thin re-export from `contour_search.algorithms`
- `tests/test_algorithms.py` — 203 tests, all algorithms
- `tests/test_contour_stress.py` — 59 stress tests, 20 random seeds vs Dijkstra

## Key Facts

- **Champion M146**: inline single-entry expansion + degree-bounded (≤4) batch dispatch. ~0.14ms geo-mean.
- **Interface**: all algorithms return `Optional[List[str]]` (path or None).
- **No external deps** except pytest. Pure Python, stdlib only (heapq, collections, math, typing).
- **No CI**, no build system, no requirements.txt.

## Editing Rules

- `contour_search` lives in `src/contour_search/algorithms.py`. Never break the public interface.
- Every mutation must pass `pytest tests/ -q -m "not slow"` before commit.
- Stale-entry check must be preserved for non-consistent heuristics.
- New candidates go in `alphaevolve/candidates/` with `M<number>_<description>.py` naming.
- Benchmark before promoting: `python alphaevolve/run_evolution.py -q --candidate <file>` for correctness; `python alphaevolve/evaluator.py <file>` for scoring.
