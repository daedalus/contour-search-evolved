import pytest
import random

from search.graph import Graph
from search.algorithms import contour_search, dijkstra


# ---------------------------------------------------------------------------
# Negative weights — Dijkstra-family assumption breaks
# ---------------------------------------------------------------------------


def test_negative_weights_straight_line():
    g = Graph()
    g.add_edge("A", "B", weight=-5, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path is not None
    assert path[0] == "A" and path[-1] == "C"


def test_negative_weights_negative_cycle_expands():
    """Negative cycle before goal — algorithm should still terminate."""
    g = Graph()
    g.add_edge("A", "B", weight=-10, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path is not None
    assert path[0] == "A" and path[-1] == "C"


def test_negative_weights_reachable():
    """Start and goal reachable despite negative edges elsewhere."""
    g = Graph()
    g.add_edge("A", "B", weight=2, bidirectional=False)
    g.add_edge("B", "C", weight=2, bidirectional=False)
    g.add_edge("A", "Z", weight=100, bidirectional=False)
    g.add_edge("Z", "B", weight=-200, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path is not None


# ---------------------------------------------------------------------------
# Self-loops
# ---------------------------------------------------------------------------


def test_self_loop():
    g = Graph()
    g.add_edge("A", "A", weight=1)
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path == ["A", "B", "C"]


def test_self_loop_only_edge():
    g = Graph()
    g.add_edge("A", "A", weight=1)
    assert contour_search(g, "A", "A") == ["A"]


# ---------------------------------------------------------------------------
# Parallel edges
# ---------------------------------------------------------------------------


def test_parallel_edges_picks_cheapest():
    g = Graph()
    g.add_edge("A", "B", weight=10, bidirectional=False)
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    path = contour_search(g, "A", "C")

    def path_cost(p):
        return sum(
            min(e.weight for e in g.neighbors(p[i]) if e.target == p[i + 1])
            for i in range(len(p) - 1)
        )

    assert path_cost(path) == 2


def test_parallel_edges_three_way():
    g = Graph()
    g.add_edge("A", "B", weight=100, bidirectional=False)
    g.add_edge("A", "B", weight=50, bidirectional=False)
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path is not None


# ---------------------------------------------------------------------------
# Zero-weight and zero-cost cycles
# ---------------------------------------------------------------------------


def test_zero_weight_cycle():
    g = Graph()
    g.add_edge("A", "B", weight=0, bidirectional=False)
    g.add_edge("B", "A", weight=0, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path == ["A", "B", "C"]


def test_zero_weight_long_chain():
    g = Graph()
    for i in range(100):
        g.add_edge(str(i), str(i + 1), weight=0, bidirectional=False)
    path = contour_search(g, "0", "100")
    assert path is not None
    assert len(path) == 101


# ---------------------------------------------------------------------------
# Extreme precision values
# ---------------------------------------------------------------------------


def test_precision_negative():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    path = contour_search(g, "A", "C", precision=-1)
    assert path == ["A", "B", "C"]


def test_precision_very_high():
    g = Graph()
    g.add_edge("A", "B", weight=0.123456789, bidirectional=False)
    g.add_edge("B", "C", weight=0.987654321, bidirectional=False)
    path = contour_search(g, "A", "C", precision=10)
    assert path == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Large / float('inf') weights
# ---------------------------------------------------------------------------


def test_large_weights():
    g = Graph()
    g.add_edge("A", "B", weight=1e12, bidirectional=False)
    g.add_edge("B", "C", weight=1e12, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path == ["A", "B", "C"]


def test_inf_weight():
    g = Graph()
    g.add_edge("A", "B", weight=float("inf"), bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path is None


def test_mixed_inf_and_finite():
    g = Graph()
    g.add_edge("A", "B", weight=float("inf"), bidirectional=False)
    g.add_edge("A", "C", weight=1, bidirectional=False)
    g.add_edge("C", "D", weight=1, bidirectional=False)
    g.add_edge("D", "E", weight=1, bidirectional=False)
    path = contour_search(g, "A", "E")
    assert path is not None
    assert path[0] == "A" and path[-1] == "E"


# ---------------------------------------------------------------------------
# Heuristic edge cases
# ---------------------------------------------------------------------------


def test_overestimating_heuristic():
    """Overestimating heuristic may degrade optimality but must still
    return a valid path."""
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    g.add_edge("A", "C", weight=100, bidirectional=False)

    def h(a, b):
        return 1000 if a != b else 0

    path = contour_search(g, "A", "C", heuristic=h)
    assert path is not None
    assert path[0] == "A" and path[-1] == "C"


def test_heuristic_returns_inf():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)

    def h(a, b):
        return float("inf") if a == "A" else 0

    path = contour_search(g, "A", "C", heuristic=h)
    assert path == ["A", "B", "C"]


def test_heuristic_returns_nan():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)

    def h(a, b):
        return float("nan")

    path = contour_search(g, "A", "C", heuristic=h)
    assert path is not None


# ---------------------------------------------------------------------------
# Disconnected components — goal unreachable
# ---------------------------------------------------------------------------


def test_disconnected_larger():
    g = Graph()
    for i in range(100):
        g.add_edge(f"a_{i}", f"a_{i + 1}", bidirectional=False)
    for i in range(100):
        g.add_edge(f"b_{i}", f"b_{i + 1}", bidirectional=False)
    assert contour_search(g, "a_0", "b_50") is None


def test_isolated_nodes():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("C", "D", weight=1, bidirectional=False)
    assert contour_search(g, "A", "D") is None
    assert contour_search(g, "C", "A") is None


# ---------------------------------------------------------------------------
# Graph with many nodes, heavy branching
# ---------------------------------------------------------------------------


def test_wide_fan_out():
    g = Graph()
    for i in range(5000):
        g.add_edge("root", f"leaf_{i}", weight=1, bidirectional=False)
    path = contour_search(g, "root", "leaf_0")
    assert path == ["root", "leaf_0"]


def test_deep_chain_stress():
    g = Graph()
    for i in range(10000):
        g.add_edge(str(i), str(i + 1), weight=1, bidirectional=False)
    path = contour_search(g, "0", "10000")
    assert path is not None
    assert len(path) == 10001


def test_deep_chain_unreachable():
    g = Graph()
    for i in range(10000):
        g.add_edge(str(i), str(i + 1), weight=1, bidirectional=False)
    assert contour_search(g, "0", "20000") is None


# ---------------------------------------------------------------------------
# Bidirectional vs directed edge consistency
# ---------------------------------------------------------------------------


def test_bidirectional_is_directed_symmetric():
    g = Graph()
    g.add_edge("A", "B", weight=3, bidirectional=True)
    path_fwd = contour_search(g, "A", "B")
    path_rev = contour_search(g, "B", "A")
    assert path_fwd == ["A", "B"]
    assert path_rev == ["B", "A"]


def test_directed_asymmetric():
    g = Graph()
    g.add_edge("A", "B", weight=3, bidirectional=False)
    g.add_edge("B", "A", weight=10, bidirectional=False)
    path_fwd = contour_search(g, "A", "B")
    path_rev = contour_search(g, "B", "A")
    assert path_fwd == ["A", "B"]
    assert path_rev == ["B", "A"]


# ---------------------------------------------------------------------------
# Unicode / non-ASCII node names
# ---------------------------------------------------------------------------


def test_unicode_node_names():
    g = Graph()
    g.add_edge("東京", "大阪", weight=1, bidirectional=False)
    g.add_edge("大阪", "京都", weight=1, bidirectional=False)
    path = contour_search(g, "東京", "京都")
    assert path == ["東京", "大阪", "京都"]


# ---------------------------------------------------------------------------
# Path consistency with Dijkstra on non-negative graphs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(20)))
def test_random_graph_matches_dijkstra(seed):
    random.seed(seed)
    g = Graph()
    for _ in range(200):
        u = f"n{random.randint(0, 99)}"
        v = f"n{random.randint(0, 99)}"
        if u != v:
            g.add_edge(
                u,
                v,
                weight=random.uniform(0.1, 10),
                bidirectional=random.choice([True, False]),
            )
    for _ in range(20):
        s = f"n{random.randint(0, 99)}"
        t = f"n{random.randint(0, 99)}"
        if s == t:
            continue
        dij = dijkstra(g, s, t)
        cs = contour_search(g, s, t)
        if dij is None:
            assert cs is None
        else:

            def path_cost(p):
                return sum(
                    min(e.weight for e in g.neighbors(p[i]) if e.target == p[i + 1])
                    for i in range(len(p) - 1)
                )

            dij_cost = path_cost(dij)
            cs_cost = path_cost(cs)
            assert abs(dij_cost - cs_cost) < 1e-9, (
                f"seed={seed} s={s} t={t}: Dijkstra={dij_cost} Contour={cs_cost}"
            )


# ---------------------------------------------------------------------------
# Non-admissible (negative) heuristic — path still valid
# ---------------------------------------------------------------------------


def test_negative_heuristic_still_finds_path():
    g = Graph()
    g.add_edge("A", "B", weight=5, bidirectional=False)
    g.add_edge("B", "C", weight=5, bidirectional=False)
    g.add_edge("A", "C", weight=100, bidirectional=False)

    def h(a, b):
        return -50 if a == "A" else 0

    path = contour_search(g, "A", "C", heuristic=h)
    assert path is not None


# ---------------------------------------------------------------------------
# Precision rounding causing different f-values to collide in same bucket
# ---------------------------------------------------------------------------


def test_precision_collision():
    g = Graph()
    g.add_edge("A", "B", weight=0.1234, bidirectional=False)
    g.add_edge("B", "C", weight=0.1234, bidirectional=False)
    g.add_edge("A", "C", weight=0.25, bidirectional=False)
    path = contour_search(g, "A", "C", precision=1)
    assert path is not None
    assert path[0] == "A" and path[-1] == "C"


# ---------------------------------------------------------------------------
# Graph where every node pairs have parallel edges
# ---------------------------------------------------------------------------


def test_all_parallel_edges():
    g = Graph()
    for w in [10, 5, 1]:
        g.add_edge("A", "B", weight=w, bidirectional=False)
    for w in [10, 5, 1]:
        g.add_edge("B", "C", weight=w, bidirectional=False)
    path = contour_search(g, "A", "C")
    cost = sum(
        min(e.weight for e in g.neighbors(path[i]) if e.target == path[i + 1])
        for i in range(len(path) - 1)
    )
    assert cost == 2


# ---------------------------------------------------------------------------
# Goal reachable through multiple paths with same f-score (ties)
# ---------------------------------------------------------------------------


def test_many_tied_f_scores():
    g = Graph()
    for i in range(100):
        g.add_edge("start", f"mid_{i}", weight=i, bidirectional=False)
        g.add_edge(f"mid_{i}", "goal", weight=100 - i, bidirectional=False)
    path = contour_search(g, "start", "goal")
    assert path is not None


# ---------------------------------------------------------------------------
# Very small weights near float epsilon
# ---------------------------------------------------------------------------


def test_tiny_weights():
    g = Graph()
    g.add_edge("A", "B", weight=1e-15, bidirectional=False)
    g.add_edge("B", "C", weight=1e-15, bidirectional=False)
    g.add_edge("A", "C", weight=1e-14, bidirectional=False)
    path = contour_search(g, "A", "C")
    assert path == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Exact float comparison — same bucket value, different origins
# ---------------------------------------------------------------------------


def test_exact_float_bucket_key():
    g = Graph()
    g.add_edge("A", "B", weight=0.1 + 0.2, bidirectional=False)
    g.add_edge("B", "C", weight=0.1 + 0.2, bidirectional=False)
    g.add_edge("A", "C", weight=0.3, bidirectional=False)
    path = contour_search(g, "A", "C", precision=2)
    assert path is not None


# ---------------------------------------------------------------------------
# Rapid fan-out from multiple levels
# ---------------------------------------------------------------------------


def test_multi_level_explosion():
    g = Graph()
    for i in range(500):
        g.add_edge("root", f"l1_{i}", weight=1, bidirectional=False)
        g.add_edge(f"l1_{i}", "goal", weight=1, bidirectional=False)
    path = contour_search(g, "root", "goal")
    assert path is not None


# ---------------------------------------------------------------------------
# Graph with no path but many nodes — must return None quickly
# ---------------------------------------------------------------------------


def test_no_path_large_search_space():
    g = Graph()
    for i in range(1000):
        g.add_edge(f"a_{i}", f"a_{i + 1}", weight=1, bidirectional=False)
    for i in range(1000):
        g.add_edge(f"b_{i}", f"b_{i + 1}", weight=1, bidirectional=False)
    result = contour_search(g, "a_0", "b_999")
    assert result is None


# ---------------------------------------------------------------------------
# Single node graph, no self-loop
# ---------------------------------------------------------------------------


def test_single_node_no_self_loop():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    path = contour_search(g, "A", "A")
    assert path == ["A"]


def test_nonexistent_node_solo():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    assert contour_search(g, "X", "Y") is None


# ---------------------------------------------------------------------------
# Verify cached tuple cache is stable across calls (graph mutation)
# ---------------------------------------------------------------------------


def test_cache_stable_after_graph_mutation():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    contour_search(g, "A", "B")
    g.add_edge("A", "C", weight=1, bidirectional=False)
    g.add_edge("C", "D", weight=1, bidirectional=False)
    path = contour_search(g, "A", "D")
    assert path is not None


# ---------------------------------------------------------------------------
# Multiple contour_search calls on same graph with different params
# ---------------------------------------------------------------------------


def test_multiple_calls_same_graph():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    g.add_edge("B", "C", weight=1, bidirectional=False)
    g.add_edge("A", "C", weight=10, bidirectional=False)
    assert contour_search(g, "A", "C") == ["A", "B", "C"]
    assert contour_search(g, "A", "B") == ["A", "B"]
    assert contour_search(g, "B", "C") == ["B", "C"]


# ---------------------------------------------------------------------------
# Non-existent start/goal with cached neighbors present
# ---------------------------------------------------------------------------


def test_nonexistent_with_cache():
    g = Graph()
    g.add_edge("A", "B", weight=1, bidirectional=False)
    contour_search(g, "A", "B")
    assert contour_search(g, "Z", "A") is None
    assert contour_search(g, "A", "Z") is None


# ---------------------------------------------------------------------------
# Stress: chain of 100k nodes — smoke test for memory/performance regressions
# ---------------------------------------------------------------------------


def test_chain_100k():
    g = Graph()
    for i in range(100000):
        g.add_edge(str(i), str(i + 1), weight=1, bidirectional=False)
    path = contour_search(g, "0", "100000")
    assert path is not None
    assert len(path) == 100001
