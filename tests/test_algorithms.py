import time

import pytest

from contour_search.graph import Edge, Graph
from contour_search.algorithms import (
    astar,
    bellman_ford,
    bfs,
    bidirectional_bfs,
    bidirectional_dijkstra,
    contour_search,
    dag_shortest_path,
    dfs,
    dijkstra,
    floyd_warshall,
    greedy_best_first_search,
    johnson,
    kruskal_mst,
    prim_mst,
    spfa,
    topological_sort,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_graph() -> Graph:
    g = Graph()
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("B", "E")
    g.add_edge("C", "F")
    g.add_edge("E", "F")
    return g


@pytest.fixture
def weighted_graph() -> Graph:
    g = Graph()
    g.add_edge("A", "B", weight=4)
    g.add_edge("A", "C", weight=2)
    g.add_edge("B", "C", weight=1)
    g.add_edge("B", "D", weight=5)
    g.add_edge("C", "D", weight=8)
    g.add_edge("C", "E", weight=10)
    g.add_edge("D", "E", weight=2)
    g.add_edge("D", "F", weight=6)
    g.add_edge("E", "F", weight=3)
    return g


@pytest.fixture
def disconnected_graph() -> Graph:
    g = Graph()
    g.add_edge("A", "B")
    g.add_edge("C", "D")
    return g


@pytest.fixture
def single_node_graph() -> Graph:
    g = Graph()
    g.add_edge("A", "A")
    return g


@pytest.fixture
def two_node_graph() -> Graph:
    g = Graph()
    g.add_edge("X", "Y")
    return g


@pytest.fixture
def cyclic_graph() -> Graph:
    g = Graph()
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")
    return g


@pytest.fixture
def self_loop_graph() -> Graph:
    g = Graph()
    g.add_edge("A", "B", bidirectional=False)
    g.add_edge("B", "B", bidirectional=False)
    g.add_edge("B", "C", bidirectional=False)
    return g


@pytest.fixture
def zero_weight_graph() -> Graph:
    g = Graph()
    g.add_edge("A", "B", weight=0)
    g.add_edge("B", "C", weight=0)
    return g


@pytest.fixture
def negative_weight_graph() -> Graph:
    g = Graph()
    g.add_edge("A", "B", weight=5)
    g.add_edge("A", "C", weight=10)
    g.add_edge("B", "C", weight=-15)
    return g


@pytest.fixture
def empty_graph() -> Graph:
    return Graph()


@pytest.fixture
def large_chain_graph() -> Graph:
    g = Graph()
    for i in range(999):
        g.add_edge(str(i), str(i + 1), bidirectional=False)
    return g


@pytest.fixture
def star_graph() -> Graph:
    g = Graph()
    for i in range(1, 500):
        g.add_edge("center", str(i), bidirectional=False)
    return g


@pytest.fixture
def grid_graph() -> Graph:
    g = Graph()
    size = 50
    for r in range(size):
        for c in range(size):
            node = f"{r},{c}"
            if c + 1 < size:
                right = f"{r},{c + 1}"
                g.add_edge(node, right, weight=1, bidirectional=False)
            if r + 1 < size:
                down = f"{r + 1},{c}"
                g.add_edge(node, down, weight=1, bidirectional=False)
    return g


@pytest.fixture
def dense_graph() -> Graph:
    g = Graph()
    nodes = [str(i) for i in range(80)]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            g.add_edge(nodes[i], nodes[j], weight=abs(i - j), bidirectional=False)
    return g


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


class TestBFS:
    def test_finds_path(self, simple_graph: Graph):
        path = bfs(simple_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_shortest_path_unweighted(self, simple_graph: Graph):
        path = bfs(simple_graph, "A", "F")
        assert path == ["A", "C", "F"]

    def test_shortest_path_minimal_edges(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        path = bfs(g, "A", "D")
        assert len(path) == 3

    def test_no_path(self, disconnected_graph: Graph):
        assert bfs(disconnected_graph, "A", "C") is None

    def test_same_node(self, simple_graph: Graph):
        assert bfs(simple_graph, "A", "A") == ["A"]

    def test_missing_goal(self, simple_graph: Graph):
        assert bfs(simple_graph, "A", "Z") is None

    def test_missing_start(self, simple_graph: Graph):
        assert bfs(simple_graph, "Z", "A") is None

    def test_both_missing(self, simple_graph: Graph):
        assert bfs(simple_graph, "Z", "Z") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert bfs(empty_graph, "A", "B") is None

    def test_single_node_same(self, single_node_graph: Graph):
        assert bfs(single_node_graph, "A", "A") == ["A"]

    def test_two_nodes_direct(self, two_node_graph: Graph):
        assert bfs(two_node_graph, "X", "Y") == ["X", "Y"]

    def test_cycle_does_not_loop(self, cyclic_graph: Graph):
        path = bfs(cyclic_graph, "A", "C")
        assert path == ["A", "C"]

    def test_self_loop_edge(self, self_loop_graph: Graph):
        path = bfs(self_loop_graph, "A", "C")
        assert path == ["A", "B", "C"]

    def test_start_not_in_graph_returns_none(self):
        g = Graph()
        g.add_edge("B", "C")
        assert bfs(g, "A", "C") is None

    def test_goal_not_in_graph_returns_none(self):
        g = Graph()
        g.add_edge("A", "B")
        assert bfs(g, "A", "C") is None

    def test_disconnected_with_extra_nodes(self):
        g = Graph()
        for i in range(5):
            g.add_edge(f"a{i}", f"a{i + 1}")
            g.add_edge(f"b{i}", f"b{i + 1}")
        assert bfs(g, "a0", "b3") is None


# ---------------------------------------------------------------------------
# DFS
# ---------------------------------------------------------------------------


class TestDFS:
    def test_finds_path(self, simple_graph: Graph):
        path = dfs(simple_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_no_path(self, disconnected_graph: Graph):
        assert dfs(disconnected_graph, "A", "C") is None

    def test_same_node(self, simple_graph: Graph):
        assert dfs(simple_graph, "A", "A") == ["A"]

    def test_missing_goal(self, simple_graph: Graph):
        assert dfs(simple_graph, "A", "Z") is None

    def test_missing_start(self, simple_graph: Graph):
        assert dfs(simple_graph, "Z", "A") is None

    def test_both_missing(self, simple_graph: Graph):
        assert dfs(simple_graph, "Z", "Z") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert dfs(empty_graph, "A", "B") is None

    def test_single_node_same(self, single_node_graph: Graph):
        assert dfs(single_node_graph, "A", "A") == ["A"]

    def test_two_nodes_direct(self, two_node_graph: Graph):
        assert dfs(two_node_graph, "X", "Y") == ["X", "Y"]

    def test_cycle_does_not_loop(self, cyclic_graph: Graph):
        path = dfs(cyclic_graph, "A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_self_loop_edge(self, self_loop_graph: Graph):
        path = dfs(self_loop_graph, "A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_start_not_in_graph(self):
        g = Graph()
        g.add_edge("B", "C")
        assert dfs(g, "A", "C") is None

    def test_goal_not_in_graph(self):
        g = Graph()
        g.add_edge("A", "B")
        assert dfs(g, "A", "C") is None

    def test_path_not_necessarily_shortest(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        path = dfs(g, "A", "D")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "D"

    def test_deep_linear_path(self):
        g = Graph()
        for i in range(100):
            g.add_edge(str(i), str(i + 1), bidirectional=False)
        path = dfs(g, "0", "100")
        assert len(path) == 101


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------


class TestDijkstra:
    def test_finds_shortest_path(self, weighted_graph: Graph):
        path = dijkstra(weighted_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_optimal_cost(self, weighted_graph: Graph):
        path = dijkstra(weighted_graph, "A", "F")
        cost = sum(
            next(
                e.weight
                for e in weighted_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        assert cost == 13

    def test_prefers_cheaper_path(self, weighted_graph: Graph):
        path = dijkstra(weighted_graph, "A", "D")
        cost = sum(
            next(
                e.weight
                for e in weighted_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        assert cost == 8

    def test_zero_weight_edges(self, zero_weight_graph: Graph):
        path = dijkstra(zero_weight_graph, "A", "C")
        cost = sum(
            next(
                e.weight
                for e in zero_weight_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        assert cost == 0

    def test_no_path(self, disconnected_graph: Graph):
        assert dijkstra(disconnected_graph, "A", "C") is None

    def test_same_node(self, weighted_graph: Graph):
        assert dijkstra(weighted_graph, "A", "A") == ["A"]

    def test_missing_goal(self, weighted_graph: Graph):
        assert dijkstra(weighted_graph, "A", "Z") is None

    def test_missing_start(self, weighted_graph: Graph):
        assert dijkstra(weighted_graph, "Z", "A") is None

    def test_both_missing(self, weighted_graph: Graph):
        assert dijkstra(weighted_graph, "Z", "Z") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert dijkstra(empty_graph, "A", "B") is None

    def test_single_node_same(self, single_node_graph: Graph):
        assert dijkstra(single_node_graph, "A", "A") == ["A"]

    def test_two_nodes_direct(self, two_node_graph: Graph):
        assert dijkstra(two_node_graph, "X", "Y") == ["X", "Y"]

    def test_cycle_does_not_loop(self, cyclic_graph: Graph):
        path = dijkstra(cyclic_graph, "A", "C")
        assert path is not None

    def test_self_loop_edge(self, self_loop_graph: Graph):
        path = dijkstra(self_loop_graph, "A", "C")
        assert path is not None

    def test_start_not_in_graph(self):
        g = Graph()
        g.add_edge("B", "C")
        assert dijkstra(g, "A", "C") is None

    def test_goal_not_in_graph(self):
        g = Graph()
        g.add_edge("A", "B")
        assert dijkstra(g, "A", "C") is None

    def test_large_weights(self):
        g = Graph()
        g.add_edge("A", "B", weight=1e9)
        g.add_edge("B", "C", weight=1e9)
        path = dijkstra(g, "A", "C")
        cost = sum(
            next(e.weight for e in g.neighbors(path[i]) if e.target == path[i + 1])
            for i in range(len(path) - 1)
        )
        assert cost == 2e9

    def test_negative_weights_produces_incorrect_result(
        self, negative_weight_graph: Graph
    ):
        path = dijkstra(negative_weight_graph, "A", "C")
        cost = sum(
            next(
                e.weight
                for e in negative_weight_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        assert cost != 0

    def test_multi_edge_between_nodes(self):
        g = Graph()
        g.adjacency.setdefault("A", [])
        g.adjacency.setdefault("B", [])
        g.adjacency["A"].append(Edge("B", 10))
        g.adjacency["A"].append(Edge("B", 1))
        g.adjacency["B"].append(Edge("A", 10))
        g.adjacency["B"].append(Edge("A", 1))
        path = dijkstra(g, "A", "B")
        assert path == ["A", "B"]
        best = min(e.weight for e in g.neighbors("A") if e.target == "B")
        assert best == 1


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------


class TestAStar:
    def test_finds_path(self, weighted_graph: Graph):
        path = astar(weighted_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_with_zero_heuristic_matches_dijkstra(self, weighted_graph: Graph):
        astar_path = astar(weighted_graph, "A", "F")
        dij_path = dijkstra(weighted_graph, "A", "F")
        assert astar_path == dij_path

    def test_with_heuristic(self, weighted_graph: Graph):
        def h(a, b):
            return 0.0

        path = astar(weighted_graph, "A", "F", heuristic=h)
        assert path is not None

    def test_no_path(self, disconnected_graph: Graph):
        assert astar(disconnected_graph, "A", "C") is None

    def test_same_node(self, weighted_graph: Graph):
        assert astar(weighted_graph, "A", "A") == ["A"]

    def test_missing_goal(self, weighted_graph: Graph):
        assert astar(weighted_graph, "A", "Z") is None

    def test_missing_start(self, weighted_graph: Graph):
        assert astar(weighted_graph, "Z", "A") is None

    def test_both_missing(self, weighted_graph: Graph):
        assert astar(weighted_graph, "Z", "Z") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert astar(empty_graph, "A", "B") is None

    def test_single_node_same(self, single_node_graph: Graph):
        assert astar(single_node_graph, "A", "A") == ["A"]

    def test_two_nodes_direct(self, two_node_graph: Graph):
        assert astar(two_node_graph, "X", "Y") == ["X", "Y"]

    def test_cycle_does_not_loop(self, cyclic_graph: Graph):
        path = astar(cyclic_graph, "A", "C")
        assert path is not None

    def test_self_loop_edge(self, self_loop_graph: Graph):
        path = astar(self_loop_graph, "A", "C")
        assert path is not None

    def test_start_not_in_graph(self):
        g = Graph()
        g.add_edge("B", "C")
        assert astar(g, "A", "C") is None

    def test_goal_not_in_graph(self):
        g = Graph()
        g.add_edge("A", "B")
        assert astar(g, "A", "C") is None

    def test_heuristic_improves_performance(self):
        g = Graph()
        for i in range(200):
            g.add_edge(str(i), str(i + 1), weight=1, bidirectional=False)

        t0 = time.perf_counter()
        astar(g, "0", "200", heuristic=lambda a, b: 0)
        t_no_heuristic = time.perf_counter() - t0

        t0 = time.perf_counter()
        astar(g, "0", "200", heuristic=lambda a, b: abs(int(a) - int(b)))
        t_with_heuristic = time.perf_counter() - t0

        assert t_with_heuristic <= t_no_heuristic * 1.1 + 0.01

    def test_admissible_heuristic_optimal_path(self):
        g = Graph()
        for i in range(10):
            g.add_edge(str(i), str(i + 1), weight=2, bidirectional=False)

        def h(a, b):
            return abs(int(a) - int(b))

        dij_path = dijkstra(g, "0", "10")
        astar_path = astar(g, "0", "10", heuristic=h)
        assert astar_path == dij_path

    def test_zero_weight_edges(self, zero_weight_graph: Graph):
        path = astar(zero_weight_graph, "A", "C")
        assert path is not None


# ---------------------------------------------------------------------------
# Contour Search
# ---------------------------------------------------------------------------


class TestContourSearch:
    def test_finds_path(self, weighted_graph: Graph):
        path = contour_search(weighted_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_matches_astar_default_precision(self, weighted_graph: Graph):
        cs = contour_search(weighted_graph, "A", "F")
        a = astar(weighted_graph, "A", "F")
        assert cs == a

    def test_coarse_precision_still_correct(self, weighted_graph: Graph):
        cs = contour_search(weighted_graph, "A", "D", precision=0)
        cost = sum(
            next(
                e.weight
                for e in weighted_graph.neighbors(cs[i])
                if e.target == cs[i + 1]
            )
            for i in range(len(cs) - 1)
        )
        assert cost == 8

    def test_no_path(self, disconnected_graph: Graph):
        assert contour_search(disconnected_graph, "A", "C") is None

    def test_same_node(self, weighted_graph: Graph):
        assert contour_search(weighted_graph, "A", "A") == ["A"]

    def test_missing_goal(self, weighted_graph: Graph):
        assert contour_search(weighted_graph, "A", "Z") is None

    def test_missing_start(self, weighted_graph: Graph):
        assert contour_search(weighted_graph, "Z", "A") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert contour_search(empty_graph, "A", "B") is None

    def test_two_nodes_direct(self, two_node_graph: Graph):
        assert contour_search(two_node_graph, "X", "Y") == ["X", "Y"]

    def test_cycle_does_not_loop(self, cyclic_graph: Graph):
        path = contour_search(cyclic_graph, "A", "C")
        assert path is not None

    def test_zero_weight_edges(self, zero_weight_graph: Graph):
        path = contour_search(zero_weight_graph, "A", "C")
        assert path is not None

    def test_admissible_heuristic_optimal_path(self):
        g = Graph()
        for i in range(10):
            g.add_edge(str(i), str(i + 1), weight=2, bidirectional=False)

        def h(a, b):
            return abs(int(a) - int(b))

        dij_path = dijkstra(g, "0", "10")
        cs_path = contour_search(g, "0", "10", heuristic=h)
        assert cs_path == dij_path

    def test_matches_self_with_different_precision(self, weighted_graph: Graph):
        cs1 = contour_search(weighted_graph, "A", "F", precision=1)
        cs3 = contour_search(weighted_graph, "A", "F", precision=3)
        assert cs1 == cs3


# ---------------------------------------------------------------------------
# Bellman-Ford
# ---------------------------------------------------------------------------


class TestBellmanFord:
    def test_finds_path(self, weighted_graph: Graph):
        path = bellman_ford(weighted_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_optimal_cost(self, weighted_graph: Graph):
        path = bellman_ford(weighted_graph, "A", "F")
        cost = sum(
            next(
                e.weight
                for e in weighted_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        assert cost == 13

    def test_matches_dijkstra_on_positive_weights(self, weighted_graph: Graph):
        bf = bellman_ford(weighted_graph, "A", "F")
        dj = dijkstra(weighted_graph, "A", "F")
        assert bf == dj

    def test_handles_negative_weights(self):
        g = Graph()
        g.add_edge("A", "B", weight=4, bidirectional=False)
        g.add_edge("A", "C", weight=2, bidirectional=False)
        g.add_edge("B", "C", weight=-3, bidirectional=False)
        g.add_edge("C", "D", weight=1, bidirectional=False)
        path = bellman_ford(g, "A", "D")
        assert path == ["A", "B", "C", "D"]
        cost = sum(
            next(e.weight for e in g.neighbors(path[i]) if e.target == path[i + 1])
            for i in range(len(path) - 1)
        )
        assert cost == 2

    def test_detects_negative_cycle(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        g.add_edge("B", "C", weight=-10, bidirectional=False)
        g.add_edge("C", "B", weight=-10, bidirectional=False)
        g.add_edge("C", "D", weight=1, bidirectional=False)
        assert bellman_ford(g, "A", "D") is None

    def test_detects_negative_cycle_self_loop(self):
        g = Graph()
        g.add_edge("A", "B", weight=1, bidirectional=False)
        g.add_edge("B", "B", weight=-1, bidirectional=False)
        g.add_edge("B", "C", weight=1, bidirectional=False)
        assert bellman_ford(g, "A", "C") is None

    def test_negative_cycle_unreachable(self):
        g = Graph()
        g.add_edge("A", "B", weight=1, bidirectional=False)
        g.add_edge("X", "Y", weight=-10, bidirectional=False)
        g.add_edge("Y", "X", weight=-10, bidirectional=False)
        path = bellman_ford(g, "A", "B")
        assert path == ["A", "B"]

    def test_no_path(self, disconnected_graph: Graph):
        assert bellman_ford(disconnected_graph, "A", "C") is None

    def test_same_node(self, weighted_graph: Graph):
        assert bellman_ford(weighted_graph, "A", "A") == ["A"]

    def test_missing_goal(self, weighted_graph: Graph):
        assert bellman_ford(weighted_graph, "A", "Z") is None

    def test_missing_start(self, weighted_graph: Graph):
        assert bellman_ford(weighted_graph, "Z", "A") is None

    def test_both_missing(self, weighted_graph: Graph):
        assert bellman_ford(weighted_graph, "Z", "Z") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert bellman_ford(empty_graph, "A", "B") is None

    def test_single_node_same(self, single_node_graph: Graph):
        assert bellman_ford(single_node_graph, "A", "A") == ["A"]

    def test_two_nodes_direct(self, two_node_graph: Graph):
        assert bellman_ford(two_node_graph, "X", "Y") == ["X", "Y"]

    def test_cycle_does_not_loop(self, cyclic_graph: Graph):
        path = bellman_ford(cyclic_graph, "A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_zero_weight_edges(self, zero_weight_graph: Graph):
        path = bellman_ford(zero_weight_graph, "A", "C")
        assert path is not None

    def test_start_not_in_graph(self):
        g = Graph()
        g.add_edge("B", "C")
        assert bellman_ford(g, "A", "C") is None

    def test_goal_not_in_graph(self):
        g = Graph()
        g.add_edge("A", "B")
        assert bellman_ford(g, "A", "C") is None

    def test_single_edge(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        assert bellman_ford(g, "A", "B") == ["A", "B"]

    def test_bellman_ford_beats_dijkstra_on_negatives(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        g.add_edge("A", "C", weight=10, bidirectional=False)
        g.add_edge("B", "D", weight=1, bidirectional=False)
        g.add_edge("C", "B", weight=-6, bidirectional=False)
        g.add_edge("D", "E", weight=1, bidirectional=False)
        bf = bellman_ford(g, "A", "E")
        dj = dijkstra(g, "A", "E")
        assert bf is not None and dj is not None
        bf_cost = sum(
            next(e.weight for e in g.neighbors(bf[i]) if e.target == bf[i + 1])
            for i in range(len(bf) - 1)
        )
        dj_cost = sum(
            next(e.weight for e in g.neighbors(dj[i]) if e.target == dj[i + 1])
            for i in range(len(dj) - 1)
        )
        assert bf_cost < dj_cost
        assert bf_cost == 6
        assert dj_cost == 7

    def test_large_graph_no_negative_cycle(self):
        g = Graph()
        for i in range(100):
            g.add_edge(str(i), str(i + 1), weight=1, bidirectional=False)
        path = bellman_ford(g, "0", "100")
        assert len(path) == 101

    def test_disconnected_node_in_graph(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("C", "D")
        assert bellman_ford(g, "A", "C") is None


# ---------------------------------------------------------------------------
# Bidirectional BFS
# ---------------------------------------------------------------------------


class TestBidirectionalBFS:
    def test_finds_path(self, simple_graph: Graph):
        path = bidirectional_bfs(simple_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_shortest_path(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        path = bidirectional_bfs(g, "A", "D")
        assert len(path) == 3

    def test_no_path(self, disconnected_graph: Graph):
        assert bidirectional_bfs(disconnected_graph, "A", "C") is None

    def test_same_node(self, simple_graph: Graph):
        assert bidirectional_bfs(simple_graph, "A", "A") == ["A"]

    def test_missing_goal(self, simple_graph: Graph):
        assert bidirectional_bfs(simple_graph, "A", "Z") is None

    def test_missing_start(self, simple_graph: Graph):
        assert bidirectional_bfs(simple_graph, "Z", "A") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert bidirectional_bfs(empty_graph, "A", "B") is None

    def test_two_nodes(self, two_node_graph: Graph):
        assert bidirectional_bfs(two_node_graph, "X", "Y") == ["X", "Y"]

    def test_matches_bfs(self, simple_graph: Graph):
        assert bidirectional_bfs(simple_graph, "A", "F") == bfs(simple_graph, "A", "F")

    def test_line_graph_undirected(self):
        g = Graph()
        for i in range(50):
            g.add_edge(str(i), str(i + 1))
        b = bidirectional_bfs(g, "0", "49")
        assert b is not None and len(b) == 50


# ---------------------------------------------------------------------------
# Bidirectional Dijkstra
# ---------------------------------------------------------------------------


class TestBidirectionalDijkstra:
    def test_finds_path(self, weighted_graph: Graph):
        path = bidirectional_dijkstra(weighted_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_optimal_cost(self, weighted_graph: Graph):
        path = bidirectional_dijkstra(weighted_graph, "A", "F")
        cost = sum(
            next(
                e.weight
                for e in weighted_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        assert cost == 13

    def test_no_path(self, disconnected_graph: Graph):
        assert bidirectional_dijkstra(disconnected_graph, "A", "C") is None

    def test_same_node(self, weighted_graph: Graph):
        assert bidirectional_dijkstra(weighted_graph, "A", "A") == ["A"]

    def test_matches_dijkstra(self, weighted_graph: Graph):
        bd = bidirectional_dijkstra(weighted_graph, "A", "F")
        dj = dijkstra(weighted_graph, "A", "F")
        assert bd == dj

    def test_empty_graph(self, empty_graph: Graph):
        assert bidirectional_dijkstra(empty_graph, "A", "B") is None


# ---------------------------------------------------------------------------
# Greedy Best-First Search
# ---------------------------------------------------------------------------


class TestGreedyBestFirst:
    def test_finds_path(self, simple_graph: Graph):
        path = greedy_best_first_search(simple_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_no_path(self, disconnected_graph: Graph):
        assert greedy_best_first_search(disconnected_graph, "A", "C") is None

    def test_same_node(self, simple_graph: Graph):
        assert greedy_best_first_search(simple_graph, "A", "A") == ["A"]

    def test_missing_goal(self, simple_graph: Graph):
        assert greedy_best_first_search(simple_graph, "A", "Z") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert greedy_best_first_search(empty_graph, "A", "B") is None

    def test_with_heuristic(self):
        g = Graph()
        for i in range(10):
            g.add_edge(str(i), str(i + 1), bidirectional=False)

        def h(a, b):
            return abs(int(a) - int(b))

        path = greedy_best_first_search(g, "0", "9", heuristic=h)
        assert path is not None


# ---------------------------------------------------------------------------
# SPFA
# ---------------------------------------------------------------------------


class TestSPFA:
    def test_finds_path(self, weighted_graph: Graph):
        path = spfa(weighted_graph, "A", "F")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "F"

    def test_optimal_cost(self, weighted_graph: Graph):
        path = spfa(weighted_graph, "A", "F")
        cost = sum(
            next(
                e.weight
                for e in weighted_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        assert cost == 13

    def test_handles_negative_weights(self):
        g = Graph()
        g.add_edge("A", "B", weight=4, bidirectional=False)
        g.add_edge("A", "C", weight=2, bidirectional=False)
        g.add_edge("B", "C", weight=-3, bidirectional=False)
        path = spfa(g, "A", "C")
        cost = sum(
            next(e.weight for e in g.neighbors(path[i]) if e.target == path[i + 1])
            for i in range(len(path) - 1)
        )
        assert cost == 1

    def test_matches_bellman_ford(self, weighted_graph: Graph):
        assert spfa(weighted_graph, "A", "F") == bellman_ford(weighted_graph, "A", "F")

    def test_detects_negative_cycle(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        g.add_edge("B", "C", weight=-10, bidirectional=False)
        g.add_edge("C", "B", weight=-10, bidirectional=False)
        g.add_edge("C", "D", weight=1, bidirectional=False)
        assert spfa(g, "A", "D") is None

    def test_no_path(self, disconnected_graph: Graph):
        assert spfa(disconnected_graph, "A", "C") is None

    def test_same_node(self, weighted_graph: Graph):
        assert spfa(weighted_graph, "A", "A") == ["A"]

    def test_missing_goal(self, weighted_graph: Graph):
        assert spfa(weighted_graph, "A", "Z") is None

    def test_empty_graph(self, empty_graph: Graph):
        assert spfa(empty_graph, "A", "B") is None


# ---------------------------------------------------------------------------
# Topological Sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_linear_dag(self):
        g = Graph()
        for i in range(5):
            g.add_edge(str(i), str(i + 1), bidirectional=False)
        order = topological_sort(g)
        assert order == ["0", "1", "2", "3", "4", "5"]

    def test_diamond_dag(self):
        g = Graph()
        g.add_edge("A", "B", bidirectional=False)
        g.add_edge("A", "C", bidirectional=False)
        g.add_edge("B", "D", bidirectional=False)
        g.add_edge("C", "D", bidirectional=False)
        order = topological_sort(g)
        assert order is not None
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_cycle_returns_none(self, cyclic_graph: Graph):
        assert topological_sort(cyclic_graph) is None

    def test_single_node(self):
        g = Graph()
        g.adjacency["A"] = []
        order = topological_sort(g)
        assert order == ["A"]

    def test_two_independent_chains(self):
        g = Graph()
        g.add_edge("A1", "A2", bidirectional=False)
        g.add_edge("B1", "B2", bidirectional=False)
        order = topological_sort(g)
        assert order is not None
        assert len(order) == 4
        assert order.index("A1") < order.index("A2")
        assert order.index("B1") < order.index("B2")

    def test_empty_graph(self, empty_graph: Graph):
        assert topological_sort(empty_graph) == []

    def test_self_loop_returns_none(self):
        g = Graph()
        g.add_edge("A", "B", bidirectional=False)
        g.add_edge("B", "B", bidirectional=False)
        assert topological_sort(g) is None


# ---------------------------------------------------------------------------
# DAG Shortest Path
# ---------------------------------------------------------------------------


class TestDAGShortestPath:
    def test_finds_path(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        g.add_edge("A", "C", weight=3, bidirectional=False)
        g.add_edge("B", "D", weight=2, bidirectional=False)
        g.add_edge("C", "D", weight=1, bidirectional=False)
        path = dag_shortest_path(g, "A", "D")
        assert path == ["A", "C", "D"]

    def test_negative_weights(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        g.add_edge("A", "C", weight=2, bidirectional=False)
        g.add_edge("B", "D", weight=1, bidirectional=False)
        g.add_edge("C", "B", weight=-3, bidirectional=False)
        path = dag_shortest_path(g, "A", "D")
        cost = sum(
            next(e.weight for e in g.neighbors(path[i]) if e.target == path[i + 1])
            for i in range(len(path) - 1)
        )
        assert cost == 0

    def test_cycle_returns_none(self, cyclic_graph: Graph):
        assert dag_shortest_path(cyclic_graph, "A", "C") is None

    def test_no_path(self):
        g = Graph()
        g.add_edge("A", "B", bidirectional=False)
        g.add_edge("C", "D", bidirectional=False)
        assert dag_shortest_path(g, "A", "C") is None

    def test_same_node(self):
        g = Graph()
        g.add_edge("A", "B", bidirectional=False)
        assert dag_shortest_path(g, "A", "A") == ["A"]

    def test_matching_dijkstra(self):
        g = Graph()
        g.add_edge("A", "B", weight=4, bidirectional=False)
        g.add_edge("A", "C", weight=2, bidirectional=False)
        g.add_edge("B", "D", weight=5, bidirectional=False)
        g.add_edge("C", "D", weight=8, bidirectional=False)
        assert dag_shortest_path(g, "A", "D") == dijkstra(g, "A", "D")

    def test_missing_goal(self):
        g = Graph()
        g.add_edge("A", "B", bidirectional=False)
        assert dag_shortest_path(g, "A", "Z") is None


# ---------------------------------------------------------------------------
# Prim's MST
# ---------------------------------------------------------------------------


class TestPrimMST:
    def test_simple_mst(self):
        g = Graph()
        g.add_edge("A", "B", weight=2)
        g.add_edge("A", "C", weight=3)
        g.add_edge("B", "C", weight=1)
        mst = prim_mst(g)
        assert mst is not None
        assert len(mst) == 2
        total = sum(w for _, _, w in mst)
        assert total == 3

    def test_disconnected_returns_none(self, disconnected_graph: Graph):
        assert prim_mst(disconnected_graph) is None

    def test_single_node(self):
        g = Graph()
        g.add_edge("A", "A")
        mst = prim_mst(g)
        assert mst == []

    def test_two_nodes(self):
        g = Graph()
        g.add_edge("A", "B", weight=5)
        mst = prim_mst(g)
        assert len(mst) == 1
        assert mst[0][2] == 5

    def test_empty_graph(self, empty_graph: Graph):
        assert prim_mst(empty_graph) is None

    def test_start_specified(self):
        g = Graph()
        g.add_edge("A", "B", weight=2)
        g.add_edge("A", "C", weight=3)
        g.add_edge("B", "C", weight=1)
        mst = prim_mst(g, start="B")
        assert mst is not None and len(mst) == 2


# ---------------------------------------------------------------------------
# Kruskal's MST
# ---------------------------------------------------------------------------


class TestKruskalMST:
    def test_simple_mst(self):
        g = Graph()
        g.add_edge("A", "B", weight=2)
        g.add_edge("A", "C", weight=3)
        g.add_edge("B", "C", weight=1)
        mst = kruskal_mst(g)
        assert mst is not None
        assert len(mst) == 2
        total = sum(w for _, _, w in mst)
        assert total == 3

    def test_matches_prim(self):
        g = Graph()
        g.add_edge("A", "B", weight=4)
        g.add_edge("A", "C", weight=2)
        g.add_edge("B", "C", weight=1)
        g.add_edge("B", "D", weight=5)
        g.add_edge("C", "D", weight=8)
        k = kruskal_mst(g)
        p = prim_mst(g)
        assert k is not None and p is not None
        assert sum(w for _, _, w in k) == sum(w for _, _, w in p)

    def test_disconnected_returns_none(self, disconnected_graph: Graph):
        assert kruskal_mst(disconnected_graph) is None

    def test_single_node(self):
        g = Graph()
        g.add_edge("A", "A")
        mst = kruskal_mst(g)
        assert mst == []

    def test_two_nodes(self):
        g = Graph()
        g.add_edge("A", "B", weight=5)
        mst = kruskal_mst(g)
        assert len(mst) == 1
        assert mst[0][2] == 5

    def test_empty_graph(self, empty_graph: Graph):
        assert kruskal_mst(empty_graph) is None


# ---------------------------------------------------------------------------
# Floyd-Warshall
# ---------------------------------------------------------------------------


class TestFloydWarshall:
    def test_simple_graph(self, weighted_graph: Graph):
        dist = floyd_warshall(weighted_graph)
        assert dist is not None
        assert dist["A"]["F"] == 13

    def test_no_path(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("C", "D")
        dist = floyd_warshall(g)
        assert dist is not None
        assert dist["A"]["D"] == float("inf")

    def test_negative_cycle(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        g.add_edge("B", "C", weight=-10, bidirectional=False)
        g.add_edge("C", "B", weight=-10, bidirectional=False)
        assert floyd_warshall(g) is None

    def test_single_node(self):
        g = Graph()
        g.add_edge("A", "A")
        dist = floyd_warshall(g)
        assert dist is not None
        assert dist["A"]["A"] == 0

    def test_matches_dijkstra(self, weighted_graph: Graph):
        fw = floyd_warshall(weighted_graph)
        dj = dijkstra(weighted_graph, "A", "F")
        dj_cost = sum(
            next(
                e.weight
                for e in weighted_graph.neighbors(dj[i])
                if e.target == dj[i + 1]
            )
            for i in range(len(dj) - 1)
        )
        assert fw is not None
        assert fw["A"]["F"] == dj_cost

    def test_empty_graph(self, empty_graph: Graph):
        assert floyd_warshall(empty_graph) is None


# ---------------------------------------------------------------------------
# Johnson's Algorithm
# ---------------------------------------------------------------------------


class TestJohnson:
    def test_simple_graph(self, weighted_graph: Graph):
        dist = johnson(weighted_graph)
        assert dist is not None
        assert dist["A"]["F"] == 13

    def test_no_path(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("C", "D")
        dist = johnson(g)
        assert dist is not None
        assert dist["A"]["D"] == float("inf")

    def test_negative_cycle(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        g.add_edge("B", "C", weight=-10, bidirectional=False)
        g.add_edge("C", "B", weight=-10, bidirectional=False)
        assert johnson(g) is None

    def test_matches_floyd_warshall(self, weighted_graph: Graph):
        j = johnson(weighted_graph)
        f = floyd_warshall(weighted_graph)
        assert j is not None and f is not None
        for u in weighted_graph.nodes():
            for v in weighted_graph.nodes():
                assert j[u][v] == f[u][v]

    def test_matches_dijkstra(self, weighted_graph: Graph):
        j = johnson(weighted_graph)
        dj = dijkstra(weighted_graph, "A", "F")
        dj_cost = sum(
            next(
                e.weight
                for e in weighted_graph.neighbors(dj[i])
                if e.target == dj[i + 1]
            )
            for i in range(len(dj) - 1)
        )
        assert j is not None
        assert j["A"]["F"] == dj_cost

    def test_empty_graph(self, empty_graph: Graph):
        assert johnson(empty_graph) is None

    def test_negative_weights_no_cycle(self):
        g = Graph()
        g.add_edge("A", "B", weight=5, bidirectional=False)
        g.add_edge("A", "C", weight=2, bidirectional=False)
        g.add_edge("B", "C", weight=-3, bidirectional=False)
        dist = johnson(g)
        assert dist is not None
        assert dist["A"]["C"] == 2


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class TestGraph:
    def test_add_directed_edge(self):
        g = Graph()
        g.add_edge("A", "B", bidirectional=False)
        assert any(e.target == "B" for e in g.neighbors("A"))
        assert not any(e.target == "A" for e in g.neighbors("B"))

    def test_neighbors_empty(self):
        g = Graph()
        assert g.neighbors("missing") == []

    def test_nodes(self):
        g = Graph()
        g.add_edge("A", "B")
        assert set(g.nodes()) == {"A", "B"}

    def test_add_edge_creates_nodes(self):
        g = Graph()
        g.add_edge("A", "B")
        assert "A" in g.adjacency
        assert "B" in g.adjacency

    def test_add_edge_removes_duplicates_on_second_add(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("A", "B")
        matches = [e for e in g.neighbors("A") if e.target == "B"]
        assert len(matches) == 2

    def test_directed_edge_does_not_create_reverse(self):
        g = Graph()
        g.add_edge("A", "B", bidirectional=False)
        assert g.neighbors("B") == []

    def test_self_loop(self):
        g = Graph()
        g.add_edge("A", "A")
        assert any(e.target == "A" for e in g.neighbors("A"))


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

BENCHMARK_TIMEOUT = 5.0


@pytest.mark.slow
class TestBenchmarkChain:
    def test_bfs_chain_1000(self, large_chain_graph: Graph):
        t0 = time.perf_counter()
        path = bfs(large_chain_graph, "0", "998")
        dt = time.perf_counter() - t0
        assert path is not None
        assert len(path) == 999
        assert dt < BENCHMARK_TIMEOUT

    def test_dfs_chain_1000(self, large_chain_graph: Graph):
        t0 = time.perf_counter()
        path = dfs(large_chain_graph, "0", "998")
        dt = time.perf_counter() - t0
        assert path is not None
        assert len(path) == 999
        assert dt < BENCHMARK_TIMEOUT

    def test_dijkstra_chain_1000(self, large_chain_graph: Graph):
        t0 = time.perf_counter()
        path = dijkstra(large_chain_graph, "0", "998")
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT

    def test_astar_chain_1000(self, large_chain_graph: Graph):
        def h(a, b):
            return abs(int(a) - int(b))

        t0 = time.perf_counter()
        path = astar(large_chain_graph, "0", "998", heuristic=h)
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT


@pytest.mark.slow
class TestBenchmarkStar:
    def test_bfs_star_500(self, star_graph: Graph):
        t0 = time.perf_counter()
        path = bfs(star_graph, "center", "499")
        dt = time.perf_counter() - t0
        assert path is not None
        assert len(path) == 2
        assert dt < BENCHMARK_TIMEOUT

    def test_dfs_star_500(self, star_graph: Graph):
        t0 = time.perf_counter()
        path = dfs(star_graph, "center", "499")
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT


@pytest.mark.slow
class TestBenchmarkGrid:
    def test_bfs_grid_50x50(self, grid_graph: Graph):
        t0 = time.perf_counter()
        path = bfs(grid_graph, "0,0", "49,49")
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT

    def test_dfs_grid_50x50(self, grid_graph: Graph):
        t0 = time.perf_counter()
        path = dfs(grid_graph, "0,0", "49,49")
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT

    def test_dijkstra_grid_50x50(self, grid_graph: Graph):
        t0 = time.perf_counter()
        path = dijkstra(grid_graph, "0,0", "49,49")
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT

    def test_astar_grid_50x50(self, grid_graph: Graph):
        def h(a, b):
            return 0

        t0 = time.perf_counter()
        path = astar(grid_graph, "0,0", "49,49", heuristic=h)
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT


@pytest.mark.slow
class TestBenchmarkDense:
    def test_bfs_dense_80(self, dense_graph: Graph):
        t0 = time.perf_counter()
        path = bfs(dense_graph, "0", "79")
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT

    def test_dfs_dense_80(self, dense_graph: Graph):
        t0 = time.perf_counter()
        path = dfs(dense_graph, "0", "79")
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT

    def test_dijkstra_dense_80(self, dense_graph: Graph):
        t0 = time.perf_counter()
        path = dijkstra(dense_graph, "0", "79")
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT

    def test_astar_dense_80(self, dense_graph: Graph):
        def h(a, b):
            return abs(int(a) - int(b))

        t0 = time.perf_counter()
        path = astar(dense_graph, "0", "79", heuristic=h)
        dt = time.perf_counter() - t0
        assert path is not None
        assert dt < BENCHMARK_TIMEOUT


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_dijkstra_negative_weights_suboptimal(self, negative_weight_graph: Graph):
        path = dijkstra(negative_weight_graph, "A", "C")
        assert path is not None
        cost = sum(
            next(
                e.weight
                for e in negative_weight_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        A_to_B_to_C = 5 - 15
        assert cost == A_to_B_to_C

    def test_astar_with_inadmissible_heuristic(self):
        g = Graph()
        g.add_edge("A", "B", weight=10)
        g.add_edge("A", "C", weight=1)
        g.add_edge("C", "B", weight=1)

        def h(a, b):
            return 100 if a == "A" else 0

        path = astar(g, "A", "B", heuristic=h)
        assert path is not None

    def test_dfs_deep_graph_no_recursion_limit(self):
        g = Graph()
        for i in range(5000):
            g.add_edge(str(i), str(i + 1), bidirectional=False)
        path = dfs(g, "0", "5000")
        assert path is not None
        assert len(path) == 5001

    def test_bfs_partial_exploration_large_graph(self):
        g = Graph()
        n = 100_000
        for i in range(n):
            g.add_edge(str(i), str(i + 1), bidirectional=False)
        t0 = time.perf_counter()
        path = bfs(g, "0", "0")
        dt = time.perf_counter() - t0
        assert path == ["0"]
        assert dt < 0.1

    def test_all_algorithms_agree_on_unweighted(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        b = bfs(g, "A", "D")
        d = dfs(g, "A", "D")
        dj = dijkstra(g, "A", "D")
        a = astar(g, "A", "D")
        assert b == d == dj == a == ["A", "B", "C", "D"]

    def test_extremely_large_weight_values(self):
        g = Graph()
        g.add_edge("A", "B", weight=float("inf"))
        g.add_edge("B", "C", weight=1)
        path = dijkstra(g, "A", "C")
        assert path is not None
        assert "B" in path

    def test_negative_weight_edge_astar(self, negative_weight_graph: Graph):
        path = astar(negative_weight_graph, "A", "C")
        assert path is not None
        cost = sum(
            next(
                e.weight
                for e in negative_weight_graph.neighbors(path[i])
                if e.target == path[i + 1]
            )
            for i in range(len(path) - 1)
        )
        A_to_B_to_C = 5 - 15
        assert cost == A_to_B_to_C

    def test_missing_intermediate_node(self):
        g = Graph()
        g.add_edge("A", "B")
        g.adjacency["C"] = []
        assert bfs(g, "A", "C") is None
        assert dfs(g, "A", "C") is None
        assert dijkstra(g, "A", "C") is None
        assert astar(g, "A", "C") is None
