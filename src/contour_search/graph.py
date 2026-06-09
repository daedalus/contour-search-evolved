from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Edge:
    target: str
    weight: float = 1.0


@dataclass
class _ContourCache:
    idx: Dict[str, int]
    inv: List[Optional[str]]
    nb_idx: List[Optional[List]]
    h_cache: Optional[List[float]] = None
    h_goal: str = ""
    h_precision: int = 0
    h_fn: Optional[Callable] = None
    nb_f_offset: Optional[List[Optional[List]]] = None
    is_chain: Optional[bool] = None


@dataclass(slots=True)
class Graph:
    adjacency: Dict[str, List[Edge]] = field(default_factory=dict)
    _cs_cache: Optional[_ContourCache] = None

    def add_edge(
        self, u: str, v: str, weight: float = 1.0, bidirectional: bool = True
    ) -> None:
        self.adjacency.setdefault(u, []).append(Edge(v, weight))
        if bidirectional:
            self.adjacency.setdefault(v, []).append(Edge(u, weight))
        self.adjacency.setdefault(v, [])
        self.adjacency.setdefault(u, [])
        self._cs_cache = None

    def neighbors(self, node: str) -> List[Edge]:
        return self.adjacency.get(node, [])

    def nodes(self) -> List[str]:
        return list(self.adjacency.keys())
