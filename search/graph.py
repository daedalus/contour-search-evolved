from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Edge:
    target: str
    weight: float = 1.0


@dataclass(slots=True)
class Graph:
    adjacency: Dict[str, List[Edge]] = field(default_factory=dict)
    _cs_idx: Optional[Dict[str, int]] = None
    _cs_inv: Optional[List[Optional[str]]] = None
    _cs_nb_idx: Optional[List[Optional[List]]] = None
    _cs_h_cache: Optional[List[float]] = None
    _cs_h_goal: str = ""
    _cs_h_precision: int = 0
    _cs_h_fn: Optional[Callable] = None
    _cs_nb_f_offset: Optional[List[Optional[List]]] = None

    def add_edge(self, u: str, v: str, weight: float = 1.0, bidirectional: bool = True) -> None:
        self.adjacency.setdefault(u, []).append(Edge(v, weight))
        if bidirectional:
            self.adjacency.setdefault(v, []).append(Edge(u, weight))
        self.adjacency.setdefault(v, [])
        self.adjacency.setdefault(u, [])
        self._cs_idx = None
        self._cs_inv = None
        self._cs_nb_idx = None
        self._cs_h_cache = None
        self._cs_h_goal = ""
        self._cs_h_precision = 0
        self._cs_h_fn = None
        self._cs_nb_f_offset = None

    def neighbors(self, node: str) -> List[Edge]:
        return self.adjacency.get(node, [])

    def nodes(self) -> List[str]:
        return list(self.adjacency.keys())
