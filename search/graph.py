from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Edge:
    target: str
    weight: float = 1.0


@dataclass
class Graph:
    adjacency: Dict[str, List[Edge]] = field(default_factory=dict)

    def add_edge(self, u: str, v: str, weight: float = 1.0, bidirectional: bool = True) -> None:
        self.adjacency.setdefault(u, []).append(Edge(v, weight))
        if bidirectional:
            self.adjacency.setdefault(v, []).append(Edge(u, weight))
        self.adjacency.setdefault(v, [])
        self.adjacency.setdefault(u, [])
        try:
            del self._cs_nb
        except AttributeError:
            pass

    def neighbors(self, node: str) -> List[Edge]:
        return self.adjacency.get(node, [])

    def nodes(self) -> List[str]:
        return list(self.adjacency.keys())
