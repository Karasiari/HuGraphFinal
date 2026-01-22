from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Callable, Dict, FrozenSet, Hashable, List, Mapping, NewType, Optional, Sequence, Tuple
import random

import networkx as nx

DemandID = NewType("DemandID", int)
#EdgeId = NewType("EdgeID", int)
Node = Hashable
#EdgeKey = Tuple[Node, Node]
OrientedEdge = Tuple[Node, Node]
#EdgePath = List[OrientedEdge]

@dataclass(frozen=True, slots=True)
class EdgeInput:
    """Input edge specification for an undirected graph."""
    u: Node
    v: Node
    capacity: int


@dataclass(frozen=True, slots=True)
class DemandInput:
    """Input traffic demand with an initial (edge) routing path.

    `initial_edge_path` is a sequence of edges describing the demand's initial routing.
    Each element is a 2-tuple (u, v) of endpoints. The orientation does not need to be
    consistent, but the sequence must be contiguous from `source` to `target`.
    """
    demand_id: DemandID
    source: Node
    target: Node
    volume: int
    initial_edge_path: Sequence[OrientedEdge]
