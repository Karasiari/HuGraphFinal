from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Callable, Dict, FrozenSet, Hashable, List, Mapping, NewType, Optional, Sequence, Tuple
import random

import networkx as nx

DemandID = NewType("DemandID", int)
EdgeId = NewType("EdgeID", int)
Node = Hashable
EdgeKey = Tuple[Node, Node]
OrientedEdge = Tuple[Node, Node]
EdgePath = List[OrientedEdge]

@dataclass(frozen=True, slots=True)
class EdgeInput:
    """Input edge specification for an undirected graph."""
    u: Node
    v: Node
    capacity: int
