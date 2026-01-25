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


@dataclass(frozen=True, slots=True)
class SpareCapacityGreedyInput:
    """Complete input for the greedy spare-capacity allocation algorithm."""
    edges: Sequence[EdgeInput]
    demands: Sequence[DemandInput]
    epsilon: float
    random_seed: Optional[int] = None


@dataclass(frozen=True, slots=True)
class SpareCapacityGreedyOutput:
    """Greedy algorithm output.

    - `remaining_network_by_failed_edge[e]` is the remaining network for the scenario with failed edge e
    - `algorithm_failure_flag` is the flag to identify a failure of our algorithm 
       to allocate all demands in all scenarious
    - `successfully_rerouted_demands_volume` is the sum(volume of demand) of successfully rerouted demands
    - `additional_volume_by_edge[e]` is the global `add(e)` reservation for edge e.
      Keys are canonical undirected edge keys.
    - `reserve_paths_by_failed_edge[e][demand_id]` is the backup (edge) path used by
      `demand_id` when edge `e` fails.
    """
    remaining_network_by_failed_edge: Dict[EdgeKey, Tuple[nx.Graph, nx.Graph]]
    algorithm_failure_flag: bool
    successfully_rerouted_demands_volume: float
    additional_volume_by_edge: Dict[EdgeKey, int]
    reserve_paths_by_failed_edge: Dict[EdgeKey, Dict[DemandID, EdgePath]]


def canonical_edge_key(u: Node, v: Node) -> EdgeKey:
    """Return a deterministic key for an undirected edge.

    The key is independent of the (u, v) order and is stable for common
    primitive node types (ints, strings, tuples, ...).
    """
    if u == v:
        return (u, v)

    def sort_key(x: Node) -> Tuple[str, str]:
        return (type(x).__name__, repr(x))

    a, b = sorted((u, v), key=sort_key)
    return (a, b)


# ----------------------------
# Indexed data model
# ----------------------------

@dataclass(frozen=True, slots=True)
class ProcessedDemand:
    """Demand enriched with derived edge-index information."""
    demand_id: DemandID
    source: Node
    target: Node
    volume: int
    initial_edge_indices: Tuple[int, ...]
    unique_initial_edge_indices: FrozenSet[int]


@dataclass(slots=True)
class PositiveTouchedArray:
    """Mutable non-negative int array with fast reset.

    Only supports monotone (non-decreasing) updates via `increment()`.
    `clear()` resets only indices that were modified since the last clear.
    """
    values: List[int]
    touched_indices: List[int]
    was_touched: List[bool]

    @classmethod
    def zeros(cls, size: int) -> "_PositiveTouchedArray":
        """Create a zero-initialized touched array of length `size`."""
        return cls(values=[0] * size, touched_indices=[], was_touched=[False] * size)

    def increment(self, index: int, delta: int) -> None:
        """Increase values[index] by delta (delta must be >= 0)."""
        if delta == 0:
            return
        if delta < 0:
            raise ValueError("Negative increments are not supported.")

        if not self.was_touched[index]:
            self.was_touched[index] = True
            self.touched_indices.append(index)

        self.values[index] += delta

    def clear(self) -> None:
        """Reset all touched indices back to zero."""
        for idx in self.touched_indices:
            self.values[idx] = 0
            self.was_touched[idx] = False
        self.touched_indices.clear()


@dataclass(slots=True)
class PreprocessedInstance:
    """Problem instance transformed to edge-indexed structures for fast access."""
    graph: nx.Graph
    directed_graph_view: nx.DiGraph
    edge_key_by_index: List[EdgeKey]
    capacity_by_edge: List[int]
    slack_by_edge: List[int]
    demands_by_id: Dict[DemandID, ProcessedDemand]
    demands_using_edge: List[List[DemandID]]  # edge_idx -> [demand_id,...]


@dataclass(slots=True)
class FailureScenarioState:
    """Mutable state while processing one failed edge scenario."""
    failed_edge_index: int
    leftover_by_edge: PositiveTouchedArray
    routed_by_edge: PositiveTouchedArray
    add_by_edge: List[int]      # global, updated across scenarios
    slack_by_edge: List[int]    # constant (capacity - initial load)
