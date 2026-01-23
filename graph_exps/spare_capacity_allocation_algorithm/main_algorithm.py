from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Callable, Dict, FrozenSet, Hashable, List, Mapping, NewType, Optional, Sequence, Tuple
import random

import networkx as nx

from .classes_for_algorithm import *


def _canonical_edge_key(u: Node, v: Node) -> EdgeKey:
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


@dataclass(frozen=True, slots=True)
class SpareCapacityGreedyOutput:
    """Greedy algorithm output.

    - `additional_volume_by_edge[e]` is the global `add(e)` reservation for edge e.
      Keys are canonical undirected edge keys.
    - `reserve_paths_by_failed_edge[e][demand_id]` is the backup (edge) path used by
      `demand_id` when edge `e` fails.
    """
    additional_volume_by_edge: Dict[EdgeKey, int]
    reserve_paths_by_failed_edge: Dict[EdgeKey, Dict[DemandID, EdgePath]]


# ----------------------------
# Internal indexed data model
# ----------------------------

@dataclass(frozen=True, slots=True)
class _ProcessedDemand:
    """Demand enriched with derived edge-index information."""
    demand_id: DemandID
    source: Node
    target: Node
    volume: int
    initial_edge_indices: Tuple[int, ...]
    unique_initial_edge_indices: FrozenSet[int]


@dataclass(slots=True)
class _PositiveTouchedArray:
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
class _PreprocessedInstance:
    """Problem instance transformed to edge-indexed structures for fast access."""
    graph: nx.Graph
    directed_graph_view: nx.DiGraph
    edge_key_by_index: List[EdgeKey]
    capacity_by_edge: List[int]
    slack_by_edge: List[int]
    demands_by_id: Dict[DemandID, _ProcessedDemand]
    demands_using_edge: List[List[DemandID]]  # edge_idx -> [demand_id,...]


@dataclass(slots=True)
class _FailureScenarioState:
    """Mutable state while processing one failed edge scenario."""
    failed_edge_index: int
    leftover_by_edge: _PositiveTouchedArray
    routed_by_edge: _PositiveTouchedArray
    add_by_edge: List[int]      # global, updated across scenarios
    slack_by_edge: List[int]    # constant (capacity - initial load)


# ----------------------------
# Preprocessing
# ----------------------------

def _build_indexed_graph(edge_inputs: Sequence[EdgeInput]) -> Tuple[nx.Graph, List[EdgeKey], List[int]]:
    """Build an undirected NetworkX graph and assign a compact index to each edge."""
    graph = nx.Graph()
    edge_key_by_index: List[EdgeKey] = []
    capacity_by_edge: List[int] = []
    seen: Dict[EdgeKey, int] = {}

    for edge in edge_inputs:
        if edge.capacity < 0:
            raise ValueError(
                f"Edge capacity must be non-negative, got {edge.capacity} for edge {edge.u}-{edge.v}."
            )

        key = _canonical_edge_key(edge.u, edge.v)
        if key in seen:
            existing_idx = seen[key]
            if capacity_by_edge[existing_idx] != edge.capacity:
                raise ValueError(
                    f"Duplicate edge {key} with conflicting capacities: "
                    f"{capacity_by_edge[existing_idx]} vs {edge.capacity}."
                )
            continue

        idx = len(edge_key_by_index)
        seen[key] = idx
        edge_key_by_index.append(key)
        capacity_by_edge.append(edge.capacity)
        graph.add_edge(edge.u, edge.v, idx=idx, capacity=edge.capacity)

    return graph, edge_key_by_index, capacity_by_edge


def _process_demands(
    demand_inputs: Sequence[DemandInput],
    graph: nx.Graph,
    edge_count: int,
) -> Tuple[Dict[DemandID, _ProcessedDemand], List[int], List[List[DemandID]]]:
    """Validate demand paths, derive edge indices, and compute initial edge loads."""
    demands_by_id: Dict[DemandID, _ProcessedDemand] = {}
    initial_load_by_edge: List[int] = [0] * edge_count
    demands_using_edge: List[List[DemandID]] = [[] for _ in range(edge_count)]

    for demand in demand_inputs:
        if demand.demand_id in demands_by_id:
            raise ValueError(f"Duplicate demand_id: {demand.demand_id}.")
        if demand.volume < 0:
            raise ValueError(
                f"Demand volume must be non-negative, got {demand.volume} for demand {demand.demand_id}."
            )

        current_node = demand.source
        edge_indices: List[int] = []

        for step, (u, v) in enumerate(demand.initial_edge_path):
            if not graph.has_edge(u, v):
                raise ValueError(
                    f"Demand {demand.demand_id} initial_edge_path uses a non-existent edge: {u} - {v}."
                )

            if current_node == u:
                next_node = v
            elif current_node == v:
                next_node = u
            else:
                raise ValueError(
                    f"Demand {demand.demand_id} initial_edge_path is not contiguous at step {step}: "
                    f"current node {current_node}, edge endpoints {(u, v)}."
                )

            edge_idx = graph[u][v]["idx"]
            edge_indices.append(edge_idx)
            if demand.volume:
                initial_load_by_edge[edge_idx] += demand.volume

            current_node = next_node

        if current_node != demand.target:
            raise ValueError(
                f"Demand {demand.demand_id} initial_edge_path does not end at target. "
                f"Ended at {current_node}, expected {demand.target}."
            )

        unique_edge_indices = frozenset(edge_indices)
        for edge_idx in unique_edge_indices:
            demands_using_edge[edge_idx].append(demand.demand_id)

        demands_by_id[demand.demand_id] = _ProcessedDemand(
            demand_id=demand.demand_id,
            source=demand.source,
            target=demand.target,
            volume=demand.volume,
            initial_edge_indices=tuple(edge_indices),
            unique_initial_edge_indices=unique_edge_indices,
        )

    return demands_by_id, initial_load_by_edge, demands_using_edge


def _preprocess_instance(input_data: SpareCapacityGreedyInput) -> _PreprocessedInstance:
    """Transform raw input into an indexed instance and validate initial feasibility."""
    graph, edge_key_by_index, capacity_by_edge = _build_indexed_graph(input_data.edges)
    if not edge_key_by_index:
        raise ValueError("Input graph must contain at least one edge.")

    demands_by_id, initial_load_by_edge, demands_using_edge = _process_demands(
        input_data.demands, graph, edge_count=len(edge_key_by_index)
    )

    slack_by_edge: List[int] = []
    for edge_idx, (capacity, load) in enumerate(zip(capacity_by_edge, initial_load_by_edge)):
        slack = capacity - load
        if slack < 0:
            raise ValueError(
                f"Initial routing violates capacity for edge {edge_key_by_index[edge_idx]}: "
                f"capacity {capacity}, initial load {load}."
            )
        slack_by_edge.append(slack)

    return _PreprocessedInstance(
        graph=graph,
        directed_graph_view=graph.to_directed(as_view=True),
        edge_key_by_index=edge_key_by_index,
        capacity_by_edge=capacity_by_edge,
        slack_by_edge=slack_by_edge,
        demands_by_id=demands_by_id,
        demands_using_edge=demands_using_edge,
    )


# ----------------------------
# Scenario utilities
# ----------------------------

def _compute_leftover_space(
    leftover: _PositiveTouchedArray,
    affected_demand_ids: Sequence[DemandID],
    demands_by_id: Mapping[DemandID, _ProcessedDemand],
) -> None:
    """Compute per-edge freed volume when the failed edge drops `affected_demand_ids`."""
    leftover.clear()
    for demand_id in affected_demand_ids:
        demand = demands_by_id[demand_id]
        if demand.volume == 0:
            continue
        for edge_idx in demand.initial_edge_indices:
            leftover.increment(edge_idx, demand.volume)


def _make_weight1(
    scenario: _FailureScenarioState,
    demand_volume: int,
) -> Callable[[Node, Node, Mapping[str, Any]], Optional[int]]:
    """Build the Objective-1 weight function.

    For an edge f, this returns the incremental increase in add(f) required to route
    `demand_volume` through f under the current scenario state. If the edge is not
    usable (failed edge or physical capacity violation), returns None to hide the edge.
    """
    failed_edge_idx = scenario.failed_edge_index
    slack = scenario.slack_by_edge
    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge

    def weight(_u: Node, _v: Node, attrs: Mapping[str, Any]) -> Optional[int]:
        edge_idx = attrs["idx"]
        if edge_idx == failed_edge_idx:
            return None

        # Physical capacity for rerouted demands in this scenario:
        # remaining = slack + leftover - already_routed
        remaining_capacity = slack[edge_idx] + leftover[edge_idx] - routed[edge_idx]
        if remaining_capacity < demand_volume:
            return None

        # Allowance w.r.t. (initial load + add): allowance = leftover + add - routed
        allowance = leftover[edge_idx] + add[edge_idx] - routed[edge_idx]
        return 0 if allowance >= demand_volume else demand_volume - allowance

    return weight


def _find_backup_path_nodes(
    instance: _PreprocessedInstance,
    scenario: _FailureScenarioState,
    demand: _ProcessedDemand,
) -> List[Node]:
    """Compute the demand's backup path as a node sequence.

    Lexicographic objectives:
      1) minimize sum(max(0, volume - allowance(edge))) over edges in the path
      2) among Objective-1 shortest paths, minimize sum(min(allowance(edge), volume))

    Where allowance(edge) = leftover(edge) + add(edge) - routed(edge) in this scenario.
    """
    if demand.source == demand.target:
        return [demand.source]

    weight1 = _make_weight1(scenario, demand_volume=demand.volume)

    try:
        dist_from_source = nx.single_source_dijkstra_path_length(
            instance.graph, demand.source, weight=weight1
        )
        dist_to_target = nx.single_source_dijkstra_path_length(
            instance.graph, demand.target, weight=weight1
        )
    except nx.NodeNotFound as exc:
        raise ValueError(
            f"Demand {demand.demand_id} references a node that is not present in the graph."
        ) from exc

    if demand.target not in dist_from_source:
        raise ValueError(
            f"No feasible backup path for demand {demand.demand_id} under failure of edge index {scenario.failed_edge_index}."
        )

    shortest_len = dist_from_source[demand.target]

    failed_edge_idx = scenario.failed_edge_index
    slack = scenario.slack_by_edge
    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge
    volume = demand.volume

    def weight2(u: Node, v: Node, attrs: Mapping[str, Any]) -> Optional[int]:
        """Objective-2 weight, restricted to edges on Objective-1 shortest s-t paths."""
        edge_idx = attrs["idx"]
        if edge_idx == failed_edge_idx:
            return None

        remaining_capacity = slack[edge_idx] + leftover[edge_idx] - routed[edge_idx]
        if remaining_capacity < volume:
            return None

        allowance = leftover[edge_idx] + add[edge_idx] - routed[edge_idx]
        inc_add = 0 if allowance >= volume else volume - allowance

        dist_u = dist_from_source.get(u)
        dist_v_to_t = dist_to_target.get(v)
        if dist_u is None or dist_v_to_t is None:
            return None
        if dist_u + inc_add + dist_v_to_t != shortest_len:
            return None

        return allowance if allowance < volume else volume

    try:
        return nx.dijkstra_path(
            instance.directed_graph_view, demand.source, demand.target, weight=weight2
        )
    except nx.NetworkXNoPath as exc:
        raise ValueError(
            f"Objective-2 routing failed for demand {demand.demand_id} under failure of edge index {scenario.failed_edge_index}."
        ) from exc


def _apply_backup_routing(
    instance: _PreprocessedInstance,
    scenario: _FailureScenarioState,
    demand: _ProcessedDemand,
    backup_path_nodes: Sequence[Node],
) -> None:
    """Apply the chosen backup route: update global add and per-scenario routed volume."""
    if demand.volume == 0 or len(backup_path_nodes) < 2:
        return

    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge
    slack = scenario.slack_by_edge
    volume = demand.volume

    for u, v in pairwise(backup_path_nodes):
        edge_idx = instance.graph[u][v]["idx"]

        # Physical feasibility (defensive check)
        remaining_capacity = slack[edge_idx] + leftover[edge_idx] - routed[edge_idx]
        if remaining_capacity < volume:
            raise ValueError(
                f"Internal error: selected an infeasible edge for demand {demand.demand_id}. "
                f"Edge index {edge_idx} remaining capacity {remaining_capacity}, demand volume {volume}."
            )

        allowance = leftover[edge_idx] + add[edge_idx] - routed[edge_idx]
        if allowance < volume:
            add[edge_idx] += volume - allowance
            if add[edge_idx] > slack[edge_idx]:
                raise ValueError(
                    f"Internal error: add exceeds physical slack on edge {instance.edge_key_by_index[edge_idx]}. "
                    f"add={add[edge_idx]}, slack={slack[edge_idx]}."
                )

        scenario.routed_by_edge.increment(edge_idx, volume)


def _nodes_to_oriented_edge_path(nodes_path: Sequence[Node]) -> EdgePath:
    """Convert a node path [n0, n1, ..., nk] into an oriented edge path [(n0,n1),...,(n{k-1},nk)]."""
    return [(u, v) for u, v in pairwise(nodes_path)]


# ----------------------------
# Public API
# ----------------------------

def run_greedy_spare_capacity_allocation(input_data: SpareCapacityGreedyInput) -> SpareCapacityGreedyOutput:
    """Execute the greedy algorithm from the prompt.

    Processing order:
      - Edges are processed in a random order.
      - For each failed edge, the affected demands are processed in a random order.

    For each failure scenario (single failed edge), only demands that used that edge
    in the initial routing are rerouted. All other demands remain on their initial routes.

    Returns:
      A `SpareCapacityGreedyOutput` containing:
        - global per-edge reservations `add(e)`
        - per-failed-edge backup paths for affected demands
    """
    instance = _preprocess_instance(input_data)

    edge_count = len(instance.edge_key_by_index)
    add_by_edge: List[int] = [0] * edge_count

    rng = random.Random(input_data.random_seed)
    failure_edge_indices = list(range(edge_count))
    rng.shuffle(failure_edge_indices)

    leftover = _PositiveTouchedArray.zeros(edge_count)
    routed = _PositiveTouchedArray.zeros(edge_count)

    reserve_paths_by_failed_edge: Dict[EdgeKey, Dict[DemandID, EdgePath]] = {}

    for failed_edge_idx in failure_edge_indices:
        affected_demands = list(instance.demands_using_edge[failed_edge_idx])
        rng.shuffle(affected_demands)

        routed.clear()
        _compute_leftover_space(leftover, affected_demands, instance.demands_by_id)

        scenario = _FailureScenarioState(
            failed_edge_index=failed_edge_idx,
            leftover_by_edge=leftover,
            routed_by_edge=routed,
            add_by_edge=add_by_edge,
            slack_by_edge=instance.slack_by_edge,
        )

        demand_to_backup_path: Dict[DemandID, EdgePath] = {}
        for demand_id in affected_demands:
            demand = instance.demands_by_id[demand_id]
            backup_nodes = _find_backup_path_nodes(instance, scenario, demand)
            _apply_backup_routing(instance, scenario, demand, backup_nodes)
            demand_to_backup_path[demand_id] = _nodes_to_oriented_edge_path(backup_nodes)

        reserve_paths_by_failed_edge[instance.edge_key_by_index[failed_edge_idx]] = demand_to_backup_path

    additional_volume_by_edge = {
        instance.edge_key_by_index[edge_idx]: add_by_edge[edge_idx] for edge_idx in range(edge_count)
    }

    return SpareCapacityGreedyOutput(
        additional_volume_by_edge=additional_volume_by_edge,
        reserve_paths_by_failed_edge=reserve_paths_by_failed_edge,
    )
