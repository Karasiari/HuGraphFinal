"""Convert graph data (graph, demands) and the solution of the MCF problem (route_result) into SpareCapacityGreedyInput.

This module provides utilities to transform the internal graph/routing
representation into the format expected by the greedy spare capacity
allocation algorithm.
"""

from typing import Dict, List, Tuple

import networkx as nx
from .classes_for_algorithm import (
    DemandID,
    DemandInput,
    EdgeInput,
    OrientedEdge,
    SpareCapacityGreedyInput
)


def _canonical_edge_key(node_a: int, node_b: int) -> Tuple[int, int]:
    """Return a canonical key (min, max) for an undirected edge."""
    return (node_a, node_b) if node_a < node_b else (node_b, node_a)


def _aggregate_edge_capacities(graph: nx.MultiGraph) -> Dict[Tuple[int, int], int]:
    """Aggregate parallel edges into single edges with summed capacity.

    For a MultiGraph, multiple physical links may exist between the same
    node pair. This function merges them by summing their capacities.

    Parameters
    ----------
    graph:
        The source topology multigraph.

    Returns
    -------
    Dict[Tuple[int, int], int]
        Mapping from canonical edge key (min_node, max_node) to total capacity.
    """
    aggregated_capacity: Dict[Tuple[int, int], int] = {}

    for node_u, node_v, data in graph.edges(data=True):
        edge_key = _canonical_edge_key(int(node_u), int(node_v))
        if edge_key not in aggregated_capacity:
            aggregated_capacity[edge_key] = 0
        aggregated_capacity[edge_key] += data['capacity']

    return aggregated_capacity


def _build_edge_inputs(aggregated_capacity: Dict[Tuple[int, int], int]) -> List[EdgeInput]:
    """Create EdgeInput list from aggregated capacities.

    Parameters
    ----------
    aggregated_capacity:
        Mapping from canonical edge key to total capacity.

    Returns
    -------
    List[EdgeInput]
        Sorted list of EdgeInput objects for deterministic processing.
    """
    edge_inputs: List[EdgeInput] = []
    for (node_u, node_v), capacity in sorted(aggregated_capacity.items()):
        edge_inputs.append(EdgeInput(u=node_u, v=node_v, capacity=capacity))
    return edge_inputs


def _build_demand_inputs(
    demands: Dict[int, Tuple[int, int, int]],
    route_result: Dict[int, List[Tuple[int, int, int]]],
) -> List[DemandInput]:
    """Create DemandInput list from routed demands.

    Each demand's volume is taken from the original demands sequence, 
    and the initial edge path is taken from route_result.

    Parameters
    ----------
    demands:
        Source demands sequence.
    route_result:
        Routing result containing the routed paths.

    Returns
    -------
    List[DemandInput]
        List of DemandInput objects for the greedy algorithm.
    """
    demand_inputs: List[DemandInput] = []

    for demand_id, edge_path_with_keys in route_result.items():
        demand_source, demand_target, demand_capacity = demands[demand_id]
        edge_path = [(node_u, node_v) for node_u, node_v, _ in edge_path_with_keys]

        demand_inputs.append(DemandInput(
            demand_id=DemandID(demand_id),
            source=demand_source,
            target=demand_target,
            volume=demand_capacity,
            initial_edge_path=edge_path,
        ))

    return demand_inputs


def convert_to_greedy_input(
    graph: nx.MultiGraph,
    demands: Dict[int, Tuple[int, int, int]],
    route_result: Dict[int, List[Tuple[int, int, int]]],
    epsilon: float = 1.0,
    random_seed: int | None = None,
) -> SpareCapacityGreedyInput:
    """Convert topology multigraph (graph) and the result of solving MCF problem (route_result) into SpareCapacityGreedyInput.

    This function:
    1. Aggregates multi-edges into single edges with summed capacity
    2. Packages everything into the format expected by the greedy algorithm

    Parameters
    ----------
    graph:
        The source topology graph.
    demands:
        The source successfully routed demands sequence.
    route_result:
        The routing solution of MCF problem containing paths for successfully routed demands.
    epsilon:
        Scaling parameter to reserve additional demands in allocation algorithm.
    random_seed:
        Optional seed for reproducible randomization in the greedy algorithm.

    Returns
    -------
    SpareCapacityGreedyInput
        Input suitable for run_greedy_spare_capacity_allocation.
    """
    aggregated_capacity = _aggregate_edge_capacities(graph)
    edge_inputs = _build_edge_inputs(aggregated_capacity)
    demand_inputs = _build_demand_inputs(demands, route_result)

    return SpareCapacityGreedyInput(
        edges=edge_inputs,
        demands=demand_inputs,
        epsilon=epsilon,
        random_seed=random_seed,
    )
