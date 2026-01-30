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


def _get_edge_capacities(graph: nx.MultiDiGraph) -> Dict[Tuple[int, int, int], int]:
    """Get edge capacities for a directed version of a graph: 
    for each multiedge exists its doubled (capacity) version with different direction

    Parameters
    ----------
    graph:
        The source topology multigraph.

    Returns
    -------
    Dict[Tuple[int, int, int], int]
        Mapping from canonical edge key (source, target, key) to total capacity.
    """
    capacities: Dict[Tuple[int, int, int], int] = {}

    for node_u, node_v, key, data in graph.edges(keys=True, data=True):
        edge_key = (node_u, node_v, key)
        if edge_key not in capacities:
            capacities[edge_key] = 0
        capacities[edge_key] += int(data['capacity'])

    return capacities


def _build_edge_inputs(capacities: Dict[Tuple[int, int, int], int]) -> List[EdgeInput]:
    """Create EdgeInput list from capacities.

    Parameters
    ----------
    capacities:
        Mapping from canonical edge key to capacity.

    Returns
    -------
    List[EdgeInput]
        Sorted list of EdgeInput objects for deterministic processing.
    """
    edge_inputs: List[EdgeInput] = []
    for (node_u, node_v, key), capacity in sorted(capacities.items()):
        edge_inputs.append(EdgeInput(u=node_u, v=node_v, key=key, capacity=capacity))
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
        edge_path = [(node_u, node_v, key) for node_u, node_v, key in edge_path_with_keys]

        demand_inputs.append(DemandInput(
            demand_id=DemandID(demand_id),
            source=demand_source,
            target=demand_target,
            volume=demand_capacity,
            initial_edge_path=edge_path,
        ))

    return demand_inputs


def convert_to_greedy_input(
    graph: nx.MultiDiGraph,
    demands: Dict[int, Tuple[int, int, int]],
    route_result: Dict[int, List[Tuple[int, int, int]]],
    epsilon: float = 1.0,
    random_seed: int | None = None,
) -> SpareCapacityGreedyInput:
    """Convert topology multigraph (graph) and the result of solving MCF problem (route_result) into SpareCapacityGreedyInput.

    This function:
    Packages everything into the format expected by the greedy algorithm

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
    capacities = _get_edge_capacities(graph)
    edge_inputs = _build_edge_inputs(capacities)
    demand_inputs = _build_demand_inputs(demands, route_result)

    return SpareCapacityGreedyInput(
        edges=edge_inputs,
        demands=demand_inputs,
        epsilon=epsilon,
        random_seed=random_seed,
    )
