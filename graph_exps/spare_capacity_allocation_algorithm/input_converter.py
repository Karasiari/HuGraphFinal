"""Convert GraphData and RouteResult into SpareCapacityGreedyInput.

This module provides utilities to transform the internal graph/routing
representation into the format expected by the greedy spare capacity
allocation algorithm.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Dict, List, Tuple

from graph_reader.read_graph import GraphData
from spare_capacity_allocation.greedy_spare_capacity_allocation import (
    DemandID,
    DemandInput,
    EdgeInput,
    OrientedEdge,
    SpareCapacityGreedyInput,
    SpareCapacityGreedyOutput,
)
from utils.router import RouteResult


def _canonical_edge_key(node_a: int, node_b: int) -> Tuple[int, int]:
    """Return a canonical key (min, max) for an undirected edge."""
    return (node_a, node_b) if node_a < node_b else (node_b, node_a)


def _aggregate_edge_capacities(graph: GraphData) -> Dict[Tuple[int, int], int]:
    """Aggregate parallel edges into single edges with summed capacity.

    For a MultiGraph, multiple physical links may exist between the same
    node pair. This function merges them by summing their capacities.

    Parameters
    ----------
    graph:
        The source graph containing the topology multigraph.

    Returns
    -------
    Dict[Tuple[int, int], int]
        Mapping from canonical edge key (min_node, max_node) to total capacity.
    """
    capacity_per_physical_edge = graph.line_rate * graph.number_of_wavelengths
    aggregated_capacity: Dict[Tuple[int, int], int] = {}

    multigraph = graph.topology_multigraph
    for node_u, node_v, _ in multigraph.edges(keys=True):
        edge_key = _canonical_edge_key(int(node_u), int(node_v))
        if edge_key not in aggregated_capacity:
            aggregated_capacity[edge_key] = 0
        aggregated_capacity[edge_key] += capacity_per_physical_edge

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


def _node_path_to_edge_path(node_path: List[int]) -> List[OrientedEdge]:
    """Convert a node path to an oriented edge path.

    Parameters
    ----------
    node_path:
        Sequence of node indices representing the path.

    Returns
    -------
    List[OrientedEdge]
        List of (source, target) tuples for each edge in the path.
    """
    return [(node_u, node_v) for node_u, node_v in pairwise(node_path)]


def _build_demand_inputs(
    graph: GraphData,
    route_result: RouteResult,
) -> List[DemandInput]:
    """Create DemandInput list from routed demands.

    Only successfully routed demands are included. Each demand's volume
    is taken from the original GraphData, and the initial edge path is
    derived from the RouteResult's node path.

    Parameters
    ----------
    graph:
        Source graph containing demand definitions.
    route_result:
        Routing result containing the routed paths.

    Returns
    -------
    List[DemandInput]
        List of DemandInput objects for the greedy algorithm.
    """
    demand_inputs: List[DemandInput] = []

    for demand_id, node_path in route_result.routed_paths.items():
        original_demand = graph.demands[demand_id]
        edge_path = _node_path_to_edge_path(node_path)

        demand_inputs.append(DemandInput(
            demand_id=DemandID(demand_id),
            source=original_demand.source,
            target=original_demand.target,
            volume=original_demand.bitrate,
            initial_edge_path=edge_path,
        ))

    return demand_inputs


def convert_to_greedy_input(
    graph: GraphData,
    route_result: RouteResult,
    random_seed: int | None = None,
) -> SpareCapacityGreedyInput:
    """Convert GraphData and RouteResult into SpareCapacityGreedyInput.

    This function:
    1. Aggregates multi-edges into single edges with summed capacity
    2. Converts routed node paths to edge paths
    3. Packages everything into the format expected by the greedy algorithm

    Parameters
    ----------
    graph:
        The source graph containing topology and demand information.
    route_result:
        The routing solution containing paths for successfully routed demands.
    random_seed:
        Optional seed for reproducible randomization in the greedy algorithm.

    Returns
    -------
    SpareCapacityGreedyInput
        Input suitable for run_greedy_spare_capacity_allocation.
    """
    aggregated_capacity = _aggregate_edge_capacities(graph)
    edge_inputs = _build_edge_inputs(aggregated_capacity)
    demand_inputs = _build_demand_inputs(graph, route_result)

    return SpareCapacityGreedyInput(
        edges=edge_inputs,
        demands=demand_inputs,
        random_seed=random_seed,
    )


def compute_total_additional_volume(output: SpareCapacityGreedyOutput) -> int:
    """Compute the objective value: total sum of add(e) over all edges.

    Parameters
    ----------
    output:
        The output from the greedy spare capacity allocation algorithm.

    Returns
    -------
    int
        Sum of additional volume reservations across all edges.
    """
    return sum(output.additional_volume_by_edge.values())
