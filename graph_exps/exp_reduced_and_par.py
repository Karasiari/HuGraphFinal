from dataclasses import replace
import random
import pandas as pd

import pickle
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .exp_reduced import *

from .spare_capacity_allocation_algorithm.main_algorithm import run_greedy_spare_capacity_allocation
from .spare_capacity_allocation_algorithm.output_converter import convert_greedy_output_for_exp


RouteResult = Dict[int, List[Tuple[int, int, int]]]
DemandsDict = Dict[int, Tuple[int, int, int]]
EdgeKey = Tuple[int, int]
AddVolumeByEdge = Dict[EdgeKey, int]
AllocationResult = Tuple[str, Tuple[bool, float, AddVolumeByEdge]]
RemainingNetwork = Tuple[nx.DiGraph, nx.MultiDiGraph]
NetworksForExp = Tuple[nx.MultiDiGraph, Dict[EdgeKey, RemainingNetwork]]

def allocate_spare_capacity_for_reduced(
    graph_name: str,
    allocation_type: str,
    input_for_algorithm: SpareCapacityGreedyInput,
    volume_to_reroute: int
) -> AllocationResult:
    output_of_algorithm = run_greedy_spare_capacity_allocation(input_for_algorithm)
    allocation_result = convert_greedy_output_for_exp(output_of_algorithm)
    return graph_name, allocation_type, allocation_result, volume_to_reroute

def get_prepared_for_allocation(
    networks: Dict[str, NetworksForExp], 
    route_result: RouteResult,
    demands: DemandsDict,
    epsilon: float = 1.0,
    available_volumes: Tuple[Tuple[int, float], ...] = ((1, 1.0),),
    random_seed: int | None = None,
    n_jobs=-1
) -> Tuple[List[AllocationResult], int]:
    tasks_for_converting = []
    for allocation_type, network in networks.items():
      graph = network[0]
      tasks_for_converting.append((graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed))
    converted_results = Parallel(n_jobs=n_jobs)(
       delayed(convert_mcf_results_to_greedy_input)(graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed)
       for graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed in tqdm(tasks_for_converting, desc="Converting initial MCF results", total=len(tasks_for_converting))
    )

    return converted_results


def reduced_and_par_expand_test_for_graph(
    graph: HuGraphForExps, 
    additional_resources: List[float],
    allocation_types: List[str], 
    tries_for_allocation: int,
    epsilon: float = 1.0,
    available_volumes: Tuple[Tuple[int, float], ...] = ((1, 1.0),),
    random_seed: int | None = None
) -> pd.DataFrame:
    edges_with_alphas = compute_alpha_for_all_edges(graph)

    route_result, demands, _, multidigraph = graph.solve_mcf()

    unexpanded_remaining_networks, total_volume_to_reroute = convert_mcf_results_for_exp(multidigraph, route_result, demands)
    
    additional_resources.sort(reverse=True)
    expanded_networks = {}
    expanded_graphs = {}
    support_edges_dict = {}
    if ("alpha_with_support" in allocation_types) or ("alpha_mixed" in allocation_types):
      for edge, unexpanded_remaining_network in unexpanded_remaining_networks.items():
        support_edge = get_support_edge(unexpanded_remaining_network)
        support_edge = support_edge if not (support_edge is None) else edges_with_alphas[0][0]
        support_edges_dict[edge] = support_edge
    for allocation_type in allocation_types:
        expanded_networks[allocation_type] = expand_network_for_type(multidigraph, unexpanded_remaining_networks, edges_with_alphas, support_edges_dict, additional_resources, allocation_type)
        expanded_graphs[allocation_type] = expanded_networks[allocation_type][0]

    converted_results_for_allocation = get_prepared_for_allocation(expanded_networks, route_result, demands, total_volume_to_reroute, tries_for_allocation, epsilon, available_volumes, random_seed)

    return converted_results_for_allocation, total_volume_to_reroute
