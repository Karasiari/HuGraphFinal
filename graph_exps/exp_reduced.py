from dataclasses import replace
import random
import pandas as pd

import pickle
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .exp import *

RouteResult = Dict[int, List[Tuple[int, int, int]]]
DemandsDict = Dict[int, Tuple[int, int, int]]
EdgeKey = Tuple[int, int]
AddVolumeByEdge = Dict[EdgeKey, int]
AllocationResult = Tuple[str, Tuple[bool, float, AddVolumeByEdge]]
RemainingNetwork = Tuple[nx.DiGraph, nx.MultiDiGraph]
NetworksForExp = Tuple[nx.MultiDiGraph, Dict[EdgeKey, RemainingNetwork]]

def allocation_reduced_test(
    networks: Dict[str, NetworksForExp], 
    route_result: RouteResult,
    demands: DemandsDict,
    volume_to_reroute: int,
    tries_for_allocation: int,
    epsilon: float = 1.0,
    available_volumes: Tuple[Tuple[int, float], ...] = ((1, 1.0),),
    random_seed: int | None = None,
    n_jobs=-1
) -> Tuple[List[AllocationResult],
           int]:
    tasks_for_converting = []
    for allocation_type, network in networks.items():
      graph = network[0]
      tasks_for_converting.append((graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed))
    converted_results = Parallel(n_jobs=n_jobs)(
       delayed(convert_mcf_results_to_greedy_input)(graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed)
       for graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed in tqdm(tasks_for_converting, desc="Converting initial MCF results", total=len(tasks_for_converting))
    )

    tasks_for_allocation = []
    for allocation_type, input_for_algorithm in converted_results:
      initial_random_seed = input_for_algorithm.random_seed
      for try_number in range(tries_for_allocation):
        random_seed_for_try = initial_random_seed + try_number if initial_random_seed is not None else None
        new_input = replace(
          input_for_algorithm, 
          random_seed=random_seed_for_try
        )
        tasks_for_allocation.append((new_input, allocation_type))
    algorithm_results = Parallel(n_jobs=n_jobs)(
        delayed(allocate_spare_capacity)(input_for_algorithm, allocation_type)
        for input_for_algorithm, allocation_type in tqdm(tasks_for_allocation, desc="Processing allocation", total=len(tasks_for_allocation))
    )
    return algorithm_results, volume_to_reroute


def get_reduced_right_output(
    allocation_results_raw: Tuple[List[AllocationResult], 
                            int]
) -> pd.DataFrame:
    result_dict = {}
    allocation_seen = {}
    algorithm_results, volume_to_reroute = allocation_results_raw
  
    for allocation_type, result_raw in algorithm_results:
      result = {'allocation solved': result_raw[0], 'rerouted volume ratio': round(result_raw[1] / volume_to_reroute, 2)}
      if allocation_seen.get(allocation_type, False):
          allocation_seen[allocation_type] += 1
      else:
          allocation_seen[allocation_type] = 1
      result_dict[(allocation_type, allocation_seen[allocation_type])] = result.copy()
    
    result_df = pd.DataFrame(result_dict).T
    return result_df


def reduced_expand_test_for_graph(
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
    for allocation_type in allocation_types:
        expanded_networks[allocation_type] = expand_network_for_type(multidigraph, unexpanded_remaining_networks, edges_with_alphas, additional_resources, allocation_type)
        expanded_graphs[allocation_type] = expanded_networks[allocation_type][0]

    allocation_results_raw = allocation_reduced_test(expanded_networks, route_result, demands, total_volume_to_reroute, tries_for_allocation, epsilon, available_volumes, random_seed)

    allocation_results = get_reduced_right_output(allocation_results_raw)
    return allocation_results
