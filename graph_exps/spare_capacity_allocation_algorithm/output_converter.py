"""Convert SpareCapacityGreedyOutput into data format needed for main test"""

from typing import Dict, Tuple, List
import networkx as nx

from .classes_for_algorithm import (
    DemandID,
    EdgeKey,
    EdgePath,
    OrientedEdge,
    Node,
    SpareCapacityGreedyOutput
)

def convert_greedy_output_for_exp(SpareCapacityGreedyOutput) -> Tuple[Dict[Tuple[int, int], Tuple[nx.Graph, nx.Graph]], int, float]:
    remaining_network_dict: Dict[Tuple[int, int], Tuple[nx.Graph, nx.Graph]] = {}
    
    for edge_key, remaining_network in SpareCapacityGreedyOutput.remaining_network_by_failed_edge.items():
        remaining_network_dict[(edge_key[0], edge_key[1])] = remaining_network
    
    if SpareCapacityGreedyOutput.algorithm_failure_flag:
        algorithm_failure_flag = 1
    else:
        algorithm_failure_flag = 0
      
     successfully_rerouted_demands_ratio = SpareCapacityGreedyOutput.successfully_rerouted_demands_ratio
   
     return (remaining_network_dict, algorithm_failure_flag, successfully_rerouted_demands_ratio)
