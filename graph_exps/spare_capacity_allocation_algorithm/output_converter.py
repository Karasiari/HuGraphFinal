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

def convert_greedy_output_for_exp(SpareCapacityGreedyOutput) -> Tuple[int, float]:
    if SpareCapacityGreedyOutput.algorithm_failure_flag:
        algorithm_failure_flag = 1
    else:
        algorithm_failure_flag = 0
    
    successfully_rerouted_demands_volume = SpareCapacityGreedyOutput.successfully_rerouted_demands_volume
    
    return (algorithm_failure_flag, successfully_rerouted_demands_volume, SpareCapacityGreedyOutput.reserve_paths_by_failed_edge)
