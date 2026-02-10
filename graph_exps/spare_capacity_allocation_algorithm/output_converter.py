"""Convert SpareCapacityGreedyOutput into data format needed for main test"""

from typing import Tuple

from .classes_for_algorithm import (
    SpareCapacityGreedyOutput
)

AddVolumeByEdge = Dict[Tuple[int, int], int]

def convert_greedy_output_for_exp(SpareCapacityGreedyOutput) -> Tuple[bool, float, AddVolumeByEdge]:
    algorithm_solved_flag = not SpareCapacityGreedyOutput.algorithm_failure_flag
    successfully_rerouted_demands_volume = SpareCapacityGreedyOutput.successfully_rerouted_demands_volume
    additional_volume_by_edge = SpareCapacityGreedyOutput.additional_volume_by_edge
    
    return (algorithm_solved_flag, successfully_rerouted_demands_volume, additional_volume_by_edge)
