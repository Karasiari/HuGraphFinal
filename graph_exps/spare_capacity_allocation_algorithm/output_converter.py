"""Convert SpareCapacityGreedyOutput into data format needed for main test"""

from typing import Tuple

from .classes_for_algorithm import (
    SpareCapacityGreedyOutput
)

def convert_greedy_output_for_exp(SpareCapacityGreedyOutput) -> Tuple[bool, float]:
    algorithm_solved_flag = not SpareCapacityGreedyOutput.algorithm_failure_flag
    successfully_rerouted_demands_volume = SpareCapacityGreedyOutput.successfully_rerouted_demands_volume
    
    return (algorithm_solved_flag, successfully_rerouted_demands_volume)
