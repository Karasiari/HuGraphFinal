from typing import Dict, Tuple, List
import networkx as nx

# импорт вспомогательных функций под наш алгоритм
from .instruments import *

# --------------------------------------------
# основная функция для решения MCF
# комментарии на английском - оставил оригинал
# --------------------------------------------

def solve_multi_commodity_flow_problem(graph: nx.MultiDiGraph, 
                                       C_max: float, demands_raw: List[Tuple[int, int, int]], 
                                       unsatisfied_subset: List[int], 
                                       eps: float) -> Tuple[Dict[int, List[Tuple[int, int, int]]], 
                                                            Dict[int, Tuple[int, int, int]],
                                                            bool]:
  # Step 0: Get right representation for demands
  demands = []
  for source, sink, capacity in demands_raw:
    demands.append(Demand(source, sink, capacity))
                                                              
  # Step 1: Group demands and create the mapping from i to source-target pairs
  grouped_demands, demand_indices_by_group, i_to_source_target = group_demands_and_create_mapping(demands,
                                                                                                  unsatisfied_subset)
        
  # Step 2: Run the multicommodity flow procedure to generate the flow and l(e) values
  flow = multi_commodity_flow(graph, grouped_demands, C_max, eps)
                                                              
  # Step 3: Scale the flow to make it feasible (ensures flows respect edge capacities)
  scale_flows(flow, graph, C_max)

  # Step 4: Subdivide flows by paths for ungrouped demands
  flow_paths, satisfied_demands = subdivide_flows_by_paths(flow, demand_indices_by_group, demands,
                                                           i_to_source_target)

  # Step 5: Subtract the satisfied demands from the graph capacity
  graph_copy = subtract_flow_from_capacity(graph, flow_paths, demands)

  satisfied_demands_set = set(satisfied_demands)
  left_to_satisfy = unsatisfied_subset - satisfied_demands_set
        
  # Step 6: Try to fulfill remaining demands in the leftover graph
  remaining_paths, remaining_satisfied_demands = fulfill_remaining_demands(graph_copy, demands,
                                                                           demand_indices_by_group,
                                                                           i_to_source_target, left_to_satisfy)

  # Combine the satisfied demands
  satisfied_demands += remaining_satisfied_demands
  satisfied_demands_dict = {id: (demands[id].source, demands[id].target, demands[id].capacity) for id in satisfied_demands}
  flow_paths.update(remaining_paths)
  solved = unsatisfied_subset == set(satisfied_demands)

  return flow_paths, satisfied_demands_dict, solved
