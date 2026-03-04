from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .exp_reduced_and_par import *

results_by_graph = {}

tasks_for_allocation = []
for graph_name, test_input in test_graphs.items():
  graph_for_exp, additional_resources, allocation_types, tries_for_allocation, epsilon, available_volumes = test_input
  converted_results_for_allocation, volume_to_reroute = reduced_expand_test_for_graph(
      graph=graph_for_exp, 
      additional_resources=additional_resources, 
      allocation_types=allocation_types, 
      tries_for_allocation=10, 
      epsilon=1.2, 
      available_volumes=available_demand_volumes
  )
  for allocation_type, input_for_algorithm in converted_results_for_allocation:
      initial_random_seed = input_for_algorithm.random_seed
      for try_number in range(tries_for_allocation):
        random_seed_for_try = initial_random_seed + try_number if initial_random_seed is not None else None
        new_input = replace(
          input_for_algorithm, 
          random_seed=random_seed_for_try
        )
        tasks_for_allocation.append((graph_name, allocation_type, new_input, volume_to_reroute))

# тут новая параллелизация
algorithm_results_all = Parallel(n_jobs=n_jobs)(
    delayed(allocate_spare_capacity_for_reduced)(graph_name, allocation_type, input_for_algorithm, volume_to_reroute)
    for graph_name, allocation_type, input_for_algorithm, volume_to_reroute in tqdm(tasks_for_allocation, desc="Processing allocation", total=len(tasks_for_allocation))
)

for graph_name, allocation_type, allocation_result_raw, volume_to_reroute in algorithm_results_all:
  if results_by_graph.get(graph_name, False):
    pass
  else:
    results_by_graph[graph_name] = {}
  results_by_graph[graph_name]['volume'] = volume_to_reroute
  if results_by_graph[graph_name].get('allocation_results', False):
    results_by_graph[graph_name]['allocation_results'].append((allocation_type, allocation_result_raw))
  else:
    results_by_graph[graph_name]['allocation_results'] = [(allocation_type, allocation_result_raw)]

final_results_by_graph = {}

for graph_name, results_raw in results_by_graph.items():
  final_results_by_graph[graph_name] = get_reduced_right_output((results_raw['allocation_results'], results_raw['volume']))
