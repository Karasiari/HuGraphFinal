# импорт читалки ху графов
from .readers.hugraph_reader import read_hu_graphs
# импорт генерации своего трафика
from .traffic_generation.generation_for_exp import generate_own_traffic
# импорт генераци SNR-BA графа
from .SNRBA_generation.generation import generate_snr_ba_graph
# импорт окружения эксперимента
from .exp import *

# допустимые веса запросов и capacity для SNR-BA графа
available_demand_volumes = ((1, 0.9), (2, 0.1))
capacity_value = 80.0

# новые ресурсы и типы для распределения ресурсов
additional_resources = [capacity_value] * 3
allocation_types = ['alpha', 'random_alpha', 'random']

# генерация SNR-BA графа
snr_ba_graph = generate_snr_ba_graph(num_nodes=30, capacity_value=capacity_value)

# генерация своего трафика методом "alpha_with_sa" и эксперимент для графа
adj_graph_alpha, traffic_graph_alpha =  generate_own_traffic(
  snr_ba_graph, 
  available_demand_volumes, 
  generation_type="alpha_with_sa", 
  generation_params={"alpha_target": 0.35, "epsilon": 0.025, "median_weight_for_initial": 20, "var_for_initial": 1, "multi_max": 5, "t": 0.5}
)
graph_for_exp_alpha = HuGraphForExps(adj_graph_alpha, traffic_graph_alpha)
results_alpha = expand_test_for_graph(
  graph=graph_for_exp_alpha, 
  additional_resources=additional_resources, 
  allocation_types=allocation_types, 
  tries_for_allocation=10, 
  epsilon=1.2, 
  available_volumes=available_demand_volumes
)

# генерация своего трафика методом "gravity" и эксперимент для графа
adj_graph_gravity, traffic_graph_gravity =  generate_own_traffic(
  snr_ba_graph, 
  available_demand_volumes, 
  generation_type="gravity", 
  generation_params={"beta": 0.15, "dyn_k": 0.9}
)
graph_for_exp_gravity = HuGraphForExps(adj_graph_gravity, traffic_graph_gravity)
results_gravity = expand_test_for_graph(
  graph=graph_for_exp_gravity, 
  additional_resources=additional_resources, 
  allocation_types=allocation_types, 
  tries_for_allocation=10, 
  epsilon=1.2, 
  available_volumes=available_demand_volumes
)


# то же, но для ху графа

# читаем граф
hu_graphs_names = ['cola_t3']
hu_graphs = read_hu_graphs(path_to_folder, hu_graphs_names, True)
adj_graph_hu, traffic_graph_hu = hu_graphs['cola_t3']['adj_graph'], hu_graphs['cola_t3']['traffic_graph']

# проводим эксперимент
graph_for_exp_hu = HuGraphForExps(adj_graph_hu, traffic_graph_hu)
results_hu = expand_test_for_graph(
  graph=graph_for_exp_hu, 
  additional_resources=additional_resources, 
  allocation_types=allocation_types, 
  tries_for_allocation=10, 
  epsilon=1.2, 
  available_volumes=available_demand_volumes
)
