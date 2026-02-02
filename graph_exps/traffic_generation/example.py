# импорт генерации своего трафика
from .traffic_generation.generation_for_exp import generate_own_traffic
# импорт генераци SNR-BA графа
from .SNRBA_generation.generation import generate_snr_ba_graph
# импорт окружения эксперимента
from .exp import *

# допустимые веса запросов
available_demand_volumes = ((1, 0.9), (2, 0.1))

# новые ресурсы и типы для распределения ресурсов
additional_resources = [80.0, 80.0, 80.0]
allocation_types = ['alpha', 'random_alpha', 'random']

# генерация SNR-BA графа
snr_ba_graph = generate_snr_ba_graph(num_nodes=30, capacity_value=80.0)

# генерация своего трафика методом "alpha_with_sa" и эксперимент для графа
adj_graph_alpha, traffic_graph_alpha =  generate_own_traffic(
  snr_ba_graph, 
  available_demand_volumes, 
  generation_type="alpha_with_sa", 
  generation_params={"alpha": 0.35, "epsilon": 0.025, "median_weight_for_initial": 20, "var_for_initial": 1, "multi_max": 5, "t": 0.5}
)
graph_for_exp_alpha = HuGraphForExps(adj_graph_alpha, traffic_graph_alpha)
results_alpha = expand_test_for_graph(graph_for_exp_alpha, additional_resources, allocation_types, tries_for_allocation=10, epsilon=1.2, available_volumes=available_demand_volumes)

# генерация своего трафика методом "gravity" и эксперимент для графа
adj_graph_gravity, traffic_graph_gravity =  generate_own_traffic(
  snr_ba_graph, 
  available_demand_volumes, 
  generation_type="gravity", 
  generation_params={"beta": 0.15, "dyn_k": 0.9}
)
graph_for_exp_gravity = HuGraphForExps(adj_graph_gravity, traffic_graph_gravity)
results_gravity = expand_test_for_graph(graph_for_exp_gravity, additional_resources, allocation_types, tries_for_allocation=10, epsilon=1.2, available_volumes=available_demand_volumes)
