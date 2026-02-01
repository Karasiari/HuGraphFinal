from typing import Tuple, Dict, Any
import copy

import networkx as nx

from .params_for_generation import check_params

from .HuGraphForGen.core import HuGraphForGen
from .HuGraphForGen.generation.generator_with_gravity import GravitationalGenerator
from .HuGraphForGen.generation.generator import GeneratorMultiGraph
from .HuGraphForGen.generation.generator_with_sa import GeneratorMultiGraphWithSA
  

# алиасы для читаемости
VolumeWithProbability = Tuple[int, float]
GenerationParams = Dict[str, Any]
GenerationResults = Tuple[nx.MultiGraph, nx.MultiDiGraph]


# функция для нахождения greatest connected component графа связности - генерация корректна только на связных графах
def find_gcc(
  graph: nx.MultiGraph
) -> nx.MultiGraph:
  """
  Находит наибольшую компоненту связности графа - для корректной генерации трафика на input приходит связный граф смежности
  """
  gcc = max(nx.connected_components(graph), key=len)
  connected_graph = graph.subgraph(gcc).copy()
  mapping = {old: new for new, old in enumerate(connected_graph.nodes())}
  return nx.relabel_nodes(connected_graph, mapping)


# функция для агрегации мультиграфа в обычный граф

def aggregate_multigraph(
  multigraph: nx.MultiGraph | nx.MultiDiGraph
) -> nx.Graph:
  """
  Агрегация графа
  """
  graph = nx.Graph()
  graph.add_nodes_from(range(multigraph.number_of_nodes()))
  for u, v, data in multigraph.edges(data=True):
    weight = data['capacity']
    if graph.has_edge(u, v):
      graph[u][v]['weight'] += weight
    else:
      graph.add_edge(u, v, weight=weight)
  return graph


# функции генерации по типу генерации

def gravity_generation(
  graph: nx.Graph, 
  generation_params: GenerationParams,
  recommended_params: bool
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)
  valid_generation_params = check_params(graph_for_generation, generation_type, generation_params, recommended_params)
  
  generator = GravitationalGenerator(**valid_generation_params)
  generator.generate(graph_for_generation)
  traffic_graph = graph_for_generation.demands_graph
  aggregated_traffic_graph = aggregate_multigraph(traffic_graph)
  return traffic_graph

def alpha_generation(
  graph: nx.Graph, 
  generation_params: GenerationParams,
  recommended_params: bool
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)
  valid_generation_params = check_params(graph_for_generation, generation_type, generation_params, recommended_params)
  
  generator = GeneratorMultiGraph(**valid_generation_params)
  generator.generate(graph_for_generation)
  traffic_graph = graph_for_generation.demands_graph
  aggregated_traffic_graph = aggregate_multigraph(traffic_graph)
  return traffic_graph

def alpha_with_sa_generation(
  graph: nx.Graph, 
  generation_params: GenerationParams,
  recommended_params: bool
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)
  valid_generation_params = check_params(graph_for_generation, generation_type, generation_params, recommended_params)
  
  generator = GeneratorMultiGraphWithSA(**valid_generation_params)
  generator.generate(graph_for_generation)
  traffic_graph = graph_for_generation.demands_graph
  aggregated_traffic_graph = aggregate_multigraph(traffic_graph)
  return traffic_graph


# функция для дробления сгенерированного трафика

def make_multi_demands(
  aggregated_traffic_graph: nx.Graph, 
  available_demand_volumes: Tuple[VolumeWithProbability, ...]
) -> nx.MultiDiGraph:
  return aggregated_traffic_graph
  

# основная функция для генерации своего трафика

def generate_own_traffic(
  graph: nx.MultiGraph, 
  available_demand_volumes: Tuple[VolumeWithProbability, ...],
  generation_type: str,
  generation_params: GenerationParams,
  recommended_params: bool = True
) -> GenerationResults:
  """
  Функция для генерации своего трафика по графу смежности - для основного эксперимента
  """
  connected_graph = find_gcc(graph)

  aggregated_graph = aggregate_multigraph(connected_graph)
  if generation_type == "gravity":
    aggregated_traffic_graph = gravity_generation(aggregated_graph, generation_params, recommended_params)
  elif generation_type == "alpha":
    aggregated_traffic_graph = alpha_generation(aggregated_graph, generation_params, recommended_params)
  elif generation_type == "alpha_with_sa":
    aggregated_traffic_graph = alpha_with_sa_generation(aggregated_graph, generation_params, recommended_params)
  else:
      raise ValueError(f"Тип генерации {generation_type} не известен")

  traffic_graph = make_multi_demands(aggregated_traffic_graph, available_demand_volumes)
  return connected_graph, traffic_graph
