from typing import Tuple, Dict, Any
import copy

import networkx as nx

from .HuGraphForGen.core import HuGraphForGen
  

# алиасы для читаемости
VolumeWithProbability = Tuple[int, float]
GenerationParam = Dict[str, Any]
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
  generation_params: Tuple[GenerationParam, ...]
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)

  

def alpha_generation(
  graph: nx.Graph, 
  generation_params: Tuple[GenerationParam, ...]
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)

  

def alpha_with_sa_generation(
  graph: nx.Graph, 
  generation_params: Tuple[GenerationParam, ...]
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)

  
      

# основная функция для генерации своего трафика

def generate_own_traffic(
  graph: nx.MultiGraph, 
  available_demand_volumes: Tuple[VolumeWithProbability, ...],
  generation_type: str,
  generation_params: Tuple[GenerationParam, ...]: 
) -> GenerationResults:
  """
  Функция для генерации своего трафика по графу смежности - для основного эксперимента
  """
  traffic_graph: nx.MultiDiGraph = nx.MultiDiGraph()
  connected_graph = find_gcc(graph)

  aggregated_graph = aggregate_multigraph(connected_graph)
  if generation_type == "gravity":
    aggregated_traffic_graph = gravity_generation(aggregated_graph, generation_params)
  elif generation_type == "alpha":
    aggregated_traffic_graph = alpha_generation(aggregated_graph, generation_params)
  elif generation_type == "alpha_with_sa":
    aggregated_traffic_graph = alpha_with_sa_generation(aggregated_graph, generation_params)
  else:
      raise ValueError(f"Тип генерации {generation_type} не известен")
  return connected_graph, traffic_graph
