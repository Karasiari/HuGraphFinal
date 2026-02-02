from typing import Tuple, Dict, Any
import copy
import random

import networkx as nx
import numpy as np

# для подготовки гиперпараметров генерации
from .params_for_generation import check_params

# импорт генераторов
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
  Находит наибольшую компоненту связности графа - 
  для корректной генерации трафика на input приходит связный граф смежности
  Input:
        graph - граф смежности, на котором генерируем трафик
  Output:
        Наибольшая связная часть графа смежности
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
  Агрегация графа для правильного формата входа в алгоритмы генерации
  Input:
        multigraph - итоговый связный граф смежности, на котором генерируем трафик
  Output:
        Агрегированная версия графа
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
"""
Общий вид -
Input:
      graph - связный агрегированный граф смежности, на котором происходит генерация
      generation_type - тип генераии трафика
      generation_params - гиперпараметры генерации
      recommended_params - флаг, использовать ли рекомендованные гиперпараметры генерации
Output:
      Сгенерированный агрегированный граф трафика
"""

def gravity_generation(
  graph: nx.Graph, 
  generation_type: str,
  generation_params: GenerationParams,
  recommended_params: bool
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)
  valid_generation_params = check_params(graph_for_generation, generation_type, generation_params, recommended_params)
  
  generator = GravitationalGenerator(**valid_generation_params)
  generator.generate(graph_for_generation)
  traffic_graph = graph_for_generation.demands_graph
  return traffic_graph

def alpha_generation(
  graph: nx.Graph, 
  generation_type: str,
  generation_params: GenerationParams,
  recommended_params: bool
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)
  valid_generation_params = check_params(graph_for_generation, generation_type, generation_params, recommended_params)
  
  generator = GeneratorMultiGraph(**valid_generation_params)
  generator.generate(graph_for_generation)
  traffic_graph = graph_for_generation.demands_graph
  return traffic_graph

def alpha_with_sa_generation(
  graph: nx.Graph, 
  generation_type: str,
  generation_params: GenerationParams,
  recommended_params: bool
) -> nx.Graph:
  adj_matrix = nx.adjacency_matrix(graph).todense().tolist()
  graph_for_generation = HuGraphForGen(adj_matrix)
  valid_generation_params = check_params(graph_for_generation, generation_type, generation_params, recommended_params)
  
  generator = GeneratorMultiGraphWithSA(**valid_generation_params)
  generator.generate(graph_for_generation)
  traffic_graph = graph_for_generation.demands_graph
  return traffic_graph


# функция для дробления сгенерированного трафика

def make_multidemands(
  aggregated_traffic_graph: nx.Graph, 
  available_demand_volumes: Tuple[VolumeWithProbability, ...]
) -> nx.MultiDiGraph:
  """
  Функция для дробления трафика согласно допустимым значениям весов запросов
  Input:
        aggregated_traffic_graph - сгенерированный граф трафика с агрегированными запросами
        available_demand_volumes - распределение допустимых весов запросов 
                                   для дробления агрегированных запросов генерации
                                   как кортеж кортежей вида (значение, вероятность)
  Output:
        Граф трафика с ориентированными (ориентация выбирается случайно равновероятно) запросами 
        допустимых весов как nx.MultiDiGraph
  """
  import bisect
  draw = lambda probs: random.choices(*zip(*probs), k=1)[0]
  traffic_graph: nx.MultiDiGraph = nx.MultiDiGraph()
  
  def check_available_volumes(available_demand_volumes: Tuple[VolumeWithProbability, ...]) -> None:
    summed_probability = 0
    for demand_volume in available_demand_volumes:
      if not isinstance(demand_volume[0], int) or (demand_volume[0] <= 0) or (demand_volume[1] <= 0):
        raise ValueError(f"Недопустимое значение возможного веса запроса {demand_volume}")
      summed_probability += demand_volume[1]
    if summed_probability != 1:
      raise ValueError("Кортеж возможных весов запросов не является распределением")
  check_available_volumes(available_demand_volumes)
  available_demand_volume_values = sorted([value for value, probability in available_demand_volumes])

  traffic_graph.add_nodes_from(aggregated_traffic_graph)
  for u, v, data in aggregated_traffic_graph.edges(data=True):
    aggregated_volume = int(data['weight'])
    while aggregated_volume > 0:
      drawed_volume = draw(available_demand_volumes)
      while drawed_volume > aggregated_volume:
        pos = bisect.bisect_left(available_demand_volume_values, aggregated_volume)
        if pos == 0:
          aggregated_volume = 0
          break
        drawed_volume = available_demand_volume_values[pos-1]
        break
      if aggregated_volume > 0:
        if random.random() < 0.5:
          traffic_graph.add_edge(u, v, weight=drawed_volume)
        else:
          traffic_graph.add_edge(v, u, weight=drawed_volume)
        aggregated_volume -= drawed_volume
  return traffic_graph
  

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
  Input:
        graph - граф смежности, на котором генерируем свой трафик, как nx.MultiGraph
        available_demand_volumes - распределение допустимых весов запросов 
                                   для дробления агрегированных запросов генерации
                                   в make_multidemands,
                                   как кортеж кортежей вида (значение, вероятность)
        generation_type - тип генерации трафика:
                          "gravity", 
                          "alpha", 
                          "alpha_with_sa" как "alpha" с отжигом
        generation_params - гиперпараметры для генерации
        recommended_params - флаг, использовать ли рекомендованные гиперпараметры генерации
                            (см. рекомендации в params_for_generation.py)
  Output:
        Результаты генерации как кортеж из:
        - наибольшей связной части графа смежности 
        - сгенерированного на ней графа трафика как nx.MultiDiGraph
  """
  connected_graph = find_gcc(graph)
  aggregated_graph = aggregate_multigraph(connected_graph)
  
  if generation_type == "gravity":
    aggregated_traffic_graph = gravity_generation(aggregated_graph, generation_type, generation_params, recommended_params)
  elif generation_type == "alpha":
    aggregated_traffic_graph = alpha_generation(aggregated_graph, generation_type, generation_params, recommended_params)
  elif generation_type == "alpha_with_sa":
    aggregated_traffic_graph = alpha_with_sa_generation(aggregated_graph, generation_type, generation_params, recommended_params)
  else:
      raise ValueError(f"Тип генерации {generation_type} не известен")

  traffic_graph = make_multidemands(aggregated_traffic_graph, available_demand_volumes)
  return connected_graph, traffic_graph
