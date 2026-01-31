import random
import copy
import networkx as nx
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List

import pickle
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .core import HuGraphForExps
from .instruments_for_exps import * # импорт вспомогательных функций

# алиасы для читаемости outputов
EdgeWithParameter = Tuple[Tuple[int, int], float | None]
AllocationResult = Tuple[str, Tuple[bool, float]]
                                                                                                

# функция для расчета метрики α для ВСЕХ ребер графа

def compute_alpha_for_all_edges(
    graph: HuGraphForExps, 
    n_jobs=-1
) -> List[EdgeWithParameter]:
    """
    Рассчитывает с параллелизацией процесса метрику α для ВСЕХ ребер графа - для дальнейшего предпочтительного по метрике распределения новых ресурсов в эксперименте
    Input: 
          graph - граф-объект класса HuGraphsExps, 
          n_jobs
    Output: 
          Список ребер в виде EdgeWithParameter
    """
    # Проверка, что граф сериализуем
    try:
        graph_state = pickle.dumps(graph)
    except Exception as e:
        raise ValueError("Graph is not pickle-serializable. Ensure GraphMCFexps supports pickle.") from e

    # Обработка всех рёбер
    source_target_sequence = [(u, v) for u, v in graph.graph.edges()]
    
    results_all = Parallel(n_jobs=n_jobs)(
        delayed(compute_alpha_for_edge)(graph_state, u, v)
        for u, v in tqdm(source_target_sequence, desc="Processing all edges")
    )
    edges_with_alphas = [r for r in results_all if r is not None]
    return edges_with_alphas

# функция для расширения сети

def expand_network_for_type(
    graph: HuGraphForExps, 
    edges_with_alphas: List[EdgeWithParameter], 
    resources_to_add: List[int], 
    allocation_type: str
) -> HuGraphForExps:
    """
    Создает расширенный новыми ресурсами граф по типу распределения ресурсов
    Input: 
           graph - граф для расширения,
           edges_with_alphas - список всех ребер с расчитанной метрикой α,
           resources_to_add - список новых ресурсов как список capacity
           allocation_type - тип распределения ресурсов
    Output:
           расширенный граф
    """
    number_of_new_resources = len(resources_to_add)
    # добавляем новые ресурсы предпочтительно по значению метрики α ребра
    if allocation_type == "alpha":
        edges_with_alphas.sort(key=lambda x: x[1], reverse=True)
        edges_to_expand = [edge for edge, _ in edges_with_alphas[:number_of_new_resources]]
        source_target_sequence_for_new_resources = list(zip(edges_to_expand, resources_to_add))
        
    # добавляем новые ресурсы в порядке - сначала СЛУЧАЙНО выбираем ребра для расширения, потом СРЕДИ ВЫБРАННЫХ распределяем предпочтительно по значению метрики α ребра
    elif allocation_type == "random_alpha":
        random.shuffle(edges_with_alphas)
        edges_to_expand = edges_with_alphas[:number_of_new_resources]
        edges_to_expand.sort(key=lambda x: x[1], reverse=True)
        edges_to_expand = [edge for edge, _ in edges_to_expand]
        source_target_sequence_for_new_resources = list(zip(edges_to_expand, resources_to_add))

    # добавляем новые ресурсы для СЛУЧАЙНО ВЫБРАННЫХ ребер
    elif allocation_type == "random":
        random.shuffle(edges_with_alphas)
        edges_to_expand = [edge for edge, _ in edges_with_alphas[:number_of_new_resources]]
        source_target_sequence_for_new_resources = list(zip(edges_to_expand, resources_to_add))
    else:
        raise ValueError(f"Тип распределения ресурсов {allocation_type} не предусмотрен экспериментом")

    graph_copy_to_expand = graph.copy()
    expanded_graph = expand_graph(graph_copy_to_expand, source_target_sequence_for_new_resources)
    return expanded_graph

# функция для теста на перепрокладку при падении ребер

def allocation_test(
    graphs: Dict[str, HuGraphForExps], 
    tries_for_allocation: int,
    epsilon: float = 1.0,
    available_volumes: Tuple[Tuple[int, float], ...] = ((1, 1.0),),
    random_seed: int | None = None,
    n_jobs=-1
) -> Tuple[List[AllocationResult], 
           Dict[str, List[EdgeWithParameter]],
           Dict[str, int]]:
    """
    Функция для проведения перепрокладки на уже расширенном графе:
    1) Решает задачу MCF на расширенном графе:
       -- Получает результаты решения
       -- По результатам решения задачи MCF для каждого ребра рассматривает сценарий его падения и формирует остаточную сеть для этого сценария
    2) Для каждой остаточной сети решает max concurrent flow problem - нужный нам результат gamma остаточной сети
    3) Использует результаты решения задачи MCF для задачи перепрокладки - задача решается нашим алгоритмом spare capacity allocation
    4) Возвращает общие результаты решений двух задач
    Input:
          graphs - словарь расширенных графов по типу распределения ресурсов
          tries_for_allocation - количество запусков жадного алгоритма spare capacity allocation
          epsilon - scaling параметр для алгоритма перепрокладки для резервирования дополнительных запросов
          available_volumes - распределение возможных весов резервных запросов в алгоритме перепрокладки 
                             как кортеж кортежей вида (значение, вероятность)
          n_jobs
    Output:
          algorithm_results - результаты решений задачи перепрокладки
          remaining_networks_gammas_by_type - словарь по типу распределения ресурсов 
                                              списков gamma остаточной сети по упавшему ребру
          volume_to_reroute_by_type - суммарное значение volume запросов для перепрокладки 
                                      для всех сценариев падений ребер
                                      по типу распределения ресурсов,
                                      знаменатель для нормировки rerouted volume ratio
    """
    # решаем исходный MCF с помощью параллельного расчета на всех расширенных графах
    tasks_for_mcf = []
    for allocation_type, graph in graphs.items():
      graph_copy = graph.copy()
      tasks_for_mcf.append((graph_copy, allocation_type, epsilon, available_volumes))
    mcf_results = Parallel(n_jobs=n_jobs)(
       delayed(solve_mcf_for_exp)(graph, allocation_type, epsilon, available_volumes)
       for graph, allocation_type, epsilon, available_values in tqdm(tasks_for_mcf, desc="Solving initial MCF", total=len(tasks_for_mcf))
    )

    # параллельно считаем gamma для остаточных сетей
    remaining_networks_gammas_by_type = {}
    volume_to_reroute_by_type = {}
    for allocation_type, _, remaining_networks, volume_to_reroute in mcf_results:
      remaining_networks_by_failed_edge = [(edge, remaining_network) for edge, remaining_network in remaining_networks.items()]
      remaining_networks_gammas = Parallel(n_jobs=n_jobs)(
          delayed(solve_mcfp_wrapper)(edge, network)
          for edge, network in tqdm(remaining_networks_by_failed_edge, desc=f"Solving remaining network MCFPs for {allocation_type}", total=len(remaining_networks_by_failed_edge))
      )
      remaining_networks_gammas_by_type[allocation_type] = remaining_networks_gammas
      volume_to_reroute_by_type[allocation_type] = volume_to_reroute

    # параллельно запускаем алгоритм spare capacity allocation
    tasks_for_allocation = []
    for allocation_type, input_for_algorithm, _, _ in mcf_results:
      for try_number in range(tries_for_allocation):
        tasks_for_allocation.append((input_for_algorithm, allocation_type))
    algorithm_results = Parallel(n_jobs=n_jobs)(
        delayed(allocate_spare_capacity)(input_for_algorithm, allocation_type)
        for graph, allocation_type in tqdm(tasks_for_allocation, desc="Processing allocation", total=len(tasks_for_allocation))
    )
    return algorithm_results, remaining_networks_gammas_by_type, volume_to_reroute_by_type

# функция для получения итоговых результатов эксперимента по графу в нужном формате

def get_right_output(
    allocation_results_raw: Tuple[List[AllocationResult], 
                            Dict[str, List[EdgeWithParameter]],
                            Dict[str, int]]
) -> pd.DataFrame:
    """
    По результатам эксперимента формируем табличку pandas DataFrame
    Input:
          allocation_results_raw 
          - кортеж из
          -- algorithm_results - результаты решений задачи перепрокладки
          -- remaining_networks_gammas_by_type - словарь по типу распределения ресурсов 
                                                 списков gamma остаточной сети по упавшему ребру
          -- volume_to_reroute_by_type - суммарное значение volume запросов для перепрокладки 
                                         для всех сценариев падений ребер
                                         по типу распределения ресурсов,
                                         знаменатель для нормировки rerouted volume ratio
    Output:
          результаты в виде pandas DataFrame
    """
    result_dict = {}
    gammas_dict = {}
    allocation_seen = {}
    algorithm_results, remaining_networks_gammas_by_type, volume_to_reroute_by_type = allocation_results_raw
  
    for allocation_type, remaining_networks_gammas in remaining_networks_gammas_by_type.items():
      gammas_dict[allocation_type] = {}
      for edge, remaining_network_gamma in remaining_networks_gammas:
        gammas_dict[allocation_type][f'gamma for failed {edge}'] = round(remaining_network_gamma, 2)
  
    for allocation_type, result_raw in algorithm_results:
      result = {'allocation solved': result_raw[0], 'rerouted volume ratio': round(result_raw[1] / volume_to_reroute_by_type[allocation_type], 2)}
      result |= gammas_dict[allocation_type]
      if allocation_seen.get(allocation_type, False):
          allocation_seen[allocation_type] += 1
      else:
          allocation_seen[allocation_type] = 1
      result_dict[(allocation_type, allocation_seen[allocation_type])] = result.copy()
    
    result_df = pd.DataFrame(result_dict).T
    return result_df
  
    
# основная функция для эксперимента по расширению для ОДНОГО графа

def expand_test_for_graph(
    graph: HuGraphForExps, 
    additional_resources: List[float], 
    allocation_types: List[str], 
    tries_for_allocation: int,
    epsilon: float = 1.0,
    available_volumes: Tuple[Tuple[int, float], ...] = ((1, 1.0),),
    random_seed: int | None = None
) -> pd.DataFrame:
    """
    Функция для проведения основного эксперимента по расширению на одном графе
    Input:
          graph - граф для эксперимента ка объект класса HuGraphForExps
          additional_resources - список новых ресурсов как список capacity
          allocation_types - список типов распределения ресурсов
          tries_for_allocation - количество запусков алгоритма перепрокладки распределенных ресурсов
          epsilon - scaling параметр для алгоритма перепрокладки для резервирования дополнительных запросов
          available_volumes - распределение возможных весов резервных запросов в алгоритме перепрокладки 
                             как кортеж кортежей вида (значение, вероятность)
    Output:
          табличка с результатами эксперимента как pandas DataFrame
    """
    # рассчитываем метрику α для ребер графа
    edges_with_alphas = compute_alpha_for_all_edges(graph)
    
    # распределяем новые ресурсы согласно типу аллокации и получаем расширенные сети
    additional_resources.sort(reverse=True)
    expanded_graphs = {}
    for allocation_type in allocation_types:
        expanded_graph = expand_network_for_type(graph, edges_with_alphas, additional_resources, allocation_type)
        expanded_graphs[allocation_type] = expanded_graph.copy()

    # проводим эксперимент на расширенных графах
    allocation_results_raw = allocation_test(expanded_graphs, tries_for_allocation, epsilon, available_volumes, random_seed)

    # получаем нужный формат output
    allocation_results = get_right_output(allocation_results_raw)
    return allocation_results
