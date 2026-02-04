import random
import copy

import pandas as pd

import pickle
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# импорт вспомогательных функций
from .instruments_for_exps import *

# алиасы для читаемости
EdgeWithParameter = Tuple[Tuple[int, int], float | None]
RouteResult = Dict[int, List[Tuple[int, int, int]]]
DemandsDict = Dict[int, Tuple[int, int, int]]
EdgeKey = Tuple[int, int]
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
      for u, v in tqdm(source_target_sequence, desc="Processing all edges", total=len(source_target_sequence))
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


# функция для выборки ребер, для которых мы будем рассматривать остаточные сети в ходе эксперимента

def get_edges_for_remaining_networks(
  expanded_graphs: Dict[str, nx.MultiDiGraph], 
  remaining_networks_pref: str, 
  route_result: RouteResult, 
  demands: DemandsDict
) -> Set[EdgeKey]:
  """
  Функция для формирования выборки ребер, 
  для которых мы будем рассматривать остаточные сети в эксперименте
  Input:
        expanded_graphs - словарь расширенных графов
        remaining_networks_pref - тип выборки ребер
                                -- "all" - рассматриваем все сценарии
                                -- "mincuts" - рассматриваем ребра из разрезов 
                                    остаточных по решению исходного MCF расширенных графов
        route_result - результат решения исходного MCF 
                       как словарь проложенных по ребрам путей 
                       по индексу запроса
        demands - информация по проложенным в ходе решения исходного MCF запросам
                  как словарь по индексу запроса
  Output:
        Множество ребер, для которых мы будем рассматривать остаточные сети в эксперименте
  """
  edges_for_remaining_networks: Set[EdgeKey] = set()
    
  for _, multidigraph in expanded_graphs.items():
    if remaining_networks_pref == "all":
      edges_for_remaining_networks = set(tuple(sorted(edge[:2])) for edge in multidigraph.edges())
      break
    elif remaining_networks_pref == "mincuts":
      slack_graph = get_slack_graph(multidigraph, route_result, demands)
      almost_zero_traffic_graph = nx.MultiDiGraph()
      almost_zero_traffic_graph.add_nodes_from(slack_graph)
      #almost_zero_traffic_graph.add_edge(0, 1, weight=1.0)
      slack_hugraph = HuGraphForExps(slack_graph, almost_zero_traffic_graph)
      edges_in_cut = slack_hugraph.generate_cut()
      for edge in edges_in_cut:
        edges_for_remaining_networks.add(sorted(edge))
    else:
      raise ValueError(f"Тип выборки {remaining_networks_pref} для ребер под остаточные сети не предусмотрен")

  return edges_for_remaining_networks
      
  
# функция для теста на перепрокладку при падении ребер

def allocation_test(
    graphs: Dict[str, nx.MultiDiGraph], 
    route_result: RouteResult,
    demands: DemandsDict,
    edges_for_remaining_networks: Set[EdgeKey],
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
    1) По результатам решения исходной задачи MCF для каждого ребра рассматривает сценарий его падения и формирует остаточную сеть для этого сценария
    2) Для каждой остаточной сети решает max concurrent flow problem - нужный нам результат gamma остаточной сети
    3) Использует результаты решения задачи MCF для задачи перепрокладки - задача решается нашим алгоритмом spare capacity allocation
    4) Возвращает общие результаты
    Input:
          graphs - словарь расширенных графов по типу распределения ресурсов
          route_result - результат решения исходного MCF 
                         как словарь проложенных по ребрам путей 
                         по индексу запроса
          demands - информация по проложенным в ходе решения исходного MCF запросам
                    как словарь по индексу запроса
          edges_for_remaining_networks - множество ребер, для которых мы будем рассматривать 
                                         остаточные сети в эксперименте
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
    # преобразуем решение исходного MCF и находим остаточные сети
    tasks_for_converting = []
    for allocation_type, graph in graphs.items():
      tasks_for_converting.append((graph, route_result, demands, allocation_type, epsilon, available_volumes))
    converted_results = Parallel(n_jobs=n_jobs)(
       delayed(convert_mcf_results_for_exp)(graph, route_result, demands, allocation_type, epsilon, available_volumes)
       for graph, route_result, demands, allocation_type, epsilon, available_values in tqdm(tasks_for_converting, desc="Converting initial MCF results", total=len(tasks_for_converting))
    )

    # параллельно считаем gamma для остаточных сетей
    remaining_networks_gammas_by_type = {}
    volume_to_reroute_by_type = {}
    for allocation_type, _, remaining_networks, volume_to_reroute in converted_results:
      remaining_networks_by_failed_edge = [(edge, remaining_network) for edge, remaining_network in remaining_networks.items() if edge in edges_for_remaining_networks]
      remaining_networks_gammas = Parallel(n_jobs=n_jobs)(
          delayed(solve_mcfp_for_exp)(edge, network)
          for edge, network in tqdm(remaining_networks_by_failed_edge, desc=f"Solving remaining network MCFPs for {allocation_type}", total=len(remaining_networks_by_failed_edge))
      )
      remaining_networks_gammas_by_type[allocation_type] = remaining_networks_gammas
      volume_to_reroute_by_type[allocation_type] = volume_to_reroute

    # параллельно запускаем алгоритм spare capacity allocation
    tasks_for_allocation = []
    for allocation_type, input_for_algorithm, _, _ in converted_results:
      for try_number in range(tries_for_allocation):
        tasks_for_allocation.append((input_for_algorithm, allocation_type))
    algorithm_results = Parallel(n_jobs=n_jobs)(
        delayed(allocate_spare_capacity)(input_for_algorithm, allocation_type)
        for input_for_algorithm, allocation_type in tqdm(tasks_for_allocation, desc="Processing allocation", total=len(tasks_for_allocation))
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
    random_seed: int | None = None,
    remaining_networks_pref: str = "all"
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
          random_seed - для перепрокладки
          remaining_networks_pref - тип выборки ребер, для которых рассматриваем 
                                    остаточные сети в эксперименте
                                    -- "all" - рассматриваем все сценарии
                                    -- "mincuts" - рассматриваем ребра из разрезов 
                                       остаточных по решению исходного MCF расширенных графов
    Output:
          табличка с результатами эксперимента как pandas DataFrame
    """
    # рассчитываем метрику α для ребер графа
    edges_with_alphas = compute_alpha_for_all_edges(graph)

    # решаем задачу MCF на исходном графе - получаем начальную маршрутизацию трафика
    route_result, demands, _, multidigraph = graph.solve_mcf()
    
    # распределяем новые ресурсы согласно типу аллокации и получаем расширенные сети
    additional_resources.sort(reverse=True)
    expanded_graphs = {}
    for allocation_type in allocation_types:
        expanded_graphs[allocation_type] = expand_network_for_type(multidigraph, edges_with_alphas, additional_resources, allocation_type)

    # определяем, для каких ребер рассматриваем остаточные сети
    edges_for_remaining_networks = get_edges_for_remaining_networks(expanded_graphs, remaining_networks_pref, route_result, demands)

    # проводим эксперимент на расширенных графах
    allocation_results_raw = allocation_test(expanded_graphs, route_result, demands, edges_for_remaining_networks, tries_for_allocation, epsilon, available_volumes, random_seed)

    # получаем нужный формат output
    allocation_results = get_right_output(allocation_results_raw)
    return allocation_results
