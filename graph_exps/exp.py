from dataclasses import replace
import random
import pandas as pd

import pickle
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# импорт вспомогательных функций
from .instruments_for_exps import *

# алиасы для читаемости
EdgeKey = Tuple[int, int]
EdgeWithParameter = Tuple[EdgeKey, float | None]
RouteResult = Dict[int, List[Tuple[int, int, int]]]
DemandsDict = Dict[int, Tuple[int, int, int]]
RemainingNetwork = Tuple[nx.DiGraph, nx.MultiDiGraph]
AddVolumeByEdge = Dict[EdgeKey, int]
AllocationResult = Tuple[str, Tuple[bool, float, AddVolumeByEdge]]
NetworksForExp = Tuple[nx.MultiDiGraph, Dict[EdgeKey, RemainingNetwork]]
                                                                                                

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
    source_target_sequence = [(min(u, v), max(u, v)) for u, v in graph.graph.edges()]

    results_all = Parallel(n_jobs=n_jobs)(
      delayed(compute_alpha_for_edge)(graph_state, u, v)
      for u, v in tqdm(source_target_sequence, desc="Processing all edges", total=len(source_target_sequence))
    )
    edges_with_alphas = [r for r in results_all if r is not None]
    return edges_with_alphas


# функция для обработки решения исходного MCF в рамках эксперимента

def convert_mcf_results_for_exp(
    graph: nx.MultiDiGraph, 
    route_result: RouteResult,
    demands: DemandsDict
) -> Tuple[Dict[EdgeKey, RemainingNetwork], int]:
    """
    Обрабатывает решение исходной задачи MCF на нерасширенном графе
    под основной эксперимент расширения
    По результатам решения исходной задачи MCF
    - для каждого ребра рассматривает сценарий его падения и формирует остаточную сеть для этого сценария
    - вычисляет total volume запросов для потенциальной перепрокладки
    Input:
          graph - нерасширенная версия графа, 
                  на котором проводится эксперимент
          route_result - результат решения исходного MCF 
                         как словарь проложенных по ребрам путей 
                         по индексу запроса
          demands - информация по проложенным в ходе решения исходного MCF запросам
                    как словарь по индексу запроса
    Output:
          Возвращает
          - словарь нерасширенных остаточных сетей по EdgeKey ребра, для которого остаточная сеть считается
          - суммарный volume запросов для потенциальной перепрокладки для ВСЕХ сценариев падений ребер 
    """
    remaining_networks, volume_to_reroute = get_remaining_networks_and_volume_to_reroute(graph, route_result, demands)
    return remaining_networks, volume_to_reroute


# функция для расширения сети

def expand_network_for_type(
    graph: nx.MultiDiGraph, 
    unexpanded_remaining_networks: Dict[EdgeKey, RemainingNetwork],
    edges_with_alphas: List[EdgeWithParameter], 
    resources_to_add: List[int], 
    allocation_type: str
) -> Tuple[nx.MultiDiGraph, 
           Dict[EdgeKey, RemainingNetwork]]:
    """
    Создает расширенный новыми ресурсами граф по типу распределения ресурсов,
    получает расширенные остаточные сети для каждого типа распределения ресурсов
    Input: 
           graph - граф для расширения
           unexpanded_remaining_networks - нерасширенные остаточные сети 
                                           как словарь сетей по EdgeKey
           edges_with_alphas - список всех ребер с расчитанной метрикой α
           resources_to_add - список новых ресурсов как список capacity
           allocation_type - тип распределения ресурсов
    Output:
           expanded_graph - расширенный граф
           expanded_remaining_networks - расширенные остаточные сети в форме исходного словаря
    """
    expanded_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    expanded_remaining_networks: Dict[EdgeKey, RemainingNetwork] = {}

    def filter_edges(edges, min_ratio, max_ratio):
      if min_ratio < 0 or max_ratio > 1:
        raise ValueError("Некорректные значения min и max")
    
      sorted_edges = sorted(edges, key=lambda x: x[1])
      n = len(sorted_edges)
    
      min_count = int(n * min_ratio)
      max_count = int(n * max_ratio)
    
      start_index = min_count
      end_index = n - max_count
      return sorted_edges[start_index:end_index]
  
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

    # пробный тип
    elif allocation_type == "alpha_mid_random":
        mid_edges = filter_edges(edges_with_alphas, 0.7, 0.2)
        random.shuffle(mid_edges)
        edges_to_expand = [edge for edge, _ in mid_edges[:number_of_new_resources]]
        source_target_sequence_for_new_resources = list(zip(edges_to_expand, resources_to_add)) 
    elif allocation_type == "alpha_high_random":
        mid_edges = filter_edges(edges_with_alphas, 0.8, 0.1)
        random.shuffle(mid_edges)
        edges_to_expand = [edge for edge, _ in mid_edges[:number_of_new_resources]]
        source_target_sequence_for_new_resources = list(zip(edges_to_expand, resources_to_add)) 
    elif allocation_type == "alpha_with_support";
        continue
    else:
        raise ValueError(f"Тип распределения ресурсов {allocation_type} не предусмотрен экспериментом")

    expanded_graph = expand_graph(graph, source_target_sequence_for_new_resources)
    for failed_edge, unexpanded_remaining_network in unexpanded_remaining_networks.items():
      resources_for_remaining_network = []
      for additional_edge, capacity in source_target_sequence_for_new_resources:
        if additional_edge != failed_edge:
          resources_for_remaining_network.append((additional_edge, capacity))
      unexpanded_remaining_topology_graph, remaining_traffic_graph = unexpanded_remaining_network
      remaining_topology_graph = expand_remaining_network_graph(unexpanded_remaining_topology_graph, resources_for_remaining_network)
      expanded_remaining_networks[failed_edge] = (remaining_topology_graph.copy(), remaining_traffic_graph.copy())
    return expanded_graph, expanded_remaining_networks


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
      zero_traffic_graph = nx.MultiDiGraph()
      zero_traffic_graph.add_nodes_from(slack_graph)
      slack_hugraph = HuGraphForExps(slack_graph, zero_traffic_graph)
      edges_in_cut = slack_hugraph.generate_cut()
      for edge in edges_in_cut:
        edges_for_remaining_networks.add(tuple(sorted(edge)))
    else:
      raise ValueError(f"Тип выборки {remaining_networks_pref} для ребер под остаточные сети не предусмотрен")

  return edges_for_remaining_networks
      
  
# функция для теста на перепрокладку при падении ребер

def allocation_test(
    networks: Dict[str, NetworksForExp], 
    route_result: RouteResult,
    demands: DemandsDict,
    volume_to_reroute: int,
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
    Функция для проведения перепрокладки на уже расширенных графах:
    1) Для каждой расширенной остаточной сети решает max concurrent flow problem - нужный нам результат gamma остаточной сети
    2) Использует результаты решения задачи MCF для задачи перепрокладки - задача решается нашим алгоритмом spare capacity allocation
    3) Возвращает общие результаты
    Input:
          networks     - словарь расширенных сетей по типу распределения ресурсов,
                         каждая расширенная сеть это:
                         -- расширенный исходный граф, на котором проводится эксперимент
                         -- словарь расширенных остаточных сетей по EdgeKey
          route_result - результат решения исходного MCF 
                         как словарь проложенных по ребрам путей 
                         по индексу запроса
          demands      - информация по проложенным в ходе решения исходного MCF запросам
                         как словарь по индексу запроса
          volume_to_reroute - суммарное значение volume запросов для перепрокладки 
                              для всех сценариев падений ребер
                              по типу распределения ресурсов,
                              знаменатель для нормировки rerouted volume ratio
          edges_for_remaining_networks - множество ребер, для которых мы будем рассматривать 
                                         остаточные сети в эксперименте
          tries_for_allocation - количество запусков жадного алгоритма spare capacity allocation
          epsilon - scaling параметр для алгоритма перепрокладки для резервирования дополнительных запросов
          available_volumes - распределение возможных весов резервных запросов в алгоритме перепрокладки 
                             как кортеж кортежей вида (значение, вероятность)
          random_seed
          n_jobs
    Output:
          algorithm_results - результаты решений задачи перепрокладки
          remaining_networks_gammas_by_type - словарь по типу распределения ресурсов 
                                              списков gamma остаточной сети по упавшему ребру
          volume_to_reroute - передаем в эту функцию для удобства
    """
    # преобразуем решение исходного MCF под алгоритм spare capacity allocation
    tasks_for_converting = []
    for allocation_type, network in networks.items():
      graph = network[0]
      tasks_for_converting.append((graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed))
    converted_results = Parallel(n_jobs=n_jobs)(
       delayed(convert_mcf_results_to_greedy_input)(graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed)
       for graph, allocation_type, demands, route_result, epsilon, available_volumes, random_seed in tqdm(tasks_for_converting, desc="Converting initial MCF results", total=len(tasks_for_converting))
    )

    # параллельно считаем gamma для остаточных сетей
    remaining_networks_gammas_by_type = {}
    for allocation_type, network in networks.items():
      remaining_networks = network[1]
      remaining_networks_by_failed_edge = [(edge, remaining_network) for edge, remaining_network in remaining_networks.items() if edge in edges_for_remaining_networks]
      remaining_networks_gammas = Parallel(n_jobs=n_jobs)(
          delayed(solve_mcfp_for_exp)(edge, remaining_network)
          for edge, remaining_network in tqdm(remaining_networks_by_failed_edge, desc=f"Solving remaining network MCFPs for {allocation_type}", total=len(remaining_networks_by_failed_edge))
      )
      remaining_networks_gammas_by_type[allocation_type] = remaining_networks_gammas

    # параллельно запускаем алгоритм spare capacity allocation
    tasks_for_allocation = []
    for allocation_type, input_for_algorithm in converted_results:
      initial_random_seed = input_for_algorithm.random_seed
      for try_number in range(tries_for_allocation):
        random_seed_for_try = initial_random_seed + try_number if initial_random_seed is not None else None
        new_input = replace(
          input_for_algorithm, 
          random_seed=random_seed_for_try
        )
        tasks_for_allocation.append((new_input, allocation_type))
    algorithm_results = Parallel(n_jobs=n_jobs)(
        delayed(allocate_spare_capacity)(input_for_algorithm, allocation_type)
        for input_for_algorithm, allocation_type in tqdm(tasks_for_allocation, desc="Processing allocation", total=len(tasks_for_allocation))
    )
    return algorithm_results, remaining_networks_gammas_by_type, volume_to_reroute


# функция для получения итоговых результатов эксперимента по графу в нужном формате

def get_right_output(
    allocation_results_raw: Tuple[List[AllocationResult], 
                            Dict[str, List[EdgeWithParameter]],
                            int]
) -> pd.DataFrame:
    """
    По результатам эксперимента формируем табличку pandas DataFrame
    Input:
          allocation_results_raw 
          - кортеж из
          -- algorithm_results - результаты решений задачи перепрокладки
          -- remaining_networks_gammas_by_type - словарь по типу распределения ресурсов 
                                                 списков gamma остаточной сети по упавшему ребру
          -- volume_to_reroute - суммарное значение volume запросов для перепрокладки 
                                 для всех сценариев падений ребер,
                                 знаменатель для нормировки rerouted volume ratio
    Output:
          результаты в виде pandas DataFrame
    """
    result_dict = {}
    gammas_dict = {}
    allocation_seen = {}
    algorithm_results, remaining_networks_gammas_by_type, volume_to_reroute = allocation_results_raw
  
    for allocation_type, remaining_networks_gammas in remaining_networks_gammas_by_type.items():
      gammas_dict[allocation_type] = {}
      for edge, remaining_network_gamma in remaining_networks_gammas:
        gammas_dict[allocation_type][f'gamma for failed {edge}'] = round(remaining_network_gamma, 2) if remaining_network_gamma is not None else None
  
    for allocation_type, result_raw in algorithm_results:
      result = {'allocation solved': result_raw[0], 'rerouted volume ratio': round(result_raw[1] / volume_to_reroute, 2)}
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

    # обрабатываем решение исходной задачи MCF для нашего эксперимента
    unexpanded_remaining_networks, total_volume_to_reroute = convert_mcf_results_for_exp(multidigraph, route_result, demands)
    
    # распределяем новые ресурсы согласно типу аллокации и получаем расширенные сети
    additional_resources.sort(reverse=True)
    expanded_networks = {}
    expanded_graphs = {}
    for allocation_type in allocation_types:
        expanded_networks[allocation_type] = expand_network_for_type(multidigraph, unexpanded_remaining_networks, edges_with_alphas, additional_resources, allocation_type)
        expanded_graphs[allocation_type] = expanded_networks[allocation_type][0]

    # определяем, для каких ребер рассматриваем остаточные сети
    edges_for_remaining_networks = get_edges_for_remaining_networks(expanded_graphs, remaining_networks_pref, route_result, demands)

    # проводим эксперимент на расширенных графах
    allocation_results_raw = allocation_test(expanded_networks, route_result, demands, total_volume_to_reroute, edges_for_remaining_networks, tries_for_allocation, epsilon, available_volumes, random_seed)

    # получаем нужный формат output
    allocation_results = get_right_output(allocation_results_raw)
    return allocation_results
