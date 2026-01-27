import random
import copy
import networkx as nx
from typing import Optional, Dict, Any, Tuple, List

import pickle
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .core import HuGraphForExps
from .instruments_for_exps import * # импорт вспомогательных функций
                                                                                                

# функция для параллелизованного расчета метрики α для ВСЕХ ребер графа

def compute_alpha_for_all_edges(graph: HuGraphForExps, n_jobs=8) -> List[Tuple[Tuple[int, int], float]]:
    """
    Рассчитывает с параллелизацией процесса метрику α для ВСЕХ ребер графа - для дальнейшего предпочтительного по метрике распределения новых ресурсов в эксперименте
    Input: граф-объект класса HuGraphsExps, n_jobs
    Output: Список ребер в виде ((source, target), значение α)
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

def expand_network_for_type(graph: HuGraphForExps, edges_with_alphas: List[Tuple[Tuple[int, int], float]], resources_to_add: List[float], allocation_type: str) -> HuGraphForExps:
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
def allocation_test(graphs: Dict[str, HuGraphForExps], tries_for_allocation: int, n_jobs=-1):
    tasks = []
    for allocation_type, graph in graphs.items():
      for try_number in range(tries_for_allocation):
        graph_copy = graph.copy()
        tasks.append((graph_copy, allocation_type))

    results_all = Parallel(n_jobs=n_jobs)(
        delayed(allocate_spare_capacity)(graph, allocation_type)
        for graph, allocation_type in tqdm(tasks, desc="Processing allocation", total=len(tasks))
    )
    return results_all 
    
# основная функция для эксперимента по расширению для ОДНОГО графа

def expand_test_for_graph(graph: HuGraphForExps, additional_resources: List[float], allocation_types: List[str], tries_for_allocation: int):
    # рассчитываем метрику α для ребер графа
    edges_with_alphas = compute_alpha_for_all_edges(graph)
    
    # распределяем новые ресурсы согласно типу аллокации и получаем расширенные сети
    additional_resources.sort(reverse=True)
    expanded_graphs = {}
    for allocation_type in allocation_types:
        expanded_graph = expand_network_for_type(graph, edges_with_alphas, additional_resources, allocation_type)
        expanded_graphs[allocation_type] = expanded_graph

    # проводим тест на перепрокладку на расширенных графах
    allocation_results = allocation_test(expanded_graphs, tries_for_allocation)
    return allocation_results
