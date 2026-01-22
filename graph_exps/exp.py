import random
import copy
import networkx as nx
from typing import Optional, Dict, Any, Tuple, List

import pickle
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .core import HuGraphForExps
from .instruments_for_exps import * # импорт вспомогательных функций
from .spare_capacity_allocation_algorithm.input_converter import convert_to_greedy_input # импорт функции для преобразования данных под алгоритм перераспределения трафика

# функция для решения перераспределения трафика - в решении наш алгоритм

def allocate_spare_capacity(graph: HuGraphForExps, random_seed: int | None = None):
    route_result, demands, solved = graph.solve_mcf()
    input_for_algorithm = convert_to_greedy_input(graph.multigraph, demands, route_result, random_seed)

# функция для параллелизованного расчета метрику α для ВСЕХ ребер графа

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

# основная функция для эксперимента по расширению для ОДНОГО графа

def expand_test_for_graph(graph: HuGraphForExps, additional_resources: List[float], allocation_types: List[str]):
    # рассчитываем метрику α для ребер графа
    edges_with_alphas = compute_alpha_for_all_edges(graph)
    
    # распределяем новые ресуры согласно типу аллокации
    number_of_new_resources = len(additional_resources)
    resources_to_add = additional_resources.sort(reverse=True)
    
    for allocation_type in allocation_types:
        # добавляем новые ресурсы предпочтительно по значению метрики α ребра
        if allocation_type == "alpha":
            edges_with_alphas.sort(key=lambda x: x[1], reverse=True)
            edges_to_expand = [edge for edge, _ in edges_with_alphas[:number_of_new_resources]]
            source_target_sequence_for_new_resources = list(zip(edges_to_expand, resources_to_add))
            #!!!! функция для создания/добавления ресурсов в граф
        
        # добавляем новые ресурсы в порядке - сначала СЛУЧАЙНО выбираем ребра для расширения, потом СРЕДИ ВЫБРАННЫХ распределяем предпочтительно по значению метрики α ребра
        elif allocation_type == "random_alpha":
            random.shuffle(edges_with_alphas)
            edges_to_expand = edges_with_alphas[:number_of_new_resources]
            edges_to_expand.sort(key=lambda x: x[1], reverse=True)
            edges_to_expand = [edge for edge, _ in edges_to_expand]
            source_target_sequence_for_new_resources = list(zip(edges_to_expand, resources_to_add))
            #!!!! функция для создания/добавления ресурсов в граф

        # добавляем новые ресурсы для СЛУЧАЙНО ВЫБРАННЫХ ребер
        elif allocation_type == "random":
            random.shuffle(edges_with_alphas)
            edges_to_expand = [edge for edge, _ in edges_with_alphas[:number_of_new_resources]]
            source_target_sequence_for_new_resources = list(zip(edges_to_expand, resources_to_add))
            #!!!! функция для создания/добавления ресурсов в граф
        else:
            raise ValueError(f"Тип распределения ресурсов {allocation_type} не предусмотрен экспериментом")
