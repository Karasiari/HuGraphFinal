from typing import Dict, Tuple, List

import networkx as nx

from .core import HuGraphForExps

from .spare_capacity_allocation_algorithm.input_converter import convert_to_greedy_input # импорт функции для преобразования данных под алгоритм перераспределения трафика
from .spare_capacity_allocation_algorithm.main_algorithm import run_greedy_spare_capacity_allocation # импорт основной функции алгоритма перепрокладки
from .spare_capacity_allocation_algorithm.output_converter import convert_greedy_output_for_exp # импорт функции для преобразования результата алгоритма перепрокладки под наш эксперимент

# ----------------------------------------------------------------------------------
# вспомогательные функции для основного экспа - для расчетов, параллелизаций и проч.
# ----------------------------------------------------------------------------------

def compute_alpha_for_edge(graph_state, source, target):
    # Восстанавливаем граф из сериализованного состояния (например, pickle)
    import pickle
    graph = pickle.loads(graph_state)
    
    # Берём первый мультребро
    keys = list(graph.multigraph.get_edge_data(source, target).keys())
    if not keys:
        return ((source, target), float('nan'))
    key = keys[0]
    
    graph.change_multiedge(source, target, "insert", key, 80)
    alpha = graph.calculate_alpha()
    # graph.restore_graph() не нужен, т.к. граф временный
    return ((source, target), alpha)
    

def expand_graph(graph: HuGraphForExps, source_target_sequence_to_add: List[Tuple[Tuple[int, int], float]]) -> HuGraphForExps:
    for edge, capacity in source_target_sequence_to_add:
        graph.change_multiedge(edge[0], edge[1], type='insert', capacity=capacity)
    return graph


# функция для решения перераспределения трафика - в решении наш алгоритм

def allocate_spare_capacity(graph: HuGraphForExps, allocation_type: str, random_seed: int | None = None) -> Tuple[str, Tuple[Dict[Tuple[int, int], Tuple[nx.Graph, nx.Graph]], int, float]]:
    route_result, demands, solved = graph.solve_mcf()
    input_for_algorithm = convert_to_greedy_input(graph.multigraph, demands, route_result, random_seed)
    output_of_algorithm = run_greedy_spare_capacity_allocation(input_for_algorithm)
    allocation_results = convert_greedy_output_for_exp(output_of_algorithm)

    return (allocation_type, allocation_results)


# функция для решения max concurrent flow на остаточной сети (gamma) для параллельного расчета в рамках основного эксперимента

def solve_mcfp_wrapper(edge: Tuple[int, int], graph: HuGraphForExps) -> Tuple[Tuple[int, int], float]:
    return edge, graph.solve_mcfp()
    
