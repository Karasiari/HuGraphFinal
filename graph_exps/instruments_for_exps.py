from typing import Dict, Tuple, List

import networkx as nx
import numpy as np

from .HuGraphForExps.core import HuGraphForExps

# алиасы для читаемости outputов
EdgeWithParameter = Tuple[Tuple[int, int], float | None]
AllocationResult = Tuple[str, Tuple[bool, float]]
EdgeKey = Tuple[int, int]
RouteResult = Dict[int, List[Tuple[int, int, int]]]
DemandsDict = Dict[int, Tuple[int, int, int]]
RemainingNetwork = Tuple[nx.DiGraph, nx.MultiDiGraph]

# импорты для solve_mcfp_for_exp
from .HuGraphForExps.mcfp_algorithm.main_algorithm import solve_max_concurrent_flow_problem
from .HuGraphForExps.instruments import aggregate_graph
from .HuGraphForExps.instruments import get_laplacian


# импорт класса под нужный input в алгоритм перепрокладки
from .spare_capacity_allocation_algorithm.classes_for_algorithm import SpareCapacityGreedyInput

# импорт функции для преобразования данных под алгоритм перераспределения трафика
from .spare_capacity_allocation_algorithm.input_converter import convert_to_greedy_input

# импорт основной функции алгоритма перепрокладки
from .spare_capacity_allocation_algorithm.main_algorithm import run_greedy_spare_capacity_allocation

# импорт функции для преобразования результата алгоритма перепрокладки под наш эксперимент
from .spare_capacity_allocation_algorithm.output_converter import convert_greedy_output_for_exp

# ----------------------------------------------------------------------------------
# вспомогательные функции для основного экспа - для расчетов, параллелизаций и проч.
# ----------------------------------------------------------------------------------

# функция для расчета α для ребра

def compute_alpha_for_edge(graph_state, source, target):
    # Восстанавливаем граф из сериализованного состояния (например, pickle)
    import pickle
    graph = pickle.loads(graph_state)
    
    keys = list(graph.multigraph.get_edge_data(source, target).keys())
    if not keys:
        return ((source, target), float('nan'))
    key = keys[0]
    
    graph.change_multiedge(source, target, "insert", key, 80)
    alpha = graph.calculate_alpha()
    return ((source, target), alpha)
    
# функция для расширения графа

def expand_graph(
    graph: HuGraphForExps, 
    source_target_sequence_to_add: List[EdgeWithParameter]
) -> HuGraphForExps:
    for edge, capacity in source_target_sequence_to_add:
        graph.change_multiedge(edge[0], edge[1], type='insert', capacity=capacity)
    return graph


# функция для отдельного расчета части результатов решения исходного MCF

def get_remaining_networks_and_volume_to_reroute(
    graph: nx.MultiDiGraph, 
    route_result: RouteResult, 
    demands: DemandsDict
) -> Tuple[Dict[EdgeKey, RemainingNetwork], int]:
    slack_by_edge: Dict[EdgeKey, int] = {}
    demands_through_edge: Dict[EdgeKey, List[int]] = {}
    unique_edges: Dict[EdgeKey, bool] = {}
    remaining_networks: Dict[EdgeKey, RemainingNetwork] = {}
    volume_to_reroute: int = 0

    for u, v, data in graph.edges(data=True):
        edge_oriented = (u, v)
        edge_unoriented = min(u, v), max(u, v)
        if not unique_edges.get(edge_unoriented, False):
            unique_edges[edge_unoriented] = True
        if slack_by_edge.get(edge_oriented, False):
            slack_by_edge[edge_oriented] += int(data['capacity'])
        else:
            slack_by_edge[edge_oriented] = int(data['capacity'])
    for demand_id, demand_path in route_result.items():
        demand = demands[demand_id]
        demand_volume = demand[2]
        for u, v, _ in demand_path:
            edge_oriented = (u, v)
            slack_by_edge[edge_oriented] -= demand_volume
            if demands_through_edge.get(edge_oriented, False):
                demands_through_edge[edge_oriented].append(demand_id)
            else:
                demands_through_edge[edge_oriented] = [demand_id]

    for edge_unoriented, _ in unique_edges.items():
        slack_graph = nx.DiGraph()
        slack_demands_graph = nx.MultiDiGraph()
        slack_graph.add_nodes_from(graph)
        slack_demands_graph.add_nodes_from(graph)
        edge_reversed = edge_unoriented[1], edge_unoriented[0]
        affected_demands_ids = demands_through_edge.get(edge_unoriented, []) + demands_through_edge.get(edge_reversed, [])
        affected_demands = []
        edges = slack_by_edge.copy()
        edges_list = []
        for demand_id in affected_demands_ids:
            affected_demand = demands[demand_id]
            source, target, capacity = affected_demand
            volume_to_reroute += capacity
            affected_demands.append((source, target, {"weight": capacity}))
            edges_to_restore = route_result[demand_id]
            for u, v, _ in edges_to_restore:
                edge_oriented = (u, v)
                edges[edge_oriented] += capacity
        for edge_oriented, capacity in edges.items():
            if edge_oriented not in (edge_unoriented, edge_reversed):
                edges_list.append((edge_oriented[0], edge_oriented[1], {"weight": capacity}))
        slack_graph.add_edges_from(edges_list)
        slack_demands_graph.add_edges_from(affected_demands)
        remaining_networks[edge_unoriented] = (slack_graph, slack_demands_graph)
    
    return remaining_networks, volume_to_reroute
    

# отдельная функция для предварительного решения MCF в рамках эксперимента

def solve_mcf_for_exp(
    graph: HuGraphForExps, 
    allocation_type: str, 
    epsilon: float = 1.0, 
    available_volumes: Tuple[Tuple[int, float], ...] = ((1, 1.0),), 
    random_seed: int | None = None
) -> Tuple[str, SpareCapacityGreedyInput, Dict[EdgeKey, RemainingNetwork], int]:
    route_result, demands, solved, multidigraph = graph.solve_mcf()
    remaining_networks, volume_to_reroute = get_remaining_networks_and_volume_to_reroute(multidigraph, route_result, demands)
    input_for_allocate_spare_capacity_algorithm = convert_to_greedy_input(multidigraph, demands, route_result, epsilon, available_volumes, random_seed)
    return allocation_type, input_for_allocate_spare_capacity_algorithm, remaining_networks, volume_to_reroute


# функция для решения max concurrent flow на остаточной сети (gamma) для параллельного расчета в рамках основного эксперимента

def solve_mcfp_for_exp(
    edge: EdgeKey, 
    network: RemainingNetwork
) -> EdgeWithParameter:
    graph, demands_graph = network
    demands_laplacian = get_laplacian(aggregate_graph(demands_graph, weight_name='weight'))
    if not np.any(demands_laplacian):
     return edge, float('inf')
    gamma = solve_max_concurrent_flow_problem(graph, demands_laplacian, solver_flag=False, break_flag=True)
    return edge, gamma


# функция для решения перераспределения трафика - в решении наш алгоритм

def allocate_spare_capacity(
    input_for_algorithm: SpareCapacityGreedyInput, 
    allocation_type: str
) -> AllocationResult:
    output_of_algorithm = run_greedy_spare_capacity_allocation(input_for_algorithm)
    allocation_result = convert_greedy_output_for_exp(output_of_algorithm)
    return allocation_type, allocation_result
    
