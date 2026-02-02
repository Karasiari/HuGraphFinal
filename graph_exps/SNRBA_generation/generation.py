# импорт вспомогательных функций для генерации SNR-BA
from .instruments import *


# функция для взвешивания графа

def get_weighted_graph(
    unweighted_graph: nx.Graph,
    capacity_value: float
) -> nx.MultiGraph:
    """
    Взвешивает сгенерированный SNR-BA граф
    Input:
          unweighted_graph - сгенерированный SNR-BA граф
          capacity_value - значение веса для ребер
    Output:
          Граф с весом capacity_value для любого ребра 
          как nx.MultiGraph, но без параллельных ребер
    """
    weighted_graph: nx.MultiGraph = nx.MultiGraph()
    if capacity_value <= 0:
        raise ValueError("Значение capacity должно быть положительным")

    weighted_graph.add_nodes_from(unweighted_graph)
    for u, v in unweighted_graph.edges():
        weighted_graph.add_edge(u, v, data={"capacity": capacity_value})
    return weighted_graph


# основная функция генерации SNR-BA графа

def generate_snr_ba_graph(
    num_nodes: int, 
    capacity_value: float, 
    random_seed: Optional[int] = None
) -> nx.Graph:
    """
    Функция генерации SNR-BA графа смежности
    Input:
          num_nodes - количество вершин для генерируемого графа
          capacity_value - значение веса для ребер генерируемого графа
                           все ребра одного веса - правильный формат для 
                           эксперимента
          random_seed
    Output:
          Сгенерированный SNR-BA граф смежности
    """
    coords = generate_uniform_nodes(num_nodes, random_seed)
    snr_ba_graph_unweighted = snr_ba_from_latlon(coords, m=2, theta=5.0, seed=random_seed)
    snr_ba_graph = get_weighted_graph(snr_ba_graph_unweighted, capacity_value)
    return snr_ba_graph
