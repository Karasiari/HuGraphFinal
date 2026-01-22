from collections import defaultdict
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import networkx as nx
import scipy

from scipy.linalg import fractional_matrix_power

# ----------------------------------------------------------------
# вспомогательные функции под основной объект класса (для core.py)
# ----------------------------------------------------------------

def get_laplacian(graph: nx.Graph) -> np.ndarray:
    """
    Вычисляет матрицу Лапласа для неориентированного графа.
    
    Матрица Лапласа L = D - A, где:
    - D - диагональная матрица степеней вершин
    - A - матрица смежности
    
    Для взвешенных графов учитывает веса рёбер.
    
    Parameters
    ----------
    graph : nx.Graph
        Граф NetworkX (может быть взвешенным)
        
    Returns
    -------
    np.ndarray
        Матрица Лапласа размерности (n_nodes × n_nodes)
    """
    mat = nx.laplacian_matrix(graph)
    laplacian = mat.astype(float).toarray()
    return laplacian

def update_laplacian_on_edge(laplacian: np.ndarray, i: int, j: int, w: float) -> np.ndarray:
    """
    Обновляет матрицу Лапласа при добавлении/изменении веса ребра (i, j).
    
    При изменении веса ребра на dw:
    - L[i, i] += dw, L[j, j] += dw (увеличение степени вершин)
    - L[i, j] -= dw, L[j, i] -= dw (увеличение "отрицательной" связи)
    
    Parameters
    ----------
    laplacian : np.ndarray
        Текущая матрица Лапласа
    i, j : int
        Индексы вершин ребра (0-based)
    w : float
        Изменение веса ребра (положительное для добавления, 
                              отрицательное для уменьшения)
                              
    Returns
    -------
    np.ndarray
        Обновлённая матрица Лапласа
    """
    laplacian[i, i] += w; laplacian[j, j] += w
    laplacian[i, j] -= w; laplacian[j, i] -= w
    return laplacian

def aggregate_graph(multigraph: nx.MultiGraph | nx.MultiDiGraph, weight_name: str) -> nx.Graph:
    """
    Агрегирует мультиграф в обычный неориентированный граф.
    
    Для пар вершин, соединённых несколькими рёбрами в мультиграфе,
    создаёт одно ребро с весом, равным сумме весов всех рёбер между ними.
    
    Parameters
    ----------
    multigraph : nx.MultiGraph | nx.MultiDiGraph
        Мультиграф NetworkX (может содержать параллельные рёбра)
    weight_name : str
        Имя атрибута, содержащего вес ребра
        
    Returns
    -------
    nx.Graph
        Обычный неориентированный граф с агрегированными весами
        
    Notes
    -----
    - Для ориентированных мультиграфов преобразует в неориентированный
    - Сохраняет все вершины исходного графа
    - Если между вершинами нет рёбер, они не будут соединены в результате
    """
    # Создаём пустой неориентированный граф с теми же вершинами
    G = nx.Graph()
    G.add_nodes_from(range(multigraph.number_of_nodes())) # Сохраняем все вершины

    # Проходим по всем рёбрам мультиграфа
    for u, v, data in multigraph.edges(data=True):
      weight = data[weight_name]

      if G.has_edge(u, v):
          # Если ребро уже существует, суммируем вес
          G[u][v]['weight'] += weight
      else:
          # Создаём новое ребро с заданным весом
          G.add_edge(u, v, weight=weight)

    return G
    
def get_pinv_sqrt(laplacian: np.ndarray) -> np.ndarray:
    """
    Вычисляет квадратный корень из псевдообратной матрицы Лапласа.
    
    Выполняет: (L⁺)^(1/2), где L⁺ - псевдообратная матрица Лапласа.
    Используется в спектральной кластеризации и других задачах.
    
    Parameters
    ----------
    laplacian : np.ndarray
        Матрица Лапласа
        
    Returns
    -------
    np.ndarray
        Матрица (L⁺)^(1/2)
        
    Warnings
    --------
    - Может быть численно нестабильным для плохо обусловленных матриц
    - Для больших матриц предпочтительнее использовать разреженные методы
    - Результат может быть слегка несимметричным из-за ошибок округления
    """
    L_pinv = np.linalg.pinv(laplacian)
    L_pinv_sqrt = fractional_matrix_power(L_pinv, 0.5)
    return L_pinv_sqrt

# ------------------------------------------------------------------
# вспомогательные функции для MCFP (Maximum Concurrent Flow Problem)
# ------------------------------------------------------------------

def get_incidence_matrix_for_mcfp(graph: nx.DiGraph) -> np.ndarray:
    """
    Создаем incidence matrix
    параметры: nx.DiGraph
    return: incidence_matrix
    """
    incidence_matrix = nx.incidence_matrix(graph, edgelist=graph.edges, oriented=True)
    incidence_matrix = incidence_matrix.toarray()
    return incidence_matrix

def get_capacities_for_mcfp(graph: nx.DiGraph) -> np.ndarray:
    """
    Достаем capacities
    параметры: nx.DiGraph, граф с capacities (key: weights) на ребрах
    return: np.ndarray
    """
    edges_with_weights = [(edge, data['weight']) for edge, data in graph.edges.items()]
    edges_with_weights_dict = {key: value for key, value in edges_with_weights}
    return np.array(list(edges_with_weights_dict.values()), dtype=np.float64)
