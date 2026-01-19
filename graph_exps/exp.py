import pickle
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import networkx as nx
from typing import Optional, Dict, Any, Tuple, List

from .core import HuGraphForExps
from .instruments_for_exps import * # импорт вспомогательных функций

# функция для параллелизованного расчета меры alpha для ВСЕХ ребер графа

def compute_alpha_for_all_edges(graph: nx.Graph, n_jobs=8) -> List[Tuple[Tuple[int, int], float]]:
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
    edges_with_alphas.sort(key=lambda x: x[1], reverse=True)
    return edges_with_alphas
