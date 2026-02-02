from typing import Dict, Any

import networkx as nx
import numpy as np

from .HuGraphForGen.core import HuGraphForGen


def get_recommended_params(
    graph: HuGraphForGen, 
    generation_type: str
) -> Dict[str, Any]:
    if generation_type == "gravity":
        median_capacity = np.median([data['weight'] for u, v, data in graph.graph.edges(data=True) if 'weight' in data])
        graph_size = graph.n
        recommended = {
            "intensity": int(3*median_capacity*graph_size),
            "centrality": "pagerank",
            "edge_perc": 1/graph_size,
            "edge_mode": "dynamic",
            "dyn_max": 1,
            "dyn_law": "exponential"
        }
    elif generation_type == "alpha":
        graph_size = graph.n
        recommended = {
            "p_ER": 2/graph_size,
            "distribution": "normal",
            "num_edges": None,
            "initial_generation": "deterministic", 
            "demands_sum": int(1.5*graph_size*20)
        }
    elif generation_type == "alpha_with_sa":
        graph_size = graph.n
        recommended = {
            "p_ER": 2/graph_size,
            "distribution": "normal",
            "num_edges": None,
            "initial_generation": "deterministic", 
            "demands_sum": int(1.5*graph_size*20)
        }
    return recommended


def get_checks(
    generation_type: str
) -> Dict[str, Any]:
    if generation_type == "gravity":
        checks = {
            "beta": (0.0, 1.0),
            "intensity": lambda x: isinstance(x, int),
            "centrality": ("degree", "closeness", "harmonic_closeness", "harmonic", "pagerank"),
            "edge_perc": (0.0, 1.0),
            "edge_mode": ("dynamic", "static_top", "static_betascore"),
            "dyn_max": lambda x: isinstance(x, float),
            "dyn_law": ("exponential", "linear", None),
            "dyn_k": lambda x: isinstance(x, float)
        }
    elif generation_type == "alpha":
        checks = {
            "alpha_target": (0.0, 1.0),
            "epsilon": (0.0, 1.0),
            "p_ER": (0.0, 1.0),
            "distribution": ["normal"],
            "median_weight_for_initial": lambda x: isinstance(x, int) and (x > 0),
            "var_for_initial": lambda x: isinstance(x, int) and (x > 0),
            "multi_max": lambda x: isinstance(x, int) and (x > 0),
            "initial_generation": ("deterministic", "ER", None),
            "demands_sum": lambda x: isinstance(x, float) and (x > 0),
            "num_edges": lambda x: isinstance(x, int),
            "max_iter": lambda x: isinstance(x, int)
        }
    elif generation_type == "alpha_with_sa":
        checks = {
            "alpha_target": (0.0, 1.0),
            "epsilon": (0.0, 1.0),
            "p_ER": (0.0, 1.0),
            "distribution": ["normal"],
            "median_weight_for_initial": lambda x: isinstance(x, int) and (x > 0),
            "var_for_initial": lambda x: isinstance(x, int) and (x > 0),
            "multi_max": lambda x: isinstance(x, int) and (x > 0),
            "initial_generation": ("deterministic", "ER", None),
            "demands_sum": lambda x: isinstance(x, float) and (x > 0),
            "num_edges": lambda x: isinstance(x, int),
            "max_iter": lambda x: isinstance(x, int),
            "t": (0.0, 1.0)
        }
    return checks


def check_params(
    graph: HuGraphForGen,
    generation_type: str,
    params: Dict[str, Any],
    recommended_params: bool
) -> Dict[str, Any]:
    """
    Подготавливает гиперпараметры для генерации
    Input:
          graph - граф смежности, на котором проводится генерация 
                  (для расчета рекомендованных параметров)
          generation_type - тип генерации
          params - переданные параметры
          recommended_params - флаг использовать рекомендованные гиперпараметры
    Output:
          Валидированный словарь параметров
    """
    recommended = get_recommended_params(graph, generation_type)
    checks = get_checks(generation_type)
    if checks:
        for param, check in checks.items():
            value = params.get(param, None)
            if value is None:
                continue
            if isinstance(check, (list, tuple)) and len(check) == 2:
                # Диапазон
                min_val, max_val = check
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{param} вне диапазона [{min_val}, {max_val}]")
            elif isinstance(check, (list, set, tuple)):
                # Список допустимых значений
                if value not in check:
                    raise ValueError(f"{param} должен быть в {check}")
            elif callable(check):
                # Функция-валидатор
                if not check(value):
                    raise ValueError(f"Недопустимое значение {param}: {value}")
    if recommended_params:
        valid_params = {**params, **recommended}
    else:
        valid_params = params
    return valid_params
