beta=beta, intensity=int(3*median_capacity*graph_size), centrality='pagerank', edge_perc=1/graph_size, edge_mode='dynamic', dyn_max=1, dyn_law='exponential', dyn_k=dyn_k

from typing import Dict, Any

import networkx as nx

GRAVITY_DEFAULTS = {
    "centrality": "pagerank", 
    "edge_mode": "dynamic", 
    "dyn_max": 1, 
    "dyn_law": "exponential"
}

def get_recommended_params(
    graph: nx.Graph, 
    generation_type: str
) -> Dict[str, Any]:
    if generation_type == "gravity":
      recommended = 
    elif generation_type == "alpha":
      recommended = 
    elif generation_type == "alpha_with_sa":
      recommended = 
    return recommended

def get_checks(
    generation_type: str
) -> Dict[str, Any]:
    if generation_type == "gravity":
      checks = {
          "beta": (0.0, 1.0),
          "intensity": lambda x: type(x) is int,
          "centrality": ("degree", "closeness", "harmonic_closeness", "harmonic", "pagerank"),
          "edge_perc": (0.0, 1.0),
          "edge_mode": ("dynamic", "static_top", "static_betascore"),
          "dyn_max": lambda x: type(x) is float,
          "dyn_law": ("exponential", "linear", None),
          "dyn_k": lambda x: type(x) is float
      }
    elif generation_type == "alpha":
      checks =
    elif generation_type == "alpha_with_sa":
      checks =
    return checks

def check_params(
    graph: nx.Graph,
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
    if recommended_params:
        valid_params = {**params, **recommended}
    else:
        valid_params = params
    if checks:
        for param, check in checks.items():
            value = valid_params[param]
            
            if isinstance(check, (list, tuple)) and len(check) == 2:
                # Диапазон
                min_val, max_val = check
                if not (min_val < value < max_val):
                    raise ValueError(f"{param} вне диапазона ({min_val}, {max_val})")
            elif isinstance(check, (list, set, tuple)):
                # Список допустимых значений
                if value not in check:
                    raise ValueError(f"{param} должен быть в {check}")
            elif callable(check):
                # Функция-валидатор
                if not check(value):
                    raise ValueError(f"Недопустимое значение {param}: {value}")
    return valid_params
