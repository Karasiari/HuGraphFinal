beta=beta, intensity=int(3*median_capacity*graph_size), centrality='pagerank', edge_perc=1/graph_size, edge_mode='dynamic', dyn_max=1, dyn_law='exponential', dyn_k=dyn_k

from typing import Dict, Any

GRAVITY_DEFAULTS = {
    "centrality": "pagerank", 
    "edge_mode": "dynamic", 
    "dyn_max": 1, 
    "dyn_law": "exponential"
}
GRAVITY_CHECKS = {
    "beta": (0.0, 1.0),
    "intensity": lambda x: type(x) is int,
    "centrality": ("degree", "closeness", "harmonic_closeness", "harmonic", "pagerank"),
    "edge_perc": (0.0, 1.0),
    "edge_mode": ("dynamic", "static_top", "static_betascore"),
    "dyn_max": lambda x: type(x) is float,
    "dyn_law": ("exponential", "linear", None),
    "dyn_k": lambda x: type(x) is float
}

def check_params(
    generation_type: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Подготавливает гиперпараметры для генерации: дефолты и валидация
    Input:
          generation_type - тип генерации
          params - переданные параметры
    Output:
          Валидированный словарь параметров
    """
    if generation_type == "gravity":
      defaults, checks = GRAVITY_DEFAULTS, GRAVITY_CHECKS
    elif generation_type == "alpha":
      defaults, checks = ALPHA_DEFAULTS, ALPHA_CHECKS
    elif generation_type == "alpha_with_sa":
      defaults, checks = ALPHA_WITH_SA_DEFAULTS, ALPHA_WITH_SA_CHECKS
      
    valid_params = {**defaults, **params}
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
