beta=beta, intensity=int(3*median_capacity*graph_size), centrality='pagerank', edge_perc=1/graph_size, edge_mode='dynamic', dyn_max=1, dyn_law='exponential', dyn_k=dyn_k

from typing import Dict, Any

GRAVITY_DEFAULTS = {"lr": 0.001, "bs": 32, "opt": "adam"}
GRAVITY_CHECKS = {
    "lr": (1e-6, 1.0),          # диапазон
    "bs": {16, 32, 64, 128},    # множество допустимых
    "opt": ["adam", "sgd"]      # список допустимых
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
      
    right_params = {**defaults, **params}
    if checks:
        for param, check in checks.items():
            value = right_params[param]
            
            if isinstance(check, (list, tuple)) and len(check) == 2:
                # Диапазон (min, max)
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
    
    return right_params

valid_params = check_params(generation_type, params)
