from .core import HuGraphForExps

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
