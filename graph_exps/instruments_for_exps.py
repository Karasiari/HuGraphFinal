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
    

def expand_graph(graph: HuGraphForExps, source_target_sequence_to_add: List[Tuple[Tuple[int, int], float]]) -> HuGraphForExps:
    for edge, capacity in source_target_sequence_to_add:
        graph.change_multiedge(edge[0], edge[1], type='insert', capacity=capacity)
    return graph
