# импорт вспомогательных функций для генерации SNR-BA
from .instruments import *

# основная функция генерации SNR-BA графа

def generate_snr_ba_graph(
    num_nodes: int, 
    capacity_value: float, 
    random_seed: Optional[int] = None
) -> nx.Graph():
    coords = generate_uniform_nodes(num_nodes, random_seed)
    snr_ba_graph_unweighted = snr_ba_from_latlon(coords, m=2, theta=5.0, seed=random_seed)
    return snr_ba_graph
