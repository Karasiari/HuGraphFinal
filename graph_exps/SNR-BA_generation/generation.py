from .instruments import *

# вспомогательные функции для генерации SNR BA

def generate_uniform_nodes(n: int,
                           seed: int = 0,
                           xlim: Tuple[float, float] = (0.0, 1.0),
                           ylim: Tuple[float, float] = (0.0, 1.0),
                           ) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(*xlim), rng.uniform(*ylim)) for _ in range(n)]

def run_experiments(num_experiments=3, num_nodes=30, radii=None, capacity=80.0, seed=0):
    if radii is None:
        radii = np.linspace(0, geodesic_distance([0, 0], [1, 1]), 20).tolist()

    snr_ba_graphs = []
    for exp in range(num_experiments):

        # 1) Generate nodes
        coords = generate_uniform_nodes(num_nodes, seed=seed + exp)

        # 2) Build SNR-BA graph (reuses your snr_ba_from_latlon)
        G_snrba = snr_ba_from_latlon(coords, m=2, theta=5.0, seed=seed + exp)
        adj_matrix_nx = nx.adjacency_matrix(G_snrba, weight=None)
        adj_matrix = adj_matrix_nx.toarray()
        adj_matrix_weighted = adj_matrix * capacity
        snr_ba_graphs.append(adj_matrix_weighted)
    return snr_ba_graphs
