import .classes_for_algorithm

# ----------------------------
# Preprocessing
# ----------------------------

def _build_indexed_graph(edge_inputs: Sequence[EdgeInput]) -> Tuple[nx.Graph, List[EdgeKey], List[int]]:
    """Build an undirected NetworkX graph and assign a compact index to each edge."""
    graph = nx.Graph()
    edge_key_by_index: List[EdgeKey] = []
    capacity_by_edge: List[int] = []
    seen: Dict[EdgeKey, int] = {}

    for edge in edge_inputs:
        if edge.capacity < 0:
            raise ValueError(
                f"Edge capacity must be non-negative, got {edge.capacity} for edge {edge.u}-{edge.v}."
            )

        key = _canonical_edge_key(edge.u, edge.v)
        if key in seen:
            existing_idx = seen[key]
            if capacity_by_edge[existing_idx] != edge.capacity:
                raise ValueError(
                    f"Duplicate edge {key} with conflicting capacities: "
                    f"{capacity_by_edge[existing_idx]} vs {edge.capacity}."
                )
            continue

        idx = len(edge_key_by_index)
        seen[key] = idx
        edge_key_by_index.append(key)
        capacity_by_edge.append(edge.capacity)
        graph.add_edge(edge.u, edge.v, idx=idx, capacity=edge.capacity)

    return graph, edge_key_by_index, capacity_by_edge


def _process_demands(
    demand_inputs: Sequence[DemandInput],
    graph: nx.Graph,
    edge_count: int,
) -> Tuple[Dict[DemandID, _ProcessedDemand], List[int], List[List[DemandID]]]:
    """Validate demand paths, derive edge indices, and compute initial edge loads."""
    demands_by_id: Dict[DemandID, _ProcessedDemand] = {}
    initial_load_by_edge: List[int] = [0] * edge_count
    demands_using_edge: List[List[DemandID]] = [[] for _ in range(edge_count)]

    for demand in demand_inputs:
        if demand.demand_id in demands_by_id:
            raise ValueError(f"Duplicate demand_id: {demand.demand_id}.")
        if demand.volume < 0:
            raise ValueError(
                f"Demand volume must be non-negative, got {demand.volume} for demand {demand.demand_id}."
            )

        current_node = demand.source
        edge_indices: List[int] = []

        for step, (u, v) in enumerate(demand.initial_edge_path):
            if not graph.has_edge(u, v):
                raise ValueError(
                    f"Demand {demand.demand_id} initial_edge_path uses a non-existent edge: {u} - {v}."
                )

            if current_node == u:
                next_node = v
            elif current_node == v:
                next_node = u
            else:
                raise ValueError(
                    f"Demand {demand.demand_id} initial_edge_path is not contiguous at step {step}: "
                    f"current node {current_node}, edge endpoints {(u, v)}."
                )

            edge_idx = graph[u][v]["idx"]
            edge_indices.append(edge_idx)
            if demand.volume:
                initial_load_by_edge[edge_idx] += demand.volume

            current_node = next_node

        if current_node != demand.target:
            raise ValueError(
                f"Demand {demand.demand_id} initial_edge_path does not end at target. "
                f"Ended at {current_node}, expected {demand.target}."
            )

        unique_edge_indices = frozenset(edge_indices)
        for edge_idx in unique_edge_indices:
            demands_using_edge[edge_idx].append(demand.demand_id)

        demands_by_id[demand.demand_id] = _ProcessedDemand(
            demand_id=demand.demand_id,
            source=demand.source,
            target=demand.target,
            volume=demand.volume,
            initial_edge_indices=tuple(edge_indices),
            unique_initial_edge_indices=unique_edge_indices,
        )

    return demands_by_id, initial_load_by_edge, demands_using_edge


def _preprocess_instance(input_data: SpareCapacityGreedyInput) -> _PreprocessedInstance:
    """Transform raw input into an indexed instance and validate initial feasibility."""
    graph, edge_key_by_index, capacity_by_edge = _build_indexed_graph(input_data.edges)
    if not edge_key_by_index:
        raise ValueError("Input graph must contain at least one edge.")

    demands_by_id, initial_load_by_edge, demands_using_edge = _process_demands(
        input_data.demands, graph, edge_count=len(edge_key_by_index)
    )

    slack_by_edge: List[int] = []
    for edge_idx, (capacity, load) in enumerate(zip(capacity_by_edge, initial_load_by_edge)):
        slack = capacity - load
        if slack < 0:
            raise ValueError(
                f"Initial routing violates capacity for edge {edge_key_by_index[edge_idx]}: "
                f"capacity {capacity}, initial load {load}."
            )
        slack_by_edge.append(slack)

    return _PreprocessedInstance(
        graph=graph,
        directed_graph_view=graph.to_directed(as_view=True),
        edge_key_by_index=edge_key_by_index,
        capacity_by_edge=capacity_by_edge,
        slack_by_edge=slack_by_edge,
        demands_by_id=demands_by_id,
        demands_using_edge=demands_using_edge,
    )

