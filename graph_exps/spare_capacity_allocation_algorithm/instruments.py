from .classes_for_algorithm import *

# ----------------------------
# Preprocessing
# ----------------------------

def build_indexed_graph(edge_inputs: Sequence[EdgeInput]) -> Tuple[nx.MultiDiGraph, Dict[int, List[int, ...]], List[EdgeKey], List[int]]:
    """Build a directed NetworkX multigraph and assign a compact index to each edge, create mapping agg_index -> index."""
    graph = nx.MultiDiGraph()
    indexes_by_agg_index: Dict[int, List[int, ...]] = {}
    edge_key_by_index: List[EdgeKey] = []
    capacity_by_edge: List[int] = []
    seen: Dict[Tuple[Node, Node], int] = {}

    new_agg_index = 0
    for edge in edge_inputs:
        if edge.capacity < 0:
            raise ValueError(
                f"Edge capacity must be non-negative, got {edge.capacity} for edge {edge.u}-{edge.v}."
            )
        idx = len(edge_key_by_index)
        u_for_key, v_for_key = min(edge.u, edge.v), max(edge.u, edge.v)
        if seen.get([(u_for_key, v_for_key)], False):
            agg_index = seen[(u_for_key, v_for_key)]
            indexes_by_agg_index[agg_index].append(idx)
        else:
            agg_index = new_agg_index
            seen[(u_for_key, v_for_key)] = agg_index
            indexes_by_agg_index[agg_index] = [idx]
            new_agg_index += 1
        edge_key_by_index.append(EdgeKey(edge.u, edge.v, edge.key))
        capacity_by_edge.append(edge.capacity)
        graph.add_edge(edge.u, edge.v, edge.key, idx=idx, capacity=edge.capacity)

    return graph, indexes_by_agg_index, edge_key_by_index, capacity_by_edge


def process_demands(
    demand_inputs: Sequence[DemandInput],
    graph: nx.MultiDiGraph,
    edge_count: int,
) -> Tuple[Dict[DemandID, ProcessedDemand], List[int], List[List[DemandID]]]:
    """Validate demand paths, derive edge indices, and compute initial edge loads."""
    demands_by_id: Dict[DemandID, ProcessedDemand] = {}
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

        for step, (u, v, key) in enumerate(demand.initial_edge_path):
            if not graph.has_edge(u, v, key):
                raise ValueError(
                    f"Demand {demand.demand_id} initial_edge_path uses a non-existent edge: {u} - {v} - {key}."
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

            edge_idx = graph[u][v][key]["idx"]
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

        demands_by_id[demand.demand_id] = ProcessedDemand(
            demand_id=demand.demand_id,
            source=demand.source,
            target=demand.target,
            volume=demand.volume,
            initial_edge_indices=tuple(edge_indices),
            unique_initial_edge_indices=unique_edge_indices,
        )

    return demands_by_id, initial_load_by_edge, demands_using_edge


def preprocess_instance(input_data: SpareCapacityGreedyInput) -> PreprocessedInstance:
    """Transform raw input into an indexed instance and validate initial feasibility."""
    graph, indexes_by_agg_index, edge_key_by_index, capacity_by_edge = build_indexed_graph(input_data.edges)
    if not edge_key_by_index:
        raise ValueError("Input graph must contain at least one edge.")

    demands_by_id, initial_load_by_edge, demands_using_edge = process_demands(
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

    return PreprocessedInstance(
        graph=graph,
        indexes_by_agg_index=indexes_by_agg_index,
        edge_key_by_index=edge_key_by_index,
        capacity_by_edge=capacity_by_edge,
        slack_by_edge=slack_by_edge,
        demands_by_id=demands_by_id,
        demands_using_edge=demands_using_edge,
    )


# ----------------------------
# Scenario utilities
# ----------------------------

def compute_leftover_space(
    leftover: PositiveTouchedArray,
    affected_demand_ids: Sequence[DemandID],
    demands_by_id: Mapping[DemandID, ProcessedDemand],
) -> None:
    """Compute per-edge freed volume when the failed edge drops `affected_demand_ids`."""
    leftover.clear()
    for demand_id in affected_demand_ids:
        demand = demands_by_id[demand_id]
        if demand.volume == 0:
            continue
        for edge_idx in demand.initial_edge_indices:
            leftover.increment(edge_idx, demand.volume)
    

def make_weight1(
    scenario: FailureScenarioState,
    demand_volume: int,
) -> Callable[[Node, Node, Mapping[str, Any]], Optional[int]]:
    """Build the Objective-1 weight function.

    For an edge f, this returns the incremental increase in add(f) required to route
    `demand_volume` through f under the current scenario state. If the edge is not
    usable (failed edge or physical capacity violation), returns None to hide the edge.
    """
    failed_edges_indices = scenario.failed_edges_indices
    slack = scenario.slack_by_edge
    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge

    def weight(_u: Node, _v: Node, _key: int, attrs: Mapping[str, Any]) -> Optional[int]:
        edge_idx = attrs["idx"]
        if edge_idx in failed_edges_indices:
            return None

        # Physical capacity for rerouted demands in this scenario:
        # remaining = slack + leftover - already_routed
        remaining_capacity = slack[edge_idx] + leftover[edge_idx] - routed[edge_idx]
        if remaining_capacity < demand_volume:
            return None

        # Allowance w.r.t. (initial load + add): allowance = leftover + add - routed
        allowance = leftover[edge_idx] + add[edge_idx] - routed[edge_idx]
        return 0 if allowance >= demand_volume else demand_volume - allowance

    return weight


def find_backup_path_nodes(
    instance: PreprocessedInstance,
    scenario: FailureScenarioState,
    demand: ProcessedDemand,
) -> List[EdgeKey]:
    """Compute the demand's backup path as an edge sequence.

    Lexicographic objectives:
      1) minimize sum(max(0, volume - allowance(edge))) over edges in the path
      2) among Objective-1 shortest paths, minimize sum(min(allowance(edge), volume))

    Where allowance(edge) = leftover(edge) + add(edge) - routed(edge) in this scenario.
    """
    if demand.source == demand.target:
        return [demand.source]

    weight1 = make_weight1(scenario, demand_volume=demand.volume)

    try:
        dist_from_source = nx.single_source_dijkstra_path_length(
            instance.graph, demand.source, weight=weight1
        )
        dist_to_target = nx.single_source_dijkstra_path_length(
            instance.graph, demand.target, weight=weight1
        )
    except nx.NodeNotFound as exc:
        raise ValueError(
            f"Demand {demand.demand_id} references a node that is not present in the graph."
        ) from exc

    if demand.target not in dist_from_source:
        raise ValueError(
            f"No feasible backup path for demand {demand.demand_id} under failure of agg edge index {scenario.failed_agg_edge_index}."
        )

    shortest_len = dist_from_source[demand.target]

    failed_edges_indices = scenario.failed_edges_indices
    slack = scenario.slack_by_edge
    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge
    volume = demand.volume

    def weight2(u: Node, v: Node, key: int, attrs: Mapping[str, Any]) -> Optional[int]:
        """Objective-2 weight, restricted to edges on Objective-1 shortest s-t paths."""
        edge_idx = attrs["idx"]
        if edge_idx in failed_edges_indices:
            return None

        remaining_capacity = slack[edge_idx] + leftover[edge_idx] - routed[edge_idx]
        if remaining_capacity < volume:
            return None

        allowance = leftover[edge_idx] + add[edge_idx] - routed[edge_idx]
        inc_add = 0 if allowance >= volume else volume - allowance

        dist_u = dist_from_source.get(u)
        dist_v_to_t = dist_to_target.get(v)
        if dist_u is None or dist_v_to_t is None:
            return None
        if dist_u + inc_add + dist_v_to_t != shortest_len:
            return None

        return allowance if allowance < volume else volume

    try:
        nodes_path = nx.dijkstra_path(
            instance.graph, demand.source, demand.target, weight=weight2
        )
        edges_path = []
        for u, v in pairwise(nodes_path):
            multiedges = instance.graph[u][v]
            min_key = min(
                multiedges.keys(),
                key=lambda k: weight2(u, v, k, multiedges[k]) or float('inf')
            )
            edges_path.append(EdgeKey(u, v, min_key))
        return edges_path
    except nx.NetworkXNoPath as exc:
        raise ValueError(
            f"Objective-2 routing failed for demand {demand.demand_id} under failure of agg edge index {scenario.failed_agg_edge_index}."
        ) from exc


def apply_backup_routing(
    instance: PreprocessedInstance,
    scenario: FailureScenarioState,
    demand: ProcessedDemand,
    backup_path_edges: Sequence[EdgeKey],
) -> None:
    """Apply the chosen backup route: update global add and per-scenario routed volume."""
    if demand.volume == 0 or len(backup_path_nodes) < 2:
        return

    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge
    slack = scenario.slack_by_edge
    volume = demand.volume

    for edge in backup_path_edges:
        u, v, key = edge
        edge_idx = instance.graph[u][v][key]["idx"]

        # Physical feasibility (defensive check)
        remaining_capacity = slack[edge_idx] + leftover[edge_idx] - routed[edge_idx]
        if remaining_capacity < volume:
            raise ValueError(
                f"Internal error: selected an infeasible edge for demand {demand.demand_id}. "
                f"Edge index {edge_idx} remaining capacity {remaining_capacity}, demand volume {volume}."
            )

        allowance = leftover[edge_idx] + add[edge_idx] - routed[edge_idx]
        if allowance < volume:
            add[edge_idx] += volume - allowance
            if add[edge_idx] > slack[edge_idx]:
                add[edge_idx] -= volume - allowance
                raise ValueError(
                    f"Internal error: add exceeds physical slack on edge {instance.edge_key_by_index[edge_idx]}. "
                    f"add={add[edge_idx]}, slack={slack[edge_idx]}."
                )

        scenario.routed_by_edge.increment(edge_idx, volume)
