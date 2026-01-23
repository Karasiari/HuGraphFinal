from .instruments import *


@dataclass(frozen=True, slots=True)
class SpareCapacityGreedyOutput:
    """Greedy algorithm output.

    - `additional_volume_by_edge[e]` is the global `add(e)` reservation for edge e.
      Keys are canonical undirected edge keys.
    - `reserve_paths_by_failed_edge[e][demand_id]` is the backup (edge) path used by
      `demand_id` when edge `e` fails.
    """
    additional_volume_by_edge: Dict[EdgeKey, int]
    reserve_paths_by_failed_edge: Dict[EdgeKey, Dict[DemandID, EdgePath]]

# ----------------------------
# Scenario utilities
# ----------------------------

def _compute_leftover_space(
    leftover: _PositiveTouchedArray,
    affected_demand_ids: Sequence[DemandID],
    demands_by_id: Mapping[DemandID, _ProcessedDemand],
) -> None:
    """Compute per-edge freed volume when the failed edge drops `affected_demand_ids`."""
    leftover.clear()
    for demand_id in affected_demand_ids:
        demand = demands_by_id[demand_id]
        if demand.volume == 0:
            continue
        for edge_idx in demand.initial_edge_indices:
            leftover.increment(edge_idx, demand.volume)


def _make_weight1(
    scenario: _FailureScenarioState,
    demand_volume: int,
) -> Callable[[Node, Node, Mapping[str, Any]], Optional[int]]:
    """Build the Objective-1 weight function.

    For an edge f, this returns the incremental increase in add(f) required to route
    `demand_volume` through f under the current scenario state. If the edge is not
    usable (failed edge or physical capacity violation), returns None to hide the edge.
    """
    failed_edge_idx = scenario.failed_edge_index
    slack = scenario.slack_by_edge
    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge

    def weight(_u: Node, _v: Node, attrs: Mapping[str, Any]) -> Optional[int]:
        edge_idx = attrs["idx"]
        if edge_idx == failed_edge_idx:
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


def _find_backup_path_nodes(
    instance: _PreprocessedInstance,
    scenario: _FailureScenarioState,
    demand: _ProcessedDemand,
) -> List[Node]:
    """Compute the demand's backup path as a node sequence.

    Lexicographic objectives:
      1) minimize sum(max(0, volume - allowance(edge))) over edges in the path
      2) among Objective-1 shortest paths, minimize sum(min(allowance(edge), volume))

    Where allowance(edge) = leftover(edge) + add(edge) - routed(edge) in this scenario.
    """
    if demand.source == demand.target:
        return [demand.source]

    weight1 = _make_weight1(scenario, demand_volume=demand.volume)

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
            f"No feasible backup path for demand {demand.demand_id} under failure of edge index {scenario.failed_edge_index}."
        )

    shortest_len = dist_from_source[demand.target]

    failed_edge_idx = scenario.failed_edge_index
    slack = scenario.slack_by_edge
    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge
    volume = demand.volume

    def weight2(u: Node, v: Node, attrs: Mapping[str, Any]) -> Optional[int]:
        """Objective-2 weight, restricted to edges on Objective-1 shortest s-t paths."""
        edge_idx = attrs["idx"]
        if edge_idx == failed_edge_idx:
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
        return nx.dijkstra_path(
            instance.directed_graph_view, demand.source, demand.target, weight=weight2
        )
    except nx.NetworkXNoPath as exc:
        raise ValueError(
            f"Objective-2 routing failed for demand {demand.demand_id} under failure of edge index {scenario.failed_edge_index}."
        ) from exc


def _apply_backup_routing(
    instance: _PreprocessedInstance,
    scenario: _FailureScenarioState,
    demand: _ProcessedDemand,
    backup_path_nodes: Sequence[Node],
) -> None:
    """Apply the chosen backup route: update global add and per-scenario routed volume."""
    if demand.volume == 0 or len(backup_path_nodes) < 2:
        return

    leftover = scenario.leftover_by_edge.values
    routed = scenario.routed_by_edge.values
    add = scenario.add_by_edge
    slack = scenario.slack_by_edge
    volume = demand.volume

    for u, v in pairwise(backup_path_nodes):
        edge_idx = instance.graph[u][v]["idx"]

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
                raise ValueError(
                    f"Internal error: add exceeds physical slack on edge {instance.edge_key_by_index[edge_idx]}. "
                    f"add={add[edge_idx]}, slack={slack[edge_idx]}."
                )

        scenario.routed_by_edge.increment(edge_idx, volume)


def _nodes_to_oriented_edge_path(nodes_path: Sequence[Node]) -> EdgePath:
    """Convert a node path [n0, n1, ..., nk] into an oriented edge path [(n0,n1),...,(n{k-1},nk)]."""
    return [(u, v) for u, v in pairwise(nodes_path)]


# ----------------------------
# Public API
# ----------------------------

def run_greedy_spare_capacity_allocation(input_data: SpareCapacityGreedyInput) -> SpareCapacityGreedyOutput:
    """Execute the greedy algorithm from the prompt.

    Processing order:
      - Edges are processed in a random order.
      - For each failed edge, the affected demands are processed in a random order.

    For each failure scenario (single failed edge), only demands that used that edge
    in the initial routing are rerouted. All other demands remain on their initial routes.

    Returns:
      A `SpareCapacityGreedyOutput` containing:
        - global per-edge reservations `add(e)`
        - per-failed-edge backup paths for affected demands
    """
    instance = preprocess_instance(input_data)

    edge_count = len(instance.edge_key_by_index)
    add_by_edge: List[int] = [0] * edge_count

    rng = random.Random(input_data.random_seed)
    failure_edge_indices = list(range(edge_count))
    rng.shuffle(failure_edge_indices)

    leftover = PositiveTouchedArray.zeros(edge_count)
    routed = PositiveTouchedArray.zeros(edge_count)

    reserve_paths_by_failed_edge: Dict[EdgeKey, Dict[DemandID, EdgePath]] = {}

    for failed_edge_idx in failure_edge_indices:
        affected_demands = list(instance.demands_using_edge[failed_edge_idx])
        rng.shuffle(affected_demands)

        routed.clear()
        _compute_leftover_space(leftover, affected_demands, instance.demands_by_id)

        scenario = _FailureScenarioState(
            failed_edge_index=failed_edge_idx,
            leftover_by_edge=leftover,
            routed_by_edge=routed,
            add_by_edge=add_by_edge,
            slack_by_edge=instance.slack_by_edge,
        )

        demand_to_backup_path: Dict[DemandID, EdgePath] = {}
        for demand_id in affected_demands:
            demand = instance.demands_by_id[demand_id]
            backup_nodes = _find_backup_path_nodes(instance, scenario, demand)
            _apply_backup_routing(instance, scenario, demand, backup_nodes)
            demand_to_backup_path[demand_id] = _nodes_to_oriented_edge_path(backup_nodes)

        reserve_paths_by_failed_edge[instance.edge_key_by_index[failed_edge_idx]] = demand_to_backup_path

    additional_volume_by_edge = {
        instance.edge_key_by_index[edge_idx]: add_by_edge[edge_idx] for edge_idx in range(edge_count)
    }

    return SpareCapacityGreedyOutput(
        additional_volume_by_edge=additional_volume_by_edge,
        reserve_paths_by_failed_edge=reserve_paths_by_failed_edge,
    )
