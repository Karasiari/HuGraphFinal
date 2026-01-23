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
        compute_leftover_space(leftover, affected_demands, instance.demands_by_id)

        scenario = FailureScenarioState(
            failed_edge_index=failed_edge_idx,
            leftover_by_edge=leftover,
            routed_by_edge=routed,
            add_by_edge=add_by_edge,
            slack_by_edge=instance.slack_by_edge,
        )

        demand_to_backup_path: Dict[DemandID, EdgePath] = {}
        for demand_id in affected_demands:
            demand = instance.demands_by_id[demand_id]
            backup_nodes = find_backup_path_nodes(instance, scenario, demand)
            apply_backup_routing(instance, scenario, demand, backup_nodes)
            demand_to_backup_path[demand_id] = nodes_to_oriented_edge_path(backup_nodes)

        reserve_paths_by_failed_edge[instance.edge_key_by_index[failed_edge_idx]] = demand_to_backup_path

    additional_volume_by_edge = {
        instance.edge_key_by_index[edge_idx]: add_by_edge[edge_idx] for edge_idx in range(edge_count)
    }

    return SpareCapacityGreedyOutput(
        additional_volume_by_edge=additional_volume_by_edge,
        reserve_paths_by_failed_edge=reserve_paths_by_failed_edge,
    )
