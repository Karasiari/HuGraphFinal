from .instruments import *


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
        - algorithm failure flag
        - successfully rerouted demands volume
        - global per-edge reservations `add(e)`
        - per-failed-edge backup paths for affected demands
    """
    instance = preprocess_instance(input_data)
    #epsilon = input_data.epsilon

    successfully_rerouted_demands_volume = 0

    edge_count = len(instance.edge_key_by_index)
    add_by_edge: List[int] = [0] * edge_count

    agg_edge_count = len(instance.indexes_by_agg_index)
    rng = random.Random(input_data.random_seed)
    failure_agg_edge_indices = list(range(agg_edge_count))
    rng.shuffle(failure_agg_edge_indices)

    leftover = PositiveTouchedArray.zeros(edge_count)
    routed = PositiveTouchedArray.zeros(edge_count)

    reserve_paths_by_failed_edge: Dict[Tuple[Node, Node], Dict[DemandID, EdgePath]] = {}
    algorithm_failure_flag: bool = False

    for failed_agg_edge_idx in failure_agg_edge_indices:
        failed_edges_indices = instance.indexes_by_agg_index[failed_agg_edge_idx]
        affected_demands = []
        for failed_edges_idx in failed_edges_indices:
            affected_demands += list(instance.demands_using_edge[failed_edges_idx])
        rng.shuffle(affected_demands)

        routed.clear()
        compute_leftover_space(leftover, affected_demands, instance.demands_by_id)

        if not algorithm_failure_flag:
            scenario = FailureScenarioState(
                failed_agg_edge_index=failed_agg_edge_idx,
                failed_edges_indices=failed_edges_indices,
                leftover_by_edge=leftover,
                routed_by_edge=routed,
                add_by_edge=add_by_edge,
                slack_by_edge=instance.slack_by_edge,
            )

            demand_to_backup_path: Dict[DemandID, EdgePath] = {}
            for demand_id in affected_demands:
                demand = instance.demands_by_id[demand_id]
                try:
                    backup_edges = find_backup_path_nodes(instance, scenario, demand)
                except ValueError:
                    algorithm_failure_flag = True
                    break
                try:
                    apply_backup_routing(instance, scenario, demand, backup_edges)
                except ValueError:
                    algorithm_failure_flag = True
                    break
                demand_to_backup_path[demand_id] = backup_edges
                successfully_rerouted_demands_volume += demand.volume

            failed_edge_key = instance.edge_key_by_index[instance.indexes_by_agg_index[failed_agg_edge_idx][0]]
            u_for_key, v_for_key = min(failed_edge_key[0], failed_edge_key[1]), max(failed_edge_key[0], failed_edge_key[1])
            reserve_paths_by_failed_edge[(u_for_key, v_for_key)] = demand_to_backup_path

    additional_volume_by_edge = {
        instance.edge_key_by_index[edge_idx]: add_by_edge[edge_idx] for edge_idx in range(edge_count)
    }

    return SpareCapacityGreedyOutput(
        algorithm_failure_flag=algorithm_failure_flag,
        successfully_rerouted_demands_volume=successfully_rerouted_demands_volume,
        additional_volume_by_edge=additional_volume_by_edge,
        reserve_paths_by_failed_edge=reserve_paths_by_failed_edge,
    )
