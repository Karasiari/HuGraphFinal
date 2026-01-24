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
        - per-failed-edge remaining networks
        - algorithm failure flag
        - successfully rerouted demands ratio
        - global per-edge reservations `add(e)`
        - per-failed-edge backup paths for affected demands
    """
    instance = preprocess_instance(input_data)
    epsilon = input_data.epsilon

    total_demands_volume = sum([demand.volume for demand in input_data.demands])
    successfully_rerouted_demands_volume = 0

    edge_count = len(instance.edge_key_by_index)
    add_by_edge: List[int] = [0] * edge_count

    rng = random.Random(input_data.random_seed)
    failure_edge_indices = list(range(edge_count))
    rng.shuffle(failure_edge_indices)

    leftover = PositiveTouchedArray.zeros(edge_count)
    leftover_wo_epsilon = PositiveTouchedArray.zeros(edge_count)
    routed = PositiveTouchedArray.zeros(edge_count)

    reserve_paths_by_failed_edge: Dict[EdgeKey, Dict[DemandID, EdgePath]] = {}
    algorithm_failure_flag: bool = False
    remaining_network_by_failed_edge: Dict[EdgeKey, nx.Graph] = {}

    for failed_edge_idx in failure_edge_indices:
        affected_demands = list(instance.demands_using_edge[failed_edge_idx])
        rng.shuffle(affected_demands)

        routed.clear()
        compute_leftover_space(leftover, leftover_wo_epsilon, epsilon, affected_demands, instance.demands_by_id)

        remaining_network_for_edge = build_remaining_network_for_failed_edge(instance, failed_edge_idx, leftover_wo_epsilon)
        remaining_network_by_failed_edge[instance.edge_key_by_index[failed_edge_idx]] = remaining_network_for_edge

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
            try:
                backup_nodes = find_backup_path_nodes(instance, scenario, demand)
            except ValueError:
                algorithm_failure_flag = True
                break
            try:
                apply_backup_routing(instance, scenario, demand, backup_nodes)
            except ValueError:
                algorithm_failure_flag = True
                break
            demand_to_backup_path[demand_id] = nodes_to_oriented_edge_path(backup_nodes)
            successfully_rerouted_demands_volume += demand.volume

        reserve_paths_by_failed_edge[instance.edge_key_by_index[failed_edge_idx]] = demand_to_backup_path
        if algorithm_failure_flag:
            break

    additional_volume_by_edge = {
        instance.edge_key_by_index[edge_idx]: add_by_edge[edge_idx] for edge_idx in range(edge_count)
    }

    successfully_rerouted_demands_ratio = successfully_rerouted_demands_volume / total_demands_volume

    return SpareCapacityGreedyOutput(
        remaining_network_by_failed_edge=remaining_network_by_failed_edge,
        algorithm_failure_flag=algorithm_failure_flag,
        successfully_rerouted_demands_ratio=successfully_rerouted_demands_ratio,
        additional_volume_by_edge=additional_volume_by_edge,
        reserve_paths_by_failed_edge=reserve_paths_by_failed_edge,
    )
