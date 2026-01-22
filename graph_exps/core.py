from __future__ import annotations
import copy
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import networkx as nx
import scipy
from scipy.sparse.linalg import eigsh

import cvxpy as cp
from cvxpy import SolverError

from mcf_algorithm.main_algorithm import solve_mcf_problem

from .instruments import *

class HuGraphForExps:
    def __init__(self, multigraph: nx.MultiGraph, demands_multidigraph: nx.MultiDiGraph) -> None:
        # инициализация
        self.multigraph = multigraph
        self.demands_multidigraph = demands_multidigraph
        self.graph = aggregate_graph(multigraph, weight_name="capacity")
        self.demands_graph = aggregate_graph(demands_multidigraph, weight_name="weight")
        self.laplacian = get_laplacian(self.graph)
        self.demands_laplacian = get_laplacian(self.demands_graph)
        
        # поскольку граф будет меняться в экспериментах по расширению - храним исходный вариант
        self.multigraph_initial = self.multigraph.copy()
        self.graph_initial = self.graph.copy()

        # последние вычисленные alpha и "усредненная" L_alpha
        self.alpha: Optional[float] = None
        self.L_alpha: Optional[np.ndarray] = None

        # кэши для расчётов
        self.graph_pinv_sqrt: Optional[np.ndarray] = get_pinv_sqrt(self.laplacian)

        # атрибуты для решения MCFP
        # последнее рассчитанное gamma для MCFP
        self.gamma: Optional[float] = None

        # атрибуты для решения MCF
        # максимальная capacity мультиребра self.multigraph для MCF
        self.C_max = max([data["capacity"] for _, _, data in self.multigraph.edges(data=True)])
        # флаг - решилось ли последнее MCF
        self.mcf_solved: Optional[bool] = None

        # старые атрибуты
        # последние посчитанные разрезы self.graph (все в old.py)
        self.mincut: Optional[np.ndarray] = None
        self.cut_alpha: Optional[np.ndarray] = None

    # ----------------
    # расчет метрики α
    # ----------------
    def calculate_alpha(self) -> float:
        """
        Вычисляет метрику α = λ_max / trace для анализа устойчивости сети.
    
        α характеризует "соотношение максимальной нагрузки к средней".
    
        Математически:
        1. Создаём нормализованную матрицу: L_α = Lg_inv_sqrt @ Ld @ Lg_inv_sqrt
           где Lg_inv_sqrt = (Lg⁺)^(1/2) - нормирует по топологии графа
        2. Находим максимальное собственное значение λ_max(L_α)
        3. Вычисляем след trace(L_α) = Σ λ_i
        4. α = λ_max / trace - итоговая метрика
    
        Интерпретация:
        - Высокое α (> 0.5) - сеть неравномерно нагружена, есть "узкие места"
        - Низкое α (< 0.1) - нагрузка распределена равномерно
        - α → 1 при неустойчивости нагрузки
        - α → 0 при полностью равномерной нагрузке
    
        Returns
        -------
        float
            Метрика устойчивости α (от 0 до 1, может быть >1 в особых случаях)
        """
        Ld = self.demands_laplacian
        Lg_inv_sqrt = get_pinv_sqrt(self.laplacian)
        L_alpha = Lg_inv_sqrt @ Ld @ Lg_inv_sqrt
        self.L_alpha = L_alpha
        eig, _ = eigsh(L_alpha, k=1, which="LA")
        lam_max = float(eig[0]) if eig.size else 0.0
        tr = float(np.trace(L_alpha))
        self.alpha = lam_max / tr if tr != 0.0 else float("inf")
        return lam_max / tr if tr != 0.0 else float("inf")

    # -------------------------
    # изменения self.multigraph
    # -------------------------
    def change_multiedge(self, source: int, target: int, type: str, key: int = None, capacity: float = None) -> None:
        """
        Удаление или добавление мультиребра мультиграфа смежности self.multigraph
        type: "delete" или "insert"
        key: значение ключа удаляемого мультиребра (только для delete)
        capacity: значение capacity нового мультиребра (только для insert)
        """

        if type == "delete":
          if key is None:
            raise ValueError("Для delete необходимо указать key удаляемого мультиребра")

          edge_data = self.multigraph.get_edge_data(source, target, key=key)
          if edge_data:
              capacity_to_decrease = edge_data["capacity"]
              self.multigraph.remove_edge(source, target, key=key)
              current_capacity = self.graph.get_edge_data(source, target)["weight"]
              new_capacity = current_capacity - capacity_to_decrease
              if new_capacity > 0:
                  self.graph[source][target]["weight"] = float(new_capacity)
                  self.laplacian = update_laplacian_on_edge(self.laplacian, source, target, -capacity_to_decrease)
              else:
                  self.graph.remove_edge(source, target)
                  self.laplacian = update_laplacian_on_edge(self.laplacian, source, target, -current_capacity)
          else:
              print(f"Удаляемое мультиребро ({source}, {target}, {key}) не найдено")

        elif type == "insert":
            if capacity is None:
                raise ValueError("Для insert необходимо указать параметр capacity")
            elif capacity <= 0:
                raise ValueError("Параметр capacity должен быть положительным")

            self.multigraph.add_edge(source, target, capacity=capacity)

            if self.graph.get_edge_data(source, target):
                current_capacity = self.graph.get_edge_data(source, target)["weight"]
                new_capacity = current_capacity + capacity
                self.graph[source][target]["weight"] = float(new_capacity)
            else:
                self.graph.add_edge(source, target, weight=float(capacity))
            self.laplacian = update_laplacian_on_edge(self.laplacian, source, target, capacity)
        
        else:
          raise ValueError('type должен быть "delete" или "insert"')

    def restore_graph(self) -> None:
        """
        Восстановление self.multigraph из self.multigraph_initial
        """

        self.multigraph = self.multigraph_initial.copy()
        self.graph = self.graph_initial.copy()
        self.laplacian = get_laplacian(self.graph)

    # ---------------------------
    # основные алгоритмы на графе
    # ---------------------------

    # ------------
    # MCFP (gamma)
    #-------------
    def solve_mcfp(self, solver_flag: bool = False, **solver_kwargs) -> float:
        """
        Решение задачи максимального пропускного потока на графе self.graph + self.demands_graph с использованием CVXPY.
        solver_kwargs: параметры для solver.solve(), такие как методы решения и точность.
        return: gamma
        """
        # копируем граф и преобразуем его в ориентированный
        graph = self.graph.copy()
        graph = nx.DiGraph(graph)

        # копируем лапласиан запросов
        demands_laplacian = self.demands_laplacian.copy()

        # получаем incidence matrix и capacities рёбер
        incidence_mat = get_incidence_matrix_for_mcfp(graph)
        bandwidth = get_capacities_for_mcfp(graph)

        # определяем переменные потока и гамму
        flow = cp.Variable((len(graph.edges), len(graph.nodes)))
        gamma = cp.Variable()

        # определяем задачу
        prob = cp.Problem(
            cp.Maximize(gamma),
            [
                cp.sum(flow, axis=1) <= bandwidth,
                incidence_mat @ flow == -gamma * demands_laplacian.T,
                flow >= 0,
                gamma >= 0,
            ]
        )

        # решаем задачу
        max_for_tries = 5

        solver_error = False
        if not solver_flag:
            try:
                prob.solve(solver='CLARABEL', **solver_kwargs)
            except SolverError:
                solver_error = True
                prob.solve(solver='ECOS', **solver_kwargs)
        else:
            prob.solve(solver='CLARABEL', **solver_kwargs)
        gamma = gamma.value if gamma is not None else None
        max_gamma = gamma
        current_try = 1
        if prob.status != "optimal":
            gamma = None
        while gamma is None and (current_try <= 5 or max_gamma is None):
            if not solver_flag:
                if not solver_error:
                    try:
                        prob.solve(solver='CLARABEL', **solver_kwargs)
                    except SolverError:
                        solver_error = True
                        current_try = 1
                        max_gamma, gamma = None, None
                        prob.solve(solver='ECOS', **solver_kwargs)
                else:
                    prob.solve(solver='ECOS', **solver_kwargs)
            else:
                prob.solve(solver='CLARABEL', **solver_kwargs)
            if max_gamma is not None and gamma is not None:
                max_gamma = max(max_gamma, gamma.value)
            elif max_gamma is None and gamma is not None:
                max_gamma = gamma.value
            current_try += 1
            if prob.status != "optimal":
                gamma = None
            gamma = gamma.value if gamma is not None else None

        gamma = gamma if gamma is not None else max_gamma
        self.gamma = gamma
        
        return gamma

    # ------------------------------------------------------------------------------------------
    # MCF (проложенные запросы, проложенные запросы с индексами, флаг - проложились ли все запросы)
    # ------------------------------------------------------------------------------------------
    def solve_mcf(self, eps=0.1):
        # получаем правильный формат input под наш алгоритм
        demands = []
        index, unsatisfied_subset = 0, set()
        for source, sink, key, data in self.demands_multidigraph.edges(keys=True, data=True):
            capacity = data.get("weight", 0.0)
            demands.append(Demand(source, sink, capacity))
            unsatisfied_subset.add(index)
            index += 1
        G = nx.MultiDiGraph(self.multigraph)
        G_copy = G.copy()

        # получаем решение
        flow_paths, satisfied_demands_dict, solved = solve_mcf_problem(G_copy, self.C_max, demands, unsatisfied_subset, eps)
        self.mcf_solved = solved

        return flow_paths, satisfied_demands_dict, solved
