import networkx as nx
import numpy as np

import cvxpy as cp
from cvxpy import SolverError

# импорт вспомогательных функций под наш алгоритм
from .instruments import *

# -----------------
# основной алгоритм
# -----------------

def solve_max_concurrent_flow_problem(graph: nx.DiGraph, demands_laplacian: np.ndarray, solver_flag: bool) -> float:
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
  
  return gamma
