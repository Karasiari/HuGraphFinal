# старые методы под основной объект (core.py), которые решили оставить

import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl

from .core import HuGraphForExps

# визуализация немультиграфов

def visualise(self, version="initial", title="Граф смежности", node_size=300, font_size=10) -> None:
  if version == "current":
    graph = self.graph
  else:
    graph = self.graph_initial
  pos = nx.spring_layout(graph, seed=42)
  plt.figure(figsize=(9, 7))
  nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color="#4C79DA", alpha=0.9)
  nx.draw_networkx_edges(graph, pos, edge_color="#888", alpha=0.8)
  edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in graph.edges(data=True)}
  nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=font_size)
  plt.title(title); plt.axis("off"); plt.tight_layout(); plt.show()

def visualise_with_demands(self, version="initial", node_size: int = 110, font_size: int = 9, figsize=(14, 6),
                           demand_edge_width_range=(1.5, 6.0), node_color="dimgray", base_edge_color="gray",
                           demand_edge_cmap="viridis", edge_alpha=0.9, colorbar_label="Вес запроса") -> None:
  if version == "current":
    graph = self.graph
  else:
    graph = self.graph_initial
  DG = self.demands_graph
  pos = nx.spring_layout(graph, seed=42)
  fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

  # слева — базовый граф
  nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, alpha=0.95, ax=axL)
  nx.draw_networkx_edges(graph, pos, edge_color=base_edge_color, width=1.5, alpha=edge_alpha, ax=axL)
  labels = {(u, v): f"{d.get('weight', 0):.0f}" for u, v, d in graph.edges(data=True)}
  nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=font_size, ax=axL)
  axL.set_title("Граф смежности"); axL.axis("off")

  # справа — demands
  nx.draw_networkx_nodes(DG, pos, node_size=node_size, node_color=node_color, alpha=0.95, ax=axR)
  edgelist = list(DG.edges(data=True))
  uv = [(u, v) for u, v, _ in edgelist]
  W = np.array([float(d.get("weight", 1.0)) for _, _, d in edgelist]) if edgelist else np.array([])
  # ширины
  if W.size:
    w_min, w_max = float(W.min()), float(W.max())
    lo, hi = demand_edge_width_range
    widths = [0.5 * (lo + hi)] * len(W) if np.isclose(w_min, w_max) else list(lo + (W - w_min) * (hi - lo) / (w_max - w_min))
  else:
    widths = []
  # цвета
  cmap = mpl.cm.get_cmap(demand_edge_cmap)
  vmin, vmax = (float(W.min()), float(W.max())) if W.size else (0.0, 1.0)
  nx.draw_networkx_edges(DG, pos, edgelist=uv, width=widths, edge_color=W, edge_cmap=cmap,
                         edge_vmin=vmin, edge_vmax=vmax, alpha=edge_alpha, ax=axR)
  norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
  sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

  cbar = plt.colorbar(sm, ax=axR); cbar.set_label(colorbar_label, fontsize=font_size)
  axR.set_title("Граф запросов"); axR.axis("off")
  plt.tight_layout(); plt.show()

# старые методы

# cut

def _compute_least_nonzero_vector(self, L: np.ndarray) -> np.ndarray:
    # находим все собственные значения и векторы
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # находим индекс минимального ненулевого собственного значения
    # (первое значение в спектре для связного графа — 0)
    eps = 1e-12  # порог для сравнения с нулём
    nonzero_indices = np.where(eigenvalues > eps)[0]

    idx = nonzero_indices[0]  # минимальное ненулевое собственное значение

    # собственный вектор, соответствующий минимальному ненулевому собственному значению
    least_nonzero_vector = eigenvectors[:, idx]

    return least_nonzero_vector

def generate_cut(self, type: str = "min") -> list:
    """
    Генерирует разбиение графа на два кластера.
        - type="min": Разбиение по минимальному ненулевому вектору спектра self.laplacian
        - type="min_Lalpha": Разбиение по минимальному ненулевому вектору спектра self.L_alpha

    Аргументы:
        type (str): Тип разбиения, может быть "min" или "min_Lalpha".

    Возвращает:
        list: Список рёбер self.graph (source, target), где вершины принадлежат разным кластерам.
    """
    if type not in {"min", "min_Lalpha"}:
      raise ValueError("type должен быть 'min' или 'min_Lalpha'")

    if graph.demands_graph is None:
      raise AttributeError("demands_graph не задан")

    if type == "min":
      # для min-cut используем первый ненулевой собственный вектор лапласиана смежности
      v = self._compute_least_nonzero_vector(self.laplacian)
    if type == "min_Lalpha":
      # для min_Lalpha используем первый ненулевой собственный вектор L_alpha
      self.calculate_alpha()
      v = self._compute_least_nonzero_vector(self.L_alpha)

    # используем медиану вектора для разбиения
    med = float(np.median(v))
    # создаём разметку вершин (0 или 1)
    cluster_labels = (v <= med).astype(int)

    # список рёбер между кластерами (где метки разные)
    edges_in_cut = []

    for u, v in self.graph.edges():
      if cluster_labels[u] != cluster_labels[v]:
        edges_in_cut.append((u, v))
            
    if type == "min":
      self.mincut = edges_in_cut
    elif type == "min_Lalpha":
      self.cut_alpha = edges_in_cut

    return edges_in_cut

HuGraphForExps.visuaise = visualise
HuGraphForExps.visualise_with_demands = visualise_with_demands
HuGraphForExps._compute_least_nonzero_vector = _compute_least_nonzero_vector
HuGraphForExps.generate_cut = generate_cut
