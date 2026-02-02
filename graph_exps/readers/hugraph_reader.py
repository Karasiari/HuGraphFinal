import pandas as pd
import numpy as np
import networkx as nx
import os

# две функции - читалки hu графов в мультиграфы

BITRATE_DENOMINATOR = 100

def hu_csv_to_graphs(path, demands_path, capacity_path):
  Network = pd.read_csv(path, header=None, names = ['id', 'source', 'target', 'length'])
  Traffic = pd.read_csv(demands_path, header=None, names = ['id', 'source', 'target', 'bitrate'])
  Capacity = pd.read_csv(capacity_path, header=None, names = ['name', 'value'])
  network = Network.iloc[1:].copy()
  traffic = Traffic.iloc[1:].copy()
  capacity = Capacity.iloc[1:].copy()

  network['source'] = network['source'].astype(int)
  network['target'] = network['target'].astype(int)
  traffic['source'] = traffic['source'].astype(int)
  traffic['target'] = traffic['target'].astype(int)
  traffic['bitrate'] = traffic['bitrate'].astype(float) / BITRATE_DENOMINATOR
  capacity_value = float(capacity[capacity['name']=='NumberOfWavelengths']['value'])

  sources = network['source'].tolist()
  targets = network['target'].tolist()
  sources_traffic = traffic['source'].tolist()
  targets_traffic = traffic['target'].tolist()
  bitrates_traffic = traffic['bitrate'].tolist()

  unique_vertices = set()
  for source, target in zip(sources, targets):
    unique_vertices.add(source)
    unique_vertices.add(target)
  vertex_mapping = {old: new for new, old in enumerate(unique_vertices)}
  num_vertices = len(unique_vertices)

  adj_edges = []
  traffic_edges = []

  for source, target in zip(sources, targets):
    new_source = vertex_mapping[source]
    new_target = vertex_mapping[target]
    adj_edges.append((new_source, new_target, {"capacity": capacity_value}))
  for source, target, bitrate in zip(sources_traffic, targets_traffic, bitrates_traffic):
    new_source = vertex_mapping[source]
    new_target = vertex_mapping[target]
    traffic_edges.append((new_source, new_target, {"weight": bitrate}))

  adj_graph = nx.MultiGraph()
  adj_graph.add_edges_from(adj_edges)
  traffic_graph = nx.MultiDiGraph()
  traffic_graph.add_nodes_from(range(adj_graph.number_of_nodes()))
  traffic_graph.add_edges_from(traffic_edges)

  return adj_graph, traffic_graph, capacity_value

def read_hu_graphs(base_path, specified_graphs, specified):
  Graphs = {}
  csv_tables = []
  for folder_name in os.listdir(base_path):
    if folder_name in specified_graphs or not specified:
      folder_path = os.path.join(base_path, folder_name)
      csv_path = os.path.join(folder_path, 'links.csv')
      csv_path_demands = os.path.join(folder_path, 'demands.csv')
      csv_path_capacity = os.path.join(folder_path, 'params.csv')
      csv_tables.append(folder_name)

      adj_graph, traffic_graph, capacity_value = hu_csv_to_graphs(csv_path, csv_path_demands, csv_path_capacity)
      Graphs[folder_name] = {'adj_graph': adj_graph, 'traffic_graph': traffic_graph, 'capacity_value': capacity_value}

  return Graphs
