from typing import Tuple, Dict, Any
import networkx as nx

# алиасы для читаемости
VolumeWithProbability = Tuple[int, float]
GenerationParam = Dict[str, Any]

def generate_own_traffic(
  graph: nx.MultiGraph, 
  available_demand_volumes: Tuple[VolumeWithProbability, ...],
  generation_type: str,
  generation_params: Tuple[GenerationParam, ...]: 
) -> nx.MultiDiGraph:
  return
