from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import math
import random
import networkx as nx

Coord = [float, float] 


def geodesic_distance(a: Coord, b: Coord) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin2 = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0 * (2 * math.asin(math.sqrt(sin2)))

def euc_distance(a: Coord, b: Coord) -> float:
    return math.dist(a, b)

def etsi_fiber_km(geo_km: float) -> float:
    return geo_km
 #  if geo_km < 1000.0:
  #      return 1.5 * geo_km
 # elif geo_km <= 1200.0:
 #       return 1500.0
 #   return 1.25 * geo_km

def fiber_distance_km(a: Coord, b: Coord) -> float:
    return etsi_fiber_km(geodesic_distance(a, b))





def _centroid(coords: List[Coord]) -> Coord:
    x = sum(lat for lat, _ in coords) / len(coords)
    y = sum(lon for _, lon in coords) / len(coords)
    return (x, y)

def order_nodes_sequential(coords: List[Coord]) -> List[int]:
    n = len(coords)
    remaining = set(range(n))
    c = _centroid(coords)

    def avg_dist_to_set(idx: int, s: List[int]) -> float:
        if not s:
            return 0.0
        dsum = 0.0
        for j in s:
            dsum += fiber_distance_km(coords[idx], coords[j])
        return dsum / len(s)

    # pick first = closest to centroid
    first = min(remaining, key=lambda i: fiber_distance_km(coords[i], c))
    order = [first]
    remaining.remove(first)

    while remaining:
        nxt = min(remaining, key=lambda i: avg_dist_to_set(i, order))
        order.append(nxt)
        remaining.remove(nxt)

    return order

# -----------------------------
# SNR-BA attachment weights
# -----------------------------

class SNRBAParams:
    def __init__(self, m: Optional[int] = None, theta: float = 5.0, rng: Optional[random.Random] = None):
        self.m = m
        self.theta = theta
        self.rng = rng if rng is not None else random.Random(0)

# -----------------------------
# Attachment weights
# -----------------------------

def _normalized_inverse_distance_weights(i: int, existing_nodes: List[int], coords: List[Coord]) -> List[float]:
    eps = 1e-9
    invs = []
    for j in existing_nodes:
        d = fiber_distance_km(coords[i], coords[j])
        invs.append(1.0 / max(d, eps))
    s = sum(invs)
    if s <= 0:
        return [1.0/len(existing_nodes)] * len(existing_nodes)
    return [x / s for x in invs]

def _choose_targets(probabilities: List[float], candidates: List[int], m: int, rng: random.Random) -> List[int]:
    chosen = []
    cand = candidates[:]
    probs = probabilities[:]
    for _ in range(min(m, len(cand))):
        r = rng.random()
        cum = 0.0
        for idx, p in enumerate(probs):
            cum += p
            if r <= cum:
                chosen.append(cand[idx])
                del cand[idx]
                del probs[idx]
                s = sum(probs)
                if s > 0 and probs:
                    probs = [p/s for p in probs]
                break
        else:
            chosen.append(cand.pop())
            if probs:
                probs.pop()
            if probs:
                s = sum(probs)
                probs = [p/s for p in probs]
    return chosen

# -----------------------------
# Main generator
# -----------------------------

def generate_snr_ba(coords: Iterable[Coord],
                    params: SNRBAParams = SNRBAParams(),
                    desired_edges: Optional[int] = None,
                    distance: Callable[Coord, Coord] = euc_distance) -> nx.Graph:
    coords = list(coords)
    N = len(coords)

    order = order_nodes_sequential(coords)

    if params.m is not None:
        m = int(params.m)
    elif desired_edges is not None:
        m = max(1, desired_edges // N)
    else:
        m = 2 if N >= 8 else 1

    theta = float(params.theta)
    rng = params.rng

    G = nx.Graph()
    for idx in range(N):
        G.add_node(idx, pos=coords[idx])

    seed_a, seed_b = order[0], order[1]
    G.add_edge(seed_a, seed_b, fiber_km= fiber_distance_km(coords[seed_a], coords[seed_b]))

    present = {seed_a, seed_b}

    for i in order[2:]:
        existing = sorted(present)
        snr_weights = _normalized_inverse_distance_weights(i, existing, coords)

        degrees = [max(G.degree(j), 1) for j in existing]
        snr_term = [w ** theta for w in snr_weights]
        weights = [snr_term[k] * degrees[k] for k in range(len(existing))]
        s = sum(weights)
        if s <= 0:
            probs = [1.0 / len(existing)] * len(existing)
        else:
            probs = [w / s for w in weights]

        targets = _choose_targets(probs, existing, m, rng)

        for j in targets:
            if not G.has_edge(i, j):
                G.add_edge(i, j, fiber_km=fiber_distance_km(coords[i], coords[j]))
        present.add(i)

    if desired_edges is not None and G.number_of_edges() < desired_edges:
        attempts = 0
        while G.number_of_edges() < desired_edges and attempts < 10 * N:
            i = rng.choice(order[2:])
            existing = [v for v in G.nodes if v != i and not G.has_edge(i, v)]
            if not existing:
                attempts += 1
                continue
            snr_weights = _normalized_inverse_distance_weights(i, existing, coords)
            degrees = [max(G.degree(j), 1) for j in existing]
            snr_term = [w ** theta for w in snr_weights]
            weights = [snr_term[k] * degrees[k] for k in range(len(existing))]
            s = sum(weights)
            probs = [w / s for w in weights] if s > 0 else [1.0 / len(existing)] * len(existing)
            j = _choose_targets(probs, existing, 1, rng)[0]
            G.add_edge(i, j, fiber_km=fiber_distance_km(coords[i], coords[j]))
            attempts += 1

    return G

# -----------------------------
# Convenience wrapper
# -----------------------------

def snr_ba_from_latlon(latlons: List[Tuple[float, float]],
                       m: Optional[int] = None,
                       theta: float = 5.0,
                       desired_edges: Optional[int] = None,
                       seed: int = 0) -> nx.Graph:
    params = SNRBAParams(m=m, theta=theta, rng=random.Random(seed))
    return generate_snr_ba(latlons, params, desired_edges=desired_edges)
