# src/metaheuristic/granular.py

from dataclasses import dataclass, field
from typing import Set, Tuple, List
from src.solution import Solution
from src.instance import Instance
from src.config import *

@dataclass
class GranularSet:
    truck: Set[Tuple[int,int]]       = field(default_factory=set)
    drone: Set[Tuple[int,int,int]]   = field(default_factory=set)
    robot: Set[Tuple[int,int]]       = field(default_factory=set)


def _compute_threshold(solution: Solution, inst: Instance, beta: float) -> float:
    """
    Calcola la soglia granulare ϑ secondo Eq. (46) del paper.

    ϑ = beta * (somma costi archi truck in soluzione) / (|V_N| + |V_S| + |K|)
    """
    truck_arcs = [(route.nodes[i],route.nodes[i+1]) for route in solution.routes for i in range(len(route.nodes)-1)]
    truck_arc_cost = sum(inst.dist_T[i, j] * TRUCK_COST_PER_KM for i, j in truck_arcs)

    active_routes = [route for route in solution.routes if len(route.nodes)>2]
    denominator = inst.n_customers + inst.n_stations + len(active_routes)
    return beta * truck_arc_cost / denominator


def _collect_truck_arcs(inst: Instance, threshold: float) -> GranularSet:
    """
    Raccoglie tutti gli archi truck (i,j) con costo ≤ threshold
    dall'intero insieme A (tutti i nodi x tutti i nodi).
    """
    gs = GranularSet()
    n = inst.n_nodes
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if i == n-1 or j == 0: continue
            if i == 0 or j == n-1:
                gs.truck.add((i, j))
            elif inst.dist_T[i, j] * TRUCK_COST_PER_KM <= threshold:
                gs.truck.add((i, j))
    return gs


def _collect_drone_arcs(inst: Instance) -> GranularSet:
    """
    Raccoglie tutti gli archi drone (i, j, h) la cui energia totale
    b_on[i,j] + b_off[j,h] < DRONE_BATTERY_B.
    """
    gs = GranularSet()
    n = inst.n_nodes
    Vl = range(0, n - 1)
    Vn = range(1, inst.n_customers+1)
    Vr = range(1, n)
    for i in Vl:
        if i in inst.no_fly: continue
        for j in Vn:
            if i == j or j in inst.no_fly: continue
            for h in Vr:
                if j == h or h in inst.no_fly: continue
                if inst.enough_battery_drone(i,j,h):
                    gs.drone.add((i, j, h))
    return gs


def _collect_robot_arcs(inst: Instance) -> GranularSet:
    """
    Raccoglie tutti gli archi robot (station, j) tali che il round-trip
    (l_R + t_R[s,j] + l_R_service + t_R[j,s]) ≤ ROBOT_MAX_DRIVE_TIME.
    """
    gs = GranularSet()
    n = inst.n_nodes
    Vs = range(inst.n_customers+1, n-1)
    Vn = range(1, inst.n_customers+1)
    for s in Vs:
        for j in Vn:
            if inst.enough_range_robot(s,j):
                gs.robot.add((s, j))
    return gs


def _add_solution_arcs(solution: Solution, inst: Instance, arcs: GranularSet) -> None:
    """
    Aggiunge in-place tutti gli archi della soluzione corrente ad arcs.
    Questo garantisce che la soluzione corrente sia sempre esplorabile.
    """
    current_truck_arcs = [(route.nodes[i],route.nodes[i+1]) for route in solution.routes for i in range(len(route.nodes)-1)]
    arcs.truck.update(current_truck_arcs)
    current_sorties_arcs = [(s.launch, s.customer, s.land) for route in solution.routes for s in route.sorties]
    arcs.drone.update(current_sorties_arcs)
    current_robots_arcs = [(trip.station, cust) for route in solution.routes for trip in route.robots for cust in trip.customers]
    arcs.robot.update(current_robots_arcs)


def add_edges(solution: Solution, inst: Instance, granular_set: GranularSet) -> None:
    """
    Funzione AddEdges(·) del paper (Algorithm 1, riga 8).
    Aggiunge gli archi della soluzione corrente al set granulare esistente.
    Modifica granular_set in-place.
    """
    _add_solution_arcs(solution, inst, granular_set)


def reset_edges(solution: Solution, inst: Instance, beta: float) -> GranularSet:
    """
    Funzione ResetEdges(·) del paper (Algorithm 1, riga 2 e 18).
    1. Svuota A_g
    2. Calcola la soglia ϑ
    3. Raccoglie archi truck sotto soglia + depot + drone + robot
    4. Chiama add_edges per aggiungere gli archi della soluzione corrente
    Restituisce il nuovo GranularSet.
    """
    granular_set: GranularSet = GranularSet()

    threshold = _compute_threshold(solution, inst, beta)

    truck_arcs = _collect_truck_arcs(inst, threshold)
    granular_set.truck.update(truck_arcs.truck)

    drone_arcs = _collect_drone_arcs(inst)
    granular_set.drone.update(drone_arcs.drone)

    robot_arcs = _collect_robot_arcs(inst)
    granular_set.robot.update(robot_arcs.robot)

    _add_solution_arcs(solution, inst, granular_set)

    return granular_set


def is_granular_move(arc: tuple, granular_set: GranularSet, mode: str) -> bool:
    """
    Controlla se un nuovo arco generato da una mossa
    appartiene ad A_g.
    Usato nei neighborhood operators per filtrare le mosse.

    new_arcs: lista di tuple (i,j) o (i,j,h)
    """
    return arc in getattr(granular_set, mode)

def at_least_one_granular_move(arcs_list: List[Tuple], gs, modes: List[str]) -> bool:
    """Controlla se ALMENO un nuovo arco generato da una mossa
    appartiene ad A_g."""
    for (arc,mode) in zip(arcs_list, modes):
        if is_granular_move(arc, gs, mode):
            return True
    return False
