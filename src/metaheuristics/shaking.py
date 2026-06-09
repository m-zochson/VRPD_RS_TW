# metaheuristics/shaking.py
import random
from src.solution import Solution, TruckRoute
from src.instance import Instance
from src.metaheuristics.evaluate import evaluate_solution
from src.metaheuristics.neighborhoods import _remove_sortie, _remove_robot_customer
from src.config import MAX_TRUCKS_K

def _truck_nodes_by_route(solution: Solution, inst: Instance) -> list[list[tuple[int, int]]]:
    """
    Restituisce una lista per rotta, dove ogni elemento è (route_idx, node_idx).
    Sono inclusi solo i nodi con label 'T' (no depot, no 'C').
    """
    truck_nodes = []
    for r, route in enumerate(solution.routes):
        route_nodes = []
        for node_idx, n in enumerate(route.nodes):
            if route.labels[node_idx] == 'T' and 1 <= n <= inst.n_customers:
                route_nodes.append((r, node_idx))
        truck_nodes.append(route_nodes)
    return truck_nodes



def shaking_1(solution: Solution, inst: Instance) -> tuple[Solution, bool]:
    """
    Seleziona due nodi 'T' a caso (da rotte uguali o diverse).
    Li scambia di posizione.
    Restituisce (sol_modificata, feasible: bool).
    """
    sol = solution.copy()
    all_candidates = [node for route_nodes in _truck_nodes_by_route(sol, inst) for node in route_nodes]

    if len(all_candidates) < 2:
        return solution, False

    (r1, idx1), (r2, idx2) = random.sample(all_candidates, 2)

    sol.routes[r1].nodes[idx1], sol.routes[r2].nodes[idx2] = sol.routes[r2].nodes[idx2], sol.routes[r1].nodes[idx1]
    
    cost = evaluate_solution(sol, inst)
    if cost < float("inf"):
        return sol, True
    return solution, False

def shaking_2(solution: Solution, inst: Instance) -> tuple[Solution, bool]:
    '''Shaking 2: Prende un solo nodo visitato da un truck e lo mette
    in un'altra rotta.
    Se c'è solo una rotta, ne crea una nuova con solo quel nodo.'''
    
    sol = solution.copy()
    truck_nodes = _truck_nodes_by_route(sol, inst)
    if sum(len(nodes) for nodes in truck_nodes) < 2:
        return solution, False
    if len(sol.routes) == 1 and MAX_TRUCKS_K>=2:
        sol.routes.append(TruckRoute(nodes=[0, inst.n_nodes - 1], labels=['T', 'T']))
        r1_idx, idx1 = random.choice(truck_nodes[0])
        r2_idx, insert_pos = 1, 1
    else:
        valid_sources = [r_idx for r_idx, nodes in enumerate(truck_nodes) if len(nodes) > 0]
        if not valid_sources: return solution, False
        r1_idx = random.choice(valid_sources)
        _, idx1 = random.choice(truck_nodes[r1_idx])
        valid_destinations = [(r_idx, r) for r_idx, r in enumerate(sol.routes) if r_idx != r1_idx]
        r2_idx, _ = random.choice(valid_destinations)
        insert_pos = random.randint(1, len(sol.routes[r2_idx].nodes) - 1)
    node1 = sol.routes[r1_idx].nodes.pop(idx1)
    lbl = sol.routes[r1_idx].labels.pop(idx1)
    sol.routes[r2_idx].nodes.insert(insert_pos, node1)
    sol.routes[r2_idx].labels.insert(insert_pos, lbl)
    cost = evaluate_solution(sol, inst)
    if cost < float("inf"):
        return sol, True
    return solution, False

def shaking_3(solution: Solution, inst: Instance) -> tuple[Solution, bool]:
    '''Shaking 3: Prende una sortie, la disassembla e inserisce il nodo customer
    nella stessa o in un'altra rotta'''

    sol = solution.copy()
    drone_nodes = [(r_idx, s_idx, sortie.customer) for r_idx, route in enumerate(sol.routes) for s_idx, sortie in enumerate(route.sorties)]
    if not drone_nodes: return solution, False
    r1_idx, s_idx, customer = random.choice(drone_nodes)
    r2_idx, r2 = random.choice([(r_idx, r) for r_idx, r in enumerate(sol.routes)])
    insert_pos = random.randint(1, len(r2.nodes) - 1)
    sol.routes[r2_idx].nodes.insert(insert_pos, customer)
    sol.routes[r2_idx].labels.insert(insert_pos, 'T')
    _remove_sortie(sol.routes[r1_idx], s_idx)
    cost = evaluate_solution(sol, inst)
    if cost < float("inf"):
        return sol, True
    return solution, False

def dissemble_sortie(route: TruckRoute, s_idx: int) -> None:
    '''Disassembla una sortie: la toglie dalla rotta e rimette il cliente
    come nodo truck'''
    sortie = route.sorties[s_idx]
    launch_idx = route.nodes.index(sortie.launch)
    route.nodes.insert(launch_idx + 1, sortie.customer)
    route.labels.insert(launch_idx + 1, 'T')
    _remove_sortie(route, s_idx)


def shaking_4(solution: Solution, inst: Instance) -> tuple[Solution, bool]:
    '''Shaking 4: Prende una stazione robot e la mette in un'altra posizione
    random nella stessa o in un'altra rotta del truck.'''

    sol = solution.copy()
    robots = [(r_idx, t_idx, trip, trip.station) for r_idx, route in enumerate(sol.routes) for t_idx, trip in enumerate(route.robots)]
    if not robots: return solution, False
    r1_idx, _, _, station = random.choice(robots)
    _, route2 = random.choice([(r_idx, r) for r_idx, r in enumerate(sol.routes)])
    route1 = sol.routes[r1_idx]

    # gestisci sorties che usano questa stazione come launch/land
    station_idx = route1.nodes.index(station)
    if route1.labels[station_idx] == 'C':
        sorties_over_station = [s_idx for s_idx, s in enumerate(route1.sorties) if s.launch == station or s.land == station]
        for s_idx in sorted(sorties_over_station, reverse=True):
            dissemble_sortie(route1, s_idx)

    # sposta tutti i robot con questa stazione
    robots_to_move = [r for r in route1.robots if r.station == station]
    route1.robots = [r for r in route1.robots if r.station != station]

    # rimuovi il nodo stazione da route1
    station_idx = route1.nodes.index(station)
    route1.nodes.pop(station_idx)
    route1.labels.pop(station_idx)

    # inserisci in route2
    insert_pos = random.randint(1, len(route2.nodes) - 1)
    route2.nodes.insert(insert_pos, station)
    route2.labels.insert(insert_pos, 'T')
    route2.robots.extend(robots_to_move)

    cost = evaluate_solution(sol, inst)
    if cost < float("inf"):
        return sol, True
    return solution, False

def shaking_5(solution: Solution, inst: Instance) -> tuple[Solution, bool]:
    '''Shaking 5: Prende un cliente visitato da un round-trip di un drone
    e lo inserisce come customer truck in una rotta a caso.'''
    sol = solution.copy()
    drone_nodes = [(r_idx, s_idx, sortie.customer) for r_idx, route in enumerate(sol.routes) for s_idx, sortie in enumerate(route.sorties) if sortie.launch == sortie.land]
    if not drone_nodes: return solution, False
    r1_idx, s_idx, customer = random.choice(drone_nodes)
    r2_idx, r2 = random.choice([(r_idx, r) for r_idx, r in enumerate(sol.routes)])
    insert_pos = random.randint(1, len(r2.nodes) - 1)
    sol.routes[r2_idx].nodes.insert(insert_pos, customer)
    sol.routes[r2_idx].labels.insert(insert_pos, 'T')
    _remove_sortie(sol.routes[r1_idx], s_idx)
    cost = evaluate_solution(sol, inst)
    if cost < float("inf"):
        return sol, True
    return solution, False

def shaking_6(solution: Solution, inst: Instance) -> tuple[Solution, bool]:
    '''Shaking 6: Prende un cliente servito da un robot
            e lo inserisce come customer truck in una rotta a caso.'''
    sol = solution.copy()
    customers_served_by_robots = [(route_idx, robot_idx, cust_idx, cust)
                     for route_idx, route in enumerate(sol.routes)
                     for robot_idx, robot in enumerate(route.robots)
                     for cust_idx, cust in enumerate(robot.customers)]
    if not customers_served_by_robots: return solution, False
    route1_idx, robot_idx, cust_idx, cust = random.choice(customers_served_by_robots)
    route1, robot = sol.routes[route1_idx], sol.routes[route1_idx].robots[robot_idx]
    _, route2 = random.choice([(r_idx, r) for r_idx, r in enumerate(sol.routes)])
    if route1.is_only_one_customer_served_by_station(robot.station,inst):
        if [s for s in route1.sorties if s.launch == robot.station or s.land == robot.station]: 
            return solution, False
    _remove_robot_customer(route1,robot_idx,cust_idx)
    insert_pos = random.randint(1, len(route2.nodes) - 1)
    route2.nodes.insert(insert_pos, cust)
    route2.labels.insert(insert_pos, 'T')

    if evaluate_solution(sol, inst) < float("inf"):
        return sol, True
    return solution, False


SHAKING_PROCEDURES = [
    shaking_1,
    shaking_2,
    shaking_3,
    shaking_4,
    shaking_5,
    shaking_6,
]