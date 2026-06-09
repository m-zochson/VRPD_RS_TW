# src.metaheuristics.evaluate.py
from src.instance import Instance
from src.solution import Solution
from src.config import *
from src.metaheuristics.synchronization import synchronize_route

def evaluate_solution(solution: Solution, inst: Instance, objective: str = "cost", verbose: bool = False) -> float:
    """
    Calcola il valore dell'obiettivo per una soluzione completa.
    
    Parameters
    ----------
    solution  : Solution con una o più TruckRoute
    inst      : Instance con matrici dist_T, dist_D, dist_R e parametri di costo
    objective : "cost"  → minimizza ω (costi operativi, €)
                "makespan" → minimizza τ (tempo totale, ore)
    
    Returns
    -------
    float : valore dell'obiettivo
    """
    total_cost = 0.0
    max_time   = 0.0   # usato solo per makespan
    if len(solution.routes) > MAX_TRUCKS_K:
        if verbose: print("Too many trucks")
        return float("inf")

    for route in solution.routes:

        # Step 1: sincronizza la rotta per ottenere i tempi
        times = synchronize_route(route, inst, verbose = verbose)

        if not times["is_feasible"]:          # check di feasibility
            solution.cost = float("inf")
            return float("inf") 
        d_T   = times["d_T"]

        # Step 2: salario autista (f_T × d_T del deposito finale)
        driver_cost = TRUCK_DRIVER_WAGE * d_T[-1]

        # Step 3: costo archi truck
        truck_arc_cost = 0.0
        for k in range(len(route.nodes) - 1):
            i, j = route.nodes[k], route.nodes[k + 1]
            truck_arc_cost += TRUCK_COST_PER_KM * inst.dist_T[i, j]

        # Step 4: costo archi drone
        drone_cost = 0.0
        for sortie in route.sorties:
            i, j, h = sortie.launch, sortie.customer, sortie.land
            drone_cost += DRONE_COST_PER_KM * (inst.dist_D[i, j] + inst.dist_D[j, h])

        # Step 5: costo archi robot
        robot_cost = 0.0
        for trip in route.robots:
            station = trip.station
            for customer in trip.customers:
                robot_cost += ROBOT_COST_PER_KM * (inst.dist_R[station, customer] + inst.dist_R[customer, station])

        route_cost = driver_cost + truck_arc_cost + drone_cost + robot_cost
        total_cost += route_cost

        # Aggiorna il makespan (tempo massimo tra tutti i truck)
        max_time = max(max_time, d_T[-1])
    

    if objective == "cost":
        solution.cost = total_cost
        return solution.cost
    elif objective == "makespan":
        solution.cost = max_time / 3600.0  # converti in ore
        return solution.cost
    else:
        raise ValueError(f"Obiettivo sconosciuto: {objective}")

