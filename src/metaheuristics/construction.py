# src/metaheuristics/construction.py
from src.instance import Instance
from src.solution import Solution, TruckRoute, Sortie, Robot
from src.config import *
from src.metaheuristics.evaluate import evaluate_solution
from src.milp import solve_vrp_milp
import math

def solve_vrp(inst: Instance) -> Solution:
    """
    Costruisce una soluzione iniziale truck-only con il metodo sector sweep.
    Tutti i clienti sono assegnati a rotte truck.
    
    Returns:
        Solution con K rotte TruckRoute, ogni nodo ha label 'T',
        nessuna sortie drone, nessun robot attivo.
    """
    # All'inizio usiamo il max di trucks per garantire una sol. feasible
    n_routes = MAX_TRUCKS_K

    # 2. Recupera le coordinate del deposito di partenza (nodo 0) 
    # e le coordinate dei nodi clienti
    depot_coord = inst.coords[0]  
    nodes_to_assign = list(range(1, inst.n_customers + 1))  

    # 3. Per ogni nodo da assegnare, calcola l'angolo rispetto al deposito
    #    (in gradi, da 0 a 360).
    angles = {}   # {node_index: angle_degrees}
    for node in nodes_to_assign:
        dx = inst.coords[node, 0] - depot_coord[0]
        dy = inst.coords[node, 1] - depot_coord[1]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        angles[node] = angle

    # 4. Calcola l'ampiezza del settore
    sector_width = 360.0 / n_routes

    # 5. Assegna ogni nodo al suo settore (= indice rotta)
    #    Settore s copre gli angoli [s * sector_width, (s+1) * sector_width)
    assignment = {s: [] for s in range(n_routes)}
    for node, angle in angles.items():
        s = min(int(angle // sector_width), n_routes - 1)
        assignment[s].append(node)

    # 6. Per ogni settore, ordina i nodi in base a t_start
    routes_nodes = []
    for s in range(n_routes):
        nodes_in_sector = assignment[s]
        sorted_nodes = sorted(
            nodes_in_sector,
            key=lambda n: (inst.t_stop[n-1], inst.dist_T[0, n]), reverse=False
        )
        routes_nodes.append(sorted_nodes)

    # 7. Costruisce l'oggetto Solution
    routes = []
    for nodes in routes_nodes:
        route = TruckRoute(nodes=[0] + nodes + [inst.n_nodes - 1], labels=['T'] * (len(nodes) + 2))
        if len(route.nodes) > 2: 
            routes.append(route)

    return Solution(routes=routes)

def insert_robots(solution: Solution, inst: Instance) -> Solution:
    """
    Rimuove clienti dalle rotte truck e li assegna a stazioni robot
    seguendo la cheapest insertion rule.

    Modifica `solution` in-place (o su una copia) e la restituisce.
    """
    improved = True
    while improved:
        improved = False
        best_delta = 0.0          # accettiamo solo mosse con delta < 0
        best_move = None          # (route_idx, customer_pos, station_pos, station_node)

        for r_idx, route in enumerate(solution.routes):
            # Identifica le stazioni nella rotta corrente
            stations_in_any_route = {trip.station for route in solution.routes for trip in route.robots}
            stations_free = [station_node for station_node in range(inst.n_customers + 1, inst.n_nodes - 1) if station_node not in stations_in_any_route]
            station_candidates = set(stations_free)
            station_candidates.update({trip.station for trip in route.robots})

            # Identifica i clienti ancora serviti da truck in questa rotta
            nodes = route.nodes
            truck_customers = [
                (pos, n) for pos, n in enumerate(nodes)
                if 1 <= n <= inst.n_customers
            ]

            for pos, j in truck_customers:
                for station in station_candidates:
                    # se la stazione non è in nessuna rotta:
                    if station in stations_free:
                        nodes_no_j = nodes[:pos] + nodes[pos+1:]
                        pos_to_candidates = [(inst.dist_T[nodes_no_j[pos_to-1],station] + inst.dist_T[station,nodes_no_j[pos_to]] - inst.dist_T[nodes_no_j[pos_to-1],nodes_no_j[pos_to]], pos_to) for pos_to in range(1, len(nodes_no_j)-1)]
                        pos_to_candidates.sort()  # ordina per costo di inserimento
                        for _, pos_to in pos_to_candidates[:K_TOP]:
                            if not inst.enough_range_robot(station,j): continue
                            sol_copy = solution.copy()
                            route_copy = sol_copy.routes[r_idx]
                            route_copy.nodes.pop(pos)
                            route_copy.labels.pop(pos)
                            route_copy.nodes.insert(pos_to, station)
                            route_copy.labels.insert(pos_to, 'T')
                            route_copy.robots.append(Robot(station=station, customers=[j]))
                            new_cost = evaluate_solution(sol_copy, inst)
                            if new_cost - solution.cost < best_delta:
                                best_delta = new_cost - solution.cost
                                best_move = ("new station", r_idx, j, pos, station, pos_to)
                    
                    # se la stazione è già in una rotta:            
                    else:
                        sol_copy = solution.copy()
                        route_copy = sol_copy.routes[r_idx]
                        robots_from_station = [(trip, t_idx) for (t_idx, trip) in enumerate(route_copy.robots) if trip.station == station]
                        robots_from_station.sort(key = lambda tup: len(tup[0].customers))
                        if len(robots_from_station)<MAX_ROBOTS_R:
                            route_copy.robots.append(Robot(station=station, customers=[j]))
                            route_copy.nodes.pop(pos)
                            route_copy.labels.pop(pos)
                            new_cost = evaluate_solution(sol_copy, inst)
                            if new_cost - solution.cost < best_delta:
                                best_delta = new_cost - solution.cost
                                best_move = ("existing station new robot", r_idx, j, pos, station)
                        
                        else:
                            robot, t_idx = robots_from_station[0]
                            robot.customers.append(j)
                            route_copy.nodes.pop(pos)
                            route_copy.labels.pop(pos)
                            new_cost = evaluate_solution(sol_copy, inst)
                            if new_cost - solution.cost < best_delta:
                                best_delta = new_cost - solution.cost
                                best_move = ("existing station existing robot", r_idx, j, pos, t_idx)
                
                        
        # Applica la best_move trovata
        if best_move is not None:
            improved = True
            if best_move[0] == "new station":
                _, r_idx, j, pos, station, pos_to = best_move
                route = solution.routes[r_idx]
                route.nodes.pop(pos)
                route.labels.pop(pos)
                route.nodes.insert(pos_to, station)
                route.labels.insert(pos_to, 'T')
                route.robots.append(Robot(station=station, customers=[j]))

            elif best_move[0] == "existing station new robot":
                _, r_idx, j, pos, station = best_move
                route = solution.routes[r_idx]
                route.robots.append(Robot(station=station, customers=[j]))
                route.nodes.pop(pos)
                route.labels.pop(pos)

            elif best_move[0] == "existing station existing robot":
                _, r_idx, j, pos, t_idx = best_move
                route = solution.routes[r_idx]
                route.robots[t_idx].customers.append(j)
                route.nodes.pop(pos)
                route.labels.pop(pos)
            evaluate_solution(solution, inst)

    return solution

def make_fly(solution: Solution, inst: Instance) -> Solution:
    improved = True
    while improved:
        improved = False
        best_delta = 0.0      
        best_move = None      # (route_idx, j_pos, i_node, j_node, k_node)

        for r_idx, route in enumerate(solution.routes):
            nodes = route.nodes

            # Scorre le triple (nodes[pos-1], nodes[pos], nodes[pos+1])
            for pos in range(1, len(nodes) - 1):
                i = nodes[pos - 1]
                j = nodes[pos]
                k = nodes[pos + 1]

                if not (1 <= j <= inst.n_customers):
                    continue

                used_launches  = {s.launch   for s in route.sorties}
                used_lands     = {s.land     for s in route.sorties}
                used_customers = {s.customer for s in route.sorties}

                if i in used_launches:   # già usato come launch → continue
                    continue
                if k in used_lands:      # già usato come land → continue
                    continue
                if j in used_customers or j in used_launches or j in used_lands:  # j deve essere libero
                    continue
                if j in inst.no_fly:
                    continue

                if not inst.enough_battery_drone(i, j, k):continue

                # 2. Calcola delta
                delta = TRUCK_COST_PER_KM * (inst.dist_T[i, k] - inst.dist_T[i, j] - inst.dist_T[j, k]) + DRONE_COST_PER_KM * (inst.dist_D[i, j] + inst.dist_D[j, k])

                # 3. Aggiorna best_move se questo è il migliore
                if delta < 2*best_delta:
                    sol_copy = solution.copy()
                    route_copy = sol_copy.routes[r_idx]
                    route_copy.nodes.pop(pos)
                    route_copy.labels.pop(pos)
                    idx_i = route_copy.nodes.index(i)
                    idx_k = route_copy.nodes.index(k)
                    route_copy.labels[idx_i] = 'C'
                    route_copy.labels[idx_k] = 'C'

                    sortie = Sortie(launch=i, customer=j, land=k)
                    route_copy.sorties.append(sortie)
                    new_cost = evaluate_solution(sol_copy,inst)
                    if new_cost-solution.cost < best_delta:
                        best_delta = new_cost-solution.cost
                        best_move = (r_idx, pos, i, j, k)


        # Applica best_move se trovata
        if best_move is not None:
            improved = True
            r_idx, pos, i, j, k = best_move
            route = solution.routes[r_idx]

            # 4. Rimuove j da route.nodes e route.labels
            route.nodes.pop(pos)
            route.labels.pop(pos)

            # 5. Aggiorna i label di i e k a 'C'
            idx_i = route.nodes.index(i)
            idx_k = route.nodes.index(k)
            route.labels[idx_i] = 'C'
            route.labels[idx_k] = 'C'

            # 6. Aggiunge la sortie a route.sorties
            sortie = Sortie(launch=i, customer=j, land=k)
            route.sorties.append(sortie)    

    return solution

def phase_I(inst:Instance, gurobi_time_limit: float = None, verbose = False) -> Solution:
    '''Fase 1 dell'algoritmo: cerca una soluzione feasible di soli truck
    e poi aggiunge robot e droni se migliorano il costo'''

    # Prova a cercare una soluzione velocemente con gurobi
    sol = solve_vrp_milp(inst, time_limit = GUROBI_TIME_LIMIT, verbose=verbose)

    if sol is None:
        #Se non riesce, usa solve_vrp (sector-sweep)
        if verbose: print("Nessuna soluzione trovata con gurobi, inizio solve_vrp normale...")
        sol = solve_vrp(inst)
    if verbose: print(f"Costo sol iniziale = {evaluate_solution(sol,inst)}")

    # Inserisce i robot
    sol = insert_robots(sol,inst)
    if verbose: print(f"Costo con robots = {evaluate_solution(sol,inst)}")

    # Inserisce i droni
    sol = make_fly(sol, inst)
    if verbose: print(f"Costo con droni = {evaluate_solution(sol,inst)}")

    return sol