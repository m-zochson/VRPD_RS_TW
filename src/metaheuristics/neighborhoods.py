# src/metaheuristics/neighbourhoods.py
from itertools import combinations, product
from src.solution import *
from src.instance import Instance
from src.config import *
from src.metaheuristics.evaluate import evaluate_solution
from src.metaheuristics.granular import GranularSet, is_granular_move, at_least_one_granular_move


def _is_only_role(node: int, sortie: Sortie, route: TruckRoute) -> bool:
    """
    Restituisce True se `node` non compare come launch o land
    in nessun'altra sortie della rotta (quindi può tornare a label 'T'
    dopo aver dissemblato una sortie).
    """
    return not any(
        (s.launch == node or s.land == node)
        for s in route.sorties
    )

def push_left(solution: Solution, inst: Instance, gs: GranularSet = None) -> Solution:
    """
    Neighborhood Push Left (B.7).

    Sposta il nodo di lancio di ogni sortie (i, j, k) al predecessore nella 
    rotta truck: (i, j, k) → (prev_i, j, k). Il drone vola da un nodo più 
    a sinistra mentre il truck prosegue, estendendo la finestra della sortie.
    """

    evaluate_solution(solution, inst)   # assicura solution.cost aggiornato

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_move = None

        for r_idx, route in enumerate(solution.routes):
            nodes = route.nodes

            for s_idx, sortie in enumerate(route.sorties):
                launch_idx = nodes.index(sortie.launch)

                if launch_idx == 0:
                    continue

                used_lands = {s.land for s in route.sorties if s is not sortie}
                prev_node  = nodes[launch_idx - 1]
                i, j, k    = sortie.launch, sortie.customer, sortie.land
                new_launch = prev_node

                if i in used_lands: continue
                if not inst.enough_battery_drone(new_launch, j, k): continue
                if j in inst.no_fly: continue

                # Pre-filtro rapido: se aumenta tanto la strada, non considerare
                if inst.dist_D[new_launch, j] >= 2 * inst.dist_D[i, j]:
                    continue

                if gs is not None and not is_granular_move((new_launch, j, k), gs, "drone"):
                    continue

                # Delta completo su copia
                sol_copy   = solution.copy()
                route_copy = sol_copy.routes[r_idx]
                sortie_copy = route_copy.sorties[s_idx]
                sortie_copy.launch = new_launch
                if _is_only_role(i, sortie_copy, route_copy):
                    route_copy.labels[launch_idx] = 'T'
                route_copy.labels[launch_idx - 1] = 'C'

                new_cost = evaluate_solution(sol_copy, inst)   # inf se infeasible
                delta    = new_cost - solution.cost

                if delta < best_delta:
                    best_delta = delta
                    best_move  = (r_idx, s_idx, new_launch)

        if best_move is not None:
            improved   = True
            r_idx, s_idx, new_launch = best_move
            route      = solution.routes[r_idx]
            sortie     = route.sorties[s_idx]
            old_launch = sortie.launch

            if _is_only_role(old_launch, sortie, route):
                route.labels[route.nodes.index(old_launch)] = 'T'
            route.labels[route.nodes.index(new_launch)] = 'C'
            sortie.launch = new_launch

            evaluate_solution(solution, inst)   

    return solution

def push_right(solution: Solution, inst: Instance, gs: GranularSet = None) -> Solution:
    """
    Neighborhood Push Right (B.7).

    Sposta il nodo di atterraggio di ogni sortie (i, j, k) al successore nella 
    rotta truck: (i, j, k) → (i, j, next_k). Il drone atterra a un nodo più 
    a destra mentre il truck prosegue, estendendo la finestra della sortie.
    """

    evaluate_solution(solution, inst)   

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_move = None

        for r_idx, route in enumerate(solution.routes):
            nodes = route.nodes

            for s_idx, sortie in enumerate(route.sorties):
                land_idx = nodes.index(sortie.land)

                if land_idx >= len(nodes) - 1: continue

                used_launches = {s.launch for s in route.sorties if s is not sortie}

                next_node = nodes[land_idx + 1]
                i, j, k   = sortie.launch, sortie.customer, sortie.land
                new_land  = next_node
                if k in used_launches: continue
                if not inst.enough_battery_drone(i,j,new_land): continue
                if j in inst.no_fly: continue

                # Pre-filtro rapido: se l'approssimazione non migliora, salta
                if inst.dist_D[j,new_land] >= 2 * inst.dist_D[j, k]: continue

                if gs is not None and not is_granular_move((i,j,new_land), gs, "drone"): continue
                # Delta completo su copia
                sol_copy   = solution.copy()
                route_copy = sol_copy.routes[r_idx]
                sortie_copy = route_copy.sorties[s_idx]

                sortie_copy.land = new_land
                if _is_only_role(k, sortie_copy, route_copy):
                    route_copy.labels[land_idx] = 'T'
                route_copy.labels[land_idx + 1] = 'C'

                new_cost = evaluate_solution(sol_copy, inst)   # inf se infeasible
                delta    = new_cost - solution.cost

                if delta < best_delta:
                    best_delta = delta
                    best_move  = (r_idx, s_idx, new_land)

        if best_move is not None:
            improved = True
            r_idx, s_idx, new_land = best_move
            route    = solution.routes[r_idx]
            sortie   = route.sorties[s_idx]
            old_land = sortie.land

            if _is_only_role(old_land, sortie, route):
                route.labels[route.nodes.index(old_land)] = 'T'
            route.labels[route.nodes.index(new_land)] = 'C'
            sortie.land = new_land

            evaluate_solution(solution, inst)   

    return solution

def two_opt(solution: Solution, inst: Instance, gs: GranularSet = None) -> Solution:
    """
    Neighborhood 2-opt sulla rotta truck (B.8).
    Dati due nodi della rotta i e j, inverte il verso di percorrenza 
    dell'intero segmento [i+1, j-1] 
    """
    evaluate_solution(solution, inst)

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_move  = None

        for r_idx, route in enumerate(solution.routes):
            nodes = route.nodes
            n     = len(nodes)
            sorties = route.sorties
            for i in range(0, n - 2):         
                for j in range(i + 2, n):  

                    
                    if inst.dist_T[nodes[i],nodes[i+1]] + inst.dist_T[nodes[j-1],nodes[j]] <= 0.5 * (inst.dist_T[nodes[i],nodes[j-1]] + inst.dist_T[nodes[i+1],nodes[j]]):
                        continue
                    arcs, modes = [(nodes[i], nodes[j-1]), (nodes[i+1], nodes[j])], ["truck", "truck"]
                    if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue
                    sol_copy = solution.copy()
                    route_copy = sol_copy.routes[r_idx]
                    route_copy.nodes[i+1:j] = reversed(route_copy.nodes[i+1:j])
                    route_copy.labels[i+1:j] = reversed(route_copy.labels[i+1:j])
                    for sortie in route_copy.sorties:
                        launch_idx = nodes.index(sortie.launch)
                        land_idx   = nodes.index(sortie.land)
                        if i < launch_idx and land_idx < j:
                            sortie.launch, sortie.land = sortie.land, sortie.launch
                    new_cost = evaluate_solution(sol_copy, inst)
                    if new_cost - solution.cost < best_delta:
                        best_delta = new_cost - solution.cost
                        best_move  = (r_idx, i, j)
            
        if best_move is not None:
            improved = True
            r_idx, i, j = best_move
            route = solution.routes[r_idx]
            # Applica l'inversione sulla rotta reale
            for sortie in route.sorties:
                launch_idx = route.nodes.index(sortie.launch)
                land_idx   = route.nodes.index(sortie.land)
                if i < launch_idx and land_idx < j:
                    sortie.launch, sortie.land = sortie.land, sortie.launch     
            
            route.nodes[i+1:j] = reversed(route.nodes[i+1:j])
            route.labels[i+1:j] = reversed(route.labels[i+1:j])
            evaluate_solution(solution, inst)   
    return solution

def exchange_11(solution: Solution, inst: Instance, gs: GranularSet = None) -> Solution:
    evaluate_solution(solution, inst)

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_move  = None

        # Raccogli tutti i clienti per tipo, da tutte le rotte
        for r_idx, route in enumerate(solution.routes):
            nodes = route.nodes
            truck_nodes  = [(pos, node) for pos, (node, lbl)
                            in enumerate(zip(nodes, route.labels))
                            if lbl == 'T' and 1 <= node <= inst.n_customers]
            drone_nodes  = [(s_idx, s.customer) for s_idx, s in enumerate(route.sorties)]
            robot_nodes  = [(t_idx, p, c) for t_idx, t in enumerate(route.robots)
                            for p, c in enumerate(t.customers)]

            # ── CASO T ↔ D ──────────────────────────────────────────
            for (pos_t, t_node), (s_idx, d_node) in product(truck_nodes, drone_nodes):
                if t_node in inst.no_fly: continue
                sortie = route.sorties[s_idx]
                if not inst.enough_battery_drone(sortie.launch, t_node, sortie.land): continue
                new_arcs = [(nodes[pos_t-1], d_node), (d_node, nodes[pos_t+1]), (sortie.launch, t_node, sortie.land)]
                modes = ["truck", "truck", "drone"]
                if gs is not None and not at_least_one_granular_move(new_arcs, gs, modes): continue

                sol_copy = solution.copy()
                route_copy = sol_copy.routes[r_idx]
                sortie_copy = route_copy.sorties[s_idx]
                sortie_copy.customer = t_node
                route_copy.nodes[pos_t] = d_node
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move=("TD", r_idx, s_idx, pos_t)

            # ── CASO T ↔ R ──────────────────────────────────────────
            for (pos_t, t_node), (t_idx, p, r_node) in product(truck_nodes, robot_nodes):
                trip = route.robots[t_idx]
                if not inst.enough_range_robot(trip.station, t_node): continue
                arcs = [(nodes[pos_t-1],r_node),(r_node, nodes[pos_t+1]),(trip.station,t_node)]
                modes = ["truck", "truck", "robot"]
                if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue
                sol_copy = solution.copy()
                route_copy = sol_copy.routes[r_idx]
                trip_copy = route_copy.robots[t_idx]
                route_copy.nodes[pos_t] = r_node
                trip_copy.customers[p] = t_node
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("TR", r_idx, t_idx, pos_t, p)

            # ── CASO D ↔ R ──────────────────────────────────────────
            for (s_idx, d_node), (t_idx, p, r_node) in product(drone_nodes, robot_nodes):
                if r_node in inst.no_fly: continue
                sortie = route.sorties[s_idx]
                trip = route.robots[t_idx]
                if not inst.enough_battery_drone(sortie.launch,r_node,sortie.land): continue
                if not inst.enough_range_robot(trip.station,d_node): continue
                arcs = [(sortie.launch, r_node, sortie.land), (trip.station, d_node)]
                modes = ["drone", "robot"]
                if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue

                sol_copy = solution.copy()
                route_copy = sol_copy.routes[r_idx]
                sortie_copy = route_copy.sorties[s_idx]
                trip_copy = route_copy.robots[t_idx]
                sortie_copy.customer = r_node
                trip_copy.customers[p] = d_node
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("DR", r_idx, s_idx, t_idx, p)

            # ── CASO D ↔ D ──────────────────────────────────────────
            for (s1_idx, d1_node), (s2_idx, d2_node) in combinations(drone_nodes, 2):
                sortie1, sortie2 = route.sorties[s1_idx], route.sorties[s2_idx]
                s1_launch, s1_land = sortie1.launch, sortie1.land
                s2_launch, s2_land = sortie2.launch, sortie2.land
                if not inst.enough_battery_drone(s1_launch, d2_node, s1_land): continue
                if not inst.enough_battery_drone(s2_launch, d1_node, s2_land): continue
                arcs = [(s1_launch, d2_node, s1_land), (s2_launch, d1_node, s2_land)]
                modes = ["drone", "drone"]
                if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue

                sol_copy = solution.copy()
                route_copy = sol_copy.routes[r_idx]
                s1_copy, s2_copy = route_copy.sorties[s1_idx], route_copy.sorties[s2_idx]
                s1_copy.customer, s2_copy.customer = d2_node, d1_node
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost-solution.cost
                best_move = ("DD", r_idx, s1_idx, s2_idx)

            # ── CASO R ↔ R ──────────────────────────────────────────
            for (t1_idx, p1, r1_node), (t2_idx, p2, r2_node) in combinations(robot_nodes, 2):
                trip1, trip2 = route.robots[t1_idx], route.robots[t2_idx]
                if not inst.enough_range_robot(trip1.station,r2_node): continue
                if not inst.enough_range_robot(trip2.station,r1_node): continue
                arcs = [(trip1.station, r2_node), (trip2.station, r1_node)]
                modes = ["robot", "robot"]
                if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue
                sol_copy = solution.copy()
                route_copy = sol_copy.routes[r_idx]
                trip1_copy, trip2_copy = route_copy.robots[t1_idx], route_copy.robots[t2_idx]
                trip1_copy.customers[p1], trip2_copy.customers[p2] = r2_node, r1_node
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("RR", r_idx, t1_idx, t2_idx, p1, p2)

            # ── CASO T ↔ T stessa rotta ─────────────────────────────
            for (pos1, t1_node), (pos2, t2_node) in combinations(truck_nodes, 2):
                if abs(pos1 - pos2) <= 1: continue
                arcs = [(nodes[pos1-1], t2_node), (t2_node, nodes[pos1+1]), (nodes[pos2-1], t1_node), (t1_node, nodes[pos2+1])]
                modes = ["truck", "truck", "truck", "truck"]
                if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue
                sol_copy = solution.copy()
                route_copy = sol_copy.routes[r_idx]
                route_copy.nodes[pos1], route_copy.nodes[pos2] = t2_node, t1_node
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("TT_same", r_idx, pos1, pos2)

        # ── CASO T ↔ T rotte diverse ────────────────────────────────
        for r1_idx, r2_idx in combinations(range(len(solution.routes)), 2):
            route1 = solution.routes[r1_idx]
            route2 = solution.routes[r2_idx]
            truck_nodes_1 = [(pos, node) for pos, (node, lbl)
                            in enumerate(zip(route1.nodes, route1.labels))
                            if lbl == 'T' and 1 <= node <= inst.n_customers]
            truck_nodes_2 = [(pos, node) for pos, (node, lbl)
                            in enumerate(zip(route2.nodes, route2.labels))
                            if lbl == 'T' and 1 <= node <= inst.n_customers]
            for (pos1, node1), (pos2, node2) in product(truck_nodes_1,truck_nodes_2):
                prev1, next1 = route1.nodes[pos1-1], route1.nodes[pos1+1]
                prev2, next2 = route2.nodes[pos2-1], route2.nodes[pos2+1]
                arcs = [(prev1, node2), (node2, next1), (prev2, node1), (node1, next2)]
                modes = ["truck", "truck", "truck", "truck"]
                if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue
                sol_copy = solution.copy()
                route1_copy, route2_copy = sol_copy.routes[r1_idx], sol_copy.routes[r2_idx]
                route1_copy.nodes[pos1], route2_copy.nodes[pos2] = node2, node1
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("TT_different", r1_idx, r2_idx, pos1, pos2)
            

        if best_move is not None:
            improved = True
            move_type = best_move[0]
            if move_type == "TT_different":
                r1_idx, r2_idx, pos1, pos2 = best_move[1:]
                route1, route2 = solution.routes[r1_idx], solution.routes[r2_idx]
                route1.nodes[pos1], route2.nodes[pos2] = route2.nodes[pos2], route1.nodes[pos1]
            elif move_type == "TT_same":
                r_idx, pos1, pos2 = best_move[1:]
                route = solution.routes[r_idx]
                route.nodes[pos1], route.nodes[pos2] = route.nodes[pos2], route.nodes[pos1]
            elif move_type == "TD":
                r_idx, s_idx, pos_t = best_move[1:]
                route = solution.routes[r_idx]
                sortie = route.sorties[s_idx]
                route.nodes[pos_t], sortie.customer = sortie.customer, route.nodes[pos_t]
            elif move_type == "TR":
                r_idx, t_idx, pos_t, p = best_move[1:]
                route = solution.routes[r_idx]
                trip = route.robots[t_idx]
                route.nodes[pos_t], trip.customers[p] = trip.customers[p], route.nodes[pos_t]
            elif move_type == "DR":
                r_idx, s_idx, t_idx, p = best_move[1:]
                route = solution.routes[r_idx]
                sortie, trip = route.sorties[s_idx], route.robots[t_idx]
                sortie.customer, trip.customers[p] = trip.customers[p], sortie.customer
            elif move_type == "DD":
                r_idx, s1_idx, s2_idx = best_move[1:]
                route = solution.routes[r_idx]
                sortie1, sortie2 = route.sorties[s1_idx], route.sorties[s2_idx]
                sortie1.customer, sortie2.customer = sortie2.customer, sortie1.customer
            elif move_type == "RR":
                r_idx, t1_idx, t2_idx, p1, p2 = best_move[1:]
                route = solution.routes[r_idx]
                trip1, trip2 = route.robots[t1_idx], route.robots[t2_idx]
                trip1.customers[p1], trip2.customers[p2] = trip2.customers[p2], trip1.customers[p1]
            
            evaluate_solution(solution, inst)

    return solution

def exchange_22(solution: Solution, inst: Instance, gs: GranularSet = None) -> Solution:
    """
    Neighborhood Exchange 2.2 (B.8).
    Permuta l'ordine di 4 clienti (2+2) sulla rotta del truck.
    """
    evaluate_solution(solution, inst)
    PERMS = [(1,2,3,0), (1,3,0,2), (2,3,0,1), (2,3,1,0), (3,0,1,2), (3,2,0,1), (3,2,1,0)]
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_move  = None

        for r_idx, route in enumerate(solution.routes):
            nodes = route.nodes
            n = len(nodes)
            if n<6: continue
            for a in range(1, n - 4):        # coppia 1: posizioni a, a+1
                for b in range(a + 2, n - 2): # coppia 2: posizioni b, b+1
                    if route.labels[a] != 'T' or route.labels[a+1] != 'T': continue
                    if route.labels[b] != 'T' or route.labels[b+1] != 'T': continue
                    NODES = nodes[a], nodes[a+1], nodes[b], nodes[b+1]
                    for perm in PERMS:
                        if b == a+2:
                            arcs = [(nodes[a-1], NODES[perm[0]]),
                                    (NODES[perm[0]], NODES[perm[1]]),
                                    (NODES[perm[1]], NODES[perm[2]]),
                                    (NODES[perm[2]], NODES[perm[3]]),
                                    (NODES[perm[3]],nodes[b+2])]
                            moves = ["truck"]*5
                        else:
                            arcs = [(nodes[a-1], NODES[perm[0]]),
                                    (NODES[perm[0]], NODES[perm[1]]),
                                    (NODES[perm[1]], nodes[a+2]),
                                    (nodes[b-1], NODES[perm[2]]),
                                    (NODES[perm[2]], NODES[perm[3]]),
                                    (NODES[perm[3]], nodes[b+2])]
                            moves = ["truck"]*6
                        if gs is not None and not at_least_one_granular_move(arcs, gs, moves): continue
                        sol_copy = solution.copy()
                        route_copy = sol_copy.routes[r_idx]
                        route_copy.nodes[a], route_copy.nodes[a+1], route_copy.nodes[b], route_copy.nodes[b+1] = NODES[perm[0]], NODES[perm[1]], NODES[perm[2]], NODES[perm[3]]
                        new_cost = evaluate_solution(sol_copy, inst)
                        if new_cost - solution.cost >= best_delta: continue
                        best_delta = new_cost - solution.cost
                        best_move = (r_idx, a, b, perm)
        if best_move is not None:
            improved = True
            r_idx, a, b, perm = best_move
            route = solution.routes[r_idx]
            NODES = (route.nodes[a], route.nodes[a+1], route.nodes[b], route.nodes[b+1])
            route.nodes[a], route.nodes[a+1], route.nodes[b], route.nodes[b+1] = NODES[perm[0]], NODES[perm[1]], NODES[perm[2]], NODES[perm[3]]
            evaluate_solution(solution, inst)

    return solution

def _remove_sortie(route, sortie_idx):
    '''Rimuove la sortie di indice sortie_idx dalla rotta route in-place'''

    launch, land = route.sorties[sortie_idx].launch, route.sorties[sortie_idx].land
    route.sorties.pop(sortie_idx)
    nodes_C = {s.launch for s in route.sorties} | {s.land for s in route.sorties}

    if launch not in nodes_C: 
        launch_idx = route.nodes.index(launch)
        route.labels[launch_idx] = 'T'
    if land not in nodes_C: 
        land_idx = route.nodes.index(land)
        route.labels[land_idx] = 'T'

def _remove_robot_customer(route, robot_idx, cust_idx):
    '''Rimuove il cliente con indice cust_idx dai customers del robot
    in indice robot_idx della rotta route.
    Se il cliente è l'unico servito dal robot, elimina Robot da route.robots.
    Se il cliente è l'unico servito dalla stazione, toglie la stazione dai nodi della route.'''
    robot = route.robots[robot_idx]
    robot.customers.pop(cust_idx)
    if robot.customers == []:
        route.robots.pop(robot_idx)
        robots_same_station = route.robots_by_station(robot.station)
        customers_same_station = [cust for (_,r) in robots_same_station for cust in r.customers]
        if customers_same_station == []:
            station_idx = route.nodes.index(robot.station)
            route.nodes.pop(station_idx)
            route.labels.pop(station_idx)

def _possible_launch_land_pairs(route, inst: Instance):
    '''Ritorna tuple (launch_idx, land_idx, sortie_idx) dei possibili
    nodi di launch e land della rotta in cui si può inserire
    una sortie, avendo cura di non creare sorties accavallate.'''

    n = len(route.nodes)
    # Posizioni (L_p, R_p) ordinate (launch e land di sortie già presenti)
    positions = sorted(
        (route.nodes.index(s.launch), route.nodes.index(s.land))
        for s in route.sorties
    )
    k = len(positions)
    
    pairs = []  # tuple (launch_idx, land_idx, sortie_idx)
    prev_R = 0
    for p in range(k + 1):
        next_L = positions[p][0] if p < k else n - 1
        i_upper = next_L if p == k else next_L - 1  
        for i in range(prev_R, i_upper + 1):
            for j in range(i, next_L + 1):
                node_i, node_j = route.nodes[i], route.nodes[j]
                if node_i in inst.no_fly or node_j in inst.no_fly: continue
                pairs.append((i, j, p))
        if p < k:
            prev_R = positions[p][1]
    return pairs

def reinsertion(solution: Solution, inst: Instance, gs: GranularSet = None) -> float:
    '''Seleziona un cliente e lo inserisce in un'altra modalità di consegna'''

    evaluate_solution(solution, inst)
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_move = None 
        
        # ── destinazioni: comuni a tutti i casi ──────────────────
        dest_truck = [(rj, pos_to)
              for rj, r in enumerate(solution.routes)
              for pos_to in range(1, len(r.nodes))]

        dest_drone = [(rj, solution.routes[rj].nodes[launch_idx], solution.routes[rj].nodes[land_idx], sortie_idx)
                    for rj, r in enumerate(solution.routes)
                    for launch_idx, land_idx, sortie_idx in _possible_launch_land_pairs(r, inst)
                    ]

        dest_robot = [(rj, t_idx)
                    for rj, r in enumerate(solution.routes)
                    for t_idx, _ in enumerate(r.robots)]
        
        stations_active = {trip.station for r in solution.routes for trip in r.robots}
        stations_empty  = [s for s in range(inst.n_customers + 1, 
                                            inst.n_customers + inst.n_stations + 1)
                        if s not in stations_active]


        # ── caso D→T ─────────────────────────────────────────────
        for ri, s_idx, customer in [(ri, s_idx, s.customer) for ri, r in enumerate(solution.routes) for s_idx, s in enumerate(r.sorties)]:
            for rj, pos_to in dest_truck:
                arcs = [(solution.routes[rj].nodes[pos_to-1], customer), (customer, solution.routes[rj].nodes[pos_to])]
                moves = ["truck", "truck"]
                if gs is not None and not at_least_one_granular_move(arcs, gs, moves): continue
                sol_copy = solution.copy()
                _remove_sortie(sol_copy.routes[ri], s_idx)
                sol_copy.routes[rj].nodes.insert(pos_to, customer)
                sol_copy.routes[rj].labels.insert(pos_to, 'T')
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("DT", ri, rj, s_idx, pos_to)

        # ── caso R→T ─────────────────────────────────────────────
        for ri, t_idx, c_idx, customer in [(ri, t_idx, c_idx, c) for ri, r in enumerate(solution.routes) for t_idx, t in enumerate(r.robots) for c_idx, c in enumerate(t.customers)]:
            for rj, pos_to in dest_truck:
                arcs = [(solution.routes[rj].nodes[pos_to-1], customer), (customer, solution.routes[rj].nodes[pos_to])]
                moves = ["truck", "truck"]
                if gs is not None and not at_least_one_granular_move(arcs, gs, moves): continue
                route1 = solution.routes[ri]
                station_node = route1.robots[t_idx].station
                if route1.is_only_one_customer_served_by_station(station_node, inst):
                    if [s for s in route1.sorties if s.launch == station_node or s.land == station_node]:
                        continue
                sol_copy = solution.copy()
                sol_copy.routes[rj].nodes.insert(pos_to, customer)
                sol_copy.routes[rj].labels.insert(pos_to, 'T')
                _remove_robot_customer(sol_copy.routes[ri], t_idx, c_idx)
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("RT", ri, rj, t_idx, c_idx, pos_to)
        
        # ── caso T→D ────────────────────────────────────────────
        for ri, pos_from, customer in [(ri, pos_from, n) for ri, r in enumerate(solution.routes) for pos_from, (n, lbl) in enumerate(zip(r.nodes, r.labels)) if lbl == 'T' and 1 <= n <= inst.n_customers]:
            for rj, launch, land, sortie_idx in dest_drone:
                if customer in inst.no_fly: continue
                if not inst.enough_battery_drone(launch, customer, land): continue
                if gs is not None and not is_granular_move((launch, customer, land), gs, "drone"): continue
                route1 = solution.routes[ri]
                if customer == launch or customer == land: continue
                sol_copy = solution.copy()
                sol_copy.routes[ri].nodes.pop(pos_from)
                sol_copy.routes[ri].labels.pop(pos_from)
                sol_copy.routes[rj].sorties.insert(sortie_idx, Sortie(launch=launch, customer=customer, land=land))
                sol_copy.routes[rj].labels[sol_copy.routes[rj].nodes.index(launch)] = 'C'
                sol_copy.routes[rj].labels[sol_copy.routes[rj].nodes.index(land)] = 'C'
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("TD", ri, rj, pos_from, sortie_idx, launch, land)

        # ── caso T→R ────────────────────────────────────────────
        
        for ri, pos_from, customer in [(ri, pos_from, n) for ri, r in enumerate(solution.routes) for pos_from, (n, lbl) in enumerate(zip(r.nodes, r.labels)) if lbl == 'T' and 1 <= n <= inst.n_customers]:
            for rj, t_idx in dest_robot:
                trip = solution.routes[rj].robots[t_idx]
                station = trip.station
                if not inst.enough_range_robot(station, customer): continue
                if gs is not None and not is_granular_move((station, customer), gs, "robot"): continue
                
                robots_at_station = [r for r in solution.routes[rj].robots if r.station == station]
                
                sol_copy = solution.copy()
                sol_copy.routes[ri].nodes.pop(pos_from)
                sol_copy.routes[ri].labels.pop(pos_from)
                
                if len(robots_at_station) < MAX_ROBOTS_R:
                    # slot libero → nuovo robot
                    sol_copy.routes[rj].robots.append(Robot(station=station, customers=[customer]))
                    best_move_candidate = ("TR_new_robot", ri, rj, pos_from, station)
                else:
                    # stazione satura → robot con meno clienti
                    min_robot_idx = min(
                        [i for i, r in enumerate(sol_copy.routes[rj].robots) if r.station == station],
                        key=lambda i: len(sol_copy.routes[rj].robots[i].customers)
                    )
                    sol_copy.routes[rj].robots[min_robot_idx].customers.append(customer)
                    best_move_candidate = ("TR", ri, rj, pos_from, min_robot_idx)
                
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = best_move_candidate

            # Prova a inserire il nodo T in una delle stazioni robot vuote (senza clienti)
            # Prova su ogni rotta ad inserire la stazione, prendendo i K_TOP nodi più vicini al nodo stazione da inserire
            for rj, r in enumerate(solution.routes):
                
                for station_node in stations_empty:                
                    # calcola delta_dist per ogni pos_to in questa rotta
                    candidates = []
                    for pos_to in range(1, len(r.nodes)):
                        prev_node = r.nodes[pos_to - 1]
                        next_node = r.nodes[pos_to]
                        delta_dist = (inst.dist_T[prev_node, station_node]
                                    + inst.dist_T[station_node, next_node]
                                    - inst.dist_T[prev_node, next_node])
                        candidates.append((delta_dist, pos_to))
                    
                    candidates.sort()
                    for _, pos_to in candidates[:K_TOP]:
                        if not inst.enough_range_robot(station_node, customer): continue
                        arcs = [(r.nodes[pos_to-1], station_node), (station_node, r.nodes[pos_to]), (station_node, customer)]
                        modes = ["truck", "truck", "robot"]
                        if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue
                        actual_from = pos_from + 1 if (ri == rj and pos_from >= pos_to) else pos_from
                        sol_copy = solution.copy()
                        sol_copy.routes[rj].nodes.insert(pos_to, station_node)
                        sol_copy.routes[rj].labels.insert(pos_to, 'T')
                        sol_copy.routes[ri].nodes.pop(actual_from)
                        sol_copy.routes[ri].labels.pop(actual_from)
                        sol_copy.routes[rj].robots.append(Robot(station=station_node, customers=[customer]))
                        new_cost = evaluate_solution(sol_copy, inst)
                        if new_cost - solution.cost >= best_delta: continue
                        best_delta = new_cost - solution.cost
                        best_move = ("TR_empty", ri, rj, pos_from, station_node, pos_to)

        # ── caso D→R ────────────────────────────────────────────
        for ri, s_idx, customer in [(ri, s_idx, s.customer) for ri, r in enumerate(solution.routes) for s_idx, s in enumerate(r.sorties)]:
            for rj, t_idx in dest_robot:
                trip = solution.routes[rj].robots[t_idx]
                station = trip.station
                if not inst.enough_range_robot(station, customer): continue
                if gs is not None and not is_granular_move((station, customer), gs, "robot"): continue
                
                robots_at_station = [r for r in solution.routes[rj].robots if r.station == station]
                
                sol_copy = solution.copy()
                _remove_sortie(sol_copy.routes[ri], s_idx)
                
                if len(robots_at_station) < MAX_ROBOTS_R:
                    sol_copy.routes[rj].robots.append(Robot(station=station, customers=[customer]))
                    best_move_candidate = ("DR_new_robot", ri, rj, s_idx, station)
                else:
                    min_robot_idx = min(
                        [i for i, r in enumerate(sol_copy.routes[rj].robots) if r.station == station],
                        key=lambda i: len(sol_copy.routes[rj].robots[i].customers)
                    )
                    sol_copy.routes[rj].robots[min_robot_idx].customers.append(customer)
                    best_move_candidate = ("DR", ri, rj, s_idx, min_robot_idx)
                
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = best_move_candidate
            
            for rj, r in enumerate(solution.routes):
                for station_node in stations_empty:                
                    # calcola delta_dist per ogni pos_to in questa rotta
                    candidates = []
                    for pos_to in range(1, len(r.nodes)):
                        prev_node = r.nodes[pos_to - 1]
                        next_node = r.nodes[pos_to]
                        delta_dist = (inst.dist_T[prev_node, station_node]
                                    + inst.dist_T[station_node, next_node]
                                    - inst.dist_T[prev_node, next_node])
                        candidates.append((delta_dist, pos_to))
                    
                    candidates.sort()
                    for _, pos_to in candidates[:K_TOP]:
                        if not inst.enough_range_robot(station_node, customer): continue
                        arcs = [(r.nodes[pos_to-1], station_node), (station_node, r.nodes[pos_to]), (station_node, customer)]
                        modes = ["truck", "truck", "robot"]
                        if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue
                        sol_copy = solution.copy()
                        _remove_sortie(sol_copy.routes[ri], s_idx)
                        sol_copy.routes[rj].nodes.insert(pos_to, station_node)
                        sol_copy.routes[rj].labels.insert(pos_to, 'T')
                        sol_copy.routes[rj].robots.append(Robot(station=station_node, customers=[customer]))
                        new_cost = evaluate_solution(sol_copy, inst)
                        if new_cost - solution.cost >= best_delta: continue
                        best_delta = new_cost - solution.cost
                        best_move = ("DR_empty", ri, rj, s_idx, station_node, pos_to)

        #── caso R→D ────────────────────────────────────────────                
        for ri, t_idx, c_idx, customer in [(ri, t_idx, c_idx, c) for ri, r in enumerate(solution.routes) for t_idx, t in enumerate(r.robots) for c_idx, c in enumerate(t.customers)]:
            for rj, launch, land, sortie_idx in dest_drone:
                if customer in inst.no_fly: continue
                if not inst.enough_battery_drone(launch, customer, land): continue
                if gs is not None and not is_granular_move((launch, customer, land), gs, "drone"): continue
                route1 = solution.routes[ri]
                station_node = route1.robots[t_idx].station
                if route1.is_only_one_customer_served_by_station(station_node, inst):
                    # se è l'unico cliente del robot trip, allora non posso rimuoverlo senza rimuovere anche la stazione
                    if [s for s in route1.sorties if s.launch == station_node or s.land == station_node]: 
                        continue
                    if launch == station_node or land == station_node: continue
                sol_copy = solution.copy()
                _remove_robot_customer(sol_copy.routes[ri], t_idx, c_idx)
                sol_copy.routes[rj].sorties.insert(sortie_idx, Sortie(launch=launch, customer=customer, land=land))
                sol_copy.routes[rj].labels[sol_copy.routes[rj].nodes.index(launch)] = 'C'
                sol_copy.routes[rj].labels[sol_copy.routes[rj].nodes.index(land)] = 'C'
                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = ("RD", ri, rj, t_idx, c_idx, sortie_idx, launch, land)


        if best_move is not None:
            improved = True
            _apply_reinsertion(solution, best_move, inst)
            evaluate_solution(solution, inst)
    
    return evaluate_solution(solution, inst)

def _apply_reinsertion(solution, move, inst):
    '''Applica la best move di reinsertion'''
    move_type = move[0]
    if move_type == "DT":
        _, ri, rj, s_idx, pos_to = move
        cust = solution.routes[ri].sorties[s_idx].customer
        _remove_sortie(solution.routes[ri], s_idx)
        solution.routes[rj].nodes.insert(pos_to, cust)
        solution.routes[rj].labels.insert(pos_to, 'T')
    elif move_type == "RT":
        _, ri, rj, t_idx, c_idx, pos_to = move
        cust = solution.routes[ri].robots[t_idx].customers[c_idx]
        solution.routes[rj].nodes.insert(pos_to, cust)
        solution.routes[rj].labels.insert(pos_to, 'T')
        _remove_robot_customer(solution.routes[ri], t_idx, c_idx)
    elif move_type == "TD":
        _, ri, rj, pos_from, sortie_idx, launch, land = move
        cust = solution.routes[ri].nodes.pop(pos_from)
        solution.routes[ri].labels.pop(pos_from)
        solution.routes[rj].sorties.insert(sortie_idx, Sortie(launch=launch, customer=cust, land=land))
        solution.routes[rj].labels[solution.routes[rj].nodes.index(launch)] = 'C'
        solution.routes[rj].labels[solution.routes[rj].nodes.index(land)] = 'C'
    elif move_type == "TR":
        _, ri, rj, pos_from, t_idx = move
        cust = solution.routes[ri].nodes.pop(pos_from)
        solution.routes[ri].labels.pop(pos_from)
        solution.routes[rj].robots[t_idx].customers.append(cust)
    elif move_type == "TR_new_robot":
        _, ri, rj, pos_from, station = move
        cust = solution.routes[ri].nodes.pop(pos_from)
        solution.routes[ri].labels.pop(pos_from)
        solution.routes[rj].robots.append(Robot(station=station, customers=[cust]))
    elif move_type == "DR":
        _, ri, rj, s_idx, t_idx = move
        cust = solution.routes[ri].sorties[s_idx].customer
        trip = solution.routes[rj].robots[t_idx]
        _remove_sortie(solution.routes[ri], s_idx)
        trip.customers.append(cust)
    elif move_type == "DR_new_robot":
        _, ri, rj, s_idx, station = move
        cust = solution.routes[ri].sorties[s_idx].customer
        _remove_sortie(solution.routes[ri], s_idx)
        solution.routes[rj].robots.append(Robot(station=station, customers=[cust]))
    elif move_type == "RD":
        _, ri, rj, t_idx, c_idx, sortie_idx, launch, land = move
        cust = solution.routes[ri].robots[t_idx].customers[c_idx]
        _remove_robot_customer(solution.routes[ri], t_idx, c_idx)
        solution.routes[rj].sorties.insert(sortie_idx, Sortie(launch=launch, customer=cust, land=land))
        solution.routes[rj].labels[solution.routes[rj].nodes.index(launch)] = 'C'
        solution.routes[rj].labels[solution.routes[rj].nodes.index(land)] = 'C'
    elif move_type == "DR_empty":
        _, ri, rj, s_idx, station_node, pos_to = move
        cust = solution.routes[ri].sorties[s_idx].customer
        _remove_sortie(solution.routes[ri], s_idx)
        solution.routes[rj].nodes.insert(pos_to, station_node)
        solution.routes[rj].labels.insert(pos_to, 'T')
        solution.routes[rj].robots.append(Robot(station=station_node, customers=[cust]))
    elif move_type == "TR_empty":
        _, ri, rj, pos_from, station_node, pos_to = move
        if ri==rj and pos_from >= pos_to: pos_from += 1 
        solution.routes[rj].nodes.insert(pos_to, station_node)
        solution.routes[rj].labels.insert(pos_to, 'T')
        cust = solution.routes[ri].nodes.pop(pos_from)
        solution.routes[ri].labels.pop(pos_from)
        solution.routes[rj].robots.append(Robot(station=station_node, customers=[cust]))

def exchange_drone_robot(solution: Solution, inst: Instance, gs: GranularSet = None) -> float:
    '''Scambia tutti i clienti serviti da round-trips (launch=land) da un nodo,
    con tutti i clienti robot serviti da una station'''
    evaluate_solution(solution, inst)
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_move = None
        
        # ── raccogli candidati ────────────────────────────────────
        roundtrip_sorties = {}
        for r_idx, route in enumerate(solution.routes):
            for s_idx, s in enumerate(route.sorties):
                if s.launch == s.land:
                    key = (r_idx, s.launch)
                    if key not in roundtrip_sorties:
                        roundtrip_sorties[key] = []
                    roundtrip_sorties[key].append(s_idx)
        robots = [(r_idx, t_idx, trip) for r_idx, route in enumerate(solution.routes) for t_idx, trip in enumerate(route.robots)]
        
        for (ri, combined_node), s_idxs in roundtrip_sorties.items():
            for rj, t_idx, trip in robots:
                if ri == rj and combined_node == trip.station: continue  # guard

                drone_candidates = [s.customer for s_idx in s_idxs 
                    for s in [solution.routes[ri].sorties[s_idx]]]
                robot_candidates = list(trip.customers)

                if not all(c not in inst.no_fly for c in robot_candidates): continue
                if not all(inst.enough_battery_drone(combined_node, c, combined_node) 
                        for c in robot_candidates): continue
                if not all(inst.enough_range_robot(trip.station, c) 
                        for c in drone_candidates): continue
                arcs = [(trip.station, customer) for customer in drone_candidates]
                arcs += [(combined_node, customer, combined_node) for customer in robot_candidates]
                modes = ["robot"]*len(drone_candidates) + ["drone"]*len(robot_candidates)
                if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue
            
                sol_copy = solution.copy()
                route_copy_i = sol_copy.routes[ri]
                route_copy_j = sol_copy.routes[rj]

                # drone → robot
                for s_idx in sorted(s_idxs, reverse=True):
                    route_copy_j.robots[t_idx].customers.append(route_copy_i.sorties[s_idx].customer)
                    _remove_sortie(route_copy_i, s_idx)

                # robot → drone
                for customer in robot_candidates:
                    route_copy_i.sorties.append(Sortie(launch=combined_node, customer=customer, land=combined_node))
                    route_copy_i.labels[route_copy_i.nodes.index(combined_node)] = 'C'

                # rimuovi clienti dal robot (in reverse)
                for c_idx in range(len(robot_candidates) - 1, -1, -1):
                    _remove_robot_customer(route_copy_j, t_idx, c_idx)

                new_cost = evaluate_solution(sol_copy, inst)
                if new_cost - solution.cost >= best_delta: continue
                best_delta = new_cost - solution.cost
                best_move = (ri, combined_node, rj, t_idx, s_idxs, drone_candidates)
        
        if best_move is not None:
            improved = True
            ri, combined_node, rj, t_idx, s_idxs, drone_candidates = best_move
            route_i, route_j = solution.routes[ri], solution.routes[rj]
            robot_candidates = list(route_j.robots[t_idx].customers)

            # drone → robot
            for s_idx in sorted(s_idxs, reverse=True):
                route_j.robots[t_idx].customers.append(route_i.sorties[s_idx].customer)
                _remove_sortie(route_i, s_idx)

            # robot → drone
            for customer in robot_candidates:
                route_i.sorties.append(Sortie(launch=combined_node, customer=customer, land=combined_node))
                route_i.labels[route_i.nodes.index(combined_node)] = 'C'

            # rimuovi clienti dal robot (in reverse)
            for c_idx in range(len(robot_candidates) - 1, -1, -1):
                _remove_robot_customer(route_j, t_idx, c_idx)

            evaluate_solution(solution, inst)
    
    return evaluate_solution(solution, inst)

def exchange_robot_stations(solution: Solution, inst: Instance, gs: GranularSet = None) -> float:
    '''Prende due stazioni e le scambia di rotta'''
    evaluate_solution(solution, inst)
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_move = None  # (ri, station_A, rj, station_B)
        
        # ── raccogli tutte le coppie di stazioni attive ───────────
        all_stations = [
            (r_idx, trip.station)
            for r_idx, route in enumerate(solution.routes)
            for trip in route.robots
        ]
        # deduplica: una stazione può comparire più volte (più robot)
        all_stations = list({(r_idx, station) for r_idx, station in all_stations})
        
        for (ri, station_A), (rj, station_B) in combinations(all_stations, 2):
            if ri == rj: continue  

            nodes_i, nodes_j = solution.routes[ri].nodes, solution.routes[rj].nodes
            
            if station_A not in nodes_i or station_B not in nodes_j: continue
            idx_A, idx_B = nodes_i.index(station_A), nodes_j.index(station_B)
            
            if [s for s in solution.routes[ri].sorties if s.launch == station_A or s.land == station_A]: continue
            if [s for s in solution.routes[rj].sorties if s.launch == station_B or s.land == station_B]: continue

            arcs = [(nodes_i[idx_A-1], station_B), (station_B, nodes_i[idx_A+1]),
                    (nodes_j[idx_B-1], station_A), (station_A, nodes_j[idx_B+1])]
            modes = ["truck", "truck", "truck", "truck"]
            if gs is not None and not at_least_one_granular_move(arcs, gs, modes): continue

            sol_copy = solution.copy()
            robots_A = [r for r in sol_copy.routes[ri].robots if r.station == station_A]
            robots_B = [r for r in sol_copy.routes[rj].robots if r.station == station_B]
            
            for r in robots_A: r.station = station_B
            for r in robots_B: r.station = station_A
            
            sol_copy.routes[ri].robots = [r for r in sol_copy.routes[ri].robots if r.station != station_B] + robots_B
            sol_copy.routes[rj].robots = [r for r in sol_copy.routes[rj].robots if r.station != station_A] + robots_A

            idx_A_copy = sol_copy.routes[ri].nodes.index(station_A)
            idx_B_copy = sol_copy.routes[rj].nodes.index(station_B)
            sol_copy.routes[ri].nodes[idx_A_copy] = station_B
            sol_copy.routes[rj].nodes[idx_B_copy] = station_A

            new_cost = evaluate_solution(sol_copy, inst)
            if new_cost - solution.cost >= best_delta: continue
            best_delta = new_cost - solution.cost
            best_move = (ri, station_A, rj, station_B)
        
        if best_move is not None:
            improved = True
            ri, station_A, rj, station_B = best_move

            robots_A = [r for r in solution.routes[ri].robots if r.station == station_A]
            robots_B = [r for r in solution.routes[rj].robots if r.station == station_B]

            for r in robots_A: r.station = station_B
            for r in robots_B: r.station = station_A

            solution.routes[ri].robots = [r for r in solution.routes[ri].robots if r.station != station_B] + robots_B
            solution.routes[rj].robots = [r for r in solution.routes[rj].robots if r.station != station_A] + robots_A

            idx_A = solution.routes[ri].nodes.index(station_A)
            idx_B = solution.routes[rj].nodes.index(station_B)
            solution.routes[ri].nodes[idx_A] = station_B
            solution.routes[rj].nodes[idx_B] = station_A

            evaluate_solution(solution, inst)
    
    return evaluate_solution(solution, inst)
