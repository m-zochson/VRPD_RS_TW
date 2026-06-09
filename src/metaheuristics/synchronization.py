#src.metaheuristics.syncronization.py
from src.instance import Instance
from src.solution import TruckRoute, Sortie, Robot
from src.config import *

def compute_truck_times(route: TruckRoute, inst: Instance) -> tuple:
    """
    Calcola a_T, s_T, d_T per ogni nodo della rotta del truck.
    
    Input:
      route  : TruckRoute 
      inst   : Instance per time_T, t_start, t_stop, n_customers
    
    Output:
      a_T, s_T, d_T : liste di float, una voce per ogni nodo della rotta
    
    Note:
      - d_T[i] dipende anche dal drone: viene sincronizzato in 
      un'altra funzione
    """
    nodes = route.nodes
    n = len(nodes)
    
    a_T = [0.0] * n
    s_T = [0.0] * n
    d_T = [0.0] * n
    
    # Deposito iniziale: il truck parte a tempo 0
    a_T[0] = 0
    s_T[0] = 0
    d_T[0] = 0
    
    for idx in range(1, n):
        node = nodes[idx]
        prev = nodes[idx - 1]
        
        # Eq. 47: calcola a_T[idx]
        a_T[idx] = d_T[idx - 1] + inst.time_T[prev, node]

        # Eq. 48-49: calcola s_T[idx] in base alla finestra temporale
        
        if 1 <= node <= inst.n_customers:  # è un cliente
            t_start_node = inst.t_start[node - 1] 
            if a_T[idx] < t_start_node:
                s_T[idx] = t_start_node
            else:
                s_T[idx] = a_T[idx]
        else:
            # Se non è un cliente (es. stazione o deposito), servi subito
            s_T[idx] = a_T[idx]
        
        # Eq. 50: calcola d_T[idx]
        if idx < n - 1:  # se non è l'ultimo nodo, aggiungi tempo di servizio
            d_T[idx] = s_T[idx] + TRUCK_SERVICE_TIME
        else:       # Se è il depot finale, non serve aggiungere tempo di servizio
            d_T[idx] = s_T[idx]
        
    
    return a_T, s_T, d_T


def compute_drone_departure(
    launch_idx: int,       
    sortie: Sortie,        
    a_T: list,             
    a_D_i: float,          
    inst: Instance,
    has_roundtrips: bool,  
    r_D_last: float,       
    last_rt_customer: int, 
) -> float:
    '''Funzione che calcola la partenza di un drone, in base
    ai tempi del truck e ad eventuali round-trips precedenti presenti sullo
    stesso nodo. '''

    i = sortie.launch
    j = sortie.customer

    # t_start del cliente j
    t_start_j = inst.t_start[j - 1]  

    round_time = DRONE_LOAD_TIME + inst.time_D[i, last_rt_customer] + DRONE_SERVICE_TIME + inst.time_D[last_rt_customer, i] if has_roundtrips else 0
    
    if not has_roundtrips:
        # Senza round-trips dipende solo da quando arriva il truck o il drone dall'ultima sortie
        earliest = max( a_D_i + DRONE_LOAD_TIME, a_T[launch_idx] + DRONE_LOAD_TIME)
    else:
        # Con round-trips 
        earliest = max( r_D_last + round_time + DRONE_LOAD_TIME, a_T[launch_idx] + DRONE_LOAD_TIME)

    
    if earliest + inst.time_D[i, j] >= t_start_j:
        return earliest
    else:
        return t_start_j - inst.time_D[i, j]
    

def compute_robot_release_times(
    station_idx: int,        
    trip: Robot,    
    a_T: list,
    s_T: list,
    inst: Instance,
) -> list:
    """
    Calcola r_R[p] per p = 0, 1, 2, ...
    Restituisce lista di float (uno per ogni cliente in trip.customers).
    """
    release_times = []
    i = trip.station

    for p, customer in enumerate(trip.customers):
        j = customer
        t_start_j = inst.t_start[j - 1]  # mapping corretto

        if p == 0:
            earliest = s_T[station_idx] + TRUCK_SERVICE_TIME
        else:
            round_prev = ROBOT_LOAD_TIME + inst.time_R[i, trip.customers[p-1]]+ inst.time_R[trip.customers[p-1], i] + ROBOT_SERVICE_TIME   # round-trip verso customers[p-1]
            earliest = release_times[p-1] + round_prev

        if earliest + ROBOT_LOAD_TIME + inst.time_R[i, j] >= t_start_j:
            r = earliest
        else:
            r = t_start_j - (ROBOT_LOAD_TIME + inst.time_R[i, j])

        release_times.append(r)

    return release_times

def _propagate_times(nodes, a_T, s_T, d_T, from_idx, to_idx, inst):
    """Ricalcola a_T, s_T, d_T per nodes[from_idx : to_idx],
    questo per propagare l'attesa del drone da parte del truck
    ai nodi successivi della rotta."""

    for k in range(from_idx, to_idx):
        node_update = nodes[k]
        prev_update = nodes[k - 1]
        a_T[k] = d_T[k - 1] + inst.time_T[prev_update, node_update]
        if 1 <= node_update <= inst.n_customers:
            t_start_node = inst.t_start[node_update - 1] 
            if a_T[k] < t_start_node:
                s_T[k] = t_start_node
            else:
                s_T[k] = a_T[k]
            d_T[k] = s_T[k] + TRUCK_SERVICE_TIME
        elif node_update <= inst.n_customers + inst.n_stations:  # stazione
            s_T[k] = a_T[k]
            d_T[k] = s_T[k] + TRUCK_SERVICE_TIME
        else:  # deposito finale
            s_T[k] = a_T[k]
            d_T[k] = s_T[k]

def _check_feasibility(nodes, a_T, s_T, d_T, drone_times, robot_times, route, inst, verbose: bool = False) -> bool:
    ''' Restituisce False se la rotta è infeasible'''

    # --- 1. TRUCK: Time Windows ---
    for idx, node in enumerate(nodes):
        if 1 <= node <= inst.n_customers:
            if s_T[idx] > inst.t_stop[node - 1]:
                if verbose: print(f"Il truck arriva al nodo {node} in ritardo")
                return False

    # --- 2. DRONE: Time Windows + Batteria ---
    for sortie in route.sorties:
        i, j, h = sortie.launch, sortie.customer, sortie.land
        if j not in drone_times:
            if verbose: print(f"Il cliente {j} non è in drone_times")
            return False

        d_D_i = drone_times[j]["d_D"]
        a_D_land = drone_times[j]["a_D_land"]
        arrival_at_customer = d_D_i + inst.time_D[i, j]
        if arrival_at_customer > inst.t_stop[j - 1]:
            if verbose: print(f"La sortie {i}-{j}-{h} arriva al cliente {j} in ritardo")
            return False

        if not inst.enough_battery_drone(i, j, h):
            if verbose: print(f"La sortie {i}-{j}-{h} non ha abbastanza batteria")
            return False
        h_idx = route.nodes.index(h)
        if a_D_land < a_T[h_idx]:
            total_energy = inst.energy_on[i,j]+inst.energy_off[j,h]+(a_T[h_idx]-a_D_land)*DRONE_HOVER_POWER
            if total_energy > DRONE_BATTERY_B:
                if verbose: print(f"La sortie {i}-{j}-{h} non ha abbastanza batteria")
                return False
            
        if i == h:  # round-trip
            launch_idx = nodes.index(i)
            if a_D_land > d_T[launch_idx]:
                if verbose: print(f"La sortie {i}-{j}-{h} arriva dopo che il truck è partito")
                return False

    # --- 3. ROBOT: Time Windows + Range ---
    for robot_idx, trip in enumerate(route.robots):
        if robot_idx not in robot_times:
            if verbose: print(f"Il robot {robot_idx} della stazione {trip.station} non è in robot_times")
            return False

        r_times = robot_times[robot_idx]
        for p, customer in enumerate(trip.customers):
            if not inst.enough_range_robot(trip.station, customer):
                if verbose: print(f"Il robot {robot_idx} della stazione {trip.station} non ha abbastanza batteria")
                return False

            arrival_at_customer = r_times[p] + ROBOT_LOAD_TIME + inst.time_R[trip.station, customer]
            if arrival_at_customer > inst.t_stop[customer - 1]:
                if verbose: print(f"Il robot {robot_idx} della stazione {trip.station} arriva al cliente {customer} in ritardo")
                return False

    return True

def synchronize_route(route: TruckRoute, inst: Instance, verbose: bool = False) -> dict:
    '''Funzione principale: data una rotta, genera i valori delle variabili 
    temporali continue di truck, drone e robot in modo sincronizzato
    ed ottimale.'''
    
    _INFEASIBLE = {"is_feasible": False}
    nodes = route.nodes
    n = len(nodes)

    a_T, s_T, d_T = compute_truck_times(route, inst)

    for idx, node in enumerate(nodes):
        if 1 <= node <= inst.n_customers:
            if a_T[idx] > inst.t_stop[node - 1]:
                if verbose: print(f"Il truck arriva al nodo {node} dopo t_stop")
                return _INFEASIBLE

    sorties_by_launch = {
        launch_node: [s for s in route.sorties if s.launch == launch_node]
        for launch_node in route.nodes
        if launch_node in {s.launch for s in route.sorties}
    }
    robots_by_station = {t.station: [] for t in route.robots}
    for robot_idx, robot in enumerate(route.robots):
        robots_by_station[robot.station].append((robot,robot_idx))

    drone_times = {}   # customer → {"d_D": float, "a_D_land": float}
    robot_times = {}   # robot_idx  → List[float]

    a_D_current = 0.0

    for idx, node in enumerate(nodes):

        # --- DRONE ---
        if node in sorties_by_launch:
            sorties = sorties_by_launch[node]
            round_trips = [s for s in sorties if s.launch == s.land]
            normal      = next((s for s in sorties if s.land != s.launch), None)

            r_D_last, last_rt_customer = 0.0, 0

            for rt in round_trips:
                d_D_rt = compute_drone_departure(
                    launch_idx       = idx,
                    sortie           = rt,
                    a_T              = a_T,
                    a_D_i            = a_D_current,
                    inst             = inst,
                    has_roundtrips   = (r_D_last > 0),
                    r_D_last         = r_D_last,
                    last_rt_customer = last_rt_customer,
                )
                a_D_rt = (d_D_rt
                          + inst.time_D[rt.launch, rt.customer]
                          + DRONE_SERVICE_TIME
                          + inst.time_D[rt.customer, rt.land])
                d_T[idx] = max(d_T[idx], a_D_rt)
                drone_times[rt.customer] = {"d_D": d_D_rt, "a_D_land": a_D_rt}
                r_D_last         = d_D_rt
                last_rt_customer = rt.customer
                a_D_current      = a_D_rt
            
            if round_trips:
                _propagate_times(nodes, a_T, s_T, d_T, idx + 1, n, inst)

            if normal:
                land_idx = nodes.index(normal.land)
                d_D_i = compute_drone_departure(
                    launch_idx       = idx,
                    sortie           = normal,
                    a_T              = a_T,
                    a_D_i            = a_D_current,
                    inst             = inst,
                    has_roundtrips   = len(round_trips) > 0,
                    r_D_last         = r_D_last,
                    last_rt_customer = last_rt_customer,
                )
                a_D_land = (d_D_i
                            + inst.time_D[normal.launch, normal.customer]
                            + DRONE_SERVICE_TIME
                            + inst.time_D[normal.customer, normal.land])
                drone_times[normal.customer] = {"d_D": d_D_i, "a_D_land": a_D_land}

                d_T[idx] = max(d_T[idx], d_D_i)
                _propagate_times(nodes, a_T, s_T, d_T, idx + 1, land_idx + 1, inst)
                d_T[land_idx] = max(s_T[land_idx] + TRUCK_SERVICE_TIME, a_D_land)
                _propagate_times(nodes, a_T, s_T, d_T, land_idx + 1, n, inst)
                a_D_current = a_D_land

        else:
            a_D_current = a_T[idx]

        # --- ROBOT ---
        if node in robots_by_station:
            if len(robots_by_station[node]) > MAX_ROBOTS_R:
                if verbose: print(f"Ci sono troppi robot alla stazione {node}")
                return _INFEASIBLE
            for (robot,robot_idx) in robots_by_station[node]:
                r_times = compute_robot_release_times(
                    station_idx = idx,
                    trip        = robot,
                    a_T         = a_T,
                    s_T         = s_T,
                    inst        = inst,
                )
                robot_times[robot_idx] = r_times

    if not _check_feasibility(nodes, a_T, s_T, d_T, drone_times, robot_times, route, inst, verbose = verbose):
        return _INFEASIBLE

    return {
        "is_feasible":  True,
        "a_T":          a_T,
        "s_T":          s_T,
        "d_T":          d_T,
        "drone_times":  drone_times,
        "robot_times":  robot_times,
    }