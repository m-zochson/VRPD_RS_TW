# src/solution.py
# Architettura del file:
#
# Il file conterrà tre dataclass annidate:
# Solution
# ├── routes: List[TruckRoute]     ← una per ogni truck attivo
# │   ├── nodes: List[int]         ← sequenza nodi visitati dal truck
# │   ├── labels: List[str]        ← etichetta per ogni nodo ('C' per i nodi di launch e land, 'T' tutti gli altri)
# │   ├── sorties: List[Sortie]    ← le sorties del drone
# │   │   ├── launch: int          ← nodo da cui il drone parte
# │   │   ├── customer: int        ← cliente servito dal drone
# │   │   └── land: int            ← nodo dove il drone atterra
# │   └── robots: List[Robot]      ← i robot che ricevono pacchi dal truck
# │       ├── station: int         ← indice della stazione
# │       └── customers: List[int] ← clienti serviti dal robot
# └── cost: float                  ← valore funzione obiettivo


from dataclasses import dataclass, field
from itertools import combinations
from typing import List
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gurobipy as gp
from src.instance import Instance
from src.config import *

@dataclass
class Sortie:
    """
    Rappresenta una singola uscita del drone: parte da 'launch',
    serve 'customer', ritorna a 'land'.
    Corrisponde alla terna (i, j, h) del paper.
    """
    launch:   int
    customer: int
    land:     int


@dataclass
class Robot:
    """
    Rappresenta i clienti serviti da un robot
    che parte da una stazione 'station'.
    """
    station: int
    customers: List[int] = field(default_factory=list)


@dataclass
class TruckRoute:
    """
    La rotta completa di un truck: sequenza di nodi,
    etichetta di ciascuno, sorties del drone e robots.
    """
    nodes:       List[int]
    labels:      List[str]  
    sorties:     List[Sortie] = field(default_factory=list)
    robots: List[Robot] = field(default_factory=list)

    def __repr__(self) -> str:
        route_str = " → ".join(
            f"{'🏭' if node == 0 or node == max(self.nodes) else '✈️' if label == 'C' else '🚚'}{node}"
            for node, label in zip(self.nodes, self.labels)
        )
        lines = [route_str]
        for s in self.sorties:
            lines.append(f"  ✈️  {s.launch} ──▶ {s.customer} ──▶ {s.land}")
        for r_idx, r in enumerate(self.robots):
            customers_str = ", ".join(str(c) for c in r.customers)
            lines.append(f"🤖 Robot {r_idx} stazione {r.station} ──▶ [{customers_str}]")
        return "\n".join(lines)
    
    def robots_by_station(self,station):
        '''Ritorna una lista di [(r_idx, robot)] con tutti i robot
        della rotta che partono da una stazione'''
        rbs = []
        for (r_idx, robot) in enumerate(self.robots):
            if robot.station == station:
                rbs.append((r_idx, robot))
        return rbs
    
    def is_only_one_customer_served_by_station(self, station, inst: Instance):
        '''Piccola funzione helper, ritorna True se tra tutti i robot della
        stazione 'station' viene servito esattamente un solo cliente'''

        robots_at_station = [r for r in self.robots if r.station == station]
        customers = [c for r in robots_at_station for c in r.customers]
        return len(customers) == 1


@dataclass
class Solution:
    """
    Soluzione del VRPD-RS-TW per la metaeuristica.
    """
    routes: List[TruckRoute]
    cost:   float = field(default=float('inf'))

    def __repr__(self) -> str:
        lines = [f"Solution (cost={self.cost:.2f})"]
        for i, route in enumerate(self.routes):
            lines.append(f"--- Truck {i+1} ---")
            lines.append(repr(route))
        return "\n".join(lines)

    def copy(self) -> "Solution":
        """Ritorna una copia profonda della soluzione."""
        return copy.deepcopy(self)
    
    def check_no_duplicates(self, label: str = "") -> bool:
        '''Utile per il debug, se il truck passa due volte da un nodo'''

        for r_idx, route in enumerate(self.routes):
            seen = set()
            for node in route.nodes:
                if node in seen:
                    print(f"[{label}] DUPLICATE node {node} in route {r_idx}: {route.nodes}")
                    return False
                seen.add(node)
        return True
    
    def check_solution_integrity(self, label: str = "") -> bool:
        '''Per il debug: per controllare che un cliente non venga servito
        in più modalità'''
        for r_idx, route in enumerate(self.routes):
            truck_customers = {n for n, lbl in zip(route.nodes, route.labels) 
                            if lbl == 'T' and n not in (0, len(route.nodes))}
            drone_customers = {s.customer for s in route.sorties}
            robot_customers = {c for robot in route.robots for c in robot.customers}
            
            all_sets = [truck_customers, drone_customers, robot_customers]
            for a, b in combinations(all_sets, 2):
                overlap = a & b
                if overlap:
                    print(f"[{label}] Route {r_idx}: cliente {overlap} in due modalità!")
                    return False
        return True
    
    def _classify_truck_arcs(self, route: TruckRoute) -> dict:
        """ Ritorna un dict con due liste: 
        'truck_only': [(i,j), ...]
        'truck_drone': [(i,j), ...]
        Serve per il plot della soluzinoe"""

        nodes, sorties = route.nodes, route.sorties
        launches, lands = [s.launch for s in sorties], [s.land for s in sorties]

        if 0 in launches: drone_on_truck = False
        else: drone_on_truck = True

        n = len(route.nodes)
        arcs = {'truck_only':[], 'truck_drone':[]}

        for k in range(n-1):
            if drone_on_truck: arcs['truck_drone'].append((nodes[k],nodes[k+1]))
            else: arcs['truck_only'].append((nodes[k],nodes[k+1]))
            sorties_launching_in_k = launches.count(nodes[k+1])
            sorties_landing_in_k = lands.count(nodes[k+1])
            if sorties_launching_in_k > sorties_landing_in_k: drone_on_truck = False
            if sorties_launching_in_k < sorties_landing_in_k: drone_on_truck = True
        return arcs
    
    def plot(self, instance, ax=None, title="Solution"):
        """
        Disegna la soluzione inserendo graficamente le Time Windows dei clienti
        e i tempi di arrivo del truck calcolati tramite synchronize_route.
        """
        from src.metaheuristics.synchronization import synchronize_route  # Import locale per evitare import circolari

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10)) # Finestra leggermente più grande per i testi
        
        coords = instance.coords  # shape (n_nodes, 2)
        n_customers = instance.n_customers
        n_stations  = instance.n_stations
        n_nodes     = instance.n_nodes

        truck_only_arcs   = set()
        truck_drone_arcs  = set()
        drone_arcs        = set()  
        robot_arcs        = set()
        
        # Dizionario per mappare ogni nodo al suo tempo di arrivo a_T
        a_T_global = {}

        for route in self.routes:
            truck_arcs = self._classify_truck_arcs(route)
            truck_only_arcs.update(truck_arcs['truck_only'])
            truck_drone_arcs.update(truck_arcs['truck_drone'])
            drone_arcs.update([(s.launch, s.customer) for s in route.sorties])
            drone_arcs.update([(s.customer, s.land) for s in route.sorties])
            robot_arcs.update([(t.station, customer) for t in route.robots for customer in t.customers])
            robot_arcs.update([(customer, t.station) for t in route.robots for customer in t.customers])

            # Sincronizzazione della singola rotta
            sync_res = synchronize_route(route, instance)
            if sync_res["is_feasible"]:
                # Estrae i tempi a_T allineati sequenzialmente con i nodi della rotta
                for idx, node in enumerate(route.nodes):
                    a_T_global[node] = sync_res["a_T"][idx]
            else:
                print(f"[WARN] Rilevata rotta non fattibile durante la sincronizzazione nel plot.")

        # --- Plot degli archi ---
        _draw_arc(ax, coords, truck_only_arcs, color='steelblue', lw=3, linestyle='solid')
        _draw_arc(ax, coords, truck_drone_arcs, color='darkorange', lw=4, linestyle='solid')
        _draw_arc(ax, coords, drone_arcs, color='green', lw=2, linestyle='dashed')
        _draw_arc(ax, coords, robot_arcs, color='purple', lw=2, linestyle='dashed')

        # --- Plot dei nodi ---
        ax.scatter(coords[0,0], coords[0,1], marker='s', color='black', s=400, label='Depot', zorder=3)
        ax.scatter(coords[1:n_customers+1,0], coords[1:n_customers+1,1], marker='o', color='skyblue', s=300, label='Customers', zorder=3)
        ax.scatter(coords[n_customers+1:n_customers+n_stations+1,0], coords[n_customers+1:n_customers+n_stations+1,1], marker='^', color='salmon', s=350, label='Stations', zorder=3)

        for n in range(n_nodes):
            # Identificativo numerico del nodo al centro del marker
            ax.text(coords[n,0], coords[n,1], str(n), fontsize=12, ha='center', va='center', zorder=4)
            
            # Costruzione del box informativo sotto al nodo
            info_lines = []
            
            # Se il nodo è un cliente, mostra la sua Time Window [t_start, t_stop]
            if 1 <= n <= n_customers:
                tw_start = instance.t_start[n-1]
                tw_stop  = instance.t_stop[n-1]
                info_lines.append(f"TW: [{tw_start:.0f}, {tw_stop:.0f}]")
                
            # Se è presente un tempo di arrivo calcolato per questo nodo, mostralo
            if n in a_T_global:
                info_lines.append(f"a_T: {a_T_global[n]:.1f}")
                
            if info_lines:
                info_text = "\n".join(info_lines)
                ax.annotate(info_text,
                            xy=(coords[n,0], coords[n,1]),
                            xytext=(0, -16),  # Offset verticale in punti per non sovrapporsi al marker
                            textcoords="offset points",
                            ha='center', va='top',
                            fontsize=9,
                            color='darkred',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
                            zorder=5)

        # Etichette e legenda ---
        custom_lines = [
            Line2D([0],[0], color='steelblue',  lw=2, linestyle='solid',  label='Truck only'),
            Line2D([0],[0], color='darkorange', lw=3, linestyle='solid',  label='Truck + Drone'),
            Line2D([0],[0], color='green',     lw=2, linestyle='dashed', label='Drone sortie'),
            Line2D([0],[0], color='purple',    lw=2, linestyle='dashed', label='Robot trip')
        ]
        ax.set_title(title)
        ax.axis('equal')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + custom_lines, labels=labels + [line.get_label() for line in custom_lines], loc='best')
        plt.tight_layout()
        plt.show()
        return ax
    

def _draw_arc(ax, coords, arc_set, color, lw, linestyle):

    for (i, j) in arc_set:
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        ax.annotate("",
                    xy=(x2, y2), xycoords='data',
                    xytext=(x1, y1), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=linestyle, shrinkA=6, shrinkB=6))
        
def milp_to_solution(m: gp.Model, inst: Instance) -> "Solution":
    """
    Converte una soluzione Gurobi ottimale/feasible in un oggetto Solution.
    Precondizione: m.SolCount > 0.
    
    """
    V         = range(inst.n_nodes)
    Vc        = range(1, inst.n_customers + 1)
    Vs        = range(inst.n_customers + 1,
                      inst.n_customers + inst.n_stations + 1)
    K         = range(MAX_TRUCKS_K)
    R         = range(MAX_ROBOTS_R)
    P         = range(inst.n_customers)
    depot_out = inst.n_nodes - 1

    routes: list[TruckRoute] = []

    for k in K:

        successore: dict[int, int] = {}
        for i in V:
            for j in V:
                var = m.getVarByName(f"x_T[{i},{j},{k}]")
                if var is not None and var.X > 0.5:
                    successore[i] = j

        if 0 not in successore:
            continue

        nodes: list[int] = []
        current = 0
        while current != depot_out:
            nodes.append(current)
            current = successore[current]
        nodes.append(depot_out)

        sorties: list[Sortie] = []
        for i in V:
            for j in Vc:
                for h in V:
                    for p in P:
                        var = m.getVarByName(f"x_D[{i},{j},{h},{p},{k}]")
                        if var is not None and var.X > 0.5:
                            sorties.append(Sortie(launch=i, customer=j, land=h))

        drone_nodes: set[int] = set()
        for s in sorties:
            drone_nodes.add(s.launch)
            drone_nodes.add(s.land)

        labels: list[str] = [
            'C' if node in drone_nodes else 'T'
            for node in nodes
        ]

        station_customers: dict[int, list[int]] = {}
        for i in Vs:
            for j in Vc:
                for r in R:
                    for p in P:
                        var = m.getVarByName(f"x_R[{i},{j},{r},{p},{k}]")
                        if var is not None and var.X > 0.5:
                            station_customers.setdefault(i, []).append(j)

        robots: list[Robot] = [
            Robot(station=s, customers=custs)
            for s, custs in station_customers.items()
        ]

        routes.append(TruckRoute(
            nodes=nodes,
            labels=labels,
            sorties=sorties,
            robots=robots,
        ))

    return Solution(routes=routes, cost=m.ObjVal)  
 
def vrp_model_to_solution(m: gp.Model, inst: Instance) -> Solution:
    """
    Converte la soluzione VRP-TW truck-only in un oggetto Solution.
    Necessita di una funzione a parte per via della struttura diversa.
    Usa m.getVars() invece di getVarByName → robusto con qualsiasi modello.
    Precondizione: m.SolCount > 0.
    """
    depot_out = inst.n_nodes - 1
    K = range(MAX_TRUCKS_K)

    successori: dict[int, dict[int, int]] = {k: {} for k in K}

    for var in m.getVars():
        name = var.VarName
        if not name.startswith("x_T["):
            continue
        if var.X < 0.5:
            continue
        # "x_T[3,7,1]" → i=3, j=7, k=1
        i, j, k = (int(x) for x in name[4:-1].split(","))
        successori[k][i] = j

    routes = []
    for k in K:
        succ = successori[k]
        if 0 not in succ:
            continue                      # truck k non utilizzato

        nodes: list[int] = []
        current = 0
        while current != depot_out:
            if current not in succ:       # protezione anti-loop
                print(f"[WARN] truck {k}: rotta interrotta a nodo {current}")
                break
            nodes.append(current)
            current = succ[current]
        nodes.append(depot_out)

        labels = ['T'] * len(nodes)
        routes.append(TruckRoute(nodes=nodes, labels=labels))

    return Solution(routes=routes, cost=m.ObjVal)

def solution_to_warm_start(
    sol:  "Solution",
    inst: Instance,
    x_T, x_D, x_R
) -> None:
    """
    Imposta gli attributi .Start sulle variabili binarie Gurobi
    a partire da un oggetto Solution della metaeuristica.
    Serve per i warm-start.
    """ 
    V  = range(inst.n_nodes)
    Vc = range(1, inst.n_customers + 1)
    Vs = range(inst.n_customers + 1, inst.n_customers + inst.n_stations + 1)
    Vl = range(inst.n_nodes-1)
    Vr = range(1,inst.n_nodes)
    K  = range(MAX_TRUCKS_K)
    R  = range(MAX_ROBOTS_R)
    P  = range(inst.n_customers)

    for i in V:
        for j in V:
            for k in K:
                x_T[i, j, k].Start = 0

    for i in Vl:
        for j in Vc:
            for h in Vr:
                for p in P:
                    for k in K:
                        x_D[i, j, h, p, k].Start = 0

    for i in Vs:
        for j in Vc:
            for r in R:
                for p in P:
                    for k in K:
                        x_R[i, j, r, p, k].Start = 0

    for k, route in enumerate(sol.routes):
        if k >= MAX_TRUCKS_K:
            break

        # x_T: archi consecutivi
        for idx in range(len(route.nodes) - 1):
            i = route.nodes[idx]
            j = route.nodes[idx + 1]
            x_T[i, j, k].Start = 1

        # x_D: sorties ordinate per posizione del lancio
        node_pos = {node: pos for pos, node in enumerate(route.nodes)}
        sorties_sorted = sorted(
            route.sorties,
            key=lambda s: node_pos.get(s.launch, 0)
        )
        for p, sortie in enumerate(sorties_sorted):
            if p >= len(P):
                break
            i, j, h = sortie.launch, sortie.customer, sortie.land
            if j in Vc:
                x_D[i, j, h, p, k].Start = 1

        # x_R: robot per stazione → r=indice nella lista, p=indice cliente
        station_to_robots: dict[int, list] = {}
        for robot in route.robots:
            station_to_robots.setdefault(robot.station, []).append(robot)

        for station, robots_at_s in station_to_robots.items():
            for r_idx, robot in enumerate(robots_at_s):
                if r_idx >= MAX_ROBOTS_R:
                    break
                for p_idx, customer in enumerate(robot.customers):
                    if p_idx >= len(P):
                        break
                    if customer in Vc:
                        x_R[station, customer, r_idx, p_idx, k].Start = 1     