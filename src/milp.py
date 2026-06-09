# src/milp.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.instance import Instance
from src.solution import Solution, solution_to_warm_start, vrp_model_to_solution
from src.config import *


def build_model(inst: Instance) -> gp.Model:

    """
    Costruisce il modello Gurobi per il VRPD-RS-TW.
    Ritorna il modello con le variabili definite (senza ancora obj e vincoli).
    """
    m = gp.Model("VRPD_RS_TW")

    # --- Insiemi di indici ---
    V  = range(inst.n_nodes)                            # tutti i nodi
    Vc = range(1, inst.n_customers + 1)                 # solo clienti
    Vs = range(inst.n_customers + 1,                    # solo stazioni
               inst.n_customers + inst.n_stations + 1)  
    
    K  = range(MAX_TRUCKS_K)          # truck
    R  = range(MAX_ROBOTS_R)          # robot
    P  = range(inst.n_customers)      # posizioni sortie nella rotta 

    t_stop_ext = inst.t_stop_ext

    # --- Variabile x_T[i,j,k] ---
    # 1 se il truck k percorre il nodo i->j
    x_T = m.addVars(V, V, K, vtype=GRB.BINARY, name="x_T")

    # --- Variabile x_D[i,j,h,p,k] ---
    # 1 se la p-esima sortie del drone del truck k serve il 
    # cliente j con decollo in i e atterraggio in h (sortie i->j->h)
    x_D = m.addVars(V,Vc,V,P,K, vtype=GRB.BINARY, name="x_D") 

    # --- Variabile x_R[i,j,r,p,k] ---
    # 1 se il robot r della stazione i servita dal truck k
    # consegna al cliente j nel p-esimo round-trip
    x_R = m.addVars(Vs,Vc,R,P,K, vtype=GRB.BINARY, name="x_R")   

    # Variabili continue temporali
    # Osservazione: per ogni nodo servito dal truck, ci sono tre tempi:
    # arrivo (a_T) <= servizio (s_T) <= ripartenza (d_T)

    # --- Variabile a_T[i,k] ---
    # misura quando il truck k arriva al nodo i
    a_T = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0,
                    ub={(i,k): t_stop_ext[i] for i in V for k in K},
                    name="a_T")
    # --- Variabile s_T[i,k] ---
    # misura quando il truck k può iniziare a servire al nodo i
    s_T = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0,
                    ub={(i,k): t_stop_ext[i] for i in V for k in K},
                    name="s_T")
    # --- Variabile d_T[i,k] ---
    # misura quando il truck k parte dal nodo i
    d_T = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0,
                ub={(i, k): inst.UB_dT[i] for i in V for k in K},
                name="d_T")

    # Vriabili temporali drone, come il truck
    a_D = m.addVars(V, P, K, vtype=GRB.CONTINUOUS, lb=0,
                    ub={(i,p,k): t_stop_ext[i] for i in V for p in P for k in K},
                    name="a_D")
    d_D = m.addVars(V, P, K, vtype=GRB.CONTINUOUS, lb=0,
                    ub=float(np.max(inst.t_stop_ext)),
                    name="d_D")

    # --- Release time robot ---
    # Per il robot, ci basta una variabile che misura quando 
    # il robot lascia la stazione per andare a servire il cliente
    r_R = m.addVars(Vs, R, P, K, vtype=GRB.CONTINUOUS, lb=0, name = "r_R")   

    # --- Posizione nodo in rotta ---
    # Questo serve per la sub-tour elimination
    # posizione nodo i nella rotta del truck k  -->  z[i,k]
    z   = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0, name = "z")   

    # --- Costi ---
    depot_end = inst.n_nodes - 1
    cost_driver = gp.quicksum(TRUCK_DRIVER_WAGE * d_T[depot_end, k] for k in K)
    cost_truck = gp.quicksum(TRUCK_COST_PER_KM * inst.dist_T[i,j] * x_T[i,j,k] for i in V for j in V if i != j for k in K)
    cost_drone = gp.quicksum(DRONE_COST_PER_KM * (inst.dist_D[i,j] + inst.dist_D[j,h]) * x_D[i,j,h,p,k] for i in V for j in Vc for h in V if j != i and h != j for p in P for k in K)
    cost_robot = gp.quicksum(ROBOT_COST_PER_KM * (inst.dist_R[i,j] + inst.dist_R[j,i]) * x_R[i,j,r,p,k] for k in K for p in P for r in R for i in Vs for j in Vc)
    # Obiettivo: minimizzare il costo
    m.setObjective(cost_driver + cost_truck + cost_drone + cost_robot, GRB.MINIMIZE)

    return m, x_T, x_D, x_R, a_T, s_T, d_T, a_D, d_D, r_R, z

def add_flow_constraints(m, inst, x_T, x_D, V, Vc, Vh, Vl, Vr, K, P):
    """Vincoli Eq. 3-9: flusso truck e drone."""
    
    depot_start = 0
    depot_end   = inst.n_nodes - 1
    # --- Eq. 3: ogni truck esce dal deposito al massimo una volta ---
    for k in K:
        m.addConstr(
            gp.quicksum(x_T[depot_start, j, k] for j in V) <= 1,
            name=f"eq3_truck_departs_once[{k}]"
        )

    # --- Eq. 4: bilanciamento deposito (chi parte torna) ---
    for k in K:
        m.addConstr(
            gp.quicksum(x_T[depot_start,j,k] for j in Vh) == gp.quicksum(x_T[j,depot_end,k] for j in Vh),
            name=f"eq4_return_depot[{k}]"
        )

    # --- Eq. 5: bilanciamento flusso ai nodi intermedi ---
    for k in K:
        for i in Vh:
            m.addConstr(
                gp.quicksum(x_T[j,i,k] for j in Vl if j != i) == gp.quicksum(x_T[i,j,k] for j in Vr if j != i),
                name=f"eq5_flow_balance[{i},{k}]"
            )

    # --- Eq. 6: ogni p-esima sortie avviene al massimo una volta ---
    for k in K:
        for p in P:
            m.addConstr(
                gp.quicksum(
                    x_D[i, j, h, p, k]
                    for i in Vl for j in Vc for h in V
                    if j != i and h != j
                ) <= 1,
                name=f"eq6_sortie_once[{p},{k}]"
            )

    # --- Eq. 7: sortie parte da nodo visitato dal truck ---
    for k in K:
        for p in P:
            for i in Vl:
                m.addConstr(
                    gp.quicksum(x_D[i,j,h,p,k] for j in Vc for h in V if j != i and h != j) <= gp.quicksum(x_T[i,j,k] for j in Vr if j != i),
                    name=f"eq7_drone_launch[{i},{p},{k}]"
                )

    # --- Eq. 8: sortie atterra in nodo visitato dal truck ---
    for k in K:
        for p in P:
            for h in Vr:
                m.addConstr(gp.quicksum(x_D[i,j,h,p,k] for i in Vl for j in Vc if j!=h and j!=i) <= gp.quicksum(x_T[i,h,k] for i in Vl if i != h),
                    name=f"eq8_drone_land[{h},{p},{k}]"
                )

    # --- Eq. 9: ordinamento stretto sorties (p+1 esiste solo se p esiste) ---
    for k in K:
        for p in range(len(P) - 1):   # p in P \ {P_max}
            m.addConstr(gp.quicksum(x_D[i,j,h,p+1,k] for i in Vl for j in Vc for h in V if j!=i and h!=j) <=
                        gp.quicksum(x_D[i,j,h,p,k] for i in Vl for j in Vc for h in V if j!=i and h!=j),
                name=f"eq9_sortie_order[{p},{k}]"
            )

def add_robot_station_constraints(m, inst, x_T, x_R, V, Vs, Vc, Vl, K, P, R):

    n = inst.n_customers   

    # --- Eq. 10: ogni stazione visitata da al massimo un truck ---
    for i in Vs:
        m.addConstr(
            gp.quicksum(x_T[j,i,k] for k in K for j in Vl if j != i) <= 1,
            name=f"eq10_station_one_truck[{i}]"
        )

    # --- Eq. 11: max operazioni robot da una stazione ---
    for k in K:
        for i in Vs:
            m.addConstr(
                gp.quicksum(x_R[i,j,r,p,k] for p in P for r in R for j in Vc) <=
                n * gp.quicksum(x_T[h,i,k] for h in Vl if h != i),
                name=f"eq11_robot_ops_limit[{i},{k}]"
            )

    # --- Eq. 12: truck visita stazione solo se robot dispiegato ---
    for k in K:
        for i in Vs:
            m.addConstr(
                gp.quicksum(x_T[h,i,k] for h in Vl if h != i) <=
                gp.quicksum(x_R[i,j,r,p,k] for p in P for r in R for j in Vc),
                name=f"eq12_station_needs_robot[{i},{k}]"
            )

def add_multimodal_constraints(m, inst, x_T, x_D, x_R, V, Vc, Vs, Vl, K, P, R):

    # --- Eq. 13: ogni cliente servito esattamente una volta ---
    for j in Vc:
        m.addConstr(
            gp.quicksum(x_T[i,j,k] for k in K for i in Vl if i != j) +
            gp.quicksum(x_D[i,j,h,p,k] for k in K for p in P for i in Vl for h in V if i != j and h != j) +
            gp.quicksum(x_R[i,j,r,p,k] for k in K for p in P for r in R for i in Vs)
            == 1,
            name=f"eq13_customer_served[{j}]"
        )

    # --- Eq. 14: capacità truck ---
    for k in K:
        m.addConstr(
            gp.quicksum(x_T[i,j,k] * inst.q[j-1] for i in Vl for j in Vc if i != j) +
            gp.quicksum(x_D[i,j,h,p,k] * inst.q[j-1] for p in P for i in Vl for j in Vc for h in V if i != j and h != j) +
            gp.quicksum(x_R[i,j,r,p,k] * inst.q[j-1] for p in P for r in R for i in Vs for j in Vc)
            <= TRUCK_CAPACITY_Q,
            name=f"eq14_truck_capacity[{k}]"
        )

def add_position_constraints(m, inst, x_T, x_D, z, V, Vl, Vr, Vc, K, P):

    n  = inst.n_customers

    # --- Eq. 15: lower bound posizione ---
    for k in K:
        for i in Vl:
            for j in Vr:
                if i != j:
                    m.addConstr(
                        z[j,k] - z[i,k] >= 1 - (n+2) * (1-x_T[i,j,k]),
                        name=f"eq15_pos_lb[{i},{j},{k}]"
                    )

    # --- Eq. 16: upper bound posizione ---
    for k in K:
        for i in Vl:
            for j in Vr:
                if i != j:
                    m.addConstr(
                        z[j,k] - z[i,k] <= 1 + (n+2) * (1-x_T[i,j,k]),
                        name=f"eq16_pos_ub[{i},{j},{k}]"
                    )

    # --- Eq. 17: ordinamento sorties lungo la rotta ---
    for k in K:
        for p in range(len(P) - 1):   
            for i in Vl:
                for j in Vr:
                    if i != j:
                        m.addConstr(z[j,k] - z[i,k] >=
                                    - (n+2) * (2-
                                          gp.quicksum(x_D[h,s,i,p,k] for h in Vl for s in Vc if s!=h and s!=i)-
                                          gp.quicksum(x_D[j,s,h,p+1,k] for h in V for s in Vc if s!=h and s!=i and s!=j)),
                            name=f"eq17_sortie_order[{i},{j},{p},{k}]"
                        )

def add_sync_constraints(m, inst, x_T, x_D, a_T, s_T, d_T, a_D, d_D,
                         V, Vc, Vl, Vr, K, P):

    # Eq. 18: arrivo truck a j >= partenza da nodo prima + tempo percorso 
    for k in K:
        for i in Vl:
            for j in Vr:
                if i != j:
                    M = inst.M_T[i,j]
                    m.addConstr(
                        a_T[j,k] >= d_T[i,k] + inst.time_T[i,j] - M*(1-x_T[i,j,k]),
                        name=f"eq18[{i},{j},{k}]")

    # Eq. 19: inizio servizio truck >= arrivo del truck
    for k in K:
        for i in Vr:
            M = inst.M_node_T[i]
            m.addConstr(
                s_T[i,k] >= a_T[i,k] - M*(1-gp.quicksum(x_T[j,i,k] for j in Vl if j!=i)),
                name=f"eq19[{i},{k}]")

    # Eq. 20: partenza truck >= fine servizio 
    for k in K:
        for i in Vr:
            M = inst.M_node_T[i]
            if i<inst.n_nodes-1:
                m.addConstr(
                    d_T[i,k] >= s_T[i,k] + TRUCK_SERVICE_TIME - M*(1-gp.quicksum(x_T[j,i,k] for j in Vl if j!=i)),
                    name=f"eq20[{i},{k}]")
            else:
                m.addConstr(
                    d_T[i,k] >= s_T[i,k] - M*(1-gp.quicksum(x_T[j,i,k] for j in Vl if j!=i)),
                    name=f"eq20[{i},{k}]")

    # Eq. 21: truck aspetta drone al meeting point h
    for k in K:
        for p in P:
            for j in Vc:
                for h in V:
                    if j != h:
                        M = inst.M_D[j,h]
                        m.addConstr(
                            d_T[h,k] >= a_D[j,p,k] + DRONE_SERVICE_TIME + inst.time_D[j,h] -
                            M*(1-gp.quicksum(x_D[i,j,h,p,k] for i in Vl if i !=j)),
                            name=f"eq21[{j},{h},{p},{k}]")

    # Eq. 22: drone parte dopo caricamento 
    for k in K:
        for p in P:
            for i in Vl:
                M = inst.M_node_T[i]
                m.addConstr(
                    d_D[i,p,k] >= a_T[i,k] + DRONE_LOAD_TIME -
                    M*(1-gp.quicksum(x_D[i,j,h,p,k] for j in Vc for h in V if j!=i and h!=j)),
                    name=f"eq22[{i},{p},{k}]")

    # Eq. 23: truck non parte prima del drone 
    for k in K:
        for p in P:
            for i in Vl:
                M = inst.M_node_T[i]
                m.addConstr(
                    d_T[i,k] >= d_D[i,p,k] -
                    M*(1-gp.quicksum(x_D[i,j,h,p,k] for j in Vc for h in V if j!=i and h!=j)),
                    name=f"eq23[{i},{p},{k}]")

    # Eq. 24: arrivo drone a j 
    for k in K:
        for p in P:
            for i in Vl:
                for j in Vc:
                    if i != j:
                        M = inst.M_D[i,j]
                        m.addConstr(
                            a_D[j,p,k] >= d_D[i,p,k] + inst.time_D[i,j]
                            - M*(1-gp.quicksum(x_D[i,j,h,p,k] for h in V if h!=j)),
                            name=f"eq24[{i},{j},{p},{k}]")

    # Eq. 25: (p+1)-esima sortie parte dopo completamento p-esima
    for k in K:
        for p in range(len(P) - 1):
            for i in Vl:
                for j in Vc:
                    for h in Vr:
                        if h != j and i != j:
                            M = inst.M_25[i,j,h]
                            m.addConstr(
                                d_D[i,p+1,k] >= a_D[j,p,k] + DRONE_SERVICE_TIME + inst.time_D[j,h] + DRONE_LOAD_TIME
                                - M*(2- gp.quicksum(x_D[s,j,h,p,k] for s in Vl if s!=j)
                                    - gp.quicksum(x_D[i,s,v,p+1,k] for v in V for s in Vc if s!=h and v!=s and s!=i)),
                                name=f"eq25[{i},{j},{h},{p},{k}]")
                            
def add_robot_battery_constraints(m, inst, x_D, x_R, a_T, s_T, d_D, r_R,
                                  V, Vs, Vc, Vl, K, P, R):

    # Eq. 26: release time robot, prima consegna (p=0) dopo servizio truck
    for k in K:
        for i in Vs:
            for r in R:
                M = inst.M_26[i]
                m.addConstr(
                    s_T[i,k] + TRUCK_SERVICE_TIME <= r_R[i,r,0,k] +
                    M*(1-gp.quicksum(x_R[i,j,r,0,k] for j in Vc)),
                    name=f"eq26[{i},{r},{k}]"
                )

    # Eq. 27: release time robot, consegne successive (p>0) 
    for k in K:
        for i in Vs:
            for r in R:
                for p in range(1, len(P)):
                    m.addConstr(
                        r_R[i,r,p-1,k]+
                        gp.quicksum((ROBOT_LOAD_TIME+ROBOT_SERVICE_TIME+inst.time_R[i,j]+inst.time_R[j,i])*x_R[i,j,r,p-1,k] for j in Vc)
                        <= r_R[i,r,p,k] + inst.M_27*(1-gp.quicksum(x_R[i,j,r,p,k] for j in Vc)),
                        name=f"eq27[{i},{r},{p},{k}]"
                    )

    # Eq. 28: batteria drone
    b_sortie = inst.energy_on[:,:,np.newaxis] + inst.energy_off[np.newaxis,:,:]
    t_sortie = inst.time_D[:,:,np.newaxis] + DRONE_SERVICE_TIME + inst.time_D[np.newaxis,:,:]
    L = t_sortie + (DRONE_BATTERY_B - b_sortie) / DRONE_HOVER_POWER
    for k in K:
        for p in P:
            for i in Vl:
                for j in Vc:
                    for h in V:
                        if j != i and h != j:
                            M = inst.M_node_T[h]
                            m.addConstr(
                                a_T[h,k]-d_D[i,p,k] <= L[i,j,h] + M*(1-x_D[i,j,h,p,k]),
                                name=f"eq28[{i},{j},{h},{p},{k}]"
                            )

    # Eq. 29: range robot 
    for k in K:
        for p in P:
            for r in R:
                for i in Vs:
                    for j in Vc:
                        m.addConstr(
                            (ROBOT_LOAD_TIME+ROBOT_SERVICE_TIME+inst.time_R[i,j]+inst.time_R[j,i])*x_R[i,j,r,p,k]<=ROBOT_MAX_DRIVE_TIME,
                            name=f"eq29[{i},{j},{r},{p},{k}]"
                        )

def add_time_window_constraints(m, inst, x_T, x_D, x_R, s_T, d_D, r_R,
                                 V, Vc, Vs, Vl, Vr, K, P, R):

    # Eq. 30-31: truck rispetta finestre temporali clienti 
    for k in K:
        for i in Vc:
            incoming = gp.quicksum(x_T[j,i,k] for j in Vl if j != i)
            # Eq. 30: lower bound — M = t_start[i]
            m.addConstr(
                inst.t_start[i-1] <= s_T[i,k] + inst.M_tw30[i]*(1-incoming),
                name=f"eq30[{i},{k}]")
            # Eq. 31: upper bound — M = TRUCK_SERVICE_TIME (scalare)
            m.addConstr(
                inst.t_stop[i-1] + inst.M_tw31*(1-incoming) >= s_T[i,k] + TRUCK_SERVICE_TIME,
                name=f"eq31[{i},{k}]")

    # Eq. 32-33: drone rispetta finestre temporali clienti
    for k in K:
        for p in P:
            for i in Vl:
                for j in Vc:
                    if i != j:
                        outgoing = gp.quicksum(x_D[i,j,h,p,k] for h in V if h != j)
                        # Eq. 32: lower bound — M = M_D[i,j]
                        M32 = inst.M_tw32[i,j]
                        m.addConstr(
                            inst.t_start[j-1] <= d_D[i,p,k] + inst.time_D[i,j] + M32*(1-outgoing),
                            name=f"eq32[{i},{j},{p},{k}]")
                        # Eq. 33: upper bound — M = M_tw33[i,j]
                        M33 = inst.M_tw33[i,j]
                        m.addConstr(
                            inst.t_stop[j-1] + M33*(1-outgoing) >= d_D[i,p,k] + inst.time_D[i,j] + DRONE_SERVICE_TIME,
                            name=f"eq33[{i},{j},{p},{k}]")

    # Eq. 34-35: Robot rispetta finestre temporali clienti
    for k in K:
        for p in P:
            for r in R:
                for i in Vs:
                    for j in Vc:
                        # Eq. 34: lower bound — M = M_tw34[i,j]
                        M34 = inst.M_tw34[i,j]
                        m.addConstr(
                            inst.t_start[j-1] <= r_R[i,r,p,k] + ROBOT_LOAD_TIME + inst.time_R[i,j] + M34*(1-x_R[i,j,r,p,k]),
                            name=f"eq34[{i},{j},{r},{p},{k}]")
                        # Eq. 35: upper bound — M = M_tw35[i,j]
                        M35 = inst.M_tw35[i,j]
                        m.addConstr(
                            inst.t_stop[j-1] + M35*(1-x_R[i,j,r,p,k]) >= r_R[i,r,p,k] + (ROBOT_LOAD_TIME+inst.time_R[i,j]+ROBOT_SERVICE_TIME)*x_R[i,j,r,p,k],
                            name=f"eq35[{i},{j},{r},{p},{k}]")

def add_symmetry_and_preprocessing(m, inst, x_T, x_D, x_R,
                                    V, Vs, Vc, Vl, Vr, K, P, R):
    """Eq. 36-43: symmetry breaking e pre-processing."""

    # --- Eq. 36: simmetria truck, le rotte sono ordinate dalla piu lunga alla piu corta ---
    for k in range(1, len(K)):   
        m.addConstr(
            gp.quicksum(x_T[0,j,k] for j in Vr) <= gp.quicksum(x_T[0,j,k-1] for j in Vr),
            name=f"eq36_truck_symmetry[{k}]"
        )

    # --- Eq. 37: ordinamento posizioni robot, esiste un round-trip p se c'era un (p-1)-esimo
    for k in K:
        for i in Vs:
            for r in R:
                for p in range(1, len(P)):  
                    m.addConstr(
                        gp.quicksum(x_R[i,j,r,p,k] for j in Vc) <= gp.quicksum(x_R[i,j,r,p-1,k] for j in Vc),
                        name=f"eq37_robot_pos_order[{i},{r},{p},{k}]"
                    )

    # --- Eq. 38: ordinamento robot, per ordinare robot della stessa stazione---
    for k in K:
        for i in Vs:
            for r in range(1, len(R)):   
                m.addConstr(
                    gp.quicksum(x_R[i,j,r,0,k] for j in Vc) <= gp.quicksum(x_R[i,j,r-1,0,k] for j in Vc),
                    name=f"eq38_robot_symmetry[{i},{r},{k}]"
                )

    # --- Eq. 39-43: pre-processing (fix UB = 0)
    # Serie di disequazioni per impedire comportamenti del drone non previsti
    depot_start = 0
    for i in V:
        for j in Vc:
            for h in V:
                should_fix = (
                    (j == i) or (h == j)                                          # sortie banalmente invalida
                    or (inst.energy_on[i,j] + inst.energy_off[j,h] > DRONE_BATTERY_B)  # Eq. 39
                    or (h == depot_start)                                         # Eq. 40
                    or (i in inst.no_fly)                                         # Eq. 41
                    or (j in inst.no_fly)                                         # Eq. 42
                    or (h in inst.no_fly)                                         # Eq. 43
                )
                if should_fix:
                    for p in P:
                        for k in K:
                            x_D[i,j,h,p,k].UB = 0
    
    # Per evitare rotte degeneri (che vanno da depot_start a depot_end)
    for k in K:
        x_T[0,inst.n_nodes-1,k].UB = 0

def add_valid_inequalities(m, inst, x_T, x_D, x_R, a_T, d_T,
                           V, Vc, Vs, Vh, Vl, Vr, K, P, R,
                           which: list = None):
    """
    Valid inequalities per il VRPD-RS-TW.
    
    Parametro `which`: lista di interi in [1..8] che seleziona le VI da aggiungere.
                       Se None o vuota, le aggiunge tutte.
    
    VI 1-5 : Supplementary Material (Campuzano 2025)
    VI 6-8 : LB makespan da time windows (truck, drone depot, drone generico)
    """
    if not which:
        which = list(range(1, 9))
    active = set(which)

    depot_end   = inst.n_nodes - 1
    depot_start = 0
    n = inst.n_customers

    
    if 1 in active:
        #   VI1 — Lower bound sul makespan del truck:
        #   il tempo di ritorno al deposito deve coprire almeno la somma dei tempi
        #   di percorrenza di tutti gli archi usati, più le attese accumulate ai nodi.
        for k in K:
            m.addConstr(
                d_T[depot_end, k] >=
                gp.quicksum(inst.time_T[i, j] * x_T[i, j, k]
                            for i in Vl for j in Vr if j != i) +
                gp.quicksum(d_T[i, k] - a_T[i, k] for i in V),
                name=f"vi1[{k}]"
            )

    if 2 in active:
    #     VI2 — Lower bound sul makespan da operazioni drone:
    #   analogo a VI1 ma somma la durata completa di ogni sortie
    #   (carico + volo andata + servizio + volo ritorno).

        for k in K:
            m.addConstr(
                d_T[depot_end, k] >=
                gp.quicksum(
                    (DRONE_LOAD_TIME + inst.time_D[i, j] +
                     DRONE_SERVICE_TIME + inst.time_D[j, h]) * x_D[i, j, h, p, k]
                    for p in P for i in Vl for j in Vc for h in V
                    if j != i and h != j
                ),
                name=f"vi2[{k}]"
            )

    if 3 in active:
    #     VI3 — Limite superiore sul numero totale di missioni secondarie:
    #   sorties drone + trips robot non possono superare n,
    #   perché ogni cliente è servito al più una volta da un mezzo secondario.
        m.addConstr(
            gp.quicksum(
                x_D[i, j, h, p, k]
                for k in K for p in P for i in Vl for j in Vc for h in V
                if j != i and h != j
            ) +
            gp.quicksum(
                x_R[i, j, r, p, k]
                for k in K for p in P for r in R for i in Vs for j in Vc
            ) <= n,
            name="vi3"
        )

    if 4 in active:
    #     VI4 — Almeno un truck deve uscire dal deposito:
    #   taglia soluzioni banalmente infeasible in cui nessun veicolo parte.
        m.addConstr(
            gp.quicksum(x_T[0, j, k] for k in K for j in Vh) >= 1,
            name="vi4"
        )

    if 5 in active:
    #     VI5 — Ogni nodo è visitato da al più un truck:
    #   impedisce che due truck percorrano lo stesso arco entrante su j.
        for j in Vh:
            m.addConstr(
                gp.quicksum(x_T[i, j, k] for k in K for i in Vl if i != j) <= 1,
                name=f"vi5[{j}]"
            )

    if 6 in active:
    #     VI6 — Lower bound sul makespan da finestre temporali (truck):
    #   se il truck k visita j direttamente, deve rientrare non prima di
    #   t_start[j] + servizio + viaggio di ritorno al deposito.
        for k in K:
            for j in Vc:
                lb_j = (inst.t_start[j - 1]
                        + TRUCK_SERVICE_TIME
                        + inst.time_T[j, depot_end])
                m.addConstr(
                    d_T[depot_end, k] >= lb_j * gp.quicksum(
                        x_T[i, j, k] for i in Vl if i != j),
                    name=f"vi6[{j},{k}]"
                )

    if 7 in active:
    #     VI7 — Lower bound sul makespan da finestre temporali (drone → deposito):
    #   come VI6, ma per sorties drone che rientrano direttamente al deposito
    #   (meeting point h = deposito).
        for k in K:
            for p in P:
                for j in Vc:
                    lb_j = (inst.t_start[j - 1]
                            + DRONE_SERVICE_TIME
                            + inst.time_D[j, depot_end])
                    m.addConstr(
                        d_T[depot_end, k] >= lb_j * gp.quicksum(
                            x_D[i, j, depot_end, p, k] for i in Vl if i != j),
                        name=f"vi7[{j},{p},{k}]"
                    )

    if 8 in active:
    #     VI8 — Lower bound sul makespan da finestre temporali (drone → nodo generico):
    #   versione più stretta di VI7 per meeting point h qualsiasi:
    #   tiene conto del volo drone j→h e del viaggio truck h→deposito.
        for k in K:
            for p in P:
                for i in Vl:
                    for j in Vc:
                        for h in V:
                            if i == j or h == j:
                                continue
                            if (inst.energy_on[i, j] + inst.energy_off[j, h] > DRONE_BATTERY_B
                                    or h == depot_start
                                    or i in inst.no_fly
                                    or j in inst.no_fly
                                    or h in inst.no_fly):
                                continue
                            lb = (inst.t_start[j - 1]
                                  + DRONE_SERVICE_TIME
                                  + inst.time_D[j, h]
                                  + inst.time_T[h, depot_end])
                            m.addConstr(
                                d_T[depot_end, k] >= lb * x_D[i, j, h, p, k],
                                name=f"vi8[{i},{j},{h},{p},{k}]"
                            )

def solve(inst: Instance, verbose: bool = True,
          use_valid_inequalities = False,
          warm_start: "Solution" = None, max_time = None, tuned = False) -> gp.Model:

    V  = range(inst.n_nodes)
    Vc = range(1, inst.n_customers + 1)
    Vs = range(inst.n_customers + 1,
               inst.n_customers + inst.n_stations + 1)
    Vh = range(1, inst.n_nodes - 1)     # Tutti i nodi tranne primo e ultimo
    Vl = range(0, inst.n_nodes - 1)     # Tutti i nodi tranne l'ultimo (posizioni buone per decollo)
    Vr = range(1, inst.n_nodes)         # Tutti i nodi tranne il primo (posizioni buone per atterraggio)
    K  = range(MAX_TRUCKS_K)
    R  = range(MAX_ROBOTS_R)
    P  = range(inst.n_customers)

    m, x_T, x_D, x_R, a_T, s_T, d_T, a_D, d_D, r_R, z = build_model(inst)

    add_flow_constraints(m, inst, x_T, x_D, V, Vc, Vh, Vl, Vr, K, P)
    add_robot_station_constraints(m, inst, x_T, x_R, V, Vs, Vc, Vl, K, P, R)
    add_multimodal_constraints(m, inst, x_T, x_D, x_R, V, Vc, Vs, Vl, K, P, R)
    add_position_constraints(m, inst, x_T, x_D, z, V, Vl, Vr, Vc, K, P)
    add_sync_constraints(m, inst, x_T, x_D, a_T, s_T, d_T, a_D, d_D,
                         V, Vc, Vl, Vr, K, P)
    add_robot_battery_constraints(m, inst, x_D, x_R, a_T, s_T, d_D, r_R,
                                  V, Vs, Vc, Vl, K, P, R)
    add_time_window_constraints(m, inst, x_T, x_D, x_R, s_T, d_D, r_R,
                                V, Vc, Vs, Vl, Vr, K, P, R)
    add_symmetry_and_preprocessing(m, inst, x_T, x_D, x_R,
                                   V, Vs, Vc, Vl, Vr, K, P, R)

    if use_valid_inequalities:
        add_valid_inequalities(m, inst, x_T, x_D, x_R, a_T, d_T,
                            V, Vc, Vs, Vh, Vl, Vr, K, P, R,
                            which=use_valid_inequalities if isinstance(use_valid_inequalities, list) else None)
        
    if tuned:
        m.Params.CutPasses = 5          # Fa 5 passate di tagli
        m.Params.SimplexPricing = 2     # Strategia di Devex Pricing nel metodo del simplesso - viene da un tuning
        m.Params.PreSparsify = 1        # Sparsifica la matrice per alleggerire il B&B
        m.Params.BranchDir = 1          # Esplora prima i nodi di destra, buono per scoprire subito soluzioni infeasible e fare tagli
        m.Params.AggFill = 0            # Impedisce di ingrandire la matrice in presolve
        m.Params.Heuristics = 0.01      # Riduci tempo su euristiche (di solito è più difficile alzare best bound)

    if warm_start is not None:
        solution_to_warm_start(warm_start, inst, x_T, x_D, x_R)

    m.Params.MIPGap    = 0
    m.Params.LogFile   = ""
    m.Params.OutputFlag = 1 if verbose else 0

    if max_time: m.Params.TimeLimit = max_time

    m.optimize()
    if m.status == GRB.INFEASIBLE:
        print("\nLa soluzione euristica viola un vincolo matematico! Calcolo l'IIS...")
        m.computeIIS()
        m.write("debug_infeasible.ilp")
        print("Controlla il file 'debug_infeasible.ilp' per vedere i vincoli contraddittori.")
    return m, x_T, x_D, x_R

#  VRP-TW TRUCK-ONLY — usato come costruttore di soluzione iniziale feasible
# nelle metaeuristiche (in phase I)

def build_model_vrp_only(inst: Instance):
    m = gp.Model("VRP_TW")
    depot_end = inst.n_nodes-1

    V  = [node for node in range(inst.n_customers +1)] + [depot_end]
    K  = range(MAX_TRUCKS_K)
    t_stop_ext = inst.t_stop_ext

    x_T = m.addVars(V, V, K, vtype=GRB.BINARY, name="x_T")
    a_T = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0,
                    ub={(i,k): t_stop_ext[i] for i in V for k in K}, name="a_T")
    s_T = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0,
                    ub={(i,k): t_stop_ext[i] for i in V for k in K}, name="s_T")
    d_T = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0,
                    ub={(i, k): inst.UB_dT[i] for i in V for k in K}, name="d_T")

    n   = inst.n_customers
    z   = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0, ub=n+2, name="z")

    cost_truck = gp.quicksum(
        TRUCK_COST_PER_KM * inst.dist_T[i, j] * x_T[i, j, k]
        for i in V for j in V if i != j for k in K
    )
    m.setObjective(cost_truck, GRB.MINIMIZE)

    return m, x_T, a_T, s_T, d_T, z   # ← z aggiunto al return

def add_vrp_tw_constraints(m, inst, x_T, a_T, s_T, d_T, z, V, Vc, K):
    """
    Vincoli VRP-TW truck-only con sub-tour elimination (Eq. 15-16).
    """
    depot_end = inst.n_nodes-1
    Vl = range(0, inst.n_customers +1)
    Vr = [node for node in range(1,inst.n_customers +1)] + [depot_end]
    n  = inst.n_customers

    # (1) no self-loop
    for i in V:
        for k in K:
            m.addConstr(x_T[i, i, k] == 0)

    # (2) ogni cliente visitato esattamente una volta
    for j in Vc:
        m.addConstr(gp.quicksum(x_T[i, j, k] for i in V for k in K if i != j) == 1)
        m.addConstr(gp.quicksum(x_T[j, i, k] for i in V for k in K if i != j) == 1)

    # (3) flow conservation ai clienti
    for v in Vc:
        for k in K:
            m.addConstr(
                gp.quicksum(x_T[i, v, k] for i in V if i != v) ==
                gp.quicksum(x_T[v, j, k] for j in V if j != v)
            )

    # (4) ogni truck parte/torna al depot al più una volta
    for k in K:
        m.addConstr(gp.quicksum(x_T[0, j, k] for j in V if j != 0) <= 1)
        m.addConstr(gp.quicksum(x_T[i, depot_end, k] for i in V if i != depot_end) <= 1)
        m.addConstr(
            gp.quicksum(x_T[0, j, k] for j in V if j != 0) ==
            gp.quicksum(x_T[i, depot_end, k] for i in V if i != depot_end)
        )

    # (5) capacità
    for k in K:
        m.addConstr(
            gp.quicksum(inst.q[j-1] * gp.quicksum(x_T[i, j, k] for i in V if i != j)
                        for j in Vc) <= TRUCK_CAPACITY_Q
        )

    # (6) propagazione temporale (Eq. 18 del paper)
    for i in Vl:
        for j in Vr:
            if i == j: continue
            for k in K:
                m.addConstr(
                    a_T[j, k] >= d_T[i, k] + inst.time_T[i, j]
                               - inst.M_T[i, j] * (1 - x_T[i, j, k])
                )

    # (7) s_T >= a_T
    for i in V:
        for k in K:
            m.addConstr(s_T[i, k] >= a_T[i, k])

    # (8) time window lower bound (Eq. 30 del paper)
    for j in Vc:
        for k in K:
            m.addConstr(
                s_T[j, k] >= inst.t_start_ext[j] *
                             gp.quicksum(x_T[i, j, k] for i in V if i != j)
            )

    # (9) service time
    for i in V:
        for k in K:
            if i in Vc:
                m.addConstr(d_T[i, k] == s_T[i, k] + TRUCK_SERVICE_TIME)
            else:
                m.addConstr(d_T[i, k] == s_T[i, k])

    # (10) nessun arco da depot_end o verso depot_start
    for k in K:
        m.addConstr(gp.quicksum(x_T[depot_end, j, k] for j in V) == 0)
        m.addConstr(gp.quicksum(x_T[j, 0, k] for j in V) == 0)

    # ── (11) SUB-TOUR ELIMINATION — Eq. 15-16 del paper ─────────────────
    for k in K:
        for i in Vl:
            for j in Vr:
                if i == j: continue
                m.addConstr(z[j,k] - z[i,k] >= 1 - (n+2) * (1 - x_T[i,j,k]))
                m.addConstr(z[j,k] - z[i,k] <= 1 + (n+2) * (1 - x_T[i,j,k]))

    # (12) symmetry breaking truck
    for k1 in range(MAX_TRUCKS_K - 1):
        for k2 in range(k1 + 1, MAX_TRUCKS_K):
            m.addConstr(
                gp.quicksum(x_T[i,j,k1] for i in V for j in V if i != j) >=
                gp.quicksum(x_T[i,j,k2] for i in V for j in V if i != j)
            )
    for k in K:
        m.addConstr(x_T[0,depot_end,k]==0)

    m.addConstr(
            gp.quicksum(x_T[0, j, k] for k in K for j in Vc) >= 1,
            name="vi4"
        )
    
    for j in Vc:
            m.addConstr(
                gp.quicksum(x_T[i, j, k] for k in K for i in Vl if i != j) <= 1,
                name=f"vi5[{j}]"
            )

def solve_vrp_milp(inst: Instance,
                   time_limit: int = GUROBI_TIME_LIMIT,
                   verbose: bool = False) -> "Solution | None":
    depot_end = inst.n_nodes-1
    V  = [node for node in range(inst.n_customers +1)] + [depot_end]
    Vc = range(1, inst.n_customers + 1)
    K  = range(MAX_TRUCKS_K)

    m, x_T, a_T, s_T, d_T, z = build_model_vrp_only(inst)   # ← z
    add_vrp_tw_constraints(m, inst, x_T, a_T, s_T, d_T, z, V, Vc, K)  # ← z

    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit  = time_limit
    m.Params.MIPGap     = 0.1
    m.Params.LogFile    = ""
    m.Params.MIPFocus = 1
    m.Params.SolutionLimit = 2


    m.optimize()

    if m.SolCount == 0:
        return None

    return vrp_model_to_solution(m, inst)

def print_solution(inst, m, x_T, x_D, x_R):
    """Stampa la soluzione in linguaggio naturale."""
    
    V  = range(inst.n_nodes)
    Vc = range(1, inst.n_customers + 1)
    Vs = range(inst.n_customers + 1,
               inst.n_customers + inst.n_stations + 1)
    Vl = range(0, inst.n_customers + inst.n_stations + 1)
    Vr = range(1, inst.n_nodes)
    K  = range(MAX_TRUCKS_K)
    R  = range(MAX_ROBOTS_R)
    P  = range(inst.n_customers)

    print("\n========== SOLUZIONE ==========")
    for k in K:
        # Ricostruisci la rotta del truck k
        route = []
        current = 0
        visited = set()
        while True:
            next_node = None
            for j in Vr:
                if j != current and j not in visited:
                    if x_T[current, j, k].X > 0.5:
                        next_node = j
                        break
            if next_node is None:
                break
            route.append(next_node)
            visited.add(next_node)
            current = next_node

        if not route:
            continue

        print(f"\n🚚 TRUCK {k}: 0 → {' → '.join(map(str, route))}")

        # Sorties drone da questo truck
        for p in P:
            for i in Vl:
                for j in Vc:
                    for h in V:
                        if j != i and h != j:
                            if x_D[i, j, h, p, k].X > 0.5:
                                print(f"   ✈️  Sortie p={p}: "
                                      f"lancia da {i} → serve cliente {j} "
                                      f"→ atterra in {h}")

        # Consegne robot
        for p in P:
            for r in R:
                for i in Vs:
                    for j in Vc:
                        if x_R[i, j, r, p, k].X > 0.5:
                            print(f"   🤖 Robot r={r}, p={p}: "
                                  f"stazione {i} → cliente {j}")
                            
def tune(inst: Instance, warm_start: "Solution" = None) -> gp.Model:
    """Esegue il tuning automatico dei parametri Gurobi."""

    m, x_T, x_D, x_R, a_T, s_T, d_T, a_D, d_D, r_R, z = build_model(inst)
    V  = range(inst.n_nodes)
    Vc = range(1, inst.n_customers + 1)
    Vs = range(inst.n_customers + 1,
               inst.n_customers + inst.n_stations + 1)
    Vh = range(1, inst.n_customers + inst.n_stations + 1)
    Vl = range(0, inst.n_customers + inst.n_stations + 1)
    Vr = range(1, inst.n_nodes)
    K  = range(MAX_TRUCKS_K)
    R  = range(MAX_ROBOTS_R)
    P  = range(inst.n_customers)

    add_flow_constraints(m,inst,x_T,x_D,V,Vc,Vh,Vl,Vr,K,P)
    add_robot_station_constraints(m, inst, x_T, x_R, V, Vs, Vc, Vl, K, P, R)
    add_multimodal_constraints(m, inst, x_T, x_D, x_R, V, Vc, Vs, Vl, K, P, R)
    add_position_constraints(m, inst, x_T, x_D, z, V, Vl, Vr, Vc, K, P)
    add_sync_constraints(m, inst, x_T, x_D, a_T, s_T, d_T, a_D, d_D,
                         V, Vc, Vl, Vr, K, P)
    add_robot_battery_constraints(m, inst, x_D, x_R, a_T, s_T, d_D, r_R,
                                  V, Vs, Vc, Vl, K, P, R)
    add_time_window_constraints(m, inst, x_T, x_D, x_R, s_T, d_D, r_R,
                                 V, Vc, Vs, Vl, Vr, K, P, R)
    add_symmetry_and_preprocessing(m, inst, x_T, x_D, x_R,
                                    V, Vs, Vc, Vl, Vr, K, P, R)

    # parametri tuning — NON impostare parametri algoritmici
    m.Params.TimeLimit     = 30         # max per ogni singola prova
    m.Params.TuneTimeLimit = 40*60      # sec totali
    m.Params.TuneCriterion = 3          # cerca miglior gap
    m.Params.TuneTrials = 3             # 3 prove per combinazione
    if warm_start is not None:
        solution_to_warm_start(warm_start, inst, x_T, x_D, x_R)

    m.tune()

    if m.TuneResultCount > 0:
        m.getTuneResult(0)
        m.write("best.prm")
        print("Parametri migliori salvati in best.prm")
    else:
        print("Nessun miglioramento trovato rispetto ai default")

    return m
