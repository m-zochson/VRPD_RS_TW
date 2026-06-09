# src/instance.py
from dataclasses import dataclass, field
from typing import Set
import numpy as np
from src.config import *

@dataclass
class Instance:
    # --- Numero di nodi ---
    n_customers: int          # numero di clienti (|V_c|)
    n_stations: int           # numero di stazioni robot (|V_s|)
    no_fly: Set[int]          # indici dei clienti in zona no-fly

    # --- Coordinate dei nodi ---
    coords: np.ndarray

    # --- Matrici distanze (km) e tempi (s) ---
    dist_T:  np.ndarray       # [n_nodes x n_nodes]
    time_T:  np.ndarray       # [n_nodes x n_nodes]
    dist_D:  np.ndarray       # [n_nodes x n_nodes]
    time_D:  np.ndarray       # [n_nodes x n_nodes]
    dist_R:  np.ndarray       # [n_nodes x n_nodes]
    time_R:  np.ndarray       # [n_nodes x n_nodes]

    # --- Dati per cliente ---
    q:        np.ndarray             # [n_customers] peso dei pacchi (kg)
    t_start:  np.ndarray             # [n_customers] inizio finestra temporale
    t_stop:   np.ndarray             # [n_customers] fine finestra temporale

    # --- Matrici dell'energia dei droni
    energy_on: np.ndarray           # [n_nodes x n_nodes]
    energy_off: np.ndarray          # [n_nodes x n_nodes]

    # --- Proprietà derivate (calcolate in post_init) ---
    n_nodes: int = field(init=False)   # numero totale di nodi

    def __post_init__(self):
        # Qui calcoliamo n_nodes a partire dagli altri campi
        self.n_nodes = 2 + self.n_customers + self.n_stations
        self.t_start = self.t_start.tolist()
        self.t_stop  = self.t_stop.tolist()
        self.q       = self.q.tolist()

        # In post_init precalcoliamo anche i valori dei Big M per il milp
        self.precompute_big_m() 
    
    def enough_battery_drone(self, i: int, j: int, h: int):
        '''Funzione helper per calcolare velocemente se il drone
        ha abbastanza batteria per la sortie'''
        return self.energy_on[i,j] + self.energy_off[j,h] <= DRONE_BATTERY_B
    
    def enough_range_robot(self, station: int, customer: int):
        '''Funzione helper per calcolare velocemente se il robot
        ha abbastanza batteria per la consegna'''
        trip_time = (ROBOT_LOAD_TIME
                    + self.time_R[station, customer]
                    + ROBOT_SERVICE_TIME
                    + self.time_R[customer, station])
        return trip_time <= ROBOT_MAX_DRIVE_TIME
        
    def precompute_big_m(self):
        """Precalcola i Big-M tight per ogni famiglia di vincoli.
        Valori negativi (vincolo naturalmente soddisfatto) vengono sostituiti
        con t_stop_ext[i], che è sempre positivo e molto più stretto di BIG_M.
        """
        n = self.n_nodes

        # Estende t_start / t_stop a tutti i nodi (stazioni e depositi usano max(t_stop))
        t_max = np.max(self.t_stop)
        t_start_ext, t_stop_ext = np.zeros(n), np.full(n, t_max)
        t_stop_ext [1 : self.n_customers + 1] = self.t_stop
        t_start_ext[1 : self.n_customers + 1] = self.t_start
        self.t_stop_ext  = t_stop_ext
        self.t_start_ext = t_start_ext

        fb_row = t_stop_ext[:, np.newaxis]  # fallback per clip, shape (n,1)
        fb_col = t_stop_ext[np.newaxis, :]  # fallback per clip, shape (1,n)

        # UB su d_T[i]: massimo tra fine servizio truck e fine attesa drone (Eq. 20-21)
        UB_dT = np.maximum(
            t_stop_ext + TRUCK_SERVICE_TIME,
            np.max(t_stop_ext[:, np.newaxis] + DRONE_SERVICE_TIME + self.time_D, axis=0),
        )
        self.UB_dT = UB_dT

        # M_T[i,j]  — vincoli di sincronizzazione truck (Eq. 18)
        M_T_raw = UB_dT[:, np.newaxis] + self.time_T - t_start_ext[np.newaxis, :]
        self.M_T = np.where(M_T_raw > 0, M_T_raw, fb_row)

        # M_node_T  — vincoli Eq. 19-23 (scalare per nodo i)
        self.M_node_T = np.maximum(
            t_stop_ext + max(TRUCK_SERVICE_TIME, DRONE_LOAD_TIME),
            t_max,
        )

        # M_D[i,j]  — sincronizzazione drone (Eq. 21, 24)
        M_D_raw = (np.maximum(t_stop_ext + DRONE_SERVICE_TIME, t_max)[:, np.newaxis]
                + self.time_D)
        self.M_D = np.where(M_D_raw > 0, M_D_raw, fb_row)

        # M_25[i,j,h]  — dipende solo da j e h; replicato su i (Eq. 25)
        M_25_jh = t_stop_ext[:, np.newaxis] + DRONE_SERVICE_TIME + self.time_D + DRONE_LOAD_TIME
        M_25_raw = np.tile(M_25_jh[np.newaxis, :, :], (n, 1, 1))
        self.M_25 = np.where(M_25_raw > 0, M_25_raw, 1.0)

        # Vincoli di finestra temporale (Eq. 30-35)
        self.M_tw30 = t_start_ext.copy()
        self.M_tw31 = float(TRUCK_SERVICE_TIME)

        M_tw32_raw = t_start_ext[np.newaxis, :] - self.time_D
        self.M_tw32 = np.where(M_tw32_raw > 0, M_tw32_raw, 1.0)

        M_tw33_raw = t_max + self.time_D + DRONE_SERVICE_TIME - t_stop_ext[np.newaxis, :]
        self.M_tw33 = np.where(M_tw33_raw > 0, M_tw33_raw, 1.0)

        M_tw34_raw = t_start_ext[np.newaxis, :] - ROBOT_LOAD_TIME - self.time_R
        self.M_tw34 = np.where(M_tw34_raw > 0, M_tw34_raw, fb_col)

        M_tw35_raw = (t_stop_ext[:, np.newaxis] + ROBOT_LOAD_TIME
                    + self.time_R + ROBOT_SERVICE_TIME - t_stop_ext[np.newaxis, :])
        self.M_tw35 = np.where(M_tw35_raw > 0, M_tw35_raw, fb_row)

        # Vincoli batteria robot (Eq. 26-27)
        self.M_26 = t_stop_ext + TRUCK_SERVICE_TIME
        self.M_27 = float(ROBOT_MAX_DRIVE_TIME)