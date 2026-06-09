# src/instance_generator.py
import numpy as np
from src.instance import Instance
from src.config import *
from scipy.spatial.distance import cdist

def generate_random_instance(
    n_customers: int,
    n_stations: int,
    area_km: float = 20.0,
    seed: int = 42
) -> Instance:
    """
    Genera un'istanza sintetica casuale del VRPD-RS-TW.
    Coordinate e distanze in km, tempi in secondi.
    """
    rng = np.random.default_rng(seed)
    n_nodes = n_customers + n_stations + 2

    # --- 1. Coordinate (n_nodes x 2), in km ---
    # Ordine: [deposito_start,  clienti ,  stazioni ,  deposito_end]
    # deposito_start e deposito_end hanno le STESSE coordinate (stesso luogo fisico)
    coords = np.zeros(shape = (n_nodes,2))
    coords[0,:] = [area_km/2, area_km/2]
    coords[1:-1,:] = rng.random(size = (n_nodes-2,2)) * area_km
    coords[n_nodes-1,:] = [area_km/2, area_km/2]

    # --- 2. Matrice distanze euclidee (n_nodes x n_nodes), in km ---
    # dist[i,j] = norma euclidea tra coords[i] e coords[j]
    dist = cdist(coords, coords, metric='euclidean')

    # --- 3. Matrici tempo TRUCK (n_nodes x n_nodes), in secondi ---
    TRUCK_SPEED_KMS = TRUCK_SPEED_KMH / 3600  # km/s
    time_truck = dist / TRUCK_SPEED_KMS

    # --- 4. Matrici tempo DRONE (n_nodes x n_nodes), in secondi ---
    # Il tempo totale drone include decollo + volo + atterraggio:
    time_drone_horizontal = 1000 * dist / DRONE_SPEED
    time_drone = DRONE_TAKEOFF_TIME + DRONE_LANDING_TIME + time_drone_horizontal
    np.fill_diagonal(time_drone, 0)

    # --- 5. Matrici tempo ROBOT (n_nodes x n_nodes), in secondi ---
    ROBOT_SPEED_KMS = ROBOT_SPEED_KMH / 3600  
    time_robot = dist / ROBOT_SPEED_KMS

    # --- 6. Matrici energia DRONE on/off (n_nodes x n_nodes), in KJ ---
    # Energia consumata durante il volo di andata, con pacco (on) e durante il ritorno, senza pacco (off)
    # Per semplicità abbiamo scelto di calcolare l'energia usando il peso medio dei pacchi
    mean_weight = (PARCEL_MAX_WEIGHT + PARCEL_MIN_WEIGHT)/2
    energy_on  = (DRONE_HOVER_POWER + DRONE_POWER_PER_KG * mean_weight) * time_drone
    energy_off = DRONE_HOVER_POWER * time_drone

    # --- 7. Peso del pacco q per ogni cliente (in kg) ---
    q = rng.uniform(PARCEL_MIN_WEIGHT, PARCEL_MAX_WEIGHT, size=n_customers)

    # --- 8. Finestre temporali per ogni cliente (in secondi) ---
    t_start = rng.uniform(0, DELIVERY_TIME_RANGE_S, size = n_customers)
    t_stop  = t_start + TIME_WINDOW_RANGE_S

    return Instance(
        n_customers=n_customers,
        n_stations=n_stations,
        coords = coords,
        dist_T=dist,
        time_T=time_truck,
        dist_D=dist,
        time_D=time_drone,
        dist_R=dist,
        time_R=time_robot,
        energy_on=energy_on,
        energy_off=energy_off,
        q=q,
        t_start=t_start,
        t_stop=t_stop,
        no_fly=set()  
    )