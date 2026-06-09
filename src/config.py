# src/config.py
# Parametri globali del VRPD-RS-TW
# Riferimento: Campuzano et al. (2025), Section 5.1

# ---------------------------------------------------------------
# PARAMETRI DEL TRUCK
# ---------------------------------------------------------------
TRUCK_CAPACITY_Q = 1000             # Capacità massima del truck (kg)
TRUCK_SERVICE_TIME = 30             # Tempo di servizio del truck al cliente (secondi)
TRUCK_COST_PER_KM = 0.8             # Costo per km percorso dal truck (€/km)
TRUCK_DRIVER_WAGE = 25/3600         # Salario orario del conducente (€/s)
TRUCK_SPEED_KMH = 25                # velocità media urbana del truck (km/h)

# ---------------------------------------------------------------
# PARAMETRI DEL DRONE
# ---------------------------------------------------------------
DRONE_BATTERY_B = 1000              # Capacità batteria drone (KJ) - vedi paper
DRONE_SPEED = 40/3.6                # v_D velocità orrizontale (m/s)
DRONE_TAKEOFF_TIME = 60             # t_tk tempo di decollo (secondi)
DRONE_LANDING_TIME = 30             # t_l tempo di atterraggio (secondi)
DRONE_SERVICE_TIME = 30             # t_D tempo di servizio drone (secondi)
DRONE_LOAD_TIME = 20                # t_L tempo di carico del drone (secondi)   
DRONE_COST_PER_KM = 0.5             # Costo per km percorso dal drone (€/km)
DRONE_HOVER_POWER = 2               # p_h(0) potenza hover a carico 0 (KJ/s)
DRONE_POWER_PER_KG = 0.17           # potenza aggiuntiva per kg trasportato (KJ/(kg·s))

# ---------------------------------------------------------------
# PARAMETRI DEL ROBOT
# ---------------------------------------------------------------
ROBOT_MAX_DRIVE_TIME = 1500         # Tempo massimo di guida del robot (secondi)
ROBOT_SERVICE_TIME = 40             # Tempo di consegna del robot al cliente (secondi)
ROBOT_COST_PER_KM = 0.3             # Costo per km percorso dal robot (€/km)
ROBOT_LOAD_TIME = 10                # Tempo di carico del robot (secondi)
ROBOT_SPEED_KMH = 6                 # Velocità media del robot (km/h)

# ---------------------------------------------------------------
# PARAMETRI DELL'ISTANZA
# ---------------------------------------------------------------
MAX_TRUCKS_K = 20                   # Numero massimo di truck disponibili
MAX_ROBOTS_R = 3                    # Numero di robot per stazione
PARCEL_MIN_WEIGHT = 0.1             # Peso minimo di un pacco (kg)
PARCEL_MAX_WEIGHT = 8               # Peso massimo di un pacco (kg)

#----------------------------------------------------------------
# PARAMETRI MILP
#----------------------------------------------------------------
TIME_WINDOW_RANGE_S = 1.5 * 3600    # Range finestre temporali cliente(secondi)
DELIVERY_TIME_RANGE_S = 8 * 3600    # Range massimo della finestra temporale di consegna (secondi)
BIG_M = (DELIVERY_TIME_RANGE_S + TIME_WINDOW_RANGE_S) * 1.1

#----------------------------------------------------------------
# PARAMETRI METAEURISTICA
#----------------------------------------------------------------
BETA = 3                            # Parametro di sparsificazione
K_TOP = 15                          # Numero di migliori nodi da considerare per le mosse di inserimento stazione
MAX_MULTISTART_R = 25               # Numero di restart dell'algoritmo
MAX_ITERATIONS = 1000               # Numero massimo di iterazioni per la fase II e III
GUROBI_TIME_LIMIT = 10              # Tempo massimo per ricerca soluzione iniziale (fase I) in secondi

