# src/metaheuristic/intensification.py

import random
from src.solution import Solution
from src.instance import Instance
from src.metaheuristics.granular import GranularSet, add_edges, reset_edges
from src.metaheuristics.neighborhoods import (
    push_left, push_right, two_opt,
    exchange_11, exchange_22,
    reinsertion, exchange_drone_robot,
    exchange_robot_stations
)

ALL_NEIGHBORHOODS = [
    push_left, push_right, two_opt,
    exchange_11, exchange_22,
    reinsertion, exchange_drone_robot,
    exchange_robot_stations
]

def phase_III(
    granular_set: GranularSet,
    beta: float,
    incumbent: Solution,        
    current: Solution,          
    inst: Instance,
    k: int
) -> tuple[Solution, Solution, int]:
    """
    Algorithm 4 del paper — Phase III: Intensification.
    
    Esplora i neighborhood in ordine randomizzato finché non trova
    un ottimo locale. Aggiorna incumbent e granular_set in-place.
    
    Restituisce (current, incumbent, k) aggiornati.
    """
    
    f_prev = float("inf")  
    
    while current.cost < f_prev:
        sol = current.copy()
        f_prev = current.cost
        
        neighborhoods = ALL_NEIGHBORHOODS.copy()
        random.shuffle(neighborhoods)
        for neighborhood in neighborhoods:  
            
            debug_sol = current.copy()
            neighborhood(current, inst, granular_set)
            if not current.check_solution_integrity(neighborhood.__name__):
                print(f"Integrity check failed after neighborhood {neighborhood.__name__}.")
                print(f"Before move: {debug_sol}")
                print(f"After move: {current}")
                raise ValueError(f"Neighborhood {neighborhood.__name__} produced an invalid solution.")
            if not current.check_no_duplicates(neighborhood.__name__):
                print(f"Prima della mossa {neighborhood.__name__}, soluzione: {debug_sol}")
                print(f"Dopo la mossa {neighborhood.__name__}, soluzione: {current}")
                raise ValueError(f"Neighborhood {neighborhood.__name__} generated a solution with duplicate nodes.")
                
            if not isinstance(current, Solution):
                print(f"Warning: neighborhood {neighborhood.__name__} did not return a Solution object.")
            
            if current.cost <= sol.cost:
                add_edges(current, inst, granular_set)
                break

        
        if current.cost < incumbent.cost:
            reset_edges(current, inst, beta)
            incumbent = current.copy()
            k = 0
        else:
            k = k + 1
    return current, incumbent, k