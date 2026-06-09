# src/metaheuristic/diversification.py
import random
from src.solution import Solution
from src.instance import Instance
from src.metaheuristics.granular import GranularSet, reset_edges
from src.metaheuristics.shaking import shaking_1, shaking_2, shaking_3, shaking_4, shaking_5, shaking_6
from src.config import MAX_MULTISTART_R

ALL_SHAKINGS = [shaking_1, shaking_2, shaking_3, shaking_4, shaking_5, shaking_6]

def phase_II(
    restarting_list: list[Solution],
    granular_set: GranularSet,
    beta: float,
    incumbent: Solution,       
    current: Solution,         
    inst: Instance,
    k: int,
    p: int,
    verbose: bool = False
) -> tuple[Solution, Solution, GranularSet, int, int]:
    """
    Algorithm 3 del paper — Phase II: Diversification.
    Restituisce (current, incumbent, granular_set, k, p) aggiornati.
    L'obiettivo è uscire dai minimi locali applicando shakings casuali.
    """
    random.shuffle(ALL_SHAKINGS)
    n_shakings = len(ALL_SHAKINGS)

    if k < n_shakings:
        # --- Modalità A: shaking sequenziale ---
        while k < n_shakings:
            shaking_fn = ALL_SHAKINGS[k]
            new_sol, feasible = shaking_fn(current, inst)
            if feasible:
                restarting_list.append(current)
                current = new_sol
                break
            else:
                k += 1
    else:
        # --- Modalità B: multi-start ---
        if not restarting_list:
            p += 1
            if verbose:print(f"Restart {p}/{MAX_MULTISTART_R}")
            k = 0
            return current, incumbent, granular_set, k, p
        current_copy = current.copy()
        current = restarting_list.pop()
        k = random.randint(0,5)
        shaking_fn = ALL_SHAKINGS[k]
        current, feasible = shaking_fn(current,inst)
        current.check_no_duplicates(shaking_fn.__name__)
        restarting_list.append(current_copy)
        k = 0
        p += 1
        if verbose:print(f"Restart {p}/{MAX_MULTISTART_R}")

    # --- Dopo entrambe le modalità ---
    granular_set = reset_edges(current,inst,beta)
    if current.cost < incumbent.cost: incumbent=current
    
    return current, incumbent, granular_set, k, p