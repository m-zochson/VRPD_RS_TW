# src/metaheuristic/framework.py
from src.metaheuristics.evaluate import evaluate_solution
from src.solution import Solution
from src.instance import Instance
from src.config import MAX_MULTISTART_R, BETA, MAX_ITERATIONS, GUROBI_TIME_LIMIT
from src.metaheuristics.granular import reset_edges
from src.metaheuristics.construction import phase_I
from src.metaheuristics.diversification import phase_II
from src.metaheuristics.intensification import phase_III
import random
import time

def run_3p_gms_ils(
    inst: Instance,
    R: int = MAX_MULTISTART_R,
    max_iterations: int = MAX_ITERATIONS,
    beta: float = BETA,
    seed: int | None = None,
    track_history: bool = False,
    verbose : bool = False
) -> "Solution | tuple[Solution, dict]":
    """
    Algorithm 1 del paper — framework principale 3P-GMS-ILS.
 
    Args:
        inst:            istanza del problema
        R:               numero massimo di restart (criterio di stop)
        max_iterations:  numero massimo di iterazioni totali
        beta:            parametro di sparsificazione granulare
        seed:            seed per la riproducibilità. Se None, ne genera uno casuale.
        track_history:   se True, restituisce (incumbent, history_dict) invece di incumbent
 
    Returns:
        Se track_history=False: incumbent (Solution)
        Se track_history=True:  (incumbent, history) dove history è un dict con:
            "iteration_costs"  : [float]         costo current ad ogni iterazione
            "incumbent_costs"  : [float]         costo incumbent (non-crescente)
            "elapsed_times"    : [float]         secondi dall'inizio
            "phase_markers"    : [(int, str)]    (iterazione, fase) quando l'incumbent migliora
            "phase_I_cost"     : float           costo dopo la costruzione iniziale
    """
    if seed is None:
        seed = random.randint(1, 10_000)
    random.seed(seed)
    if verbose: print(f"Seed: {seed}")
 
    restarting_list: list[Solution] = []
    time_start = time.time()
 
    # ── Phase I: costruzione soluzione iniziale ──────────────────────────────
    current = phase_I(inst,gurobi_time_limit=GUROBI_TIME_LIMIT, verbose = verbose)
    evaluate_solution(current, inst)
    if verbose: print(f"[Phase I]   cost = {current.cost:.2f}, "
                      f"time spent = {(time.time() - time_start):.2f}s")
    incumbent = current.copy()
 
    granular_set = reset_edges(current, inst, beta)
    i, p, k = 0, 0, 0
 
    # ── Inizializzazione history ─────────────────────────────────────────────
    if track_history:
        history: dict = {
            "phase_I_cost":     incumbent.cost,
            "iteration_costs":  [],
            "incumbent_costs":  [],
            "elapsed_times":    [],
            "phase_markers":    [],   
        }
 
    # ── Loop principale, fasi II e III──────────────────────────────────────────────────
    while p < R and i < max_iterations:
        incumbent_before_II = incumbent.cost

        # Fase II
        current, incumbent, granular_set, k, p = phase_II(
            restarting_list, granular_set, beta, incumbent, current, inst, k, p, verbose = verbose
        )
 
        if track_history and incumbent.cost < incumbent_before_II:
            history["phase_markers"].append((i, "II"))
 
        incumbent_before_III = incumbent.cost

        # Fase III
        current, incumbent, k = phase_III(
            granular_set, beta, incumbent, current, inst, k
        )
 
        if track_history and incumbent.cost < incumbent_before_III:
            history["phase_markers"].append((i, "III"))
 
        # Registra stato a fine iterazione
        if track_history:
            history["iteration_costs"].append(current.cost)
            history["incumbent_costs"].append(incumbent.cost)
            history["elapsed_times"].append(round(time.time() - time_start, 3))
 
        i += 1
 
    # ──Debug ──────────────────────────────────────────────────────
    if incumbent.cost == float("inf"):
        if verbose: print("Warning: incumbent solution is infeasible.")
        evaluate_solution(incumbent, inst, verbose=True)
 
    # Rimuovi rotte vuote
    incumbent.routes = [r for r in incumbent.routes if len(r.nodes) >= 3]
 
    if verbose: print(f"Terminated after {i} iterations and {p} restarts. "
                      f"Total time = {(time.time() - time_start):.2f}s")
 
    if track_history:
        return incumbent, history
    return incumbent