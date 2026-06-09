"""
test.py — VRPD-RS-TW (Campuzano et al., 2025)
==============================================
Esegue tutti gli algoritmi del paper sull'istanza reale di Trieste:
  · 3P-GMS-ILS  : metaeuristica (costruzione + diversificazione + intensificazione)
  · MILP        : modello esatto risolto con Gurobi (con/senza Valid Inequalities)

La soluzione della metaeuristica viene usata come warm-start per il MILP,
riducendo il tempo necessario a Gurobi per trovare la soluzione ottima.

Per la scalability analysis vedere scalability.py.
"""

import os, time
from data.trieste.visualize     import visualize_solution
from src.milp                   import solve, print_solution
from src.solution import milp_to_solution
from src.metaheuristics.framework import run_3p_gms_ils
from data.trieste.instance_trieste_generator import generate_trieste_istance

# ===========================================================================
# CONFIGURAZIONE  ← unico punto in cui modificare i parametri
# ===========================================================================

# --- Istanze ---
N_CUSTOMERS_MILP      = 5       # numero di clienti da servire (consigliato max 10)
N_CUSTOMERS_META      = 20      # numero di clienti per la metaeuristica (consigliato, max 100)
INSTANCE_SEED         = 69

# --- Metaeuristica ---
META_R             = 25       # numero di multi-start (restart dalla lista di soluzioni)
META_MAX_ITER      = 1000     # iterazioni massime dell'ILS
META_BETA          = 3        # parametro di sparsificazione; valori alti = spazio di ricerca più ampio

# --- MILP (Gurobi) ---
MILP_TIME_LIMIT    = 300      # secondi massimi per ogni run di Gurobi
MILP_VERBOSE       = True     # True → mostra il log completo di Gurobi a terminale

# --- Output ---
OUTPUT_DIR         = "maps_outputs"     # cartella in cui salvare le mappe HTML
PRINT_ROUTES       = True               # True → stampa le rotte in dettaglio dopo il MILP

# ===========================================================================


def run_meta(inst):
    """Esegue la metaeuristica 3P-GMS-ILS e restituisce (soluzione, tempo)."""
    print("=" * 55)
    print("  3P-GMS-ILS — metaeuristica")
    print("=" * 55)
    t0  = time.time()
    sol = run_3p_gms_ils(inst, R=META_R, max_iterations=META_MAX_ITER, beta=META_BETA)
    elapsed = time.time() - t0
    print(f"\n  Costo: {sol.cost:.4f}   Tempo: {elapsed:.1f}s\n")
    return sol, elapsed


def run_milp(inst, label, vi=False, warm_start=None):
    """
    Esegue il MILP con le opzioni specificate.

    label      : stringa identificativa per la tabella risultati
    vi         : False / True / [1,2,..] — valid inequalities da aggiungere
                 False = nessuna  |  True = tutte (1-8)  |  lista = sottoinsieme
    warm_start : soluzione iniziale (da metaeuristica) da passare a Gurobi
    """
    print(f"  MILP: {label}  ", end="", flush=True)
    t0 = time.time()
    m, x_T, x_D, x_R = solve(inst,
                               verbose=MILP_VERBOSE,
                               use_valid_inequalities=vi,
                               warm_start=warm_start,
                               max_time=MILP_TIME_LIMIT, tuned=True)
    elapsed = time.time() - t0
    print(f"  ObjVal={m.ObjVal:.4f}  Gap={m.MIPGap:.2%}  ({elapsed:.0f}s)")
    return m, x_T, x_D, x_R, elapsed


def print_results(results, meta_cost, meta_time):
    """
    Stampa la tabella comparativa: meta + configurazioni MILP.
 
    results   : dict  label → (ObjVal, MIPGap, tempo_s)
    meta_cost : float costo soluzione 3P-GMS-ILS
    meta_time : float tempo di esecuzione 3P-GMS-ILS (secondi)
    """
    print(f"\n{'═'*62}")
    print(f"  {'Algoritmo':<32} {'ObjVal':>9} {'Gap%':>7} {'Tempo':>8}")
    print(f"  {'-'*58}")
 
    print(f"  {'3P-GMS-ILS':<32} {meta_cost:>9.4f} {'—':>7} {meta_time:>7.1f}s")
 
    for label, (objval, gap, t) in results.items():
        print(f"  {label:<32} {objval:>9.4f} {gap*100:>6.2f}% {t:>7.0f}s")
 
    best_milp = min(v[0] for v in results.values())
    gap_meta  = (meta_cost - best_milp) / best_milp * 100
    print(f"  {'-'*58}")
    print(f"  Gap metaeuristica vs miglior MILP: {gap_meta:+.2f}%")
    print(f"{'═'*62}\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Carica istanza reale di Trieste
    # Può richiedere del tempo per il caricamento dei dati, salva
    # in cache i dati pre-calcolati (per i run successivi con lo stesso n_customers)
    inst_meta = generate_trieste_istance(n_customers=N_CUSTOMERS_META, seed = INSTANCE_SEED)

    # 2. Istanza Metaeuristica
    sol_meta, _ = run_meta(inst_meta)
    visualize_solution(sol_meta, inst_meta, output_path=os.path.join(OUTPUT_DIR, f"solution_meta_trieste_{N_CUSTOMERS_META}.html"))

    # 3. Istanza MILP in tre configurazioni + metaeuristica
    inst_milp = generate_trieste_istance(n_customers=N_CUSTOMERS_MILP)  
    print("=" * 55)
    print("  MILP — Gurobi")
    print("=" * 55)
    results = {}
    
    sol_ws, t_ws = run_meta(inst_milp)
 
    print(f"\n{'─'*55}")
    print("  MILP — Gurobi")
    print(f"{'─'*55}")
 
    m, *_, t = run_milp(inst_milp, "base (no VI)")
    results["MILP base"] = (m.ObjVal, m.MIPGap, t)
 
    m, *_, t = run_milp(inst_milp, "VI 1-5 (paper)", vi=[1, 2, 3, 4, 5])
    results["MILP + VI 1-5 (paper)"] = (m.ObjVal, m.MIPGap, t)
 
    m, xT, xD, xR, t = run_milp(inst_milp, "VI 1-8 + warm-start",
                                  vi=True, warm_start=sol_ws)
    results["MILP + VI 1-8 + warm-start"] = (m.ObjVal, m.MIPGap, t)
 
    # Visualizza e stampa la soluzione della configurazione migliore
    sol_milp = milp_to_solution(m, inst_milp)
    visualize_solution(sol_milp, inst_milp, output_path=os.path.join(OUTPUT_DIR, f"solution_milp_trieste_{N_CUSTOMERS_MILP}.html"))
    
    if PRINT_ROUTES:
        print_solution(inst_milp, m, xT, xD, xR)
 
    print_results(results, sol_ws.cost, t_ws)
 


if __name__ == "__main__":
    main()
