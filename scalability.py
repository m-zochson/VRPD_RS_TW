# scalability.py
"""
Runner centralizzato per tutti gli esperimenti VRPD-RS-TW.

Esempi di uso:
    python scalability.py --exp E1                  # esegui E1
    python scalability.py --exp E1 E2 E4            # esegui più esperimenti
    python scalability.py --exp all                 # tutti gli esperimenti
    python scalability.py --exp E1 --dry-run        # mostra combinazioni senza eseguire
    python scalability.py --exp E1 --force          # riesegui anche combinazioni già nel CSV

Ogni esperimento scrive in results/csv/.
Se il processo viene interrotto e riavviato, riprende da dove era rimasto.

Elenco esperimenti:
  E1  — Scalabilità MILP: tempo e gap al crescere di n_customers (3→12)
  E2  — Valid inequalities: confronto 4 configurazioni su istanze n=5,7,9,11
  E3  — Scalabilità metaeuristica: tempo e qualità al crescere di n_customers (3→12)
  E4  — MILP vs Metaeuristica: qualità e tempo su n=5..9, stazioni=2,3,5
  E5  — Convergenza: curva costo vs iterazioni su istanza small e medium
  E6  — Warm start: MILP con/senza soluzione iniziale della metaeuristica
  E7  — Sensitivity β: effetto della soglia granular su n=25 clienti
  E8  — Sensitivity R: effetto del numero di multi-start su n=5 clienti
  E9  — Istanza reale Trieste: 5 configurazioni modali (default, no drone, ...)
  E10 — Ablation study: valore marginale di drone/robot vs truck-only
  E11 — Robustezza al seed: 30 run su istanza fissa (n=20, stazioni=3)
"""

import argparse
import random
import time
from typing import Any
import copy
import numpy as np

# ── Import progetto ──────────────────────────────────────────────────────────
from src.instance_generator import generate_random_instance
from src.milp import solve
from src.metaheuristics.framework import run_3p_gms_ils
from src.solution import Solution
from src.instance import Instance
from src.config import *
from src.helpers.results_manager import (
    append_row, is_already_done, setup_logger,
    extract_milp_metrics, make_run_id, ensure_dirs, summary,
)
from gurobipy import GRB

# ── Configurazioni VI ────────────────────────────────────────────────────────
# Mappa stringa → kwargs da passare a solve()
VI_CONFIGS: dict[str, dict] = {
    "none":        {"use_valid_inequalities": False, "tuned": True},
    "vi_paper":    {"use_valid_inequalities": [1, 2, 3, 4, 5], "tuned": True},
    "vi_custom":   {"use_valid_inequalities": [6, 7, 8], "tuned": True},
    "vi_all":      {"use_valid_inequalities": True, "tuned": True},
}

# ── Parametri degli esperimenti ───────────────────────────────────────────────
# Modificare qui per cambiare le combinazioni senza toccare le funzioni.
EXP_PARAMS = {
    "E1": {
        "n_customers":  [3,4,5,6,7,8,9,10,11,12],
        "n_stations":   [2],
        "seeds":        [99],
        "time_limit":   3600,
        "vi_config":    "vi_all",
        "area_km":      10.0,
    },
    "E2": {
        "n_customers":  [5, 7, 9, 11],
        "n_stations":   [2],
        "seeds":        [42, 99, 123],
        "time_limit":   300,
        "vi_configs":   ["none", "vi_paper", "vi_custom", "vi_all"],
        "area_km":      10.0,
    },
    "E3": {
    "n_customers": [5,10,15,20,25,30],
    "n_stations":  [2],
    "seeds":       [42,99,123],
    "meta_reps":   10,           
    "R":           MAX_MULTISTART_R,
    "beta":        BETA,
    "area_km":     10.0,
    },
    "E4": {
        "n_customers":  [5, 6, 7,8,9],
        "n_stations":   [2, 3, 5],
        "seeds":        [42, 99, 123],
        "milp_time_limit": 1800,
        "meta_reps":    5,
        "R":            MAX_MULTISTART_R,
        "beta":         BETA,
        "area_km":      10.0,
    },
    "E5": {   
        "instances": [
            {"n_customers": 8,  "n_stations": 2, "seed": 42, "label": "small"},
            {"n_customers": 15, "n_stations": 2, "seed": 42, "label": "medium"},
        ],
        "meta_seeds": [42, 99, 123],
        "R":          MAX_MULTISTART_R,
        "beta":       BETA,
        "area_km":    10.0,
    },
    "E6": {
        "n_customers":  [7, 10, 12, 15],
        "n_stations":   [2],
        "seeds":        [42, 99, 123, 321],
        "time_limit":   600,
        "configs":      ["milp_base", "milp_ws", "milp_vi", "milp_ws_vi"],
        "area_km":      10.0,
    },
    "E7": {
    "instances": [
        {"n_customers": 25, "n_stations": 4, "seed": 42},
    ],
    "betas":      [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    "meta_seeds": list(range(10,15)),
    "R":          25,
    "area_km":    10.0,
    },
    "E8": {
        "instances": [
        {"n_customers": 5, "n_stations": 2, "seed": 42},
    ],
        "R_values":   [3, 5, 10, 17, 25, 50, 100],
        "meta_seeds": list(range(10,15)),
        "beta":       3,
        "area_km":    10.0,
    },
    "E9": {   # istanza reale Trieste — gestita separatamente
        "meta_seeds": [42, 99, 123, 345, 456],
        "configs":    ["default", "no_drone", "no_robot", "no_no_fly", "low_battery"],
        "R":          MAX_MULTISTART_R,
        "beta":       BETA,
    },
    "E10": {
        "n_customers":  [15],
        "n_stations":   [4],
        "seeds":        [42, 99, 123],
        "configs":      ["all", "no_drone", "no_robot", "truck_only"],
        "meta_reps":    10,
        "R":            MAX_MULTISTART_R,
        "beta":         BETA,
        "area_km":      10.0,
    },
    "E11": {
        "n_customers":  20,
        "n_stations":   3,
        "inst_seed":    420,
        "meta_seeds":   list(range(30)),
        "R":            40,
        "beta":         3,
        "area_km":      10.0,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  FUNZIONI BASE
# ═══════════════════════════════════════════════════════════════════════════════

def run_milp_experiment(
    inst: Instance,
    vi_config: str = "none",
    time_limit: int = 600,
    warm_start: Solution | None = None,
) -> dict[str, Any]:
    """
    Esegue una singola run del MILP e restituisce un dizionario di metriche.

    Args:
        inst:        istanza del problema
        vi_config:   chiave di VI_CONFIGS
        time_limit:  time limit Gurobi in secondi
        warm_start:  soluzione iniziale (opzionale)

    Returns:
        Dict con tutte le metriche estratte via extract_milp_metrics()
        più n_vars_bin e n_vars_cont per la dimensione del modello.
    """
    kwargs = VI_CONFIGS[vi_config].copy()
    m, x_T, x_D, x_R = solve(
        inst,
        verbose=True,
        max_time=time_limit,
        warm_start=warm_start,
        **kwargs,
    )
    metrics = extract_milp_metrics(m)
    metrics["vi_config"] = vi_config
    return metrics


def run_meta_experiment(
    inst: Instance,
    meta_seed: int,
    R: int = MAX_MULTISTART_R,
    beta: float = BETA,
    max_iterations: int = MAX_ITERATIONS,
) -> dict[str, Any]:
    """
    Esegue una singola run della metaeuristica e restituisce metriche.

    Imposta random.seed(meta_seed) prima di chiamare il framework,
    rendendo il risultato deterministico per quel seed.

    Returns:
        Dict con: cost, runtime_s, meta_seed
    """
    random.seed(meta_seed)
    t0 = time.time()
    sol = run_3p_gms_ils(inst, R=R, max_iterations=max_iterations,
                         beta=beta, seed=meta_seed, verbose=False)
    elapsed = time.time() - t0
    return {
        "meta_seed":  meta_seed,
        "cost":       round(sol.cost, 4),
        "runtime_s":  round(elapsed, 2),
        "n_routes":   len(sol.routes),
        **count_modal_split(sol, inst),
    }



def count_modal_split(sol: Solution, inst: Instance) -> dict[str, int]:
    """
    Conta quanti clienti sono serviti da ciascuna modalità.

    Returns:
        {"n_truck": int, "n_drone": int, "n_robot": int}
    """
    n_truck = n_drone = n_robot = 0
    for route in sol.routes:
        for node, label in zip(route.nodes, route.labels):
            if 1 <= node <= inst.n_customers:
                n_truck += 1
        n_drone += len(route.sorties)
        for robot in route.robots:
            n_robot += len(robot.customers)
    return {"n_truck": n_truck, "n_drone": n_drone, "n_robot": n_robot}

# ── Loader Trieste (lazy, caricato una volta sola) ────────────────────────────

_trieste_graphs: dict | None = None

def _get_trieste_graphs() -> dict:
    """Carica i grafi OSM di Trieste una volta sola, poi li riusa."""
    global _trieste_graphs
    if _trieste_graphs is None:
        from data.trieste.loader import get_drive_graph, get_walk_graph
        _trieste_graphs = {
            "drive": get_drive_graph(),
            "walk":  get_walk_graph(),
        }
    return _trieste_graphs


def _build_trieste_instance_base(n_customers: int | None = None,
                                  n_stations:  int | None = None,
                                  seed: int = 42):
    """
    Costruisce l'istanza base di Trieste con le coordinate reali.
    n_customers / n_stations: se None, usa tutti quelli in locations.json.
    """
    from data.trieste.node_selector  import get_node_ids, load_locations
    from data.trieste.matrix_builder import build_real_instance

    graphs    = _get_trieste_graphs()
    locations = load_locations(n_customers=n_customers, n_stations=n_stations)
    node_ids  = get_node_ids(
        graphs["drive"], graphs["walk"],
        n_customers=n_customers, n_stations=n_stations,
    )
    return build_real_instance(
        graphs["drive"], graphs["walk"], node_ids, locations, seed=seed
    ), node_ids, locations


def _apply_trieste_config(inst_base, config: str):
    """
    Ritorna una copia dell'istanza modificata per la configurazione richiesta.

    Configs disponibili:
        "default"    — nessuna modifica
        "no_robot"   — forza tempo robot oltre il range: nessun robot ammesso
        "no_no_fly"  — rimuove tutte le zone no-fly
        "low_battery"— dimezza la capacità effettiva raddoppiando i consumi
    """
    inst = copy.deepcopy(inst_base)

    if config == "default":
        pass

    elif config == "no_drone":
        # Qualunque sortie viola il vincolo di batteria → nessun drone usato
        inst.energy_on  = np.full_like(inst.energy_on,  DRONE_BATTERY_B + 1.0)
        inst.energy_off = np.full_like(inst.energy_off, DRONE_BATTERY_B + 1.0)

    elif config == "no_robot":
        # Qualunque trip supera il range massimo → nessun robot usato
        inst.time_R = np.full_like(inst.time_R, ROBOT_MAX_DRIVE_TIME + 1.0)
        inst.dist_R = np.full_like(inst.dist_R, ROBOT_MAX_DRIVE_TIME + 1.0)

    elif config == "no_no_fly":
        inst.no_fly = set()

    elif config == "low_battery":
        # Raddoppio dei consumi energetici → batteria effettiva dimezzata
        inst.energy_on  = inst.energy_on  * 2.0
        inst.energy_off = inst.energy_off * 2.0

    else:
        raise ValueError(f"Config Trieste sconosciuta: '{config}'")

    # Ricalcola big_M se l'istanza è stata modificata (necessario per MILP)
    if config != "default":
        inst.precompute_big_m()

    return inst

def _apply_synthetic_config(inst: Instance, config: str) -> Instance:
    """
    Ritorna una copia dell'istanza sintetica modificata per la configurazione richiesta.
    Analogo di _apply_trieste_config per istanze generate da generate_random_instance.

    Configs disponibili:
        "all"        — nessuna modifica (tutte le modalità attive)
        "no_drone"   — energia drone oltre soglia: nessun drone ammesso
        "no_robot"   — tempo robot oltre range massimo: nessun robot ammesso
        "truck_only" — sia drone che robot disabilitati
    """
    if config == "all":
        return inst

    inst = copy.deepcopy(inst)

    if config == "no_drone":
        inst.energy_on  = np.full_like(inst.energy_on,  DRONE_BATTERY_B + 1.0)
        inst.energy_off = np.full_like(inst.energy_off, DRONE_BATTERY_B + 1.0)

    elif config == "no_robot":
        inst.time_R = np.full_like(inst.time_R, ROBOT_MAX_DRIVE_TIME + 1.0)
        inst.dist_R = np.full_like(inst.dist_R, ROBOT_MAX_DRIVE_TIME + 1.0)

    elif config == "truck_only":
        inst.energy_on  = np.full_like(inst.energy_on,  DRONE_BATTERY_B + 1.0)
        inst.energy_off = np.full_like(inst.energy_off, DRONE_BATTERY_B + 1.0)
        inst.time_R = np.full_like(inst.time_R, ROBOT_MAX_DRIVE_TIME + 1.0)
        inst.dist_R = np.full_like(inst.dist_R, ROBOT_MAX_DRIVE_TIME + 1.0)

    else:
        raise ValueError(f"Config E10 sconosciuta: '{config}'")

    inst.precompute_big_m()
    return inst

# ═══════════════════════════════════════════════════════════════════════════════
#  ESPERIMENTI
# ═══════════════════════════════════════════════════════════════════════════════

def exp_E1(dry_run: bool = False, force: bool = False) -> None:
    """
    E1 — Scalabilità del MILP.
    Mostra come il tempo di risoluzione esplode al crescere di n.
    """
    p = EXP_PARAMS["E1"]
    combos = [
        (n, s)
        for n in p["n_customers"]
        for s in p["seeds"]
    ]
    print(f"\n[E1] {len(combos)} combinazioni × 1 config = {len(combos)} run")
    if dry_run:
        for n, s in combos:
            print(f"  n={n:2d}  seed={s:3d}  vi={p['vi_config']}  tl={p['time_limit']}s")
        return

    for n, seed in combos:
        key = {"n_customers": n, "seed": seed}
        if not force and is_already_done("E1", key):
            print(f"  [skip] n={n} seed={seed} già nel CSV")
            continue

        run_id = make_run_id(f"n{n}", f"seed{seed}")
        log = setup_logger("E1", run_id)
        log.info(f"Avvio n={n} n_stat={p['n_stations'][0]} seed={seed}")

        inst = generate_random_instance(n, p["n_stations"][0], p["area_km"], seed)
        metrics = run_milp_experiment(inst, p["vi_config"], p["time_limit"])

        row = {
            "n_customers": n,
            "n_stations":  p["n_stations"][0],
            "seed":        seed,
            **metrics,
        }
        append_row("E1", row)
        log.info(f"  status={metrics['status']}  obj={metrics['obj_val']}  "
                 f"gap={metrics['mip_gap_pct']}%  t={metrics['runtime_s']}s")


def exp_E2(dry_run: bool = False, force: bool = False) -> None:
    """
    E2 — Effetto delle Valid Inequalities sul gap.
    Per ogni istanza, 5 configurazioni di VI.
    """
    p = EXP_PARAMS["E2"]
    combos = [
        (n, s, vc)
        for n  in p["n_customers"]
        for s  in p["seeds"]
        for vc in p["vi_configs"]
    ]
    n_inst = len(p["n_customers"]) * len(p["seeds"])
    print(f"\n[E2] {n_inst} istanze × {len(p['vi_configs'])} config = {len(combos)} run")
    if dry_run:
        for n, s, vc in combos:
            print(f"  n={n:2d}  seed={s:3d}  vi={vc}  tl={p['time_limit']}s")
        return

    for n, seed, vi_config in combos:
        key = {"n_customers": n, "seed": seed, "vi_config": vi_config}
        if not force and is_already_done("E2", key):
            print(f"  [skip] n={n} seed={seed} vi={vi_config}")
            continue

        run_id = make_run_id(f"n{n}", f"seed{seed}", vi_config)
        log = setup_logger("E2", run_id)
        log.info(f"Avvio n={n} seed={seed} vi={vi_config}")

        inst = generate_random_instance(n, p["n_stations"][0], p["area_km"], seed)
        metrics = run_milp_experiment(inst, vi_config, p["time_limit"])

        row = {"n_customers": n, "n_stations": p["n_stations"][0],
               "seed": seed, **metrics}
        append_row("E2", row)
        log.info(f"  {metrics['status']}  obj={metrics['obj_val']}  "
                 f"gap={metrics['mip_gap_pct']}%  bound={metrics['obj_bound']}")


def exp_E3(dry_run: bool = False, force: bool = False) -> None:
    """
    E3 — Scalabilità della metaeuristica.
    Speculare a E1: mostra come il tempo di esecuzione cresce con n,
    con media e std su meta_reps run indipendenti.
    """
    p = EXP_PARAMS["E3"]
    meta_seeds = list(range(p["meta_reps"]))
    combos = [
        (n, s, ms)
        for n  in p["n_customers"]
        for s  in p["seeds"]
        for ms in meta_seeds
    ]
    print(f"\n[E3] {len(combos)} run "
          f"({len(p['n_customers'])} taglie × {len(p['seeds'])} seed × {p['meta_reps']} rep)")
    if dry_run:
        for n, s, ms in combos:
            print(f"  n={n:2d}  inst_seed={s}  meta_seed={ms}")
        return

    for n, seed, meta_seed in combos:
        key = {"n_customers": n, "seed": seed, "meta_seed": meta_seed}
        if not force and is_already_done("E3", key):
            print(f"  [skip] n={n} seed={seed} meta_seed={meta_seed} già nel CSV")
            continue

        run_id = make_run_id(f"n{n}", f"seed{seed}", f"meta{meta_seed}")
        log = setup_logger("E3", run_id)
        log.info(f"Avvio n={n} n_stat={p['n_stations'][0]} "
                 f"inst_seed={seed} meta_seed={meta_seed}")

        inst = generate_random_instance(n, p["n_stations"][0], p["area_km"], seed)
        m    = run_meta_experiment(inst, meta_seed, p["R"], p["beta"])

        row = {
            "n_customers": n,
            "n_stations":  p["n_stations"][0],
            "seed":        seed,
            **m,
        }
        append_row("E3", row)
        log.info(f"  cost={m['cost']}  t={m['runtime_s']}s")


def exp_E4(dry_run: bool = False, force: bool = False) -> None:
    """
    E4 — MILP vs Metaeuristica.
    Per ogni istanza: 1 MILP (con VI e tuning) + n_reps run meta con seed diversi.
    """
    p = EXP_PARAMS["E4"]
    combos = [(n, station, s) for n in p["n_customers"] for station in p["n_stations"] for s in p["seeds"]]
    n_total = len(combos) * (1 + p["meta_reps"])
    print(f"\n[E4] {len(combos)} istanze × (1 MILP + {p['meta_reps']} meta) = {n_total} run")
    if dry_run:
        for n, n_stations, s in combos:
            print(f"  n={n:2d}  n_stations = {n_stations} seed={s:3d}  → 1 MILP + {p['meta_reps']} meta")
        return

    for n, n_stations, seed in combos:
        inst = generate_random_instance(n, n_stations, p["area_km"], seed)

        # --- MILP ---
        milp_key = {"n_customers": n,"n_stations": n_stations, "seed": seed, "run_type": "milp"}
        if force or not is_already_done("E4", milp_key):
            log = setup_logger("E4", make_run_id(f"n{n}", f"n_stations{n_stations}",f"seed{seed}", "milp"))
            log.info(f"MILP n={n} n_stations = {n_stations} seed={seed}")
            metrics = run_milp_experiment(inst, "vi_all", p["milp_time_limit"])
            row = {"n_customers": n, "n_stations": n_stations,
                   "seed": seed, "run_type": "milp", **metrics}
            append_row("E4", row)
            log.info(f"  MILP done: {metrics['status']} obj={metrics['obj_val']}")

        # --- Meta (n_reps ripetizioni con meta_seed diversi) ---
        meta_seeds = list(range(p["meta_reps"]))
        for meta_seed in meta_seeds:
            meta_key = {"n_customers": n, "n_stations": n_stations, "seed": seed,
                        "run_type": "meta", "meta_seed": meta_seed}
            if not force and is_already_done("E4", meta_key):
                continue
            log = setup_logger("E4", make_run_id(f"n{n}", f"n_stations{n_stations}", f"seed{seed}", f"meta{meta_seed}"))
            log.info(f"Meta n={n} n_stations = {n_stations} seed={seed} meta_seed={meta_seed}")
            m = run_meta_experiment(inst, meta_seed, p["R"], p["beta"])
            row = {"n_customers": n, "n_stations": n_stations,
                   "seed": seed, "run_type": "meta", **m}
            append_row("E4", row)
            log.info(f"  Meta done: cost={m['cost']}  t={m['runtime_s']}s")


def exp_E5(dry_run: bool = False, force: bool = False) -> None:
    """
    E5 — Convergenza della metaeuristica.
    Per ogni istanza e meta_seed, registra costo current e incumbent
    ad ogni iterazione → usato per la curva di convergenza (figura F4).

    Il CSV ha una riga per iterazione: può essere grande.
    is_already_done controlla sulla combinazione (instance_label, meta_seed).
    """
    p = EXP_PARAMS["E5"]
    combos = [
        (inst_cfg, ms)
        for inst_cfg in p["instances"]
        for ms       in p["meta_seeds"]
    ]
    print(f"\n[E5] {len(combos)} run "
          f"({len(p['instances'])} istanze × {len(p['meta_seeds'])} meta_seed)")
    if dry_run:
        for inst_cfg, ms in combos:
            print(f"  {inst_cfg['label']:8s}  n={inst_cfg['n_customers']:2d}  "
                  f"inst_seed={inst_cfg['seed']}  meta_seed={ms}")
        return

    for inst_cfg, meta_seed in combos:
        label = inst_cfg["label"]
        n     = inst_cfg["n_customers"]
        n_stat = inst_cfg["n_stations"]
        inst_seed = inst_cfg["seed"]

        # is_already_done controlla a livello di run (label + meta_seed),
        # non di singola iterazione — evita di riscrivere migliaia di righe
        key = {"instance_label": label, "meta_seed": meta_seed}
        if not force and is_already_done("E5", key):
            print(f"  [skip] {label} meta_seed={meta_seed}")
            continue

        run_id = make_run_id(label, f"seed{meta_seed}")
        log = setup_logger("E5", run_id)
        log.info(f"Avvio {label} (n={n}) meta_seed={meta_seed}")

        inst = generate_random_instance(n, n_stat, p["area_km"], inst_seed)
        sol, history = run_3p_gms_ils(
            inst,
            R=p["R"],
            beta=p["beta"],
            seed=meta_seed,
            track_history=True,
        )

        # Scrivi UNA riga per iterazione
        phase_marker_map = dict(history["phase_markers"])  # {iter: "II"/"III"}
        for i, (curr, inc, elapsed) in enumerate(zip(
            history["iteration_costs"],
            history["incumbent_costs"],
            history["elapsed_times"],
        )):
            append_row("E5", {
                "instance_label": label,
                "n_customers":    n,
                "n_stations":     n_stat,
                "inst_seed":      inst_seed,
                "meta_seed":      meta_seed,
                "phase_I_cost":   history["phase_I_cost"],
                "iteration":      i,
                "current_cost":   curr,
                "incumbent_cost": inc,
                "elapsed_s":      elapsed,
                "improved_by":    phase_marker_map.get(i, ""),
            })

        log.info(f"  Done: {len(history['iteration_costs'])} iterazioni, "
                 f"cost finale={sol.cost:.2f}")

def exp_E6(dry_run: bool = False, force: bool = False) -> None:
    """
    E6 — Effetto del Warm Start.
    4 configurazioni: base / +WS / +VI / +WS+VI+tuned
    """
    p = EXP_PARAMS["E6"]
    combos = [
        (n, s, cfg)
        for n   in p["n_customers"]
        for s   in p["seeds"]
        for cfg in p["configs"]
    ]
    print(f"\n[E6] {len(combos)} run ({len(p['n_customers'])} n × "
          f"{len(p['seeds'])} seed × {len(p['configs'])} config)")
    if dry_run:
        for n, s, cfg in combos:
            print(f"  n={n:2d}  seed={s:3d}  config={cfg}")
        return

    for n, seed, config in combos:
        key = {"n_customers": n, "seed": seed, "ws_config": config}
        if not force and is_already_done("E6", key):
            print(f"  [skip] n={n} seed={seed} config={config}")
            continue

        inst = generate_random_instance(n, p["n_stations"][0], p["area_km"], seed)

        # Genera warm start (soluzione meta) se necessario
        warm_start = None
        if "ws" in config:
            random.seed(0)
            warm_start = run_3p_gms_ils(inst, R=25, beta=3)

        # Determina vi_config e tuned
        if config == "milp_base":
            vi_config = "none"
        elif config == "milp_ws":
            vi_config = "none"
        elif config == "milp_vi":
            vi_config = "vi_all"
        else:  # milp_ws_vi
            vi_config = "vi_all"

        run_id = make_run_id(f"n{n}", f"seed{seed}", config)
        log = setup_logger("E6", run_id)
        log.info(f"Avvio n={n} seed={seed} config={config}")

        metrics = run_milp_experiment(inst, vi_config, p["time_limit"], warm_start)
        row = {
            "n_customers": n, "n_stations": p["n_stations"][0],
            "seed": seed, "ws_config": config, **metrics,
        }
        append_row("E6", row)
        log.info(f"  {metrics['status']}  gap={metrics['mip_gap_pct']}%  t={metrics['runtime_s']}s")


def exp_E7(dry_run: bool = False, force: bool = False) -> None:
    """
    E7 — Sensitivity su beta (parametro di sparsificazione granulare).
    """
    p = EXP_PARAMS["E7"]
    combos = [
        (inst_cfg, beta, ms)
        for inst_cfg in p["instances"]
        for beta     in p["betas"]
        for ms       in p["meta_seeds"]
    ]
    print(f"\n[E7] {len(combos)} run "
          f"({len(p['instances'])} istanze × {len(p['betas'])} beta × {len(p['meta_seeds'])} seed)")
    if dry_run:
        for ic, b, ms in combos:
            print(f"  n={ic['n_customers']:2d}  inst_seed={ic['seed']}  beta={b:4.1f}  meta_seed={ms}")
        return

    for inst_cfg, beta, meta_seed in combos:
        n, inst_seed = inst_cfg["n_customers"], inst_cfg["seed"]
        key = {"n_customers": n, "inst_seed": inst_seed, "beta": beta, "meta_seed": meta_seed}
        if not force and is_already_done("E7", key):
            continue

        inst = generate_random_instance(n, inst_cfg["n_stations"], p["area_km"], inst_seed)
        m = run_meta_experiment(inst, meta_seed, p["R"], beta)
        row = {"n_customers": n, "n_stations": inst_cfg["n_stations"],
               "inst_seed": inst_seed, "beta": beta, **m}
        append_row("E7", row)
        print(f"  n={n} beta={beta} meta_seed={meta_seed} → cost={m['cost']}")


def exp_E8(dry_run: bool = False, force: bool = False) -> None:
    """
    E8 — Sensitivity su R (numero di restart).
    """
    p = EXP_PARAMS["E8"]
    combos = [
        (inst_cfg, R_val, ms)
        for inst_cfg in p["instances"]
        for R_val    in p["R_values"]
        for ms       in p["meta_seeds"]
    ]
    print(f"\n[E8] {len(combos)} run")
    if dry_run:
        for ic, R_val, ms in combos:
            print(f"  n={ic['n_customers']:2d}  R={R_val:3d}  meta_seed={ms}")
        return

    for inst_cfg, R_val, meta_seed in combos:
        n, inst_seed = inst_cfg["n_customers"], inst_cfg["seed"]
        key = {"n_customers": n, "inst_seed": inst_seed, "R": R_val, "meta_seed": meta_seed}
        if not force and is_already_done("E8", key):
            continue

        inst = generate_random_instance(n, inst_cfg["n_stations"], p["area_km"], inst_seed)
        m = run_meta_experiment(inst, meta_seed, R_val, p["beta"])
        row = {"n_customers": n, "n_stations": inst_cfg["n_stations"],
               "inst_seed": inst_seed, "R": R_val, **m}
        append_row("E8", row)
        print(f"  n={n} R={R_val} meta_seed={meta_seed} → cost={m['cost']}")


def exp_E9(dry_run: bool = False, force: bool = False) -> None:
    """
    E9 — Modal split su istanza reale Trieste.

    Per ogni config (default / no_drone / no_robot / no_no_fly / low_battery)
    e per ogni meta_seed, esegue run_meta_experiment e registra:
    - costo, runtime, split modale (n_truck / n_drone / n_robot)
    - config usata
    """
    p = EXP_PARAMS["E9"]
    combos = [
        (cfg, ms)
        for cfg in p["configs"]
        for ms  in p["meta_seeds"]
    ]
    n_configs  = len(p["configs"])
    n_seeds    = len(p["meta_seeds"])
    print(f"\n[E9] Trieste — {n_configs} config × {n_seeds} seed = {len(combos)} run")

    if dry_run:
        for cfg, ms in combos:
            print(f"  config={cfg:<14s}  meta_seed={ms}")
        return

    # Carica l'istanza base una volta sola (OSM può essere lento)
    print("  Caricamento grafi OSM Trieste...")
    inst_base, node_ids, locations = _build_trieste_instance_base(n_customers=25)
    n_c = inst_base.n_customers
    n_s = inst_base.n_stations
    print(f"  Istanza caricata: {n_c} clienti, {n_s} stazioni, "
          f"{len(inst_base.no_fly)} nodi no-fly")

    for config, meta_seed in combos:
        key = {"config": config, "meta_seed": meta_seed}
        if not force and is_already_done("E9", key):
            print(f"  [skip] config={config} meta_seed={meta_seed}")
            continue

        run_id = make_run_id("trieste", config, f"seed{meta_seed}")
        log    = setup_logger("E9", run_id)
        log.info(f"Avvio config={config} meta_seed={meta_seed}")

        inst = _apply_trieste_config(inst_base, config)
        m    = run_meta_experiment(inst, meta_seed, p["R"], p["beta"])

        row = {
            "n_customers": n_c,
            "n_stations":  n_s,
            "config":      config,
            **m,
        }
        append_row("E9", row)
        log.info(
            f"  config={config}  cost={m['cost']}  "
            f"truck={m['n_truck']}  drone={m['n_drone']}  robot={m['n_robot']}  "
            f"t={m['runtime_s']}s"
        )


def exp_E10(dry_run: bool = False, force: bool = False) -> None:
    """
    E10 — Ablation: misura il valore marginale di ogni modalità.
    Usa istanze sintetiche in 4 configurazioni: all, no_drone, no_robot, truck_only.
    """
    p = EXP_PARAMS["E10"]
    combos = [
        (n, s, cfg, ms)
        for n   in p["n_customers"]
        for s   in p["seeds"]
        for cfg in p["configs"]
        for ms  in range(p["meta_reps"])
    ]
    print(f"\n[E10] {len(combos)} run "
          f"({len(p['n_customers'])} taglie × {len(p['seeds'])} seed × "
          f"{len(p['configs'])} config × {p['meta_reps']} rep)")
    if dry_run:
        for n, s, cfg, ms in combos:
            print(f"  n={n:2d}  seed={s:3d}  config={cfg:<12s}  meta_seed={ms}")
        return

    for i, (n, seed, config, meta_seed) in enumerate(combos):
        key = {"n_customers": n, "seed": seed, "config": config, "meta_seed": meta_seed}
        if not force and is_already_done("E10", key):
            print(f"  [skip] n={n} seed={seed} config={config} meta_seed={meta_seed}")
            continue

        run_id = make_run_id(f"n{n}", f"seed{seed}", config, f"meta{meta_seed}")
        log = setup_logger("E10", run_id)
        log.info(f"Avvio n={n} seed={seed} config={config} meta_seed={meta_seed} "
                 f"[{i+1}/{len(combos)}]")

        inst_base = generate_random_instance(n, p["n_stations"][0], p["area_km"], seed)
        inst = _apply_synthetic_config(inst_base, config)
        m = run_meta_experiment(inst, meta_seed, p["R"], p["beta"])

        row = {"n_customers": n, "n_stations": p["n_stations"][0],
               "seed": seed, "config": config, **m}
        append_row("E10", row)
        log.info(f"  config={config}  cost={m['cost']}  "
                 f"truck={m['n_truck']}  drone={m['n_drone']}  robot={m['n_robot']}  "
                 f"t={m['runtime_s']}s")
        

def exp_E11(dry_run: bool = False, force: bool = False) -> None:
    """
    E11 — Robustezza al seed: 30 run su un'istanza fissa.
    Caratterizza la varianza stocastica della metaeuristica.
    """
    p = EXP_PARAMS["E11"]
    print(f"\n[E11] {len(p['meta_seeds'])} run su istanza fissa "
          f"(n={p['n_customers']}, inst_seed={p['inst_seed']})")
    if dry_run:
        for ms in p["meta_seeds"]:
            print(f"  meta_seed={ms}")
        return

    inst = generate_random_instance(
        p["n_customers"], p["n_stations"], p["area_km"], p["inst_seed"]
    )
    for meta_seed in p["meta_seeds"]:
        key = {"n_customers": p["n_customers"], "inst_seed": p["inst_seed"],
               "meta_seed": meta_seed}
        if not force and is_already_done("E11", key):
            continue

        m = run_meta_experiment(inst, meta_seed, p["R"], p["beta"])
        row = {"n_customers": p["n_customers"], "n_stations": p["n_stations"],
               "inst_seed": p["inst_seed"], **m}
        append_row("E11", row)
        print(f"  meta_seed={meta_seed:2d}  cost={m['cost']}  t={m['runtime_s']}s")


# ═══════════════════════════════════════════════════════════════════════════════
#  DISPATCH TABLE e MAIN
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "E1":  exp_E1,
    "E2":  exp_E2,
    "E3":  exp_E3,
    "E4":  exp_E4,
    "E5":  exp_E5,
    "E6":  exp_E6,
    "E7":  exp_E7,
    "E8":  exp_E8,
    "E9":  exp_E9,
    "E10": exp_E10,
    "E11": exp_E11,
}

# Ordine raccomandato per "all" (dipendenze prima)
ALL_ORDER = ["E1", "E2", "E3", "E4", "E6", "E7", "E8", "E11", "E10", "E5", "E9"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner esperimenti VRPD-RS-TW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python scalability.py --exp E1
  python scalability.py --exp E1 E2 E4
  python scalability.py --exp all
  python scalability.py --exp E1 --dry-run
  python scalability.py --exp E1 --force
  python scalability.py --status
""",
    )
    parser.add_argument(
        "--exp", nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Esperimento/i da eseguire (es. E1 E2, oppure 'all')",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostra le combinazioni che verranno eseguite senza eseguirle",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Riesegui anche le combinazioni già presenti nel CSV",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Mostra lo stato attuale di tutti gli esperimenti e termina",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.status:
        summary()
        return

    if not args.exp:
        parser.print_help()
        return

    exps_to_run = ALL_ORDER if "all" in args.exp else args.exp

    if args.dry_run:
        print("🔍 DRY-RUN — nessuna computazione verrà eseguita\n")

    t_global = time.time()
    for exp_id in exps_to_run:
        fn = EXPERIMENTS[exp_id]
        fn(dry_run=args.dry_run, force=args.force)

    if not args.dry_run:
        print(f"\n✅ Completato in {time.time() - t_global:.1f}s")
        summary()


if __name__ == "__main__":
    main()
