# VRPD-RS-TW
## *Vehicle Routing Problem with Drones, Robots, and Time Windows.*

### Tonetto Giovanni, Zoch Matteo
Article of reference:

> Campuzano, G., Lalla-Ruiz, E., Mes, M. (2025).  
> *A matheuristic for the vehicle routing problem with drones and robots under time windows.*  
> Transportation Research Part E.

---

## Problem Description

A depot serves a set of customers using three coordinated vehicle types:

- **Truck** — follows the main route; carries the drone and all parcels.
- **Drone** — launched from the truck to serve customers via aerial sorties (round-trip or multi-leg), subject to battery constraints and no-fly zones.
- **Robot** — dispatched from fixed robot stations to serve nearby customers on foot; limited by a maximum travel time.

The objective is to minimise total operational cost: driver wage, truck routing cost, drone routing cost, and robot routing cost.

---

## Repository Structure

```
.
├── paper.pdf                          # Reference paper (Campuzano et al., 2025)
├── supplementary_material.pdf         # Additional material from the paper
├── test.py                            # Quick demo: runs 3P-GMS-ILS + MILP on Trieste
├── scalability.py                     # Scalability / sensitivity experiments (E1–E11)
├── plot_results.py                    # Generates all figures (F1–F11) from experiment CSVs
│
├── data/
│   └── trieste/
│       ├── locations.json             # Geographic coordinates of nodes in Trieste
│       ├── loader.py                  # Downloads / caches OSM road graphs (osmnx)
│       ├── matrix_builder.py          # Builds distance & time matrices from OSM graphs
│       ├── node_selector.py           # Selects customers and robot stations from OSM nodes
│       ├── no_fly_zones.py            # Defines no-fly polygons over Trieste
│       ├── instance_trieste_generator.py  # Assembles the full Trieste Instance object
│       └── visualize.py              # Renders interactive HTML maps (folium)
│
├── results/
│   ├── csv/
│   │   ├── milp/                      # E1 (MILP scalability), E2 (VI), E6 (warm start)
│   │   ├── meta/                      # E3, E5 (convergence), E7 (β), E8 (R), E11 (robustness)
│   │   ├── comparison/                # E4 (MILP vs meta)
│   │   └── trieste/                   # E9 (modal split), E10 (ablation)
│   ├── plots/
│   │   ├── milp/                      # Figures F1, F2, F6
│   │   ├── meta/                      # Figures F3, F5, F7, F8, F11
│   │   ├── comparison/                # Figure F4
│   │   └── trieste/                   # Figures F9, F10
│   └── logs/                          # Per-run log files
│
└── src/
    ├── config.py                      # Global parameters (speeds, costs, Big-M, …)
    ├── instance.py                    # Instance dataclass (distance/time matrices, TW, …)
    ├── instance_generator.py          # Random synthetic instance generator
    ├── milp.py                        # Gurobi MILP model (variables, constraints, valid ineqs)
    ├── solution.py                    # Solution dataclass + MILP→Solution converter
    ├── run_milp_trieste.py            # Standalone MILP run on Trieste
    ├── run_meta_trieste.py            # Standalone metaheuristic run on Trieste
    │
    ├── helpers/
    │   └── results_manager.py        # CSV logging, directory management, Gurobi metrics
    │
    └── metaheuristics/
        ├── construction.py            # Phase I  — sector-sweep VRP + robot insertion + MakeFly
        ├── intensification.py         # Phase III — local search (9 neighbourhood structures)
        ├── diversification.py         # Phase II  — shaking + multi-start restart
        ├── neighborhoods.py           # All neighbourhood move operators
        ├── shaking.py                 # Shaking perturbation procedures
        ├── granular.py                # Granular arc set management (β threshold)
        ├── synchronization.py         # Vehicle synchronisation (truck / drone / robot times)
        ├── evaluate.py                # Feasibility check + cost evaluation
        └── framework.py              # 3P-GMS-ILS main loop
```

---

## Installation

**Python 3.11 or later is required.**

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Gurobi licence

Gurobi requires a valid licence. We used an Academic licence (WLS).

---

## How to Run

### 1. Quick demo — `test.py`

```bash
python test.py
```

This script runs all algorithms end-to-end on the real Trieste instance and prints a comparison table. It is the recommended entry point for exploring the code.

**What it does, step by step:**

1. Loads the Trieste instance via OSM (downloads and caches road graphs on the first run; subsequent runs load from `.pkl` cache in `data/trieste/` and take under 1 second).
2. Runs **3P-GMS-ILS** on a larger instance (`N_CUSTOMERS_META = 20`) and saves an interactive HTML map to `maps_outputs/`.
3. Runs **Gurobi MILP** on a smaller instance (`N_CUSTOMERS_MILP = 5`) in three configurations:
 - base
 - with valid inequalities proposed by the authors of the article (VI 1–5)
 - with valid inequalities proposed by the authors (VI 1–5) + our valid inequalities (VI 6-8) + the metaheuristic solution as warm start.
4. Prints a comparison table: objective value, optimality gap, and runtime for each configuration.

Key parameters can be changed at the top of `test.py`:

| Variable | Default | Meaning |
|---|---|---|
| `N_CUSTOMERS_MILP` | 5 | Customers for the exact MILP (recommended max ≈ 10) |
| `N_CUSTOMERS_META` | 20 | Customers for the metaheuristic (recommended max ≈ 100) |
| `INSTANCE_SEED` | 69 | Random seed for node selection |
| `META_R` | 25 | Number of multi-start restarts |
| `META_MAX_ITER` | 1000 | ILS iterations |
| `META_BETA` | 3 | Granular sparsification threshold |
| `MILP_TIME_LIMIT` | 300 | Gurobi time limit (seconds) |
| `MILP_VERBOSE` | True | Show full Gurobi log |

**Expected runtime:** 1–5 minutes depending on hardware and `N_CUSTOMERS_MILP`.

---

### 2. Scalability and sensitivity experiments — `scalability.py`

> ⚠️ **Warning: long runtimes.** Running all experiments takes many hours (E1 alone runs for up to 10 hours, E6 runs 64 MILP solves of 600 seconds each). **Pre-computed results for all experiments are already available in `results/csv/`** and can be used directly to generate figures with `plot_results.py` without re-running anything.

```bash
# Run a single experiment
python scalability.py --exp E1

# Run multiple experiments
python scalability.py --exp E3 E7 E8

# Run all eleven experiments (takes many hours — not recommended)
python scalability.py --exp all

# Dry run: print all combinations that would be executed, without running them
python scalability.py --exp E2 --dry-run

# Force re-run even if results already exist in CSV
python scalability.py --exp E1 --force

# Show current status of all experiments (how many rows each CSV has)
python scalability.py --status
```

Results are written as CSV files to `results/csv/`. **Interrupted runs resume automatically** from where they stopped, each combination is checked against the CSV before executing.

| Experiment | Description | Estimated runtime |
|---|---|---|
| E1 | MILP scalability: time and gap vs. number of customers (3→12) | ~4 h |
| E2 | Valid inequalities: 4 configurations on n=5,7,9,11 | ~4 h |
| E3 | Metaheuristic scalability: time and quality (5→30 customers) | ~35 min |
| E4 | MILP vs metaheuristic: quality and time on small instances | ~3 h |
| E5 | Convergence: cost vs. iterations curve, with per-iteration history | ~10 min |
| E6 | Warm start: MILP with/without metaheuristic initial solution | ~12 h |
| E7 | Sensitivity β: effect of granular threshold (n=25) | ~30 min |
| E8 | Sensitivity R: effect of multi-start count (n=5) | ~10 min |
| E9 | Real Trieste instance: 5 modal configurations | ~10 min |
| E10 | Ablation study: marginal value of drone/robot vs truck-only | ~10 min |
| E11 | Robustness to seed: 30 runs on a fixed instance (n=20) | ~35 min |
| |**Totale stimato:** | **~25h**|

---

### 3. Generate figures — `plot_results.py`

Reads the CSV files from `results/csv/` and saves PDF and PNG figures to `results/plots/`. Each figure F{n} corresponds to experiment E{n}.

```bash
# Generate a single figure
python plot_results.py --fig F4

# Generate multiple figures
python plot_results.py --fig F1 F3 F4

# Generate all figures from available CSVs
python plot_results.py --fig all

# Open an interactive matplotlib window after saving
python plot_results.py --fig F9 --show
```

Since pre-computed CSVs are already present in `results/csv/`, **all implemented figures can be generated immediately** without running any experiment first.

| Figure | Experiment | Description |
|---|---|---|
| F1 | E1 | MILP scalability: runtime and MIP gap |
| F2 | E2 | Valid inequalities: MIP gap and lower bound |
| F3 | E3 | Metaheuristic scalability: runtime and cost |
| F4 | E4 | MILP vs metaheuristic: optimality gap and speedup |
| F5 | E5 | Convergence curve: cost vs. time |
| F6 | E6 | Warm start effect on MILP |
| F7 | E7 | Sensitivity analysis: parameter β |
| F8 | E8 | Sensitivity analysis: number of restarts R |
| F9 | E9 | Modal split on real Trieste instance |
| F10 | E10 | Ablation study: marginal value of drone/robot |
| F11 | E11 | Robustness to seed: cost distribution over 30 runs |

Figures are saved to `results/plots/<category>/` where category is `milp`, `meta`, `comparison`, or `trieste`.

---

## Algorithms Implemented

### MILP (`src/milp.py`)

Full mixed-integer linear programming formulation following Campuzano et al. (2025), solved with Gurobi:

- **Binary variables:** truck arcs `x_T`, drone sorties `x_D`, robot trips `x_R`
- **Continuous variables:** arrival/departure times, service times, node positions
- **Constraints:** flow balance, multimodality, capacity, time windows, battery, synchronisation, symmetry breaking, valid inequalities (VI 1–8)

### 3P-GMS-ILS Metaheuristic (`src/metaheuristics/`)

Three-phase Granular Multi-Start Iterated Local Search:

- **Phase I — Construction:** sector-sweep truck routing, cheapest-insertion robots, greedy drone sortie assignment
- **Phase II — Diversification:** 6 shaking procedures; multi-start restart from a solution pool when no improvement is found
- **Phase III — Intensification:** randomised exploration of 9 neighbourhood structures with granular arc filtering

---

## Notes

- Distances are in **kilometres**; times are in **seconds** throughout.
- OSM graphs for Trieste are downloaded automatically on the first run and cached as `.pkl` files inside `data/trieste/`. Subsequent runs load from cache (< 1 s).
- The MILP is tractable up to approximately **10 customers**. For larger instances, the metaheuristic algorithm is preferred.
- The metaheuristic scales to **50+ customers** in a few minutes.
- The implementation uses Gurobi/Python; the original paper uses CPLEX/C++. Results are comparable on small instances.
