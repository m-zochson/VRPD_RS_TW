# run_milp_trieste.py
from data.trieste.loader import get_drive_graph, get_walk_graph
from data.trieste.node_selector import get_node_ids, load_locations
from data.trieste.matrix_builder import build_real_instance
from src.milp import solve, print_solution
from src.solution import Solution, milp_to_solution
from data.trieste.visualize import visualize_solution
from src.metaheuristics.framework import run_3p_gms_ils


def main(n_customers = 5):
    print("Caricamento grafi OSM...")
    G_drive = get_drive_graph()
    G_walk  = get_walk_graph()

    print("Selezione nodi...")
    locations = load_locations(n_customers)
    node_ids  = get_node_ids(G_drive, G_walk, n_customers)
    osm_ids_drive = (
    [node_ids['depot']['drive']]
    + [c['drive'] for c in node_ids['customers']]
    + [s['drive'] for s in node_ids['stations']]
    + [node_ids['depot']['drive']]
    )
    osm_ids_walk = (
        [node_ids['depot']['walk']]
        + [c['walk'] for c in node_ids['customers']]
        + [s['walk'] for s in node_ids['stations']]
        + [node_ids['depot']['walk']]
    )

    print("Costruzione istanza reale...")
    inst = build_real_instance(G_drive, G_walk, node_ids, locations)

    print(f"Istanza: {inst.n_customers} clienti, {inst.n_stations} stazioni")

    print("Avvio MILP...")
    sol_ws = run_3p_gms_ils(inst,R=50,beta=3)
    m, x_T, x_D, x_R = solve(inst, use_valid_inequalities=True, tuned = True, warm_start=sol_ws)
    sol = milp_to_solution(m,inst)
    sol.plot(inst)

    print_solution(inst, m, x_T, x_D, x_R)
    visualize_solution(sol, inst, locations, G_drive, G_walk, osm_ids_drive, osm_ids_walk, output_path="results/solution_trieste.html")


if __name__ == "__main__":
    main()