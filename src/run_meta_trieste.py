# run_meta_trieste.py
from data.trieste.loader       import get_drive_graph, get_walk_graph
from data.trieste.node_selector import get_node_ids, load_locations
from data.trieste.matrix_builder import load_or_build_real_instance
from data.trieste.visualize     import visualize_solution
from src.metaheuristics.framework import run_3p_gms_ils

def main(n_customers = 20):
    

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

    print("Caricamento/costruzione istanza...")
    inst = load_or_build_real_instance(G_drive, G_walk, node_ids, locations, seed=42)
    print(f"Istanza pronta: {inst.n_customers} clienti, {inst.n_stations} stazioni")

    print("Avvio 3P-GMS-ILS...")
    sol = run_3p_gms_ils(inst,verbose=True)
    print(f"Costo soluzione: {sol.cost:.4f}")
    sol.plot(inst)

    visualize_solution(sol, inst, output_path="results/prove/solution_meta_trieste.html")

if __name__ == "__main__":
    main()