# results_manager.py
"""
Gestione centralizzata dell'output degli esperimenti VRPD-RS-TW.

Tutto ciò che riguarda percorsi, scrittura CSV, logging e checkpoint
passa da qui. Gli script di esperimento non toccano mai i path direttamente.
"""

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gurobipy as gp
from gurobipy import GRB

# ── Radice del progetto ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent

# ── Mappa experiment_id → sottocartella ─────────────────────────────────────
# Raggruppa per natura dell'esperimento, non per numero.
EXPERIMENT_CATEGORY: dict[str, str] = {
    "E1":  "milp",        # scalabilità MILP
    "E2":  "milp",        # effetto valid inequalities
    "E3":  "meta",        # scalabilità meta
    "E4":  "comparison",  # MILP vs metaeuristica
    "E5":  "meta",        # convergenza metaeuristica
    "E6":  "milp",        # warm start
    "E7":  "meta",        # sensitivity beta
    "E8":  "meta",        # sensitivity R
    "E9":  "trieste",     # modal split Trieste
    "E10": "trieste",     # ablation modalità
    "E11": "meta",        # robustezza al seed
}

# Nome del file CSV per ogni esperimento (senza percorso)
EXPERIMENT_FILENAME: dict[str, str] = {
    "E1":  "E1_scalability_milp.csv",
    "E2":  "E2_valid_inequalities.csv",
    "E3":  "E3_scalability_meta.csv",
    "E4":  "E4_milp_vs_meta.csv",
    "E5":  "E5_convergence.csv",
    "E6":  "E6_warm_start.csv",
    "E7":  "E7_beta_sensitivity.csv",
    "E8":  "E8_R_sensitivity.csv",
    "E9":  "E9_modal_split.csv",
    "E10": "E10_ablation.csv",
    "E11": "E11_seed_robustness.csv",
}

# ── Costruzione dell'albero di directory ─────────────────────────────────────

def _build_tree() -> dict[str, Path]:
    """
    Costruisce il dizionario name→Path di tutte le directory necessarie.
    Usa un loop sulle categorie invece di hardcodare 20 path.
    """
    base = ROOT / "results"
    categories = {"milp", "meta", "comparison", "trieste"}

    dirs: dict[str, Path] = {}

    # csv/<categoria> e plots/<categoria> per ogni categoria
    for cat in categories:
        dirs[f"csv_{cat}"]   = base / "csv"   / cat
        dirs[f"plots_{cat}"] = base / "plots" / cat

    # directory singole
    dirs["logs"]        = base / "logs"

    return dirs


DIRS = _build_tree()


def ensure_dirs() -> None:
    """Crea tutte le directory se non esistono (idempotente)."""
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


# ── Percorsi ─────────────────────────────────────────────────────────────────

def csv_path(experiment_id: str) -> Path:
    """Ritorna il Path completo del CSV per l'esperimento dato."""
    if experiment_id not in EXPERIMENT_CATEGORY:
        raise ValueError(f"Esperimento sconosciuto: '{experiment_id}'. "
                         f"Validi: {sorted(EXPERIMENT_CATEGORY)}")
    cat      = EXPERIMENT_CATEGORY[experiment_id]
    filename = EXPERIMENT_FILENAME[experiment_id]
    return DIRS[f"csv_{cat}"] / filename


def plots_dir(experiment_id: str) -> Path:
    """Ritorna la directory dei plot per l'esperimento dato."""
    cat = EXPERIMENT_CATEGORY[experiment_id]
    return DIRS[f"plots_{cat}"]


def log_path(experiment_id: str, run_id: str) -> Path:
    """Path del file di log per una run specifica."""
    return DIRS["logs"] / f"{experiment_id}_{run_id}.log"


# ── CSV append-only ───────────────────────────────────────────────────────────

def append_row(experiment_id: str, row: dict[str, Any]) -> None:
    """
    Aggiunge UNA riga al CSV dell'esperimento.

    - Se il file non esiste, scrive prima l'header.
    - Non sovrascrive mai i dati esistenti.
    - Aggiunge automaticamente 'timestamp' a ogni riga.

    Args:
        experiment_id: es. "E1"
        row: dizionario colonna→valore. Le colonne vengono determinate
             dal primo row scritto; righe successive devono avere le
             stesse chiavi (DictWriter gestisce campi mancanti con '').
    """
    ensure_dirs()
    path = csv_path(experiment_id)

    # Aggiungi timestamp automaticamente
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}

    file_exists = path.exists() and path.stat().st_size > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
            extrasaction="ignore",   # ignora chiavi extra senza crashare
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv(experiment_id: str) -> list[dict[str, str]]:
    """Legge il CSV dell'esperimento e lo restituisce come lista di dict."""
    path = csv_path(experiment_id)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_already_done(experiment_id: str, key: dict[str, Any]) -> bool:
    """
    Controlla se questa combinazione di parametri è già nel CSV.
    Permette di riprendere un batch interrotto senza duplicati.

    Args:
        experiment_id: es. "E1"
        key: sottoinsieme di colonne da matchare, es.
             {"n_customers": 10, "seed": 42, "vi_config": "base"}

    Returns:
        True se esiste già almeno una riga con tutti i key=value di `key`.

    Esempio:
        if is_already_done("E1", {"n_customers": n, "seed": seed}):
            print("già fatto, salto")
            continue
    """
    rows = read_csv(experiment_id)
    key_str = {str(k): str(v) for k, v in key.items()}  # CSV è tutto stringhe
    for row in rows:
        if all(row.get(k) == v for k, v in key_str.items()):
            return True
    return False



# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(
    experiment_id: str,
    run_id: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Crea e configura un logger che scrive sia su file sia su console.

    Il logger è nominato 'vrpd.<experiment_id>.<run_id>' per evitare
    conflitti tra logger di run diverse.

    Args:
        experiment_id: es. "E1"
        run_id:        stringa identificativa della run, es. "n10_seed42"
        level:         livello di logging (default: INFO)

    Returns:
        Un logging.Logger già configurato.
    """
    ensure_dirs()
    logger_name = f"vrpd.{experiment_id}.{run_id}"
    logger = logging.getLogger(logger_name)

    # Evita di aggiungere handler duplicati se chiamato più volte
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handler su file
    fh = logging.FileHandler(log_path(experiment_id, run_id), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Handler su console (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ── Estrazione metriche Gurobi ────────────────────────────────────────────────

def extract_milp_metrics(m: gp.Model) -> dict[str, Any]:
    """
    Estrae dal modello Gurobi risolto tutte le metriche standard
    usate negli esperimenti E1, E2, E3, E4, E6.

    Gestisce in modo sicuro i casi in cui il modello non ha trovato
    nessuna soluzione feasible (es. TIME_LIMIT senza incumbent).

    Returns:
        Dizionario con le chiavi:
            status          "OPTIMAL" | "TIME_LIMIT" | "INFEASIBLE" | "INF_OR_UNBD" | other
            obj_val         valore della miglior soluzione trovata (None se non trovata)
            obj_bound       lower bound corrente
            mip_gap_pct     gap percentuale (None se obj_val assente)
            runtime_s       tempo di risoluzione in secondi
            n_nodes         nodi B&B esplorati
            n_vars          numero di variabili nel modello
            n_constrs       numero di vincoli nel modello
            n_bin_vars      numero di variabili binarie
            n_int_vars      numero di variabili intere (include binarie)
    """
    status_map = {
        GRB.OPTIMAL:        "OPTIMAL",
        GRB.TIME_LIMIT:     "TIME_LIMIT",
        GRB.INFEASIBLE:     "INFEASIBLE",
        GRB.INF_OR_UNBD:    "INF_OR_UNBD",
        GRB.SUBOPTIMAL:     "SUBOPTIMAL",
    }
    status = status_map.get(m.Status, f"STATUS_{m.Status}")

    # ObjVal potrebbe non esistere se non è stata trovata nessuna soluzione
    has_solution = m.SolCount > 0
    obj_val   = m.ObjVal   if has_solution else None
    obj_bound = m.ObjBound if has_solution or status == "OPTIMAL" else None

    if obj_val is not None and obj_bound is not None and obj_val != 0:
        mip_gap_pct = abs(obj_val - obj_bound) / abs(obj_val) * 100
    else:
        mip_gap_pct = None

    return {
        "status":       status,
        "obj_val":      round(obj_val, 4)      if obj_val      is not None else None,
        "obj_bound":    round(obj_bound, 4)    if obj_bound    is not None else None,
        "mip_gap_pct":  round(mip_gap_pct, 2) if mip_gap_pct  is not None else None,
        "runtime_s":    round(m.Runtime, 2),
        "n_nodes":      int(m.NodeCount),
        "n_vars":       m.NumVars,
        "n_constrs":    m.NumConstrs,
        "n_bin_vars":   m.NumBinVars,
        "n_int_vars":   m.NumIntVars,
    }


# ── Run ID helper ─────────────────────────────────────────────────────────────

def make_run_id(*parts: Any) -> str:
    """
    Crea un run_id riproducibile e leggibile concatenando le parti fornite.
    Aggiunge un timestamp per unicità assoluta.

    Esempio:
        make_run_id("n10", "seed42", "vi_all")
        → "n10_seed42_vi_all_20260523_143201"
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "_".join(str(p) for p in parts)
    return f"{label}_{ts}"


# ── Utility di diagnostica ────────────────────────────────────────────────────

def summary() -> None:
    """
    Stampa un riepilogo dello stato di tutti gli esperimenti:
    quante righe ha ogni CSV, quando è stato modificato l'ultima volta.
    """
    print("\n📊 Stato esperimenti\n" + "─" * 50)
    for exp_id in sorted(EXPERIMENT_CATEGORY):
        path = csv_path(exp_id)
        if path.exists():
            rows = read_csv(exp_id)
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d/%m %H:%M")
            print(f"  {exp_id:4s}  ✅  {len(rows):4d} righe   ultimo agg. {mtime}   {path.name}")
        else:
            print(f"  {exp_id:4s}  🔲  (non ancora avviato)")
    print()


if __name__ == "__main__":
    # Se eseguito direttamente, crea le directory e mostra il riepilogo
    ensure_dirs()
    print("✅ Directory create:")
    for name, path in sorted(DIRS.items()):
        print(f"   {path.relative_to(ROOT)}")
    print()
    summary()
