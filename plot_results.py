# plot_results.py
"""
Generazione di tutte le figure per la presentazione VRPD-RS-TW.
Legge i CSV prodotti da scalability.py e salva PDF + PNG in results/plots/.

Convenzione: figura F{n} corrisponde all'esperimento E{n}.

Uso:
    python plot_results.py --fig F4             # genera solo F4
    python plot_results.py --fig F2 F3 F4       # genera più figure
    python plot_results.py --fig all            # tutte
    python plot_results.py --fig F4 --show      # apre finestra interattiva

Mappa figure → esperimento:
  F1  — E1  — Scalabilità MILP: runtime e MIP gap al crescere di n
  F2  — E2  — Valid Inequalities: MIP gap e lower bound
  F3  — E3  — Scalabilità metaeuristica: runtime e costo al crescere di n
  F4  — E4  — MILP vs Metaeuristica: qualità e tempo a confronto
  F5  — E5  — Convergenza: curva costo vs iterazioni
  F6  — E6  — Warm start: effetto della soluzione iniziale sul MILP
  F7  — E7  — Sensitivity β: effetto della soglia granulare
  F8  — E8  — Sensitivity R: effetto del numero di multi-start
  F9  — E9  — Modal split Trieste: 5 configurazioni modali
  F10 — E10 — Ablation study: valore marginale di drone/robot
  F11 — E11 — Robustezza al seed: distribuzione costo su 30 run
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.helpers.results_manager import read_csv, plots_dir, ensure_dirs, DIRS

# ── Parametri grafici globali ─────────────────────────────────────────────────
# Modifica qui per adattare tutta la presentazione (es. font size per Beamer)
PLOT_PARAMS = {
    "figsize_single":  (7, 4.5),      # figura a colonna singola
    "figsize_double":  (12, 4.5),     # figura a due colonne affiancate
    "figsize_table":   (10, 5),       # tabelle
    "dpi":             150,
    "font_size":       10,
    "title_size":      11,
    "label_size":      10,
    "legend_size":     9,
    "colors_seeds":    ["#2196F3", "#FF5722", "#4CAF50"],   # blu, arancio, verde
    "colors_vi":       ["#90A4AE", "#42A5F5", "#AB47BC", "#EF5350", "#FF7043"],
    "colors_config":   ["#546E7A", "#26A69A", "#5C6BC0", "#EF5350"],
    "color_milp":      "#5C6BC0",
    "color_meta":      "#EF5350",
    "color_best":      "black",
}

def _apply_style() -> None:
    """Applica lo stile matplotlib globale."""
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         PLOT_PARAMS["font_size"],
        "axes.titlesize":    PLOT_PARAMS["title_size"],
        "axes.labelsize":    PLOT_PARAMS["label_size"],
        "legend.fontsize":   PLOT_PARAMS["legend_size"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })


def _save(fig: plt.Figure, name: str, exp_id: str, show: bool) -> None:
    """Salva la figura in PDF e PNG nella cartella corretta."""
    ensure_dirs()
    out_dir = plots_dir(exp_id)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=PLOT_PARAMS["dpi"])
        print(f"  Salvato: {path}")
    if show:
        plt.show()
    plt.close(fig)


def _check_csv(exp_id: str, fig_name: str) -> pd.DataFrame | None:
    """Legge il CSV; se non esiste stampa un avviso e ritorna None."""
    rows = read_csv(exp_id)
    if not rows:
        print(f"  ⚠️  [{fig_name}] CSV di {exp_id} non trovato o vuoto. "
              f"Lancia prima: python scalability.py --exp {exp_id}")
        return None
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def fig_F1(show: bool = False) -> None:
    """
    F1 — Scalabilità del MILP: grafico bidirezionale runtime / MIP gap.

    Asse superiore (log): per le istanze OPTIMAL, barra alta = runtime veloce.
    Asse inferiore (lineare invertito): per le istanze TIME_LIMIT,
        altezza = 100% - MIP gap (alta = vicino all'ottimo, bassa = lontano).

    Le due metà condividono la linea di separazione (midline = gap 0% = ottimo).

    Input:  E1_scalability_milp.csv
    Output: results/plots/milp/F1_scalability_milp.pdf
    """
    from matplotlib.patches import Patch

    _apply_style()
    df = _check_csv("E1", "F1")
    if df is None:
        return

    for col in ["obj_val", "mip_gap_pct", "runtime_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].str.strip()
    df["n_customers"] = pd.to_numeric(df["n_customers"])
    ns = sorted(df["n_customers"].unique())
    x  = np.arange(len(ns))
    w  = 0.60

    C_OPT  = "#2E7D32"
    C_TIME = "#C62828"

    # ── Aggrega per n (media su seed) ─────────────────────────────────────────
    opt_rt: dict = {}
    tl_gap: dict = {}
    for n, g in df.groupby("n_customers"):
        opt = g[g["status"] == "OPTIMAL"]
        tl  = g[g["status"] == "TIME_LIMIT"]
        opt_rt[n] = opt["runtime_s"].mean()  if len(opt) > 0 else None
        tl_gap[n] = tl["mip_gap_pct"].mean() if len(tl)  > 0 else None

    # ── Layout: due assi adiacenti ────────────────────────────────────────────
    fig    = plt.figure(figsize=(13, 7))
    ax_top = fig.add_axes([0.10, 0.48, 0.82, 0.38])
    ax_bot = fig.add_axes([0.10, 0.08, 0.82, 0.40])

    # ── Asse inferiore: 100 - MIP gap ─────────────────────────────────────────
    for i, n in enumerate(ns):
        gap   = tl_gap.get(n)
        is_tl = gap is not None and not np.isnan(gap)

        if is_tl:
            bar_h = 100 - gap
            ax_bot.bar(i, bar_h, bottom=gap, width=w,
                       color=C_TIME, alpha=0.85, zorder=3)
            ax_bot.text(i, gap + bar_h / 2, f"gap\n{gap:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        else:
            # OPTIMAL: sfondo verde leggero
            ax_bot.bar(i, 100, bottom=0, width=w,
                       color=C_OPT, alpha=0.15, zorder=2)

    ax_bot.set_ylim(0, 100)
    ax_bot.invert_yaxis()
    ax_bot.set_yticks([0, 25, 50, 75, 100])
    ax_bot.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_bot.set_ylabel("MIP Gap", fontsize=10)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"n={n}" for n in ns], fontsize=9)
    ax_bot.axhline(0, color="black", linewidth=1.5, zorder=5)
    ax_bot.grid(axis="y", alpha=0.2, linestyle="--")

    # ── Asse superiore: 1/runtime (log) ───────────────────────────────────────
    for i, n in enumerate(ns):
        rt = opt_rt.get(n)
        if rt is None or np.isnan(rt):
            continue
        ax_top.bar(i, 1 / rt, bottom=0, width=w,
                   color=C_OPT, alpha=0.85, zorder=3)
        ax_top.text(i, (1 / rt) * 1.6, f"{rt:.0f}s",
                    ha="center", va="bottom", fontsize=8.5,
                    color=C_OPT, fontweight="bold")

    ax_top.set_yscale("log")
    x_min, x_max = -0.5, len(ns) - 0.5
    ax_top.set_xlim(x_min, x_max)
    ax_bot.set_xlim(x_min, x_max)
    ax_top.set_yticks([0.001, 0.01, 0.1, 1.0])
    ax_top.set_yticklabels(["1000s", "100s", "10s", "1s"])
    ax_top.set_ylabel("Runtime (log)", fontsize=10)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.axhline(0, color="black", linewidth=1.5, zorder=5)
    ax_top.grid(axis="y", alpha=0.2, linestyle="--")

    # ── Linea di separazione centrale ─────────────────────────────────────────
    fig.add_artist(plt.Line2D(
        [0.10, 0.92], [0.482, 0.482],
        transform=fig.transFigure,
        color="black", linewidth=2.2, zorder=10,
    ))

    # ── Legenda e titolo ──────────────────────────────────────────────────────
    time_limit = int(df["runtime_s"].max() / 600) * 600
    n_seeds    = df["seed"].nunique()
    seed_label = f"seed = {df['seed'].iloc[0]}" if n_seeds == 1 else f"{n_seeds} seed"

    ax_top.legend(
        handles=[
            Patch(facecolor=C_OPT,  alpha=0.85, label="OPTIMAL — runtime"),
            Patch(facecolor=C_TIME, alpha=0.85, label="TIME_LIMIT — 100% − MIP gap"),
        ],
        loc="upper right", fontsize=9, framealpha=0.85,
    )

    fig.suptitle(
        f"Scalabilità del MILP  (time limit = {time_limit}s,  {seed_label})\n"
        "Sopra: runtime istanze ottimali  |  "
        "Sotto: 100% − MIP gap (più alta = più vicino all'ottimo)",
        fontsize=11, fontweight="bold", y=0.99,
    )
    ax_bot.set_xlabel("Numero di clienti", fontsize=10)

    plt.tight_layout()
    _save(fig, "F1_scalability_milp", "E1", show)


def fig_F2(show: bool = False) -> None:
    """
    F2 — Effetto delle Valid Inequalities su MIP Gap e Lower Bound.
    Due pannelli affiancati:
      - Sinistra: MIP Gap % per configurazione VI (bar chart raggruppato)
      - Destra:   Lower Bound medio con linea tratteggiata = incumbent

    Input:  E2_valid_inequalities.csv
    Output: results/plots/milp/F2_vi_effect.pdf
    """
    _apply_style()
    df = _check_csv("E2", "F2")
    if df is None:
        return

    df["n_customers"] = pd.to_numeric(df["n_customers"])
    for c in ["obj_val", "obj_bound", "mip_gap_pct"]:
        df[c] = pd.to_numeric(df[c])

    # Escludi n=5: raggiunge sempre l'ottimo, non è informativo
    df = df[df["n_customers"] > 5].copy()

    ns      = sorted(df["n_customers"].unique())
    configs = ["none", "vi_paper", "vi_custom", "vi_all"]
    labels  = ["None", "VI paper (1-5)", "VI custom (6-8)", "VI all (1-8)"]
    colors  = PLOT_PARAMS["colors_vi"]

    x      = np.arange(len(ns))
    width  = 0.18
    offset = np.linspace(-(1.5 * width), 1.5 * width, 4)
    ekw    = dict(elinewidth=0.9, capsize=3, capthick=1, ecolor="black", alpha=0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_PARAMS["figsize_double"])
    for ax in (ax1, ax2):
        ax.yaxis.grid(True, alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)

    def _add_bars(ax, metric, ylabel, title):
        suffix = "%" if metric == "mip_gap_pct" else ""
        label_offset = 1.5 if metric == "mip_gap_pct" else 0.5
        for i, (cfg, lbl, col) in enumerate(zip(configs, labels, colors)):
            means = [df[(df["n_customers"] == n) & (df["vi_config"] == cfg)][metric].mean()
                     for n in ns]
            stds  = [df[(df["n_customers"] == n) & (df["vi_config"] == cfg)][metric].std()
                     for n in ns]
            bars = ax.bar(x + offset[i], means, width, label=lbl,
                          color=col, edgecolor="white", linewidth=0.5,
                          yerr=stds, error_kw=ekw, zorder=3)
            for bar, v in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + label_offset,
                        f"{v:.0f}{suffix}",
                        ha="center", va="bottom", fontsize=7, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={n}" for n in ns], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.85)

    _add_bars(ax1, "mip_gap_pct", "MIP Gap medio (%)", "MIP Gap per configurazione VI")
    _add_bars(ax2, "obj_bound",   "Lower Bound medio (€)", "Qualità del Lower Bound")
    ax1.set_ylim(0, 105)

    # Linea tratteggiata = incumbent medio per ogni n
    for n_idx, n in enumerate(ns):
        ov = df[df["n_customers"] == n]["obj_val"].mean()
        ax2.hlines(ov, x[n_idx] - 2.2 * width, x[n_idx] + 2.2 * width,
                   colors="black", linewidths=1.5, linestyles=":", zorder=5)
    ax2.plot([], [], "k:", linewidth=1.5, label="obj_val (incumbent)")
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.85)

    fig.suptitle("Valid Inequalities: MIP Gap e qualità del Lower Bound"
                 "  (time limit=300s, media ± std su 3 seed)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
             "Linea tratteggiata = incumbent medio  |  "
             "Barre di errore = std tra i seed",
             ha="center", fontsize=8.5, color="gray")

    plt.tight_layout()
    _save(fig, "F2_vi_effect", "E2", show)


def fig_F3(show: bool = False) -> None:
    """
    F3 — Scalabilità della metaeuristica 3P-GMS-ILS.

    Due assi y sullo stesso asse x (n_customers):
      - Sinistra (log): runtime medio ± std tra instance seed
      - Destra (lineare): costo medio ± std tra instance seed

    I punti individuali (media per inst_seed) sono mostrati come scatter
    semitrasparenti dietro le linee per rendere visibile la variabilità.

    La zona n≥9 è evidenziata in grigio: è il range dove il MILP non riesce
    a trovare soluzioni ottimali entro 3600s (da E1).

    Input:  E3_scalability_meta.csv
    Output: results/plots/meta/F3_meta_scalability.pdf
    """
    _apply_style()

    path = DIRS["csv_meta"] / "E3_scalability_meta.csv"
    if not path.exists():
        print(f"  ⚠️  [F3] File non trovato: {path}")
        return

    df = pd.read_csv(path)
    df["cost"]        = pd.to_numeric(df["cost"],        errors="coerce")
    df["runtime_s"]   = pd.to_numeric(df["runtime_s"],   errors="coerce")
    df["n_customers"] = pd.to_numeric(df["n_customers"])

    # ── Aggregazione a due livelli ─────────────────────────────────────────────
    # 1. Media su meta_seed per ogni (n, inst_seed) → riduce rumore stocastico
    by_inst = (df.groupby(["n_customers", "seed"])
               .agg(cost_inst=("cost", "mean"),
                    rt_inst=("runtime_s", "mean"))
               .reset_index())

    # 2. Media e std su inst_seed per ogni n → variabilità tra istanze
    agg = (by_inst.groupby("n_customers")
           .agg(cost_mean=("cost_inst", "mean"),
                cost_std =("cost_inst", "std"),
                rt_mean  =("rt_inst",   "mean"),
                rt_std   =("rt_inst",   "std"))
           .reset_index())

    ns      = agg["n_customers"].values
    n_seeds = df["seed"].nunique()
    n_reps  = df["meta_seed"].nunique()

    C_RT   = "#1565C0"   # blu  — runtime
    C_COST = "#E65100"   # arancio — costo

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    # ── Zona MILP infeasibility (n > 8) ───────────────────────────────────────
    milp_threshold = 8.5   # da E1: OPTIMAL fino a n=8, TIME_LIMIT da n=9
    ax1.axvspan(milp_threshold, ns.max() + 0.5, alpha=0.05,
                color="gray", zorder=0)
    ax1.text(milp_threshold + 0.3,
             agg["rt_mean"].min() * 0.8,
             "MILP: TIME_LIMIT\n(n ≥ 9, tl=3600s)",
             fontsize=8, color="gray", style="italic", va="bottom")

    # ── Punti individuali (media per inst_seed) ───────────────────────────────
    for n, g in by_inst.groupby("n_customers"):
        ax1.scatter([n] * len(g), g["rt_inst"],
                    color=C_RT, alpha=0.25, s=30, zorder=3)
        ax2.scatter([n] * len(g), g["cost_inst"],
                    color=C_COST, alpha=0.25, s=30, zorder=3)

    # ── Linee principali con error band ──────────────────────────────────────
    l1 = ax1.errorbar(ns, agg["rt_mean"], yerr=agg["rt_std"],
                      fmt="o-", color=C_RT, linewidth=2.2, markersize=7,
                      capsize=4, capthick=1.5, elinewidth=1.2,
                      label="Runtime medio ± std", zorder=5)

    l2 = ax2.errorbar(ns, agg["cost_mean"], yerr=agg["cost_std"],
                      fmt="s--", color=C_COST, linewidth=2.0, markersize=6,
                      capsize=4, capthick=1.3, elinewidth=1.1,
                      label="Costo medio ± std", zorder=5)

    # Etichette runtime sulle barre
    for n, rt, std in zip(ns, agg["rt_mean"], agg["rt_std"]):
        ax1.annotate(f"{rt:.0f}s",
                     xy=(n, rt + std),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=7.5, color=C_RT, fontweight="bold")

    # ── Fit linea di tendenza runtime (scala log) ─────────────────────────────
    log_rt  = np.log(agg["rt_mean"].values)
    coeffs  = np.polyfit(ns, log_rt, 1)
    ns_fine = np.linspace(ns.min(), ns.max(), 100)
    rt_fit  = np.exp(np.polyval(coeffs, ns_fine))
    ax1.plot(ns_fine, rt_fit, color=C_RT, linewidth=0.8,
             linestyle=":", alpha=0.5, label="Trend esponenziale")

    # ── Assi ─────────────────────────────────────────────────────────────────
    ax1.set_yscale("log")
    ax1.set_xlabel("Numero di clienti", fontsize=10)
    ax1.set_ylabel("Runtime medio (s)", color=C_RT, fontsize=10)
    ax2.set_ylabel("Costo medio (€)",   color=C_COST, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=C_RT)
    ax2.tick_params(axis="y", labelcolor=C_COST)
    ax1.set_xticks(ns)
    ax1.set_xlim(ns.min() - 1, ns.max() + 1)

    # ── Linea verticale alla soglia MILP ──────────────────────────────────────
    ax1.axvline(milp_threshold, color="gray", linestyle="--",
                linewidth=1, alpha=0.6)

    # ── Legenda unificata ─────────────────────────────────────────────────────
    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left",
               fontsize=9, framealpha=0.85)

    fig.suptitle(
        f"Scalabilità della metaeuristica 3P-GMS-ILS\n"
        f"({n_seeds} istanze × {n_reps} meta-seed per taglia · "
        f"R=25, β=3.0)",
        fontsize=11, fontweight="bold",
    )
    fig.text(
        0.5, -0.04,
        "Punti piccoli = media per singola istanza  |  "
        "Barre di errore = std tra istanze  |  "
        "Zona grigia = MILP infeasibility region",
        ha="center", fontsize=8.5, color="gray",
    )

    plt.tight_layout()
    _save(fig, "F3_meta_scalability", "E3", show)


def fig_F4(show: bool = False) -> None:
    """
    F4 — MILP vs Metaeuristica: gap di ottimalità e speedup.
 
    Due pannelli affiancati:
      - Sinistra: gap di ottimalità della metaeuristica rispetto all'ottimo MILP
        (%), per n_customers e n_stations. Linea piena = media ± std su seed e
        meta-rep; linea tratteggiata = best gap (min). Punti individuali in
        trasparenza.
      - Destra: speedup (runtime MILP / runtime meta), scala logaritmica.
        Annotazioni sul valore per speedup > 5×.
 
    Nota sulla struttura del CSV: le righe meta riusano le colonne MILP
    per effetto di DictWriter (fieldnames = chiavi della riga corrente):
        colonna "obj_val"   → costo trovato dalla meta
        colonna "obj_bound" → runtime della meta (secondi)
    Tutte le istanze MILP hanno status = OPTIMAL.
 
    Input:  E4_milp_vs_meta.csv
    Output: results/plots/comparison/F4_milp_vs_meta.pdf
    """
    _apply_style()
    df = _check_csv("E4", "F4")
    if df is None:
        return
 
    # ── Pulizia ──────────────────────────────────────────────────────────────
    df["n_customers"] = pd.to_numeric(df["n_customers"])
    df["n_stations"]  = pd.to_numeric(df["n_stations"])
    df["seed"]        = pd.to_numeric(df["seed"])
    df["obj_val"]     = pd.to_numeric(df["obj_val"],   errors="coerce")
    df["obj_bound"]   = pd.to_numeric(df["obj_bound"], errors="coerce")
    df["runtime_s"]   = pd.to_numeric(df["runtime_s"], errors="coerce")
    df["run_type"]    = df["run_type"].str.strip()
 
    KEY = ["n_customers", "n_stations", "seed"]
 
    milp = (df[df["run_type"] == "milp"][KEY + ["obj_val", "runtime_s"]]
            .rename(columns={"obj_val": "milp_cost", "runtime_s": "milp_runtime"}))
 
    # obj_val  → costo meta,  obj_bound → runtime meta (secondi)
    meta = (df[df["run_type"] == "meta"][KEY + ["obj_val", "obj_bound"]]
            .rename(columns={"obj_val": "meta_cost", "obj_bound": "meta_runtime"}))
 
    merged = meta.merge(milp, on=KEY)
    merged["gap_pct"] = (
        (merged["meta_cost"] - merged["milp_cost"]) / merged["milp_cost"] * 100
    ).clip(lower=0)   # clip per arrotondamenti floating-point
    # Speedup in spazio logaritmico: evita che std > mean su scala log
    merged["log_speedup"] = np.log(merged["milp_runtime"] / merged["meta_runtime"])
 
    ns_vals = sorted(merged["n_customers"].unique())
 
    # Aggrega su tutti i seed, meta-rep e n_stations per ogni n_customers
    agg = (
        merged.groupby("n_customers")
        .agg(
            gap_mean    =("gap_pct",     "mean"),
            gap_std     =("gap_pct",     "std"),
            gap_min     =("gap_pct",     "min"),
            log_sp_mean =("log_speedup", "mean"),
            log_sp_std  =("log_speedup", "std"),
        )
        .reset_index()
        .sort_values("n_customers")
    )
    # Media geometrica e barre asimmetriche: rimangono sempre > 0 su scala log
    agg["sp_geo_mean"] = np.exp(agg["log_sp_mean"])
    agg["sp_err_up"]   = np.exp(agg["log_sp_mean"] + agg["log_sp_std"]) - agg["sp_geo_mean"]
    agg["sp_err_down"] = agg["sp_geo_mean"] - np.exp(agg["log_sp_mean"] - agg["log_sp_std"])
    ns = agg["n_customers"].values
 
    C_GAP = PLOT_PARAMS["color_meta"]   # rosso
    C_SP  = PLOT_PARAMS["color_milp"]   # blu
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_PARAMS["figsize_double"])
    for ax in (ax1, ax2):
        ax.yaxis.grid(True, alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)
 
    # ── Panel 1: Gap di ottimalità ────────────────────────────────────────────
    # Punti individuali (semitrasparenti)
    for n in ns_vals:
        pts = merged[merged["n_customers"] == n]["gap_pct"].values
        ax1.scatter([n] * len(pts), pts, color=C_GAP,
                    alpha=0.18, s=20, zorder=2)
 
    # Linea principale: media ± std
    ax1.errorbar(ns, agg["gap_mean"], yerr=agg["gap_std"],
                 fmt="o-", color=C_GAP, linewidth=2.2, markersize=7,
                 capsize=4, capthick=1.5, elinewidth=1.2,
                 label="Gap medio ± std", zorder=4)
 
    # Linea tratteggiata: best gap (min su tutte le istanze)
    ax1.plot(ns, agg["gap_min"], color=C_GAP,
             linewidth=1.0, linestyle="--", alpha=0.6, label="Best gap (min)")
 
    ax1.axhline(0, color="black", linewidth=1, linestyle=":", alpha=0.4)
    ax1.set_ylim(bottom=-1)
    ax1.set_xticks(ns_vals)
    ax1.set_xlabel("Numero di clienti", fontsize=10)
    ax1.set_ylabel("Gap di ottimalità (%) — più basso è meglio", fontsize=10)
    ax1.set_title("Qualità della metaeuristica", fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
 
    # ── Panel 2: Speedup ─────────────────────────────────────────────────────
    ax2.errorbar(ns, agg["sp_geo_mean"],
                 yerr=[agg["sp_err_down"], agg["sp_err_up"]],
                 fmt="s-", color=C_SP, linewidth=2.2, markersize=7,
                 capsize=4, capthick=1.5, elinewidth=1.2,
                 label="Speedup (media geometrica ± 1σ)", zorder=4)
 
    for n, sp in zip(ns, agg["sp_geo_mean"].values):
        if sp > 5:
            ax2.annotate(f"{sp:.0f}×",
                         xy=(n, sp),
                         xytext=(0, 7), textcoords="offset points",
                         ha="center", fontsize=7.5, color=C_SP,
                         fontweight="bold")
 
    ax2.axhline(1, color="black", linewidth=1, linestyle=":", alpha=0.4,
                label="Pareggio (1×)")
    ax2.set_yscale("log")
    ax2.set_xticks(ns_vals)
    ax2.set_xlabel("Numero di clienti", fontsize=10)
    ax2.set_ylabel("Speedup (MILP / meta) — più alto è meglio", fontsize=10)
    ax2.set_title("Velocità relativa", fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
 
    # ── Titolo e didascalia ───────────────────────────────────────────────────
    n_seeds   = df["seed"].nunique()
    n_reps    = int(merged.groupby(KEY).size().max())
    stat_vals = sorted(merged["n_stations"].unique())
    fig.suptitle(
        f"MILP vs Metaeuristica — qualità e velocità  "
        f"({n_seeds} seed × {n_reps} meta-rep × {len(stat_vals)} n_staz. · "
        f"tutti i MILP: OPTIMAL)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.03,
        "Punti piccoli = singole rep  |  "
        "Linea piena = media ± std su seed, rep e n_stations  |  "
        "Linea tratteggiata = best gap",
        ha="center", fontsize=8.5, color="gray",
    )
 
    plt.tight_layout()
    _save(fig, "F4_milp_vs_meta", "E4", show)

def fig_F5(show: bool = False) -> None:
    """
    F5 — Curve di convergenza della metaeuristica 3P-GMS-ILS.

    Legge le istanze direttamente dal CSV — nessun label hardcodato.
    Il numero di subplot si adatta a quante instance_label distinte ci sono.
    Il titolo di ogni subplot è costruito da (instance_label, n_customers).

    Per ogni istanza:
      - N linee sottili semitrasparenti = current_cost per ogni seed
      - N linee spesse = incumbent_cost per ogni seed
      - Linea tratteggiata = best cost trovato
      - Punto cerchio = costo Phase I

    Input:  E5_convergence.csv
    Output: results/plots/meta/F5_convergence.pdf
    """
    _apply_style()
    df = _check_csv("E5", "F5")
    if df is None:
        return

    for col in ("incumbent_cost", "current_cost", "elapsed_s",
                "meta_seed", "phase_I_cost", "n_customers"):
        df[col] = pd.to_numeric(df[col])

    # Leggi le istanze nell'ordine in cui compaiono nel CSV (non alfabetico)
    labels     = list(dict.fromkeys(df["instance_label"]))   # ordine di inserimento
    meta_seeds = sorted(df["meta_seed"].unique())
    colors     = PLOT_PARAMS["colors_seeds"]
    n_plots    = len(labels)

    # Se ci sono più seed dei colori definiti, estendi con un colormap
    if len(meta_seeds) > len(colors):
        cmap   = plt.cm.get_cmap("tab10", len(meta_seeds))
        colors = [cmap(i) for i in range(len(meta_seeds))]

    fig, axes = plt.subplots(1, n_plots,
                             figsize=(6 * n_plots, 4.5),
                             sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        sub          = df[df["instance_label"] == label]
        phase_I_cost = sub["phase_I_cost"].iloc[0]
        n_cust       = int(sub["n_customers"].iloc[0])
        t_max        = sub["elapsed_s"].max()

        # Titolo costruito dai dati, non hardcodato
        title = f"{label}  (n={n_cust})"
        ax.set_title(title, fontweight="bold")

        for color, ms in zip(colors, meta_seeds):
            run = sub[sub["meta_seed"] == ms].copy()
            if run.empty:
                continue
            t   = run["elapsed_s"].values
            cur = run["current_cost"].values
            inc = run["incumbent_cost"].values

            # Curva current_cost: sottile, semitrasparente
            ax.plot(t, cur, color=color, alpha=0.18, linewidth=0.8)

            # Curva incumbent_cost: spessa, solida
            ax.plot(t, inc, color=color, alpha=0.85, linewidth=1.8,
                    label=f"seed {ms}")

            # Punto Phase I
            ax.scatter(t[0], phase_I_cost, color=color,
                       s=40, zorder=5, marker="o", alpha=0.7)

        # Linea tratteggiata = best cost
        final_cost = sub["incumbent_cost"].min()
        ax.axhline(final_cost,
                   color=PLOT_PARAMS["color_best"],
                   linestyle=":", linewidth=1.2, alpha=0.6,
                   label=f"Best = {final_cost:.2f}")

        # Annotazione Phase I
        ax.annotate(
            f"Phase I\n{phase_I_cost:.1f}",
            xy=(sub["elapsed_s"].min(), phase_I_cost),
            xytext=(t_max * 0.3, phase_I_cost * 1.03),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Costo (€)")
        ax.legend(loc="upper right", framealpha=0.85)
        ax.set_xlim(left=0)

    fig.suptitle("Curva di convergenza — 3P-GMS-ILS",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.text(
        0.5, -0.04,
        "Linee sottili = costo corrente  |  Linee spesse = incumbent",
        ha="center", fontsize=8, color="gray",
    )

    plt.tight_layout()
    _save(fig, "F5_convergence", "E5", show)


def fig_F6(show: bool = False) -> None:
    """
    F6 — Effetto del Warm Start sul MILP.
 
    Due pannelli affiancati:
      - Sinistra: MIP gap % per configurazione (bar chart raggruppato).
        Celle senza soluzione → barra tratteggiata "N/F" (no feasible).
        Celle con soluzione ma lower bound = −∞ → barra punteggiata "∞".
      - Destra: qualità dell'incumbent (obj_val, media ± std tra seed).
        Celle senza soluzione → etichetta "N/F" al posto della barra.
 
    Configurazioni:
        milp_base  — MILP puro (no WS, no VI)
        milp_ws    — MILP con warm start dalla metaeuristica
        milp_vi    — MILP con valid inequalities
        milp_ws_vi — MILP con warm start + valid inequalities
 
    Input:  E6_warm_start.csv
    Output: results/plots/milp/F6_warm_start.pdf
    """
    _apply_style()
    df = _check_csv("E6", "F6")
    if df is None:
        return
 
    # ── Pulizia ──────────────────────────────────────────────────────────────
    df["n_customers"] = pd.to_numeric(df["n_customers"])
    df["obj_val"]     = pd.to_numeric(df["obj_val"],     errors="coerce")
    df["mip_gap_pct"] = pd.to_numeric(df["mip_gap_pct"], errors="coerce")
    df["ws_config"]   = df["ws_config"].str.strip()
 
    CONFIGS = ["milp_base", "milp_ws", "milp_vi", "milp_ws_vi"]
    LABELS  = ["Base", "+WS", "+VI", "+WS+VI"]
    COLORS  = PLOT_PARAMS["colors_config"]
 
    ns     = sorted(df["n_customers"].unique())
    x      = np.arange(len(ns))
    w      = 0.18
    offset = np.linspace(-(1.5 * w), 1.5 * w, 4)
    ekw    = dict(elinewidth=0.9, capsize=3, capthick=1, ecolor="black", alpha=0.5)
 
    # ── Aggregazione per (n, config) su tutti i seed ──────────────────────────
    # Classifica ogni (n, config) in tre stati:
    #   "no_sol"   — nessun seed trova soluzione (obj_val NaN per tutti)
    #   "no_bound" — soluzione trovata ma lower bound = −∞ (gap = inf)
    #   "ok"       — soluzione e gap finito disponibili
    def _cell(n, cfg):
        sub      = df[(df["n_customers"] == n) & (df["ws_config"] == cfg)]
        has_sol  = sub["obj_val"].notna()
        has_gap  = sub["mip_gap_pct"].apply(
            lambda v: pd.notna(v) and np.isfinite(v)
        )
        return {
            "state":      "ok"       if has_gap.any()
                          else "no_bound" if has_sol.any()
                          else "no_sol",
            "obj_mean":   sub.loc[has_sol, "obj_val"].mean(),
            "obj_std":    sub.loc[has_sol, "obj_val"].std(),
            "gap_mean":   sub.loc[has_gap, "mip_gap_pct"].mean(),
            "gap_std":    sub.loc[has_gap, "mip_gap_pct"].std(),
        }
 
    cells = {(n, cfg): _cell(n, cfg) for n in ns for cfg in CONFIGS}
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_PARAMS["figsize_double"])
    for ax in (ax1, ax2):
        ax.yaxis.grid(True, alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)
 
    # ── Panel 1: MIP gap ─────────────────────────────────────────────────────
    for i, (cfg, lbl, col) in enumerate(zip(CONFIGS, LABELS, COLORS)):
        for j, n in enumerate(ns):
            c    = cells[(n, cfg)]
            xpos = x[j] + offset[i]
 
            if c["state"] == "ok":
                bar = ax1.bar(xpos, c["gap_mean"], w,
                              color=col, edgecolor="white", linewidth=0.5,
                              yerr=c["gap_std"], error_kw=ekw,
                              zorder=3, alpha=0.85,
                              label=lbl if j == 0 else "_nolegend_")
                ax1.text(xpos, c["gap_mean"] + (c["gap_std"] or 0) + 1.5,
                         f"{c['gap_mean']:.0f}%",
                         ha="center", va="bottom", fontsize=6.5, color=col)
 
            elif c["state"] == "no_bound":
                # Soluzione trovata ma lower bound = −∞
                ax1.bar(xpos, 100, w, color=col, alpha=0.12,
                        edgecolor=col, linewidth=0.8,
                        linestyle=":", hatch="...", zorder=2,
                        label=lbl if j == 0 else "_nolegend_")
                ax1.text(xpos, 50, "∞", ha="center", va="center",
                         fontsize=10, color=col, fontweight="bold")
 
            else:  # no_sol
                ax1.bar(xpos, 100, w, color=col, alpha=0.10,
                        edgecolor=col, linewidth=0.8,
                        linestyle="--", hatch="///", zorder=2,
                        label=lbl if j == 0 else "_nolegend_")
                ax1.text(xpos, 50, "N/F", ha="center", va="center",
                         fontsize=6.5, color=col, fontweight="bold")
 
    ax1.set_ylim(0, 115)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"n={n}" for n in ns], fontsize=10)
    ax1.set_ylabel("MIP Gap (%) — più basso è meglio", fontsize=10)
    ax1.set_title("MIP Gap per configurazione", fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
 
    # ── Panel 2: incumbent (obj_val) ─────────────────────────────────────────
    for i, (cfg, lbl, col) in enumerate(zip(CONFIGS, LABELS, COLORS)):
        plotted_label = False
        for j, n in enumerate(ns):
            c    = cells[(n, cfg)]
            xpos = x[j] + offset[i]
 
            if pd.notna(c["obj_mean"]):
                ax2.bar(xpos, c["obj_mean"], w,
                        color=col, edgecolor="white", linewidth=0.5,
                        yerr=c["obj_std"] if pd.notna(c["obj_std"]) else 0,
                        error_kw=ekw, zorder=3, alpha=0.85,
                        label=lbl if not plotted_label else "_nolegend_")
                ax2.text(xpos,
                         c["obj_mean"] + (c["obj_std"] or 0) + 0.3,
                         f"{c['obj_mean']:.1f}",
                         ha="center", va="bottom", fontsize=6.5, color=col)
                plotted_label = True
            else:
                ax2.text(xpos, 0.5, "N/F",
                         ha="center", va="bottom", fontsize=6, color=col,
                         alpha=0.65, rotation=90)
 
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"n={n}" for n in ns], fontsize=10)
    ax2.set_ylabel("Costo incumbent (€) — più basso è meglio", fontsize=10)
    ax2.set_title("Qualità soluzione (incumbent)", fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
 
    # ── Titolo e didascalia ───────────────────────────────────────────────────
    n_seeds = df["seed"].nunique()
    fig.suptitle(
        f"Effetto del Warm Start sul MILP  (time limit=600s, {n_seeds} seed)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.03,
        "N/F = nessuna soluzione trovata entro il time limit  |  "
        "∞ = incumbent trovato ma lower bound = −∞  |  "
        "Barre di errore = std tra seed",
        ha="center", fontsize=8.5, color="gray",
    )
 
    plt.tight_layout()
    _save(fig, "F6_warm_start", "E6", show)

def fig_F7(show: bool = False) -> None:
    """
    F7 — Sensitivity analysis: parametro β di sparsificazione granulare.

    Doppio asse y: costo medio ± std (sinistra) e runtime medio ± std (destra).
    Scala logaritmica sull'asse x.
    Punti piccoli semitrasparenti = singoli seed.
    Linea tratteggiata verticale = β*=3.0 del paper.

    Input:  E7_beta_sensitivity.csv
    Output: results/plots/meta/F7_beta_sensitivity.pdf
    """
    _apply_style()
    df = _check_csv("E7", "F7")
    if df is None:
        return

    df["cost"]      = pd.to_numeric(df["cost"])
    df["runtime_s"] = pd.to_numeric(df["runtime_s"])

    df["beta"] = df["beta"].astype(float)
    betas      = sorted(df["beta"].unique())
    agg        = df.groupby("beta")[["cost", "runtime_s"]].agg(["mean", "std"])
    cost_mean  = agg[("cost",      "mean")].values
    cost_std   = agg[("cost",      "std")].values
    rt_mean    = agg[("runtime_s", "mean")].values
    rt_std     = agg[("runtime_s", "std")].values

    n       = int(df["n_customers"].iloc[0])
    n_seeds = df["meta_seed"].nunique()

    COLOR_COST = "#3F51B5"
    COLOR_RT   = "#E53935"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    # ── Costo (asse sinistro) ─────────────────────────────────────────────────
    l1 = ax1.errorbar(betas, cost_mean, yerr=cost_std,
                      fmt="o-", color=COLOR_COST, linewidth=2.2, markersize=6,
                      capsize=4, capthick=1.5, elinewidth=1.2,
                      label="Costo medio ± std", zorder=4)

    for beta in betas:
        y = df[df["beta"] == beta]["cost"].values
        ax1.scatter([beta] * len(y), y, color=COLOR_COST,
                    alpha=0.18, s=18, zorder=3)

    # ── Runtime (asse destro) ─────────────────────────────────────────────────
    l2 = ax2.errorbar(betas, rt_mean, yerr=rt_std,
                      fmt="s--", color=COLOR_RT, linewidth=1.8, markersize=5,
                      capsize=3, capthick=1.2, elinewidth=1,
                      label="Runtime medio ± std", zorder=4)

    # ── β* paper ─────────────────────────────────────────────────────────────
    paper_beta = min(betas, key=lambda b: abs(b - 3))
    paper_idx  = betas.index(paper_beta)
    ax1.axvline(paper_beta, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax1.annotate("β*=3.0\n(paper)",
             xy=(paper_beta, cost_mean[paper_idx]),
             xytext=(paper_beta * 1.2, cost_mean[paper_idx] + 2.0),
                 fontsize=8.5, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # ── Assi ─────────────────────────────────────────────────────────────────
    ax1.set_xscale("log")
    ax1.set_xlabel("β (scala logaritmica)", fontsize=10)
    ax1.set_ylabel("Costo medio (€)", color=COLOR_COST, fontsize=10)
    ax2.set_ylabel("Runtime medio (s)", color=COLOR_RT, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=COLOR_COST)
    ax2.tick_params(axis="y", labelcolor=COLOR_RT)
    ax1.set_xticks(betas)
    ax1.set_xticklabels([str(b) for b in betas], rotation=45, fontsize=8.5)

    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=9, framealpha=0.85)

    fig.suptitle(
        f"Sensitivity analysis: parametro β  (n={n}, {n_seeds} seed)",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.5, -0.04,
             "Punti piccoli = singoli seed  |  β=3.0 = default paper",
             ha="center", fontsize=8.5, color="gray")

    plt.tight_layout()
    _save(fig, "F7_beta_sensitivity", "E7", show)


def fig_F8(show: bool = False) -> None:
    """
    F8 — Sensitivity analysis: numero di restart R.

    Legge tutti i file E8_R_sensitivity_n_*.csv dalla cartella meta/
    e produce un subplot per ogni file (un pannello per valore di n).
    Doppio asse y: costo medio ± std (sinistra) e runtime medio (destra).

    Input:  results/csv/meta/E8_R_sensitivity_n_*.csv
    Output: results/plots/meta/F8_R_sensitivity.pdf
    """
    _apply_style()

    meta_dir = DIRS["csv_meta"]
    files    = sorted(meta_dir.glob("E8_R_sensitivity_n_*.csv"))

    if not files:
        print(f"  ⚠️  [F8] Nessun file E8_R_sensitivity_n_*.csv in {meta_dir}")
        return

    frames = []
    for f in files:
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)

    df["cost"]      = pd.to_numeric(df["cost"])
    df["runtime_s"] = pd.to_numeric(df["runtime_s"])
    df["R"]         = pd.to_numeric(df["R"])

    ns      = sorted(df["n_customers"].astype(int).unique())
    R_vals  = sorted(df["R"].unique())
    n_plots = len(ns)

    COLOR_COST = "#3F51B5"
    COLOR_RT   = "#E53935"

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        sub       = df[df["n_customers"] == n]
        agg       = sub.groupby("R")[["cost","runtime_s"]].agg(["mean","std"]).reindex(R_vals)
        cost_mean = agg[("cost",      "mean")].values
        cost_std  = agg[("cost",      "std")].values
        rt_mean   = agg[("runtime_s", "mean")].values

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)

        l1 = ax.errorbar(R_vals, cost_mean, yerr=cost_std,
                         fmt="o-", color=COLOR_COST, linewidth=2.2, markersize=7,
                         capsize=4, capthick=1.5, elinewidth=1.2,
                         label="Costo medio ± std", zorder=4)

        for R in R_vals:
            y = sub[sub["R"] == R]["cost"].values
            ax.scatter([R] * len(y), y, color=COLOR_COST,
                       alpha=0.2, s=20, zorder=3)

        l2 = ax2.errorbar(R_vals, rt_mean,
                          fmt="s--", color=COLOR_RT, linewidth=1.8, markersize=5,
                          capsize=3, elinewidth=1,
                          label="Runtime medio", zorder=3)

        # R*=25 paper
        if 25 in R_vals:
            idx25 = list(R_vals).index(25)
            ax.axvline(25, color="gray", linestyle="--", lw=1, alpha=0.5)
            ax.annotate("R*=25\n(paper)",
                        xy=(25, cost_mean[idx25]),
                        xytext=(30, cost_mean[idx25] - (cost_mean[0] - cost_mean[-1]) * 0.3),
                        fontsize=8.5, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        # Zona std=0 se presente
        zero_std_R = [R for R, s in zip(R_vals, cost_std)
                      if s is not None and not np.isnan(s) and s < 1e-6]
        if zero_std_R:
            ax.axvspan(min(zero_std_R), max(R_vals) * 1.05,
                       alpha=0.06, color=COLOR_COST)
            ax.annotate("std=0",
                        xy=(min(zero_std_R),
                            cost_mean[list(R_vals).index(min(zero_std_R))]),
                        xytext=(min(zero_std_R) + 5,
                                cost_mean[0] - (cost_mean[0] - cost_mean[-1]) * 0.3),
                        fontsize=8, color=COLOR_COST, alpha=0.8,
                        arrowprops=dict(arrowstyle="->", color=COLOR_COST, lw=0.7))

        ax.set_title(f"n={n}", fontweight="bold", fontsize=12)
        ax.set_xlabel("R (numero di restart)")
        ax.set_ylabel("Costo medio (€)", color=COLOR_COST)
        ax2.set_ylabel("Runtime medio (s)", color=COLOR_RT)
        ax.tick_params(axis="y", labelcolor=COLOR_COST)
        ax2.tick_params(axis="y", labelcolor=COLOR_RT)
        ax.set_xticks(R_vals)

        lines  = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.85)

    n_seeds = df["meta_seed"].nunique()
    fig.suptitle(
        f"Sensitivity analysis: numero di restart R  ({n_seeds} seed per valore)",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.5, -0.04,
             "R*=25 = default paper  |  Zona blu = std=0 (convergenza perfetta)",
             ha="center", fontsize=8.5, color="gray")

    plt.tight_layout()
    _save(fig, "F8_R_sensitivity", "E8", show)


def fig_F9(show: bool = False) -> None:
    """
    F9 — Modal split su istanza reale Trieste: 5 configurazioni a confronto.
 
    Due pannelli affiancati:
      - Sinistra: stacked bar del modal split (n_truck / n_drone / n_robot,
        media su 5 seed). Punti semitrasparenti = valori dei singoli seed.
      - Destra: costo medio ± std per configurazione (bar chart orizzontale,
        ordinato per costo crescente). Punti = singoli seed.
 
    Configurazioni:
        default     — tutte le modalità attive con zone no-fly reali
        no_drone    — drone disabilitato (energia > soglia)
        no_robot    — robot disabilitato (range > massimo)
        no_no_fly   — zone no-fly rimosse (drone libero ovunque)
        low_battery — batteria drone dimezzata
 
    Input:  E9_modal_split.csv
    Output: results/plots/trieste/F9_modal_split.pdf
    """
    _apply_style()
    df = _check_csv("E9", "F9")
    if df is None:
        return
 
    # ── Pulizia ──────────────────────────────────────────────────────────────
    for col in ["cost", "n_truck", "n_drone", "n_robot"]:
        df[col] = pd.to_numeric(df[col])
 
    CONFIG_ORDER  = ["default", "no_no_fly", "low_battery", "no_robot", "no_drone"]
    CONFIG_LABELS = ["Default", "No no-fly", "Low battery", "No robot", "No drone"]
    C_TRUCK = "#546E7A"   # grigio ardesia
    C_DRONE = "#1565C0"   # blu
    C_ROBOT = "#E65100"   # arancio
 
    # ── Aggregazione per configurazione ──────────────────────────────────────
    agg = (
        df.groupby("config")
        .agg(
            truck_mean =("n_truck", "mean"),
            drone_mean =("n_drone", "mean"),
            robot_mean =("n_robot", "mean"),
            cost_mean  =("cost",    "mean"),
            cost_std   =("cost",    "std"),
        )
        .reindex(CONFIG_ORDER)
        .reset_index()
    )
 
    x      = np.arange(len(CONFIG_ORDER))
    n_cust = int(df["n_customers"].iloc[0])
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_PARAMS["figsize_double"])
    for ax in (ax1, ax2):
        ax.yaxis.grid(True, alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)
 
    # ── Panel 1: Stacked bar modal split ─────────────────────────────────────
    bars_t = ax1.bar(x, agg["truck_mean"], color=C_TRUCK,
                     label="Truck", zorder=3, alpha=0.85)
    bars_d = ax1.bar(x, agg["drone_mean"], bottom=agg["truck_mean"],
                     color=C_DRONE, label="Drone", zorder=3, alpha=0.85)
    bars_r = ax1.bar(x, agg["robot_mean"],
                     bottom=agg["truck_mean"] + agg["drone_mean"],
                     color=C_ROBOT, label="Robot", zorder=3, alpha=0.85)
 
    # Etichette valore dentro ogni segmento
    for i, (t, d, r) in enumerate(zip(
        agg["truck_mean"], agg["drone_mean"], agg["robot_mean"]
    )):
        if t > 0.5:
            ax1.text(i, t / 2, f"{t:.1f}", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
        if d > 0.5:
            ax1.text(i, t + d / 2, f"{d:.1f}", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
        if r > 0.5:
            ax1.text(i, t + d + r / 2, f"{r:.1f}", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
 
    # Punti dei singoli seed per ogni modalità
    jitter = np.linspace(-0.18, 0.18, df["meta_seed"].nunique())
    for cfg_i, cfg in enumerate(CONFIG_ORDER):
        sub = df[df["config"] == cfg].sort_values("meta_seed")
        for j, (_, row) in enumerate(sub.iterrows()):
            xj = cfg_i + jitter[j]
            ax1.scatter(xj, row["n_truck"],
                        color=C_TRUCK, s=22, zorder=5, alpha=0.55,
                        edgecolors="white", linewidths=0.3)
            ax1.scatter(xj, row["n_truck"] + row["n_drone"],
                        color=C_DRONE, s=22, zorder=5, alpha=0.55,
                        edgecolors="white", linewidths=0.3)
            ax1.scatter(xj, row["n_truck"] + row["n_drone"] + row["n_robot"],
                        color=C_ROBOT, s=22, zorder=5, alpha=0.55,
                        edgecolors="white", linewidths=0.3)
 
    ax1.axhline(n_cust, color="black", linewidth=1, linestyle=":",
                alpha=0.4, label=f"Totale clienti ({n_cust})")
    ax1.set_ylim(0, n_cust + 3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CONFIG_LABELS, fontsize=9, rotation=15, ha="right")
    ax1.set_ylabel("Clienti serviti (media su 5 seed)", fontsize=10)
    ax1.set_title("Modal split per configurazione", fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
 
    # ── Panel 2: Costo per configurazione ────────────────────────────────────
    # Ordine per costo crescente
    agg_sorted = agg.sort_values("cost_mean").reset_index(drop=True)
    cfg_sorted = agg_sorted["config"].tolist()
    lbl_sorted = [CONFIG_LABELS[CONFIG_ORDER.index(c)] for c in cfg_sorted]
    y          = np.arange(len(cfg_sorted))
 
    ekw = dict(elinewidth=1.0, capsize=4, capthick=1.2, ecolor="black", alpha=0.6)
    bars = ax2.barh(y, agg_sorted["cost_mean"], xerr=agg_sorted["cost_std"],
                    error_kw=ekw, color=PLOT_PARAMS["color_meta"],
                    alpha=0.80, zorder=3, height=0.55)
 
    # Punti singoli seed
    for yi, cfg in enumerate(cfg_sorted):
        costs = df[df["config"] == cfg]["cost"].values
        ax2.scatter(costs, [yi] * len(costs),
                    color=PLOT_PARAMS["color_meta"], s=28, zorder=5,
                    alpha=0.45, edgecolors="white", linewidths=0.4)
 
    # Etichette valore a destra delle barre
    for yi, (mean, std) in enumerate(
        zip(agg_sorted["cost_mean"], agg_sorted["cost_std"])
    ):
        ax2.text(mean + std + 4, yi, f"{mean:.0f}",
                 va="center", ha="left", fontsize=8.5,
                 color=PLOT_PARAMS["color_meta"], fontweight="bold")
 
    # Linea di riferimento = costo default
    default_cost = agg.loc[agg["config"] == "default", "cost_mean"].iloc[0]
    ax2.axvline(default_cost, color="gray", linewidth=1,
                linestyle="--", alpha=0.6, label=f"Default ({default_cost:.0f})")
 
    ax2.set_yticks(y)
    ax2.set_yticklabels(lbl_sorted, fontsize=9)
    ax2.set_xlabel("Costo medio (€)", fontsize=10)
    ax2.set_title("Costo per configurazione", fontweight="bold")
    ax2.legend(fontsize=8.5, loc="lower right", framealpha=0.85)
    ax2.xaxis.grid(True, alpha=0.2, linestyle="--")
    ax2.yaxis.grid(False)
 
    # ── Titolo e didascalia ───────────────────────────────────────────────────
    n_seeds = df["meta_seed"].nunique()
    fig.suptitle(
        f"Modal split su istanza reale Trieste  "
        f"(n={n_cust} clienti, {n_seeds} seed per configurazione)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.03,
        "Punti piccoli = singoli seed  |  "
        "Barre = media  |  "
        "Costo ordinato crescente",
        ha="center", fontsize=8.5, color="gray",
    )
 
    plt.tight_layout()
    _save(fig, "F9_modal_split", "E9", show)


def fig_F10(show: bool = False) -> None:
    """
    F10 — Ablation study: valore marginale di drone e robot.
 
    Due pannelli affiancati:
      - Sinistra: costo medio ± std per configurazione, con tutti i punti
        visibili colorati per inst_seed. Ordine per costo crescente.
      - Destra: penalità % rispetto alla baseline "all" per inst_seed
        (media su meta_seed per ogni inst_seed). Mostra il contributo
        marginale di ciascuna modalità e la sua consistenza tra istanze.
 
    Configurazioni:
        all        — tutte le modalità attive (baseline)
        no_drone   — drone disabilitato
        no_robot   — robot disabilitato
        truck_only — solo truck (drone e robot disabilitati)
 
    Input:  E10_ablation.csv
    Output: results/plots/trieste/F10_ablation.pdf
    """
    _apply_style()
    df = _check_csv("E10", "F10")
    if df is None:
        return
 
    # ── Pulizia ──────────────────────────────────────────────────────────────
    df["cost"]        = pd.to_numeric(df["cost"])
    df["n_customers"] = pd.to_numeric(df["n_customers"])
    df["seed"]        = pd.to_numeric(df["seed"])
 
    CONFIG_ORDER  = ["all", "no_drone", "no_robot", "truck_only"]
    CONFIG_LABELS = ["All (base)", "No drone", "No robot", "Truck only"]
    C_CONFIGS     = PLOT_PARAMS["colors_config"]
    inst_seeds    = sorted(df["seed"].unique())
    C_SEEDS       = PLOT_PARAMS["colors_seeds"][: len(inst_seeds)]
    ekw           = dict(elinewidth=1.0, capsize=4, capthick=1.2,
                         ecolor="black", alpha=0.6)
 
    # ── Aggrega meta_seed → media per (inst_seed, config) ────────────────────
    by_inst = (
        df.groupby(["seed", "config"])["cost"]
        .mean()
        .reset_index()
        .rename(columns={"cost": "cost_inst"})
    )
 
    # ── Aggrega anche su inst_seed → media e std globali per config ───────────
    agg = (
        by_inst.groupby("config")["cost_inst"]
        .agg(cost_mean="mean", cost_std="std")
        .reindex(CONFIG_ORDER)
        .reset_index()
    )
 
    # ── Penalità % rispetto alla baseline "all" per ogni inst_seed ────────────
    baseline = by_inst[by_inst["config"] == "all"].set_index("seed")["cost_inst"]
    penalty  = by_inst[by_inst["config"] != "all"].copy()
    penalty["penalty_pct"] = penalty.apply(
        lambda r: (r["cost_inst"] - baseline[r["seed"]]) / baseline[r["seed"]] * 100,
        axis=1,
    )
    penalty_agg = (
        penalty.groupby("config")["penalty_pct"]
        .agg(pen_mean="mean", pen_std="std")
        .reindex([c for c in CONFIG_ORDER if c != "all"])
        .reset_index()
    )
 
    # ── Ordine panel sinistra: per costo crescente ────────────────────────────
    agg_sorted  = agg.sort_values("cost_mean").reset_index(drop=True)
    cfg_sorted  = agg_sorted["config"].tolist()
    lbl_sorted  = [CONFIG_LABELS[CONFIG_ORDER.index(c)] for c in cfg_sorted]
    col_sorted  = [C_CONFIGS[CONFIG_ORDER.index(c)] for c in cfg_sorted]
    y           = np.arange(len(cfg_sorted))
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_PARAMS["figsize_double"])
    for ax in (ax1, ax2):
        ax.xaxis.grid(True, alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)
        ax.yaxis.grid(False)
 
    # ── Panel 1: Costo per configurazione ────────────────────────────────────
    ax1.barh(y, agg_sorted["cost_mean"], xerr=agg_sorted["cost_std"],
             error_kw=ekw, color=col_sorted, alpha=0.80, zorder=3, height=0.55)
 
    # Punti per inst_seed
    jitter = np.linspace(-0.18, 0.18, len(inst_seeds))
    for yi, cfg in enumerate(cfg_sorted):
        for ji, (seed, col) in enumerate(zip(inst_seeds, C_SEEDS)):
            val = by_inst.loc[
                (by_inst["config"] == cfg) & (by_inst["seed"] == seed),
                "cost_inst"
            ]
            if not val.empty:
                ax1.scatter(val.values[0], yi + jitter[ji],
                            color=col, s=45, zorder=5,
                            edgecolors="white", linewidths=0.5,
                            label=f"seed={seed}" if yi == 0 else "_nolegend_")
 
    # Etichette valore
    for yi, (mean, std) in enumerate(
        zip(agg_sorted["cost_mean"], agg_sorted["cost_std"])
    ):
        ax1.text(mean + std + 3, yi, f"{mean:.0f}",
                 va="center", ha="left", fontsize=8.5,
                 color=col_sorted[yi], fontweight="bold")
 
    # Linea baseline "all"
    all_mean = agg.loc[agg["config"] == "all", "cost_mean"].iloc[0]
    ax1.axvline(all_mean, color="black", linewidth=1.2,
                linestyle="--", alpha=0.5, label=f"All ({all_mean:.0f})")
 
    ax1.set_yticks(y)
    ax1.set_yticklabels(lbl_sorted, fontsize=9)
    ax1.set_xlabel("Costo medio (€)", fontsize=10)
    ax1.set_title("Costo per configurazione", fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.85)
 
    # ── Panel 2: Penalità % vs baseline "all" ────────────────────────────────
    configs_pen = [c for c in CONFIG_ORDER if c != "all"]
    labels_pen  = [CONFIG_LABELS[CONFIG_ORDER.index(c)] for c in configs_pen]
    colors_pen  = [C_CONFIGS[CONFIG_ORDER.index(c)] for c in configs_pen]
    xp          = np.arange(len(configs_pen))
 
    # Barre senza yerr: con n=3 punti la std e' fuorviante,
    # la variabilita' e' gia' visibile nei punti individuali
    bars = ax2.bar(xp, penalty_agg["pen_mean"],
                   color=colors_pen, alpha=0.80, zorder=3, width=0.5)
 
    # Etichette % sopra alle barre
    for xi, mean in enumerate(penalty_agg["pen_mean"]):
        ax2.text(xi, mean + 0.8, f"+{mean:.1f}%",
                 ha="center", va="bottom", fontsize=9,
                 color=colors_pen[xi], fontweight="bold")
 
    ax2.axhline(0, color="black", linewidth=1.2, linestyle="--",
                alpha=0.5, label="Baseline (all = 0%)")
    ax2.set_xticks(xp)
    ax2.set_xticklabels(labels_pen, fontsize=9)
    ax2.set_ylabel("Penalità di costo vs baseline (%) — più alto = peggio",
                   fontsize=10)
    ax2.set_title("Contributo marginale di ogni modalità", fontweight="bold")
    ax2.set_ylim(bottom=-2)
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(True, alpha=0.2, linestyle="--")
    ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
 
    # ── Titolo e didascalia ───────────────────────────────────────────────────
    n_cust  = int(df["n_customers"].iloc[0])
    n_reps  = int(df.groupby(["seed", "config"]).size().max())
    fig.suptitle(
        f"Ablation study — valore marginale di drone e robot  "
        f"(n={n_cust}, {len(inst_seeds)} istanze × {n_reps} meta-rep)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.03,
        "Punti colorati = media per inst_seed  |  "
        "Barre = media globale ± std  |  "
        "Penalità calcolata su ogni istanza separatamente",
        ha="center", fontsize=8.5, color="gray",
    )
 
    plt.tight_layout()
    _save(fig, "F10_ablation", "E10", show)
    

def fig_F11(show: bool = False) -> None:
    """
    F11 — Robustezza al seed: confronto R=10 vs R=25 vs R=40.

    Legge tre CSV separati (uno per valore di R), ciascuno con 30 run
    su istanza fissa. Produce 4 panel:
      - Panel 0-2: dot plot ordinato per costo, uno per valore di R
      - Panel 3:   KDE overlay dei tre R a confronto

    I CSV attesi sono nella cartella results/csv/meta/:
        E11_seed_robustness_R_10.csv
        E11_seed_robustness_R_25.csv
        E11_seed_robustness_R_40.csv

    Output: results/plots/meta/F11_seed_robustness.pdf
    """
    from scipy.stats import gaussian_kde

    _apply_style()

    # ── Carica i tre CSV ──────────────────────────────────────────────────────
    R_values  = [10, 25, 40]
    colors    = ["#E53935", "#1565C0", "#2E7D32"]
    meta_dir  = DIRS["csv_meta"]

    datasets = []
    for R, col in zip(R_values, colors):
        path = meta_dir / f"E11_seed_robustness_R_{R}.csv"
        if not path.exists():
            print(f"  ⚠️  [F11] File non trovato: {path}")
            return
        df  = pd.read_csv(path)
        c   = np.sort(pd.to_numeric(df["cost"]).values)
        rt  = pd.to_numeric(df["runtime_s"]).mean()
        datasets.append((R, col, c, rt))

    def _stats(c, rt):
        return {
            "mean":   c.mean(),
            "std":    c.std(),
            "min":    c.min(),
            "max":    c.max(),
            "median": np.median(c),
            "q25":    np.percentile(c, 25),
            "q75":    np.percentile(c, 75),
            "cv":     c.std() / c.mean() * 100,
            "rt":     rt,
        }

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = fig.add_gridspec(1, 4, width_ratios=[2, 2, 2, 1.4], wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    ylim = (74, 91)

    # ── Panel 0-2: dot plot per ogni R ────────────────────────────────────────
    for ax, (R, col, c, rt) in zip(axes[:3], datasets):
        s = _stats(c, rt)

        for rank, cost in enumerate(c):
            alpha = 0.9 if cost == c.min() else 0.45
            ax.scatter(rank + 1, cost, color=col, s=55, alpha=alpha,
                       edgecolors="white", linewidths=0.4, zorder=4)

        ax.axhline(s["min"],    color="#2E7D32", ls="--", lw=1.3,
                   label=f"Best   {s['min']:.2f}")
        ax.axhline(s["median"], color=col,       ls="-.", lw=1.3,
                   label=f"Median {s['median']:.2f}")
        ax.axhline(s["mean"],   color="#6A1B9A", ls=":",  lw=1.3,
                   label=f"Mean   {s['mean']:.2f}")
        ax.axhspan(s["q25"], s["q75"], alpha=0.08, color=col)

        ax.set_title(f"R={R}  (~{rt:.0f}s/run)", fontweight="bold", color=col)
        ax.set_xlabel("Run (ordinato per costo)")
        ax.set_ylabel("Costo (€)")
        ax.set_xlim(0, 31)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=7.8, loc="upper left", framealpha=0.85)
        ax.text(0.98, 0.97, f"CV={s['cv']:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=col, fontweight="bold")

    # ── Panel 3: KDE overlay ──────────────────────────────────────────────────
    ax4   = axes[3]
    x_kde = np.linspace(74, 92, 400)

    for R, col, c, rt in datasets:
        kde = gaussian_kde(c, bw_method=0.35)
        ax4.fill_between(x_kde, kde(x_kde), alpha=0.12, color=col)
        ax4.plot(x_kde, kde(x_kde), color=col, lw=2, label=f"R={R}")
        ax4.axvline(_stats(c, rt)["mean"], color=col, ls=":", lw=1)

    ax4.set_xlabel("Costo (€)")
    ax4.set_ylabel("Densità")
    ax4.set_title("KDE confronto", fontweight="bold")
    ax4.legend(fontsize=9)

    # ── Riga riepilogativa ────────────────────────────────────────────────────
    summary_str = "  |  ".join(
        f"R={R}: mean={_stats(c,rt)['mean']:.2f}  "
        f"CV={_stats(c,rt)['cv']:.1f}%  "
        f"best={_stats(c,rt)['min']:.2f}  "
        f"~{rt:.0f}s"
        for R, col, c, rt in datasets
    )
    fig.text(0.5, -0.06, summary_str, ha="center", fontsize=9, color="gray")

    fig.suptitle(
        "F11 — Robustezza al seed: R=10 vs R=25 vs R=40  "
        f"(n={int(pd.read_csv(meta_dir / 'E11_seed_robustness_R_25.csv')['n_customers'].iloc[0])}, "
        "30 seed per valore di R)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    _save(fig, "F11_seed_robustness", "E11", show)


# ═══════════════════════════════════════════════════════════════════════════════
#  DISPATCH TABLE e MAIN
# ═══════════════════════════════════════════════════════════════════════════════

FIGURES = {
    "F1":  fig_F1,    # E1 — Scalabilità MILP
    "F2":  fig_F2,    # E2 — Valid Inequalities
    "F3":  fig_F3,    # E3 — Scalabilità metaeuristica
    "F4":  fig_F4,    # E4 — MILP vs Metaeuristica
    "F5":  fig_F5,    # E5 — Convergenza
    "F6":  fig_F6,    # E6 — Warm start
    "F7":  fig_F7,    # E7 — Sensitivity β
    "F8":  fig_F8,    # E8 — Sensitivity R
    "F9":  fig_F9,    # E9 — Modal split Trieste
    "F10": fig_F10,   # E10 — Ablation study
    "F11": fig_F11,   # E11 — Robustezza al seed
}

ALL_ORDER = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generazione figure presentazione VRPD-RS-TW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python plot_results.py --fig F5
  python plot_results.py --fig F1 F2 F3
  python plot_results.py --fig all
  python plot_results.py --fig F5 --show
""",
    )
    parser.add_argument(
        "--fig", nargs="+",
        choices=list(FIGURES.keys()) + ["all"],
        help="Figura/e da generare",
        required=True,
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Apre la finestra interattiva matplotlib dopo il salvataggio",
    )
    args = parser.parse_args()

    ensure_dirs()
    figs_to_run = ALL_ORDER if "all" in args.fig else args.fig

    for fig_id in figs_to_run:
        print(f"\n[{fig_id}]")
        FIGURES[fig_id](show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()