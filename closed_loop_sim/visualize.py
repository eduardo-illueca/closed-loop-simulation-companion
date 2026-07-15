"""Visualise the discovered process map (state-level DFG) and the DAG refinement
outcome across arms. Outputs results/process_map.png and results/dag_refinement.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from closed_loop_sim.config import base_cell
from closed_loop_sim.discovery import discover
from closed_loop_sim.eventlog import build_event_log_data
from closed_loop_sim.refinement import ARMS, INITIAL_EDGES
from closed_loop_sim.truth import TRAP_EDGES, TRUE_EDGES

ROOT = Path(__file__).resolve().parent.parent
POS = {"S0": (0.0, 0.0), "S1": (1.6, 1.05), "S2": (3.2, 0.0), "S3": (4.8, 1.05)}
LABELS = {"S0": "Drug\n(index)", "S1": "CKD", "S2": "CVAE", "S3": "Death\n(absorbing)"}
NODE_COLOR = {"S0": "#3498db", "S1": "#9b59b6", "S2": "#1abc9c", "S3": "#e74c3c"}
# curvature per edge so arrows don't overlap
RAD = {("S0", "S1"): 0.0, ("S0", "S2"): 0.0, ("S0", "S3"): 0.30,
       ("S1", "S2"): 0.0, ("S1", "S3"): 0.0, ("S2", "S3"): 0.0,
       ("S2", "S0"): -0.35, ("S2", "S1"): -0.30}


def _draw_node(ax, s):
    x, y = POS[s]
    ax.scatter([x], [y], s=2600, c=NODE_COLOR[s], zorder=3, edgecolors="k", linewidths=1.2)
    ax.text(x, y, LABELS[s], ha="center", va="center", fontsize=9,
            color="white", fontweight="bold", zorder=4)


def _draw_edge(ax, a, b, color, style="-", label=None, lw=1.8, alpha=1.0):
    rad = RAD.get((a, b), 0.0)
    arrow = FancyArrowPatch(POS[a], POS[b], connectionstyle=f"arc3,rad={rad}",
                            arrowstyle="-|>", mutation_scale=16, lw=lw, color=color,
                            linestyle=style, alpha=alpha, shrinkA=26, shrinkB=26, zorder=2)
    ax.add_patch(arrow)
    if label:
        mx, my = (POS[a][0] + POS[b][0]) / 2, (POS[a][1] + POS[b][1]) / 2
        dx, dy = POS[b][0] - POS[a][0], POS[b][1] - POS[a][1]
        nx, ny = -dy, dx
        norm = np.hypot(nx, ny) or 1
        off = 0.20 + 0.9 * abs(rad)
        sign = 1 if rad >= 0 else -1
        ax.text(mx + sign * off * nx / norm, my + sign * off * ny / norm, label,
                ha="center", va="center", fontsize=8, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def _frame(ax, title):
    for s in POS:
        _draw_node(ax, s)
    ax.set_xlim(-0.9, 5.7)
    ax.set_ylim(-0.9, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def process_map(disc):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    _frame(ax, "Discovered process map (state-level directly-follows graph, n = 5,000)")
    for (a, b), e in disc.state_edges.items():
        is_trap = (a, b) in TRAP_EDGES
        color = "#c0392b" if is_trap else "#34495e"
        style = (0, (4, 2)) if is_trap else "-"
        lab = f"{100 * e.risk:.0f}%\n({e.median_time:.1f} y)"
        _draw_edge(ax, a, b, color, style=style, label=lab,
                   lw=1.2 + 3.5 * e.risk, alpha=0.95)
    ax.scatter([], [], c="#c0392b", marker="_", s=200, label="injected artefact (trap)")
    ax.plot([], [], color="#34495e", label="recorded transition")
    ax.legend(loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, 0.02, "edge label = relative-antecedent frequency (median time); "
             "S2→S0 = drug-feedback decoy, S2→S1 = reverse coding shift",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = ROOT / "results" / "process_map.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


def _dag_panel(ax, edges, title, has_death):
    nodes = ["S0", "S1", "S2"] + (["S3"] if has_death else [])
    for s in nodes:
        _draw_node(ax, s)
    for (a, b) in edges:
        trap = (a, b) in TRAP_EDGES
        color = "#c0392b" if trap else "#27ae60"
        style = (0, (4, 2)) if trap else "-"
        _draw_edge(ax, a, b, color, style=style, lw=2.0)
    ax.set_xlim(-0.9, 5.7)
    ax.set_ylim(-0.9, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def dag_refinement(refined):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    _dag_panel(axes[0], TRUE_EDGES, "True DAG (oracle)", True)
    a_edges = refined["A"]
    b_edges = refined["B"]
    _dag_panel(axes[1], a_edges,
               f"Arm A — ungated: accepts {len(a_edges & TRAP_EDGES)} trap(s)",
               "S3" in {n for e in a_edges for n in e})
    _dag_panel(axes[2], b_edges,
               f"Arm B — governed: rejects both traps",
               "S3" in {n for e in b_edges for n in e})
    fig.suptitle("DAG refinement: starting graph {S0→S1, S0→S2, S1→S2} → refined graph per arm",
                 fontsize=12)
    fig.text(0.5, 0.02, "green = accepted edge · red dashed = injected artefact accepted (Arm A) · "
             "Arm B adds Death competing-risk node and rejects both traps",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = ROOT / "results" / "dag_refinement.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    cfg = base_cell()
    data = build_event_log_data(cfg.n, cfg, seed=1)
    disc = discover(data.log, data.visit_counts, compute_activity_dfg=False)
    process_map(disc)
    
    from closed_loop_sim.rdf_export import export_graphs_to_rdf
    refined_b = ARMS["B"](disc)
    export_graphs_to_rdf(cfg, disc, refined_b, ROOT / "results")

    refined = {arm: set(fn(disc).edges) for arm, fn in ARMS.items()}
    dag_refinement(refined)
    print("initial edges:", sorted(INITIAL_EDGES))
    for arm in "ABC":
        print(f"Arm {arm} edges:", sorted(refined[arm]), "| traps:", sorted(refined[arm] & TRAP_EDGES))


if __name__ == "__main__":
    main()
