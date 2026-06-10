"""Structural and estimand metrics, graded against the oracle."""
from __future__ import annotations

from closed_loop_sim.refinement import RefinementResult
from closed_loop_sim.truth import TRAP_EDGES, TRUE_EDGES


def structural_metrics(res: RefinementResult) -> dict:
    recovered = set(res.edges)
    tp = len(recovered & TRUE_EDGES)
    fp = len(recovered - TRUE_EDGES)
    fn = len(TRUE_EDGES - recovered)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # SHD = wrong edges (extra + missing), undirected-direction handled by edge set.
    shd = fp + fn
    traps_accepted = sorted(recovered & TRAP_EDGES)
    return {
        "precision": precision, "recall": recall, "f1": f1, "shd": shd,
        "death_node_added": "S3" in res.nodes,
        "n_traps_accepted": len(traps_accepted),
        "accepted_S2_S0": ("S2", "S0") in recovered,
        "accepted_S2_S1": ("S2", "S1") in recovered,
        "accepts_any_trap": len(traps_accepted) >= 1,
    }
