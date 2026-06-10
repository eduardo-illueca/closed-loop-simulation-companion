"""Phase 5: run replications of a cell and aggregate the metrics.

Reproducibility: one master SeedSequence -> deterministic per-replication seeds
(spec 7.4). The mediation oracle is computed once at large n, not per replication.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from closed_loop_sim.config import SimConfig, base_cell
from closed_loop_sim.discovery import discover
from closed_loop_sim.estimation import estimate
from closed_loop_sim.eventlog import build_event_log_data
from closed_loop_sim.metrics import structural_metrics
from closed_loop_sim.refinement import ARMS
from closed_loop_sim.truth import mediated_proportion, true_beta_E


def run_replication(cfg: SimConfig, seed: int, with_estimation: bool = True) -> dict:
    data = build_event_log_data(cfg.n, cfg, seed)
    disc = discover(data.log, data.visit_counts, compute_activity_dfg=False)
    out: dict = {"seed": seed, "structural": {}}
    for arm, fn in ARMS.items():
        out["structural"][arm] = structural_metrics(fn(disc))
    if with_estimation:
        out["estimation"] = estimate(data, cfg.dgp.t_max)
    return out


def _wilson(k: int, n: int) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def run_cell(cfg: SimConfig, M: int, master_seed: int = 20260610,
             with_estimation: bool = True) -> dict:
    seeds = np.random.SeedSequence(master_seed).generate_state(M)
    reps = [run_replication(cfg, int(s), with_estimation) for s in seeds]

    arms = ["A", "B", "C"]
    struct = {a: {} for a in arms}
    for a in arms:
        for key in ["precision", "recall", "f1", "shd", "n_traps_accepted"]:
            vals = [r["structural"][a][key] for r in reps]
            struct[a][key + "_mean"] = float(np.mean(vals))
            struct[a][key + "_sd"] = float(np.std(vals))
        k = sum(r["structural"][a]["accepts_any_trap"] for r in reps)
        struct[a]["accept_any_trap_rate"] = k / M
        struct[a]["accept_any_trap_ci"] = _wilson(k, M)
        struct[a]["reject_S2_S0_rate"] = float(np.mean([not r["structural"][a]["accepted_S2_S0"] for r in reps]))
        struct[a]["reject_S2_S1_rate"] = float(np.mean([not r["structural"][a]["accepted_S2_S1"] for r in reps]))
        struct[a]["death_node_rate"] = float(np.mean([r["structural"][a]["death_node_added"] for r in reps]))

    result = {"M": M, "n": cfg.n, "structural": struct}

    if with_estimation:
        truth01 = true_beta_E(cfg, "S0->S1")
        truth12 = true_beta_E(cfg, "S1->S2")
        est = {"truth_S0S1": truth01, "truth_S1S2": truth12}
        for key, truth in [("s0s1_latent", truth01), ("s0s1_naive", truth01),
                           ("s0s1_adjusted", truth01), ("s1s2", truth12)]:
            betas = np.array([r["estimation"][key]["beta_E"] for r in reps])
            cover = np.mean([
                r["estimation"][key]["ci_low"] <= truth <= r["estimation"][key]["ci_high"]
                for r in reps])
            est[key] = {
                "mean_beta": float(betas.mean()),
                "sd": float(betas.std()),
                "p2_5": float(np.percentile(betas, 2.5)),
                "p97_5": float(np.percentile(betas, 97.5)),
                "bias": float(betas.mean() - truth),
                "rmse": float(np.sqrt(np.mean((betas - truth) ** 2))),
                "coverage": float(cover),
            }
        result["estimation"] = est

    # Per-replication rows for archival (all M; no second pass).
    per_rep: list = []
    for r in reps:
        base = {"seed": r["seed"]}
        if with_estimation:
            for key in ["s0s1_latent", "s0s1_naive", "s0s1_adjusted", "s1s2"]:
                base[f"beta_{key}"] = r["estimation"][key]["beta_E"]
        for a in arms:
            per_rep.append({**base, "arm": a, **r["structural"][a]})
    result["_per_rep"] = per_rep

    return result


def reps_to_frame(cfg: SimConfig, M: int, master_seed: int = 20260610) -> pd.DataFrame:
    seeds = np.random.SeedSequence(master_seed).generate_state(M)
    rows = []
    for s in seeds:
        r = run_replication(cfg, int(s), with_estimation=True)
        for a in ["A", "B", "C"]:
            row = {"seed": int(s), "arm": a, **r["structural"][a]}
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = base_cell()
    res = run_cell(cfg, M=20, with_estimation=True)
    import json
    print(json.dumps(res, indent=2, default=str))
