"""Run the full base-cell study and write results + metadata to results/.

Produces: aggregated A/B/C structural + estimand metrics over M replications, the
g-computation mediated-proportion oracle, the observed-log Table-4 calibration
report, a representative single-run DFG (pm4py), and a per-replication CSV.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from closed_loop_sim.calibration import latent_plausibility_report, observed_calibration_report
from closed_loop_sim.config import base_cell
from closed_loop_sim.discovery import discover
from closed_loop_sim.eventlog import build_event_log_data
from closed_loop_sim.experiment import run_cell
from closed_loop_sim.truth import mediated_proportion

M = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def main() -> None:
    cfg = base_cell()

    print(f"[1/5] base cell M={M} n={cfg.n} ...", flush=True)
    cell = run_cell(cfg, M=M, with_estimation=True)

    print("[2/5] mediation oracle (n=50000) ...", flush=True)
    med = mediated_proportion(cfg, n=50000, seed=999, horizon=cfg.dgp.t_max)

    print("[3/5] calibration reports ...", flush=True)
    obs_cal = observed_calibration_report(cfg, n=20000, seed=7, tol=0.08)
    lat = latent_plausibility_report(cfg, n=20000, seed=7)

    print("[4/5] representative DFG (pm4py) ...", flush=True)
    data = build_event_log_data(cfg.n, cfg, seed=1)
    disc = discover(data.log, data.visit_counts, compute_activity_dfg=True)
    state_edges = {f"{a}->{b}": {"risk": round(e.risk, 3), "count": e.count,
                                 "median_time": round(e.median_time, 2),
                                 "src": sorted(e.source_activities),
                                 "attenuation": round(e.monitoring_attenuation, 2)}
                   for (a, b), e in disc.state_edges.items()}

    print("[5/5] per-replication CSV (all M) ...", flush=True)
    per_rep = cell.pop("_per_rep")
    pd.DataFrame(per_rep).to_csv(OUT / "replications.csv", index=False)

    import lifelines
    import pm4py
    meta = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "lifelines": lifelines.__version__,
            "pm4py": pm4py.__version__, "master_seed": 20260610, "M": M, "n": cfg.n}

    result = {"meta": meta, "config": cfg.model_dump(), "cell": cell,
              "mediation_oracle": med, "observed_calibration": obs_cal,
              "latent_plausibility": lat, "representative_state_edges": state_edges,
              "representative_activity_dfg": {f"{a}->{b}": int(c)
                                              for (a, b), c in disc.activity_dfg.items()}}
    (OUT / "base_cell_results.json").write_text(json.dumps(result, indent=2, default=str))
    print("DONE -> results/base_cell_results.json", flush=True)


if __name__ == "__main__":
    main()
