"""Calibration against manuscript Table 4.

Per review fix 3, Table 4 reflects OBSERVED process-mining transitions, so the
authoritative gate compares the recorded event log (after visit-gated detection
and artifact injection) to Table 4 -- not the latent rates. A latent plausibility
report is also provided as a sanity check.
"""
from __future__ import annotations

import pandas as pd

from closed_loop_sim.config import SimConfig
from closed_loop_sim.dgp import PopulationData
from closed_loop_sim.eventlog import build_event_log_data
from closed_loop_sim.truth import marginal_transition_rates

# Manuscript Table 4 marginal risks (proportion), PPI and H2B.
TABLE4_TARGETS = {
    "S0->S1": {"PPI": 0.253, "H2B": 0.198},
    "S0->S2": {"PPI": 0.045, "H2B": 0.036},
    "S0->S3": {"PPI": 0.051, "H2B": 0.027},
    "S1->S2": {"PPI": 0.167, "H2B": 0.149},
    "S1->S3": {"PPI": 0.239, "H2B": 0.179},
    "S2->S3": {"PPI": 0.470, "H2B": 0.379},
}
# Table 4 also reports the artifactual reverse transition CVAE->CKD (~0.19/0.20).
CODING_SHIFT_TARGET = {"PPI": 0.190, "H2B": 0.201}


def observed_transition_risks(df: pd.DataFrame) -> dict:
    """Per-arm observed transition risk from the event log (eventually-follows;
    denominator = patients who entered the origin activity). Approximate -- the
    full DFG-based version arrives with discovery.py in Plan 2."""
    out: dict = {}
    keys = ["S0->S1", "S0->S2", "S0->S3", "S1->S2", "S1->S3", "S2->S3", "S2->S1"]
    for arm in ("PPI", "H2B"):
        g = df[df.group == arm].sort_values(["case", "time"])
        denom = {"S0": 0, "S1": 0, "S2": 0}
        num = {k: 0 for k in keys}
        for _, sub in g.groupby("case"):
            acts = sub["activity"].tolist()
            idx = {a: i for i, a in enumerate(acts)}
            ckd = "CKD_recorded" in idx
            cvae = "CVAE_recorded" in idx
            death = "Death" in idx
            denom["S0"] += 1
            if ckd:
                num["S0->S1"] += 1
            if cvae and (not ckd or idx["CVAE_recorded"] < idx["CKD_recorded"]):
                num["S0->S2"] += 1
            if death and not ckd and not cvae:
                num["S0->S3"] += 1
            if ckd:
                denom["S1"] += 1
                if cvae and idx["CVAE_recorded"] > idx["CKD_recorded"]:
                    num["S1->S2"] += 1
                if death and idx["Death"] > idx["CKD_recorded"]:
                    num["S1->S3"] += 1
            if cvae:
                denom["S2"] += 1
                if death and idx["Death"] > idx["CVAE_recorded"]:
                    num["S2->S3"] += 1
                if "CKD_recode" in idx:
                    num["S2->S1"] += 1
        out[arm] = {tr: (num[tr] / denom[tr.split("->")[0]] if denom[tr.split("->")[0]] else 0.0)
                    for tr in num}
    return out


def observed_calibration_report(cfg: SimConfig, n: int, seed: int, tol: float = 0.08) -> dict:
    data = build_event_log_data(n, cfg, seed)
    obs = observed_transition_risks(data.log)
    rows = []
    worst = 0.0
    for tr, target in TABLE4_TARGETS.items():
        for arm in ("PPI", "H2B"):
            got = obs[arm][tr]
            gap = abs(got - target[arm])
            worst = max(worst, gap)
            rows.append((tr, arm, target[arm], round(got, 3), round(gap, 3)))
    return {"rows": rows, "worst_gap": worst, "within_tol": worst <= tol,
            "observed": obs, "coding_shift": {a: obs[a]["S2->S1"] for a in ("PPI", "H2B")}}


def latent_plausibility_report(cfg: SimConfig, n: int, seed: int) -> dict:
    pop = PopulationData(*(lambda d: (d.L, d.L_std, d.E, d.histories))(
        build_event_log_data(n, cfg, seed)))
    rates = marginal_transition_rates(pop)
    ok = (rates["PPI"]["S0->S1"] > rates["H2B"]["S0->S1"]
          and all(0.0 <= v <= 1.0 for arm in rates.values() for v in arm.values()))
    return {"rates": rates, "plausible": ok}
