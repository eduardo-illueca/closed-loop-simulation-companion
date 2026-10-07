"""Phase 2 discovery: directly-follows graph and the state-level transition
summary that carries the falsification evidence consumed by the gate.

pm4py discovers the activity-level DFG (the same engine the deployed tool wraps);
the state-level summary (risks, median times, monitoring-stratified strengths,
generating activities) is computed directly from the log for refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Map recorded activities to latent-state nodes (deterministic spec A.1).
ACTIVITY_TO_NODE = {
    "IndexDrug": "S0",
    "CKD_recorded": "S1",
    "CVAE_recorded": "S2",
    "Death": "S3",
    "MedicationChange": "S0",   # decoy: medication change maps to the drug node
    "CKD_recode": "S1",          # coding shift maps to the CKD node
    "Lab_Reassay": "S2",         # Action A10 held-out artifact class
}
ADMIN_ACTIVITIES = {"MedicationChange", "Lab_Reassay"}  # administrative/lab, not biological transitions



@dataclass
class StateEdge:
    risk: float                       # relative-antecedent frequency
    count: int                        # cases with the directly-follows pair
    median_time: float                # median (t_to - t_from) over occurrences
    source_activities: set            # activities that generated this edge
    risk_high_mon: float              # risk among high-monitoring patients
    risk_low_mon: float               # risk among low-monitoring patients

    @property
    def monitoring_attenuation(self) -> float:
        """Fractional drop in risk from high- to low-monitoring strata. Large for
        surveillance-driven edges; ~0 for monitoring-insensitive ones."""
        if self.risk_high_mon <= 0:
            return 0.0
        return (self.risk_high_mon - self.risk_low_mon) / self.risk_high_mon


@dataclass
class DiscoveryResult:
    state_edges: dict                 # (a, b) -> StateEdge
    activity_dfg: dict = field(default_factory=dict)  # pm4py activity-level DFG


def _to_pm4py(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"case": "case:concept:name", "activity": "concept:name"}).copy()
    out["case:concept:name"] = out["case:concept:name"].astype(str)  # pm4py requires string case id
    out["time:timestamp"] = pd.Timestamp("2010-01-01") + pd.to_timedelta(out["time"] * 365.25, unit="D")
    return out


def _activity_dfg(df: pd.DataFrame) -> dict:
    import pm4py
    dfg, _start, _end = pm4py.discover_directly_follows_graph(_to_pm4py(df))
    return {k: int(v) for k, v in dfg.items()}


def discover(df: pd.DataFrame, visit_counts: np.ndarray,
             compute_activity_dfg: bool = True) -> DiscoveryResult:
    df = df.sort_values(["case", "time"])
    mon_median = float(np.median(visit_counts))
    high_mon = set(np.where(visit_counts > mon_median)[0].tolist())

    origin_cases: dict = {}              # node -> set of cases where it occurs
    pair_cases: dict = {}                # (a,b) -> set of cases with the pair
    pair_delays: dict = {}               # (a,b) -> list of delays
    pair_src: dict = {}                  # (a,b) -> set of target activities

    for case, sub in df.groupby("case"):
        rows = list(zip(sub["activity"].tolist(), sub["time"].tolist()))
        nodes_seen = set()
        for act, _t in rows:
            nodes_seen.add(ACTIVITY_TO_NODE.get(act, "S0"))
        for nd in nodes_seen:
            origin_cases.setdefault(nd, set())  # ensure key exists
        for nd in nodes_seen:
            origin_cases[nd].add(case)
        for (a_act, a_t), (b_act, b_t) in zip(rows, rows[1:]):
            a, b = ACTIVITY_TO_NODE.get(a_act, "S0"), ACTIVITY_TO_NODE.get(b_act, "S0")

            if a == b:
                continue
            edge = (a, b)
            pair_cases.setdefault(edge, set()).add(case)
            pair_delays.setdefault(edge, []).append(b_t - a_t)
            pair_src.setdefault(edge, set()).add(b_act)

    state_edges: dict = {}
    for edge, cases in pair_cases.items():
        a = edge[0]
        denom = len(origin_cases.get(a, set()))
        risk = len(cases) / denom if denom else 0.0
        hi = origin_cases.get(a, set()) & high_mon
        lo = origin_cases.get(a, set()) - high_mon
        rh = len(cases & high_mon) / len(hi) if hi else 0.0
        rl = len(cases - high_mon) / len(lo) if lo else 0.0
        state_edges[edge] = StateEdge(
            risk=risk, count=len(cases),
            median_time=float(np.median(pair_delays[edge])),
            source_activities=pair_src[edge], risk_high_mon=rh, risk_low_mon=rl,
        )
    activity_dfg = _activity_dfg(df) if compute_activity_dfg else {}
    return DiscoveryResult(state_edges=state_edges, activity_dfg=activity_dfg)
