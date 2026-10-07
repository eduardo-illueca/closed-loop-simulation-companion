"""Observation process: visits, detection delays, and artifact injection.

Review fixes: (1) no events recorded after death -- follow-up is truncated at
min(t_max, death) and disease detection is guarded to occur strictly before
death; (2) deterministic activity ordering via ACTIVITY_ORDER for same-time ties.

This layer MANUFACTURES the biases. It must never feed back into the DGP.
"""
from __future__ import annotations

import numpy as np

from closed_loop_sim.config import ObservationConfig

History = list
Events = list  # list[tuple[str, float]]

_STATE_ACTIVITY = {"S1": "CKD_recorded", "S2": "CVAE_recorded"}

# Deterministic tie-break for events sharing a timestamp (review fix 2).
ACTIVITY_ORDER = {
    "IndexDrug": 0,
    "CKD_recorded": 1,
    "CVAE_recorded": 2,
    "MedicationChange": 3,
    "CKD_recode": 4,
    "Death": 999,
}


def _sort_events(events: Events) -> Events:
    return sorted(events, key=lambda e: (e[1], ACTIVITY_ORDER.get(e[0], 500)))


def generate_visits(E: int, cfg: ObservationConfig, horizon: float,
                    rng: np.random.Generator) -> list:
    """Homogeneous Poisson visits on [0, horizon); rate scales with kappa for PPI."""
    rate = cfg.lambda_base * (cfg.kappa if E == 1 else 1.0)
    visits: list = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= horizon:
            break
        visits.append(t)
    return visits


def _first_visit_at_or_after(visits: list, onset: float):
    return next((v for v in visits if v >= onset), None)


def observe_trajectory(history: History, E: int, visits: list) -> Events:
    """Visit-gated detection of CKD/CVAE (only if the detection visit is before
    death); exact recording of Death."""
    onset = {s: t for s, t in history}
    death_time = onset.get("S3")
    events: Events = [("IndexDrug", 0.0)]
    for state, activity in _STATE_ACTIVITY.items():
        if state in onset:
            tt = _first_visit_at_or_after(visits, onset[state])
            if tt is not None and (death_time is None or tt < death_time):
                events.append((activity, tt))
    if death_time is not None:
        events.append(("Death", death_time))
    return _sort_events(events)


def inject_artifacts(events: Events, cfg: ObservationConfig, visits: list,
                     rng: np.random.Generator) -> Events:
    """Decoy (MedicationChange, S2->S0) and coding shift (CKD_recode, S2->S1),
    each at the next visit after a recorded CVAE. Visits are already truncated at
    follow-up end, so artifacts cannot land after death.
    
    Action A10 Challenge Scenario: Supports held-out artifact class (Lab_Reassay)
    unknown to standard keyword-matching gates.
    """
    cvae_time = next((t for a, t in events if a == "CVAE_recorded"), None)
    if cvae_time is None:
        return events
    next_visit = next((v for v in visits if v > cvae_time), None)
    out = list(events)
    if next_visit is not None:
        if rng.random() < cfg.p_switch:
            out.append(("MedicationChange", next_visit))
        if rng.random() < cfg.p_recode:
            out.append(("CKD_recode", next_visit))
        if cfg.challenge_scenario == "held_out_artifact" or getattr(cfg, "p_held_out", 0.0) > 0:
            if rng.random() < getattr(cfg, "p_held_out", 0.35):
                out.append(("Lab_Reassay", next_visit))
    return _sort_events(out)

