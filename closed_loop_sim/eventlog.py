"""Assemble the patient-level activity event log.

Wires simulate_population -> generate_visits -> observe_trajectory ->
inject_artifacts with separate DGP/observation RNG streams (review fix 5d) and
death-truncated follow-up (review fix 1). Returns a tidy DataFrame and the latent
covariate/exposure arrays needed downstream by estimation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from closed_loop_sim.config import GRANULARITY_STEP, SimConfig
from closed_loop_sim.dgp import (
    assign_exposure, draw_covariates, simulate_trajectory, spawn_patient_seeds,
    standardize,
)
from closed_loop_sim.observation import (
    ACTIVITY_ORDER, generate_visits, inject_artifacts, observe_trajectory,
)


@dataclass
class EventLogData:
    log: pd.DataFrame          # columns: case, activity, time, group
    L: np.ndarray              # (n, 5) raw covariates
    L_std: np.ndarray          # (n, 5) standardized
    E: np.ndarray              # (n,) exposure
    histories: list            # latent histories (for oracle cross-checks)
    visit_counts: np.ndarray   # (n,) number of visits per patient (monitoring intensity)


def _death_time(history) -> float | None:
    return next((t for s, t in history if s == "S3"), None)


def build_event_log_data(n: int, cfg: SimConfig, seed: int) -> EventLogData:
    setup_rng, patient_seqs = spawn_patient_seeds(seed, n)
    L = draw_covariates(n, cfg.dgp.covariates, setup_rng)
    L_std = standardize(L)
    E = assign_exposure(L_std, cfg.dgp.propensity, setup_rng)

    records: list = []
    histories: list = []
    visit_counts = np.zeros(n, dtype=int)
    for i in range(n):
        dgp_seed, obs_seed = patient_seqs[i].spawn(2)
        dgp_rng = np.random.default_rng(dgp_seed)
        obs_rng = np.random.default_rng(obs_seed)

        history = simulate_trajectory(L_std[i], int(E[i]), cfg.dgp, dgp_rng)
        histories.append(history)

        death = _death_time(history)
        followup_end = min(cfg.dgp.t_max, death) if death is not None else cfg.dgp.t_max
        visits = generate_visits(int(E[i]), cfg.obs, followup_end, obs_rng)
        visit_counts[i] = len(visits)

        events = observe_trajectory(history, int(E[i]), visits)
        events = inject_artifacts(events, cfg.obs, visits, obs_rng)

        group = "PPI" if E[i] == 1 else "H2B"
        for activity, t in events:
            records.append({"case": i, "activity": activity, "time": float(t), "group": group})

    df = pd.DataFrame.from_records(records, columns=["case", "activity", "time", "group"])
    step = GRANULARITY_STEP[cfg.obs.time_granularity]
    if step is None:
        # Exact: deterministic, disease-order tie-break (reproducible base study).
        df["_ord"] = df["activity"].map(ACTIVITY_ORDER).fillna(500)
        df = df.sort_values(["case", "time", "_ord"]).drop(columns="_ord")
    else:
        # Coarse: round recorded timestamps; same-bin events have ambiguous order,
        # modelled by a reproducible random tie-break (loss of temporal precedence).
        df["time"] = np.round(df["time"] / step) * step
        df["_tb"] = np.random.default_rng(seed ^ 0x9E3779B9).random(len(df))
        df = df.sort_values(["case", "time", "_tb"]).drop(columns="_tb")
    df = df.reset_index(drop=True)
    return EventLogData(log=df, L=L, L_std=L_std, E=E,
                        histories=histories, visit_counts=visit_counts)


def build_event_log(n: int, cfg: SimConfig, seed: int) -> pd.DataFrame:
    return build_event_log_data(n, cfg, seed).log
