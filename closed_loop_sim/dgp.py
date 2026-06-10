"""Latent data-generating process: the ground truth.

Weibull-PH cause-specific hazards, a clock-reset competing-risks trajectory
sampler, and a population simulator. Per-patient reproducibility uses numpy
``SeedSequence`` with separate DGP and observation streams (review fix 5d), so
the latent truth is invariant to changes in the observation model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from closed_loop_sim.config import (
    CovariateConfig, DGPConfig, PropensityConfig, SimConfig,
)

# True allowed transitions out of each latent state (spec section 2.1).
TRUE_TRANSITIONS: dict[str, list[str]] = {
    "S0": ["S1", "S2", "S3"],
    "S1": ["S2", "S3"],
    "S2": ["S3"],
    "S3": [],
}

History = list  # list[tuple[str, float]] of (state, entry_time)


def sample_transition_time(lam: float, rho: float, linpred: float,
                           rng: np.random.Generator) -> float:
    """Inverse-CDF draw from a Weibull-PH cause-specific hazard (clock reset at entry).

    h(t)=lam*rho*t^(rho-1)*exp(linpred) => S(t)=exp(-a*t^rho), a=lam*exp(linpred).
    """
    a = lam * np.exp(linpred)
    u = rng.random()
    return float((-np.log(u) / a) ** (1.0 / rho))


def draw_covariates(n: int, cfg: CovariateConfig,
                    rng: np.random.Generator) -> np.ndarray:
    """Columns: age, sex(1=female), diabetes(0/1), hypertension(0/1), egfr."""
    age = rng.normal(cfg.age_mean, cfg.age_sd, n)
    sex = (rng.random(n) < cfg.sex_female_prob).astype(float)
    diabetes = (rng.random(n) < cfg.diabetes_prob).astype(float)
    hypertension = (rng.random(n) < cfg.hypertension_prob).astype(float)
    egfr = rng.normal(cfg.egfr_mean, cfg.egfr_sd, n)
    return np.column_stack([age, sex, diabetes, hypertension, egfr])


def standardize(L: np.ndarray) -> np.ndarray:
    mu = L.mean(axis=0)
    sd = L.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (L - mu) / sd


def assign_exposure(L_std: np.ndarray, cfg: PropensityConfig,
                    rng: np.random.Generator) -> np.ndarray:
    logit = cfg.alpha0 + L_std @ np.asarray(cfg.alpha)
    p = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(len(p)) < p).astype(int)


def _linpred(tp, e: int, L_std: np.ndarray) -> float:
    return tp.beta_E * e + float(np.asarray(tp.beta_L) @ L_std)


def simulate_trajectory(L_std: np.ndarray, E, cfg: DGPConfig,
                        rng: np.random.Generator) -> History:
    """Clock-reset competing-risks walk until Death or t_max.

    ``E`` may be an int (constant exposure) or a callable transition->0/1 (used by
    the mediation oracle to override exposure per transition).
    """
    e_func: Callable[[str], int] = E if callable(E) else (lambda _t: int(E))
    state, t = "S0", 0.0
    history: History = [("S0", 0.0)]
    while state != "S3" and t < cfg.t_max:
        outs = TRUE_TRANSITIONS[state]
        if not outs:
            break
        best_k, best_dt = None, np.inf
        for k in outs:
            tp = cfg.transitions[f"{state}->{k}"]
            dt = sample_transition_time(tp.lam, tp.rho,
                                        _linpred(tp, e_func(f"{state}->{k}"), L_std), rng)
            if dt < best_dt:
                best_k, best_dt = k, dt
        if t + best_dt > cfg.t_max:
            break  # administrative censoring at the horizon
        t += best_dt
        state = best_k
        history.append((state, t))
    return history


def spawn_patient_seeds(seed: int, n: int):
    """Return (setup_rng, patient_seqs).

    Each ``patient_seqs[i].spawn(2)`` yields (dgp_seed, obs_seed). Because the DGP
    seed is derived identically here and in eventlog.build_event_log, latent
    trajectories are identical across the two for the same master seed.
    """
    ss = np.random.SeedSequence(seed)
    setup_seed, patient_root = ss.spawn(2)
    setup_rng = np.random.default_rng(setup_seed)
    patient_seqs = patient_root.spawn(n)
    return setup_rng, patient_seqs


@dataclass
class PopulationData:
    L: np.ndarray            # (n, 5) raw covariates
    L_std: np.ndarray        # (n, 5) standardized
    E: np.ndarray            # (n,) exposure 0/1
    histories: list          # latent (state, entry_time) per patient


def simulate_population(n: int, cfg: SimConfig, seed: int) -> PopulationData:
    setup_rng, patient_seqs = spawn_patient_seeds(seed, n)
    L = draw_covariates(n, cfg.dgp.covariates, setup_rng)
    L_std = standardize(L)
    E = assign_exposure(L_std, cfg.dgp.propensity, setup_rng)
    histories = []
    for i in range(n):
        dgp_seed, _obs_seed = patient_seqs[i].spawn(2)
        rng = np.random.default_rng(dgp_seed)
        histories.append(simulate_trajectory(L_std[i], int(E[i]), cfg.dgp, rng))
    return PopulationData(L=L, L_std=L_std, E=E, histories=histories)
