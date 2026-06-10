"""Phase 4 estimation: cause-specific Cox models (lifelines).

The headline estimand is the exposure log-HR on S0->S1 (CKD onset). The *naive*
model regresses recorded time-to-CKD on exposure + confounders; it is inflated by
surveillance bias (PPI users are monitored more, so CKD is detected sooner and
more often). The *monitoring-adjusted* model adds visit intensity, attenuating the
detection-driven component toward the true latent beta_E. S1->S2 (true beta_E=0)
is also estimated as a check that the mediator-rate effect is null.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from closed_loop_sim.eventlog import EventLogData

_COVS = ["age", "sex", "diabetes", "hypertension", "egfr"]


def _first_time(log: pd.DataFrame, activity: str) -> pd.Series:
    return log[log.activity == activity].groupby("case")["time"].min()


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def s0s1_frame(data: EventLogData, t_max: float) -> pd.DataFrame:
    n = len(data.E)
    ckd = _first_time(data.log, "CKD_recorded")
    death = _first_time(data.log, "Death")
    dur = np.empty(n)
    ev = np.zeros(n, dtype=int)
    for i in range(n):
        ct = ckd.get(i, np.nan)
        if not np.isnan(ct):
            dur[i], ev[i] = ct, 1
        else:
            dt = death.get(i, np.nan)
            dur[i] = dt if not np.isnan(dt) else t_max
    dur = np.clip(dur, 1e-6, None)
    frame = {"duration": dur, "event": ev, "E": data.E.astype(float)}
    for j, c in enumerate(_COVS):
        frame[c] = _zscore(data.L[:, j])
    frame["visits"] = _zscore(data.visit_counts.astype(float))
    return pd.DataFrame(frame)


def s0s1_latent_frame(data: EventLogData, t_max: float) -> pd.DataFrame:
    """Cause-specific Cox frame for S0->S1 on the TRUE latent onset times (no
    detection delay). This is the surveillance-bias-free target the recorded
    estimate is compared against; it validates that cause-specific Cox recovers
    the true beta_E when the observation process is clean."""
    n = len(data.E)
    dur = np.empty(n)
    ev = np.zeros(n, dtype=int)
    for i, hist in enumerate(data.histories):
        onset = {s: t for s, t in hist}
        if "S1" in onset:
            dur[i], ev[i] = onset["S1"], 1
        else:
            dur[i] = onset.get("S3", t_max)
    dur = np.clip(dur, 1e-6, None)
    frame = {"duration": dur, "event": ev, "E": data.E.astype(float)}
    for j, c in enumerate(_COVS):
        frame[c] = _zscore(data.L[:, j])
    return pd.DataFrame(frame)


def s1s2_frame(data: EventLogData, t_max: float) -> pd.DataFrame:
    """Among patients who entered CKD: time from CKD to CVAE (clock reset)."""
    ckd = _first_time(data.log, "CKD_recorded")
    cvae = _first_time(data.log, "CVAE_recorded")
    death = _first_time(data.log, "Death")
    rows = []
    for i in ckd.index:
        ct = ckd[i]
        vt = cvae.get(i, np.nan)
        if not np.isnan(vt) and vt > ct:
            dur, ev = vt - ct, 1
        else:
            dt = death.get(i, np.nan)
            end = dt if not np.isnan(dt) else t_max
            dur, ev = max(end - ct, 1e-6), 0
        row = {"duration": dur, "event": ev, "E": float(data.E[i])}
        for j, c in enumerate(_COVS):
            row[c] = data.L[i, j]
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in _COVS:
        df[c] = _zscore(df[c].to_numpy())
    return df


def _fit_betaE(df: pd.DataFrame, covariates: list) -> dict:
    cph = CoxPHFitter(penalizer=0.01)
    cols = ["duration", "event", "E"] + covariates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df[cols], duration_col="duration", event_col="event")
    coef = cph.params_["E"]
    lo = cph.confidence_intervals_.loc["E"].iloc[0]
    hi = cph.confidence_intervals_.loc["E"].iloc[1]
    return {"beta_E": float(coef), "ci_low": float(lo), "ci_high": float(hi)}


def estimate(data: EventLogData, t_max: float) -> dict:
    s01 = s0s1_frame(data, t_max)
    s01_latent = s0s1_latent_frame(data, t_max)
    s12 = s1s2_frame(data, t_max)
    return {
        "s0s1_latent": _fit_betaE(s01_latent, _COVS),       # surveillance-bias-free target
        "s0s1_naive": _fit_betaE(s01, _COVS),               # recorded: surveillance-inflated
        "s0s1_adjusted": _fit_betaE(s01, _COVS + ["visits"]),  # visit-count adj: over-corrects
        "s1s2": _fit_betaE(s12, _COVS),                     # true beta_E = 0
    }
