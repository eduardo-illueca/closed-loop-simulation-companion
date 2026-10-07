"""Phase 4 estimation: cause-specific Cox models (lifelines).

The headline estimand is the exposure log-HR on S0->S1 (CKD onset). The *naive*
model regresses recorded time-to-CKD on exposure + confounders; it is inflated by
surveillance bias (PPI users are monitored more, so CKD is detected sooner and
more often). The *monitoring-adjusted* model adds visit intensity, attenuating the
detection-driven component toward the true latent beta_E. S1->S2 (true beta_E=0)
is also estimated as a check that the mediator-rate effect is null.
"""
from __future__ import annotations

import typing
import warnings

import numpy as np
import pandas as pd
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
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        return {"beta_E": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    cph = CoxPHFitter(penalizer=0.01)
    cols = ["duration", "event", "E"] + covariates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df[cols], duration_col="duration", event_col="event")
    coef = cph.params_["E"]
    lo = cph.confidence_intervals_.loc["E"].iloc[0]
    hi = cph.confidence_intervals_.loc["E"].iloc[1]
    return {"beta_E": float(coef), "ci_low": float(lo), "ci_high": float(hi)}


def s0s1_arm_frame(data: EventLogData, t_max: float, ref: typing.Any = None) -> pd.DataFrame:
    """Build cause-specific Cox frame for S0->S1 derived from an arm's refined graph structure.
    
    Graph-derived estimation rules:
      - Death S3 competing risk censoring: applied if 'S3' is present in ref.nodes (Arm B, C)
        or omitted if absent (Initial Phase 1 DAG).
      - Reverse coding shift (S2->S1): included if ('S2', 'S1') is accepted in ref.edges (Arm A ungated),
        contaminating the CKD onset set.
    """
    n = len(data.E)
    ckd = _first_time(data.log, "CKD_recorded")
    
    # If arm accepts reverse trap S2->S1 (Arm A ungated), recoded events contaminate CKD onset
    include_recode = (ref is None) or (hasattr(ref, "edges") and ("S2", "S1") in ref.edges)
    if include_recode and "CKD_recode" in data.log.activity.values:
        recode = _first_time(data.log, "CKD_recode")
        ckd = pd.concat([ckd, recode]).groupby("case").min()
        
    # If arm graph includes Death S3 node (Arm B governed, Arm C oracle)
    has_death_node = (ref is None) or (hasattr(ref, "nodes") and "S3" in ref.nodes)
    death = _first_time(data.log, "Death") if has_death_node else pd.Series(dtype=float)
    
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


def cif_contrasts(df: pd.DataFrame, horizon: float = 10.0) -> dict:
    """Standardized Cumulative Incidence Function (CIF) contrasts at horizon t (Action A14).
    
    Returns Risk Difference (RD = CIF_1 - CIF_0) and Risk Ratio (RR = CIF_1 / CIF_0).
    """
    e1 = df[df.E == 1.0]
    e0 = df[df.E == 0.0]
    cif1 = float(np.mean((e1.event == 1) & (e1.duration <= horizon))) if len(e1) > 0 else 0.0
    cif0 = float(np.mean((e0.event == 1) & (e0.duration <= horizon))) if len(e0) > 0 else 0.0
    rd = cif1 - cif0
    rr = cif1 / cif0 if cif0 > 0 else np.nan
    return {"cif1": cif1, "cif0": cif0, "risk_difference": rd, "risk_ratio": rr}


def rmst_contrasts(df: pd.DataFrame, horizon: float = 10.0) -> dict:
    """Restricted Mean Survival Time / Loss (RMST / RMTL) contrasts at horizon t (Action A14).
    
    RMTL is time lost to event before horizon t.
    """
    e1 = df[df.E == 1.0]
    e0 = df[df.E == 0.0]
    rmst1 = float(np.mean(np.minimum(e1.duration, horizon))) if len(e1) > 0 else horizon
    rmst0 = float(np.mean(np.minimum(e0.duration, horizon))) if len(e0) > 0 else horizon
    rmtl1 = horizon - rmst1
    rmtl0 = horizon - rmst0
    return {"rmst1": rmst1, "rmst0": rmst0, "rmtl_diff": rmtl1 - rmtl0, "rmst_diff": rmst1 - rmst0}


def iiw_weighted_cox(df: pd.DataFrame, visit_counts: np.ndarray, covariates: list) -> dict:
    """Inverse-Intensity-of-Visit Weighting (IIW; Lin 2004) to adjust for monitoring-differential (Action A19)."""
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        return {"beta_E": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    
    mean_visits = np.mean(visit_counts) if len(visit_counts) > 0 else 1.0
    weights = mean_visits / np.maximum(visit_counts.astype(float), 1.0)
    weights = weights / np.mean(weights)
    
    df_w = df.copy()
    df_w["weights"] = weights
    cph = CoxPHFitter(penalizer=0.01)
    cols = ["duration", "event", "E", "weights"] + covariates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df_w[cols], duration_col="duration", event_col="event", weights_col="weights")
    coef = cph.params_["E"]
    lo = cph.confidence_intervals_.loc["E"].iloc[0]
    hi = cph.confidence_intervals_.loc["E"].iloc[1]
    return {"beta_E": float(coef), "ci_low": float(lo), "ci_high": float(hi)}


def quantitative_bias_analysis(recorded_hr: float, kappa: float = 2.0) -> dict:
    """Quantitative Bias Analysis (QBA; Action A19) for monitoring-driven surveillance bias.
    
    Applies monitoring intensity attenuation factor ln(kappa) to recorded hazard ratio.
    """
    if np.isnan(recorded_hr) or recorded_hr <= 0:
        return {"adjusted_hr": np.nan, "adjusted_beta_E": np.nan, "bias_attenuation": np.nan}
    log_hr = np.log(recorded_hr)
    attenuation = np.log(kappa) * 0.5
    adj_beta = log_hr - attenuation
    return {"adjusted_hr": float(np.exp(adj_beta)), "adjusted_beta_E": float(adj_beta), "bias_attenuation": float(attenuation)}


def estimate_arm(data: EventLogData, t_max: float, ref: typing.Any) -> dict:
    """Derive arm-specific Cox and primary estimand (CIF/RMST) estimates based on the arm's refined graph."""
    arm_df = s0s1_arm_frame(data, t_max, ref)
    naive = _fit_betaE(arm_df, _COVS)
    adjusted = _fit_betaE(arm_df, _COVS + ["visits"])
    cif = cif_contrasts(arm_df, horizon=t_max)
    rmst = rmst_contrasts(arm_df, horizon=t_max)
    iiw = iiw_weighted_cox(arm_df, data.visit_counts, _COVS)
    qba = quantitative_bias_analysis(np.exp(naive["beta_E"]) if not np.isnan(naive["beta_E"]) else np.nan, kappa=2.0)
    return {
        "beta_E": naive["beta_E"],
        "ci_low": naive["ci_low"],
        "ci_high": naive["ci_high"],
        "naive": naive,
        "adjusted": adjusted,
        "cif": cif,
        "rmst": rmst,
        "iiw": iiw,
        "qba": qba,
    }


def estimate(data: EventLogData, t_max: float) -> dict:
    s01 = s0s1_frame(data, t_max)
    s01_latent = s0s1_latent_frame(data, t_max)
    s12 = s1s2_frame(data, t_max)
    return {
        "s0s1_latent": _fit_betaE(s01_latent, _COVS),       # surveillance-bias-free target
        "s0s1_naive": _fit_betaE(s01, _COVS),               # recorded: surveillance-inflated
        "s0s1_adjusted": _fit_betaE(s01, _COVS + ["visits"]),  # visit-count adj: over-corrects
        "s1s2": _fit_betaE(s12, _COVS),                     # true beta_E = 0
        "cif_latent": cif_contrasts(s01_latent, t_max),
        "cif_naive": cif_contrasts(s01, t_max),
        "rmst_latent": rmst_contrasts(s01_latent, t_max),
        "rmst_naive": rmst_contrasts(s01, t_max),
    }
