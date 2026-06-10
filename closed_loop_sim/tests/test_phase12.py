"""Phase 1-2 tests: DGP, oracle, observation, event log, and review-fix behaviors."""
from __future__ import annotations

import numpy as np

from closed_loop_sim.config import base_cell
from closed_loop_sim.dgp import (
    sample_transition_time, simulate_population, simulate_trajectory, TRUE_TRANSITIONS,
)
from closed_loop_sim.observation import generate_visits, observe_trajectory, inject_artifacts
from closed_loop_sim.eventlog import build_event_log, build_event_log_data
from closed_loop_sim.truth import marginal_transition_rates, mediated_proportion


# --- hazard sampling -------------------------------------------------------
def test_exponential_mean():
    rng = np.random.default_rng(0)
    draws = np.array([sample_transition_time(0.5, 1.0, 0.0, rng) for _ in range(200_000)])
    assert abs(draws.mean() - 2.0) < 0.03


def test_weibull_survival_matches_theory():
    rng = np.random.default_rng(1)
    lam, rho, lp, t0 = 0.3, 1.4, 0.2, 2.0
    a = lam * np.exp(lp)
    draws = np.array([sample_transition_time(lam, rho, lp, rng) for _ in range(200_000)])
    assert abs((draws > t0).mean() - np.exp(-a * t0 ** rho)) < 0.01


# --- trajectory ------------------------------------------------------------
def test_trajectory_monotone_absorbing_true_edges_only():
    cfg = base_cell()
    rng = np.random.default_rng(0)
    for _ in range(300):
        hist = simulate_trajectory(np.zeros(5), 1, cfg.dgp, rng)
        states = [s for s, _ in hist]
        times = [t for _, t in hist]
        assert states[0] == "S0" and times[0] == 0.0
        assert times == sorted(times)
        assert all(t <= cfg.dgp.t_max for t in times)
        for a, b in zip(states, states[1:]):
            assert b in TRUE_TRANSITIONS[a]


def test_population_reproducible():
    cfg = base_cell()
    a = simulate_population(300, cfg, seed=42)
    b = simulate_population(300, cfg, seed=42)
    assert (a.E == b.E).all()
    assert a.histories[7] == b.histories[7]


# --- oracle ----------------------------------------------------------------
def test_marginal_rates_ppi_gt_h2b_on_ckd():
    cfg = base_cell()
    rates = marginal_transition_rates(simulate_population(8000, cfg, seed=1))
    assert rates["PPI"]["S0->S1"] > rates["H2B"]["S0->S1"]


def test_mediated_proportion_unit_interval_and_positive_indirect():
    cfg = base_cell()
    res = mediated_proportion(cfg, n=6000, seed=4, horizon=5.0)
    assert 0.0 <= res["mediated_proportion"] <= 1.0
    assert res["total"] > 0 and res["indirect"] > 0


# --- observation (review fixes) -------------------------------------------
def test_visit_rate_scales_with_kappa():
    cfg = base_cell()
    rng = np.random.default_rng(0)
    ppi = np.mean([len(generate_visits(1, cfg.obs, cfg.dgp.t_max, rng)) for _ in range(2000)])
    h2b = np.mean([len(generate_visits(0, cfg.obs, cfg.dgp.t_max, rng)) for _ in range(2000)])
    assert ppi > h2b * 1.5


def test_no_disease_event_recorded_after_death():
    # CKD latent onset 5.0, death 6.0, but the only visit after onset is at 7.0 (> death).
    history = [("S0", 0.0), ("S1", 5.0), ("S3", 6.0)]
    events = dict(observe_trajectory(history, 1, visits=[1.0, 7.0]))
    assert "CKD_recorded" not in events
    assert events["Death"] == 6.0


def test_detection_delay_shorter_for_ppi():
    # Actual delay = first_visit_after(onset) - onset, controlling onset (review fix 4).
    cfg = base_cell()
    onset = 3.0

    def mean_delay(E):
        rng = np.random.default_rng(100 + E)
        d = []
        for _ in range(5000):
            visits = generate_visits(E, cfg.obs, cfg.dgp.t_max, rng)
            tt = next((v for v in visits if v >= onset), None)
            if tt is not None:
                d.append(tt - onset)
        return float(np.mean(d))

    assert mean_delay(1) < mean_delay(0)


def test_both_artifacts_at_next_visit_with_deterministic_order():
    cfg = base_cell()
    cfg.obs.p_switch = 1.0
    cfg.obs.p_recode = 1.0
    rng = np.random.default_rng(1)
    events = [("IndexDrug", 0.0), ("CVAE_recorded", 2.0)]
    out = inject_artifacts(events, cfg.obs, visits=[1.0, 2.5, 4.0], rng=rng)
    acts = [a for a, _ in out]
    assert acts.index("MedicationChange") < acts.index("CKD_recode")  # ACTIVITY_ORDER
    assert dict(out)["MedicationChange"] == 2.5


# --- event log + phase-2 gate ---------------------------------------------
def test_event_log_schema_sorted_reproducible():
    cfg = base_cell()
    a = build_event_log(300, cfg, seed=11)
    b = build_event_log(300, cfg, seed=11)
    assert list(a.columns) == ["case", "activity", "time", "group"]
    assert a.equals(b)
    for _, g in a.groupby("case"):
        assert g["time"].tolist() == sorted(g["time"].tolist())


def test_no_activity_after_death_in_log():
    cfg = base_cell()
    df = build_event_log(6000, cfg, seed=21)
    for _, g in df.groupby("case"):
        g = g.sort_values("time")
        if "Death" in g["activity"].values:
            dtime = g.loc[g["activity"] == "Death", "time"].iloc[0]
            after = g[g["time"] > dtime]
            assert after.empty, "no activity may follow Death"


def test_both_traps_present_in_directly_follows():
    cfg = base_cell()
    df = build_event_log(8000, cfg, seed=23).sort_values(["case", "time"])
    follows = set()
    for _, g in df.groupby("case"):
        acts = g["activity"].tolist()
        follows.update(zip(acts, acts[1:]))
    assert ("CVAE_recorded", "MedicationChange") in follows
    assert ("CVAE_recorded", "CKD_recode") in follows


def test_ppi_ckd_recorded_time_earlier_than_h2b():
    cfg = base_cell()
    df = build_event_log(8000, cfg, seed=22)
    ckd = df[df["activity"] == "CKD_recorded"]
    assert ckd[ckd["group"] == "PPI"]["time"].median() < ckd[ckd["group"] == "H2B"]["time"].median()


def test_visit_counts_returned():
    cfg = base_cell()
    data = build_event_log_data(500, cfg, seed=5)
    assert data.visit_counts.shape == (500,)
    assert data.visit_counts[data.E == 1].mean() > data.visit_counts[data.E == 0].mean()
