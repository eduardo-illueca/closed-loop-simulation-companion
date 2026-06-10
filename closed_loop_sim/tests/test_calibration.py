"""Calibration gate: the recorded event log must approximate Table 4 (review fix 3:
the authoritative target is OBSERVED, not latent)."""
from __future__ import annotations

from closed_loop_sim.calibration import (
    latent_plausibility_report, observed_calibration_report,
)
from closed_loop_sim.config import base_cell


def test_observed_log_within_table4_tolerance():
    rep = observed_calibration_report(base_cell(), n=12000, seed=7, tol=0.10)
    assert rep["within_tol"], f"worst observed gap {rep['worst_gap']:.3f} exceeds tol"


def test_latent_rates_plausible():
    assert latent_plausibility_report(base_cell(), n=8000, seed=3)["plausible"]
