"""Timestamp-granularity option (Table 5 sweep mechanism)."""
from __future__ import annotations

import numpy as np

from closed_loop_sim.config import GRANULARITY_STEP, base_cell
from closed_loop_sim.eventlog import build_event_log


def test_exact_is_unchanged_default():
    cfg = base_cell()
    assert cfg.obs.time_granularity == "exact"
    a = build_event_log(300, cfg, seed=3)
    b = build_event_log(300, cfg, seed=3)
    assert a.equals(b)  # exact stays deterministic + reproducible


def test_yearly_rounds_timestamps_to_grid():
    cfg = base_cell(); cfg.obs.time_granularity = "yearly"
    df = build_event_log(800, cfg, seed=3)
    step = GRANULARITY_STEP["yearly"]
    # every recorded time is an integer multiple of the grid step
    assert np.allclose(df["time"].to_numpy() % step, 0.0, atol=1e-9)


def test_coarser_granularity_increases_ties():
    def tie_rate(gran):
        cfg = base_cell(); cfg.obs.time_granularity = gran
        df = build_event_log(2000, cfg, seed=3).sort_values(["case", "time"])
        c = df["case"].to_numpy(); t = df["time"].to_numpy()
        adj = c[1:] == c[:-1]
        return (adj & (t[1:] == t[:-1])).sum() / max(adj.sum(), 1)
    assert tie_rate("yearly") > tie_rate("exact")
