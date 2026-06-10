"""Phase 3 integration: discovery -> refinement arms -> structural metrics."""
from __future__ import annotations

from closed_loop_sim.config import base_cell
from closed_loop_sim.discovery import discover
from closed_loop_sim.eventlog import build_event_log_data
from closed_loop_sim.metrics import structural_metrics
from closed_loop_sim.refinement import ARMS


def _run_arms(seed=1, n=6000):
    cfg = base_cell()
    data = build_event_log_data(n, cfg, seed)
    disc = discover(data.log, data.visit_counts)
    return {arm: structural_metrics(fn(disc)) for arm, fn in ARMS.items()}


def test_dfg_surfaces_both_traps_as_candidates():
    cfg = base_cell()
    data = build_event_log_data(8000, cfg, seed=2)
    disc = discover(data.log, data.visit_counts)
    assert ("S2", "S0") in disc.state_edges
    assert ("S2", "S1") in disc.state_edges
    assert disc.state_edges[("S2", "S0")].risk >= 0.10


def test_arm_A_accepts_at_least_one_trap():
    m = _run_arms()["A"]
    assert m["accepts_any_trap"], "ungoverned arm should accept >=1 trap"


def test_arm_B_rejects_both_traps_and_adds_death():
    m = _run_arms()["B"]
    assert m["n_traps_accepted"] == 0, "governed arm must reject both traps"
    assert m["death_node_added"]
    assert m["accepted_S2_S0"] is False and m["accepted_S2_S1"] is False


def test_arm_B_precision_perfect_and_recovers_discoverable():
    m = _run_arms()["B"]
    assert m["precision"] == 1.0           # no false edges (both traps rejected)
    # Recovers all discoverable true edges; misses only S0->S3 (4.8%, below the
    # 10% Fig-4 support threshold) -> recall 5/6, SHD 1.
    assert abs(m["recall"] - 5 / 6) < 1e-9
    assert m["shd"] == 1


def test_arm_A_precision_degraded_by_traps():
    res = _run_arms()
    assert res["A"]["precision"] < res["B"]["precision"]


def test_arm_C_oracle_is_perfect():
    m = _run_arms()["C"]
    assert m["f1"] == 1.0 and m["shd"] == 0


def test_A_vs_B_contrast():
    res = _run_arms()
    assert res["A"]["n_traps_accepted"] > res["B"]["n_traps_accepted"]
