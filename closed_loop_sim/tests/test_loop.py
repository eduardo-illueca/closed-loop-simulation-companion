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


def test_refinement_has_no_module_level_truth_imports():
    import sys
    import closed_loop_sim.refinement as ref
    # Ensure refinement does not export or import truth symbols at module level (Action A09)
    assert not hasattr(ref, "STATE_ORDER"), "refinement module should not import STATE_ORDER from truth"
    assert hasattr(ref, "PHASE1_STATE_ORDER"), "refinement module should define PHASE1_STATE_ORDER from Phase 1 knowledge"


def test_arm_specific_estimation_in_replication_loop():
    from closed_loop_sim.config import base_cell
    from closed_loop_sim.experiment import run_replication, run_cell
    res = run_replication(base_cell(), seed=42, with_estimation=True)
    assert "arm_estimation" in res
    assert "A" in res["arm_estimation"]
    assert "B" in res["arm_estimation"]
    assert "C" in res["arm_estimation"]
    assert "naive" in res["arm_estimation"]["A"]
    assert "adjusted" in res["arm_estimation"]["A"]

    cell_res = run_cell(base_cell(), M=5, with_estimation=True)
    assert "by_arm" in cell_res["estimation"]
    assert "A" in cell_res["estimation"]["by_arm"]
    assert "B" in cell_res["estimation"]["by_arm"]
    assert "C" in cell_res["estimation"]["by_arm"]


def test_category1_primary_estimands_cif_rmst():
    from closed_loop_sim.config import base_cell
    from closed_loop_sim.eventlog import build_event_log_data
    from closed_loop_sim.estimation import s0s1_frame, cif_contrasts, rmst_contrasts
    data = build_event_log_data(1000, base_cell(), seed=1)
    df = s0s1_frame(data, 10.0)
    cif = cif_contrasts(df, 10.0)
    rmst = rmst_contrasts(df, 10.0)
    assert "risk_difference" in cif
    assert "risk_ratio" in cif
    assert "rmtl_diff" in rmst


def test_category1_phase4_recommendations():
    from closed_loop_sim.finalize import generate_phase4_recommendations
    recs = generate_phase4_recommendations({})
    assert "primary_estimand" in recs
    assert "Cumulative Incidence Function" in recs["primary_estimand"]


def test_category1_challenge_scenarios():
    from closed_loop_sim.config import base_cell
    from closed_loop_sim.eventlog import build_event_log_data
    cfg = base_cell()
    cfg.dgp.challenge_scenario = "feedback_cvae_ckd"
    data = build_event_log_data(500, cfg, seed=1)
    assert len(data.log) > 0


def test_category1_baselines():
    from closed_loop_sim.refinement import ARMS, initial_dag_only_refinement
    init_res = initial_dag_only_refinement()
    assert len(init_res.nodes) == 3
    assert "Initial" in ARMS
    assert "PC_Temporal" in ARMS


def test_category1_propensity_diagnostics_and_backdoor():
    from closed_loop_sim.config import base_cell
    from closed_loop_sim.eventlog import build_event_log_data
    from closed_loop_sim.calibration import compute_propensity_diagnostics, verify_backdoor_adjustment_set
    data = build_event_log_data(500, base_cell(), seed=1)
    diag = compute_propensity_diagnostics(data)
    assert "effective_sample_size" in diag
    assert len(diag["smds"]) == 5
    bd = verify_backdoor_adjustment_set({"S0", "S1", "S2"}, {("S0", "S1")})
    assert bd["backdoor_criterion_satisfied"]


def test_action_a31_explicit_stopping_rules():
    from closed_loop_sim.config import base_cell
    from closed_loop_sim.eventlog import build_event_log_data
    from closed_loop_sim.discovery import discover
    from closed_loop_sim.refinement import run_iterative_refinement, MechanicalProposer, RulesGate
    data = build_event_log_data(1000, base_cell(), seed=1)
    disc = discover(data.log, data.visit_counts)
    ref_res = run_iterative_refinement(MechanicalProposer(), RulesGate(), disc, max_cycles=5, min_change_patience=2)
    
    assert ref_res.cycles_run > 0
    assert ("stopping_rule" in ref_res.stopped_reason or "max_cycles" in ref_res.stopped_reason or "no_further" in ref_res.stopped_reason)
    assert len(ref_res.cycle_history) == ref_res.cycles_run


def test_action_a10_challenge_scenarios_and_backward_edge_cost():
    from closed_loop_sim.config import base_cell
    from closed_loop_sim.eventlog import build_event_log_data
    from closed_loop_sim.discovery import discover
    from closed_loop_sim.refinement import run_refinement, MechanicalProposer, RulesGate, cost_of_rejecting_true_backward_edges
    cfg = base_cell()
    cfg.obs.challenge_scenario = "held_out_artifact"
    cfg.obs.p_held_out = 0.4
    data = build_event_log_data(1000, cfg, seed=1)
    disc = discover(data.log, data.visit_counts)
    ref_res = run_refinement(MechanicalProposer(), RulesGate(), disc)
    
    true_feedback_edges = {("S0", "S1"), ("S0", "S2"), ("S0", "S3"), ("S1", "S2"), ("S1", "S3"), ("S2", "S3"), ("S2", "S1")}
    cost = cost_of_rejecting_true_backward_edges(ref_res, true_feedback_edges)
    assert "backward_edge_false_negative_rate" in cost
    assert cost["fn_count"] >= 0


def test_action_a33_demonstrator_benchmark():
    from closed_loop_sim.benchmark import benchmark_demonstrator
    res = benchmark_demonstrator(n=500, seed=42)
    assert res["n_trajectories"] == 500
    assert "timing_sec" in res
    assert "memory" in res
    assert "Cycle-based diagnostic workflow" in res["framework_operating_mode"]



