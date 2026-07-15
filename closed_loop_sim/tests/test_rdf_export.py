from __future__ import annotations

import tempfile
from pathlib import Path

from closed_loop_sim.config import base_cell
from closed_loop_sim.discovery import discover
from closed_loop_sim.eventlog import build_event_log_data
from closed_loop_sim.refinement import ARMS
from closed_loop_sim.rdf_export import export_graphs_to_rdf


def test_rdf_export_creates_valid_files():
    cfg = base_cell()
    data = build_event_log_data(500, cfg, seed=1)  # small n for speed
    disc = discover(data.log, data.visit_counts, compute_activity_dfg=False)
    refined_b = ARMS["B"](disc)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        export_graphs_to_rdf(cfg, disc, refined_b, tmp_path)

        dgp_file = tmp_path / "dgp.ttl"
        pm_file = tmp_path / "process_map.ttl"
        enhanced_file = tmp_path / "enhanced_graph.ttl"

        assert dgp_file.exists()
        assert pm_file.exists()
        assert enhanced_file.exists()

        # Check DGP content
        dgp_content = dgp_file.read_text(encoding="utf-8")
        assert "@prefix : <https://w3id.org/closed-loop-sim/> ." in dgp_content
        assert ":DGPGraph a :Graph ;" in dgp_content
        assert ":S0 a :State ;" in dgp_content
        assert ":edge_dgp_S0_S1 a :Edge ;" in dgp_content
        assert ":beta_E 0.25" in dgp_content

        # Check Process Map content
        pm_content = pm_file.read_text(encoding="utf-8")
        assert ":ProcessMapGraph a :Graph ;" in pm_content
        assert ":edge_pm_S0_S1 a :Edge ;" in pm_content
        assert ":isTrap" in pm_content

        # Check Enhanced Graph content
        enhanced_content = enhanced_file.read_text(encoding="utf-8")
        assert ":EnhancedGraph a :Graph ;" in enhanced_content
        assert ":edge_enhanced_S0_S1 a :Edge ;" in enhanced_content
        assert ":decision_S2_S0 a :RefinementDecision ;" in enhanced_content
        assert ":accepted false ;" in enhanced_content
        assert "rejected" in enhanced_content
