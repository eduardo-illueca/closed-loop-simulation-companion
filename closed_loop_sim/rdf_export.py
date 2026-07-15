"""RDF Exporter for simulation graphs.

Serializes DGP, Process Map, and Enhanced Graph into RDF Turtle (.ttl) files.
"""
from __future__ import annotations

from pathlib import Path
from closed_loop_sim.config import SimConfig
from closed_loop_sim.discovery import DiscoveryResult
from closed_loop_sim.refinement import RefinementResult
from closed_loop_sim.truth import TRAP_EDGES

# Map of nodes to labels
NODE_LABELS = {
    "S0": "Drug (index)",
    "S1": "CKD",
    "S2": "CVAE",
    "S3": "Death (absorbing)"
}


def format_literal(val) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    # String: escape backslashes and double quotes
    escaped = str(val).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def export_dgp(cfg: SimConfig, out_path: Path) -> None:
    lines = [
        "@prefix : <https://w3id.org/closed-loop-sim/> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        ":DGPGraph a :Graph ;",
        '    rdfs:label "Data Generating Process (True DAG)" ;',
    ]
    
    # List edges
    edge_uris = []
    for key in sorted(cfg.dgp.transitions.keys()):
        src, target = key.split("->")
        edge_uris.append(f":edge_dgp_{src}_{target}")
    
    lines.append("    :hasEdge " + " , ".join(edge_uris) + " .")
    lines.append("")
    
    # State nodes
    for state, label in NODE_LABELS.items():
        lines.append(f":{state} a :State ;")
        lines.append(f"    rdfs:label {format_literal(label)} .")
        lines.append("")
        
    # Edges
    for key in sorted(cfg.dgp.transitions.keys()):
        params = cfg.dgp.transitions[key]
        src, target = key.split("->")
        edge_uri = f":edge_dgp_{src}_{target}"
        lines.append(f"{edge_uri} a :Edge ;")
        lines.append(f"    :source :{src} ;")
        lines.append(f"    :target :{target} ;")
        lines.append(f"    :beta_E {format_literal(params.beta_E)} ;")
        lines.append(f"    :lambda {format_literal(params.lam)} ;")
        lines.append(f"    :rho {format_literal(params.rho)} .")
        lines.append("")
        
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_process_map(disc: DiscoveryResult, out_path: Path) -> None:
    lines = [
        "@prefix : <https://w3id.org/closed-loop-sim/> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        ":ProcessMapGraph a :Graph ;",
        '    rdfs:label "Discovered Process Map (Directly-Follows Graph)" ;',
    ]
    
    edge_uris = []
    for a, b in sorted(disc.state_edges.keys()):
        edge_uris.append(f":edge_pm_{a}_{b}")
        
    if edge_uris:
        lines.append("    :hasEdge " + " , ".join(edge_uris) + " .")
    else:
        lines.append("    :hasEdge () .")
    lines.append("")
    
    # State nodes
    for state, label in NODE_LABELS.items():
        lines.append(f":{state} a :State ;")
        lines.append(f"    rdfs:label {format_literal(label)} .")
        lines.append("")
        
    # Edges
    for (a, b) in sorted(disc.state_edges.keys()):
        e = disc.state_edges[(a, b)]
        edge_uri = f":edge_pm_{a}_{b}"
        is_trap = (a, b) in TRAP_EDGES
        lines.append(f"{edge_uri} a :Edge ;")
        lines.append(f"    :source :{a} ;")
        lines.append(f"    :target :{b} ;")
        lines.append(f"    :risk {format_literal(e.risk)} ;")
        lines.append(f"    :count {format_literal(e.count)} ;")
        lines.append(f"    :medianTime {format_literal(e.median_time)} ;")
        lines.append(f"    :monitoringAttenuation {format_literal(e.monitoring_attenuation)} ;")
        lines.append(f"    :isTrap {format_literal(is_trap)} .")
        lines.append("")
        
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_enhanced_graph(refined_arm_b: RefinementResult, out_path: Path) -> None:
    lines = [
        "@prefix : <https://w3id.org/closed-loop-sim/> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        ":EnhancedGraph a :Graph ;",
        '    rdfs:label "Enhanced Graph (Governed Causal Model)" ;',
    ]
    
    edge_uris = []
    for (a, b) in sorted(refined_arm_b.edges):
        edge_uris.append(f":edge_enhanced_{a}_{b}")
        
    if edge_uris:
        lines.append("    :hasEdge " + " , ".join(edge_uris) + " ;")
    
    decision_uris = []
    # Sort decisions to ensure deterministic Turtle output order
    sorted_decisions = sorted(refined_arm_b.decisions, key=lambda x: (x[0].frm, x[0].to))
    for cand, dec in sorted_decisions:
        decision_uris.append(f":decision_{cand.frm}_{cand.to}")
        
    if decision_uris:
        lines.append("    :hasDecision " + " , ".join(decision_uris) + " .")
    else:
        lines.append("    :hasDecision () .")
    lines.append("")
    
    # State nodes
    for state in sorted(refined_arm_b.nodes):
        label = NODE_LABELS.get(state, state)
        lines.append(f":{state} a :State ;")
        lines.append(f"    rdfs:label {format_literal(label)} .")
        lines.append("")
            
    # Edges
    for (a, b) in sorted(refined_arm_b.edges):
        edge_uri = f":edge_enhanced_{a}_{b}"
        lines.append(f"{edge_uri} a :Edge ;")
        lines.append(f"    :source :{a} ;")
        lines.append(f"    :target :{b} .")
        lines.append("")
        
    # Decisions
    for cand, dec in sorted_decisions:
        dec_uri = f":decision_{cand.frm}_{cand.to}"
        lines.append(f"{dec_uri} a :RefinementDecision ;")
        lines.append(f"    :proposedEdgeSource :{cand.frm} ;")
        lines.append(f"    :proposedEdgeTarget :{cand.to} ;")
        lines.append(f"    :accepted {format_literal(dec.accept)} ;")
        lines.append(f"    :rationale {format_literal(dec.rationale)} .")
        lines.append("")
        
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_graphs_to_rdf(cfg: SimConfig, disc: DiscoveryResult, refined_arm_b: RefinementResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_dgp(cfg, out / "dgp.ttl")
    export_process_map(disc, out / "process_map.ttl")
    export_enhanced_graph(refined_arm_b, out / "enhanced_graph.ttl")
    print(f"Exported RDF files to {out}")
