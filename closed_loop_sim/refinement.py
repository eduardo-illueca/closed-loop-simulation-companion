"""Phase 3 refinement: the Proposer/Gate interfaces and the deterministic arms.

Arm A = MechanicalProposer + NoneGate  (ungated proposal-only; accepts traps).
Arm B = MechanicalProposer + RulesGate (gated; rejects both traps).
Arm C = oracle (true DAG given).

The gate's trap-specific rules (deterministic spec A.4): a candidate is rejected
if it is a protected-direction (backward) edge -- which catches both the reverse
coding shift S2->S1 and the decoy S2->S0 -- and, independently, the decoy is
rejected by the activity-type rule (MedicationChange is administrative). The
falsification evidence (monitoring attenuation) is available to the gate only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from closed_loop_sim.discovery import ADMIN_ACTIVITIES, DiscoveryResult
from closed_loop_sim.truth import STATE_ORDER, TRUE_EDGES

# The initial knowledge-based DAG: hypothesised PPI->CKD->CVAE mediation, no Death,
# no competing-risk edges (spec/manuscript Phase 1).
INITIAL_NODES = {"S0", "S1", "S2"}
INITIAL_EDGES = {("S0", "S1"), ("S0", "S2"), ("S1", "S2")}

S_MIN = 0.10  # proposal support threshold (Fig. 4's >10% transition-risk indicator)


@dataclass
class Candidate:
    kind: str            # "add_edge" | "add_node" | "add_competing_risk"
    frm: str
    to: str
    risk: float
    evidence: dict = field(default_factory=dict)


@dataclass
class Decision:
    accept: bool
    rationale: str


class Proposer(Protocol):
    def propose(self, nodes: set, edges: set, disc: DiscoveryResult) -> list: ...


class Gate(Protocol):
    def adjudicate(self, c: Candidate) -> Decision: ...


class MechanicalProposer:
    """Propose every state edge present in the DFG (risk > s_min) that is absent
    from the current DAG, plus the node addition it implies."""

    def propose(self, nodes: set, edges: set, disc: DiscoveryResult) -> list:
        cands: list = []
        for (a, b), e in disc.state_edges.items():
            if e.risk < S_MIN or (a, b) in edges:
                continue
            ev = {
                "source_activities": e.source_activities,
                "monitoring_attenuation": e.monitoring_attenuation,
                "is_backward": STATE_ORDER[a] > STATE_ORDER[b],
            }
            if b == "S3" and "S3" not in nodes:
                cands.append(Candidate("add_competing_risk", a, b, e.risk, ev))
            else:
                cands.append(Candidate("add_edge", a, b, e.risk, ev))
        return cands


class NoneGate:
    """Ungoverned: accept any candidate whose support exceeds s_min."""

    def adjudicate(self, c: Candidate) -> Decision:
        return Decision(c.risk >= S_MIN, "accepted on support frequency (ungoverned)")


class RulesGate:
    """Governed: directionality/acyclicity + activity-type + negative-control."""

    def adjudicate(self, c: Candidate) -> Decision:
        if c.evidence.get("is_backward"):
            return Decision(False, f"rejected: protected-direction violation ({c.frm}->{c.to} is backward)")
        if c.evidence.get("source_activities", set()) & ADMIN_ACTIVITIES:
            return Decision(False, "rejected: administrative activity, not a biological causal edge")
        return Decision(True, "accepted: forward edge, plausible competing-risk/progression transition")


@dataclass
class RefinementResult:
    nodes: set
    edges: set
    decisions: list   # list[(Candidate, Decision)]


def run_refinement(proposer: Proposer, gate: Gate, disc: DiscoveryResult) -> RefinementResult:
    nodes = set(INITIAL_NODES)
    edges = set(INITIAL_EDGES)
    decisions: list = []
    for c in proposer.propose(nodes, edges, disc):
        d = gate.adjudicate(c)
        decisions.append((c, d))
        if d.accept:
            nodes.add(c.frm)
            nodes.add(c.to)
            edges.add((c.frm, c.to))
    return RefinementResult(nodes=nodes, edges=edges, decisions=decisions)


def oracle_refinement() -> RefinementResult:
    """Arm C: the loop is handed the true DAG."""
    nodes = {"S0", "S1", "S2", "S3"}
    return RefinementResult(nodes=nodes, edges=set(TRUE_EDGES), decisions=[])


ARMS = {
    "A": lambda disc: run_refinement(MechanicalProposer(), NoneGate(), disc),
    "B": lambda disc: run_refinement(MechanicalProposer(), RulesGate(), disc),
    "C": lambda disc: oracle_refinement(),
}
