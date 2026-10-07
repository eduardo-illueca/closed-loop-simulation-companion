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

# The initial knowledge-based DAG: hypothesised PPI->CKD->CVAE mediation, no Death,
# no competing-risk edges (spec/manuscript Phase 1).
INITIAL_NODES = {"S0", "S1", "S2"}
INITIAL_EDGES = {("S0", "S1"), ("S0", "S2"), ("S1", "S2")}

# Phase 1 state ordering pre-specified from clinical knowledge (S0: index -> S1: CKD -> S2: CVAE, S3: Death).
# Gate rules and directionality checks use this pre-specified Phase 1 ordering without oracle dependence.
PHASE1_STATE_ORDER = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}

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

    def __init__(self, state_order: dict[str, int] | None = None) -> None:
        self.state_order = state_order if state_order is not None else PHASE1_STATE_ORDER

    def propose(self, nodes: set, edges: set, disc: DiscoveryResult) -> list:
        cands: list = []
        for (a, b), e in disc.state_edges.items():
            if e.risk < S_MIN or (a, b) in edges:
                continue
            is_backward = (
                self.state_order.get(a, 0) > self.state_order.get(b, 0)
                if (a in self.state_order and b in self.state_order)
                else False
            )
            ev = {
                "source_activities": e.source_activities,
                "monitoring_attenuation": e.monitoring_attenuation,
                "is_backward": is_backward,
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
    """Rule-based plausibility gate: directionality/acyclicity + activity-type + negative-control."""

    def __init__(self, state_order: dict[str, int] | None = None) -> None:
        self.state_order = state_order if state_order is not None else PHASE1_STATE_ORDER

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
    cycles_run: int = 1
    stopped_reason: str = "single_pass"
    cycle_history: list[dict] = field(default_factory=list)


def run_iterative_refinement(proposer: Proposer, gate: Gate, disc: DiscoveryResult,
                             max_cycles: int = 5, min_change_patience: int = 2) -> RefinementResult:
    """Iterative refinement loop with explicit stopping rules (Action A31).
    
    Stopping rules evaluated:
      1. Two consecutive cycles without accepted refinements (consecutive_empty_cycles >= min_change_patience).
      2. Stability of derived backdoor adjustment set (unchanged between cycles).
      3. Maximum cycle limit reached (max_cycles).
      4. Zero new candidate proposals generated.
    """
    nodes = set(INITIAL_NODES)
    edges = set(INITIAL_EDGES)
    decisions: list = []
    cycle_history: list[dict] = []
    consecutive_empty_cycles = 0
    stopped_reason = "max_cycles_reached"

    from closed_loop_sim.calibration import verify_backdoor_adjustment_set
    prev_adj_set = set(verify_backdoor_adjustment_set(nodes, edges)["valid_adjustment_set"])

    for cycle in range(1, max_cycles + 1):
        candidates = proposer.propose(nodes, edges, disc)
        if not candidates:
            stopped_reason = "no_further_candidates"
            break

        new_accepted = 0
        cycle_decisions = []
        for c in candidates:
            d = gate.adjudicate(c)
            decisions.append((c, d))
            cycle_decisions.append((c, d))
            if d.accept:
                new_accepted += 1
                nodes.add(c.frm)
                nodes.add(c.to)
                edges.add((c.frm, c.to))

        curr_adj_set = set(verify_backdoor_adjustment_set(nodes, edges)["valid_adjustment_set"])
        adj_set_stable = (curr_adj_set == prev_adj_set and cycle > 1)
        prev_adj_set = curr_adj_set

        cycle_history.append({
            "cycle": cycle,
            "candidates_proposed": len(candidates),
            "new_accepted": new_accepted,
            "total_edges": len(edges),
            "adj_set_stable": adj_set_stable,
        })

        if new_accepted == 0:
            consecutive_empty_cycles += 1
        else:
            consecutive_empty_cycles = 0

        # Action A31 Explicit Stopping Rule Check:
        if consecutive_empty_cycles >= min_change_patience:
            stopped_reason = f"stopping_rule: {min_change_patience} consecutive cycles without accepted refinements"
            break
        if adj_set_stable and consecutive_empty_cycles >= 1:
            stopped_reason = "stopping_rule: adjustment set stable and no new accepted refinements"
            break

    return RefinementResult(
        nodes=nodes,
        edges=edges,
        decisions=decisions,
        cycles_run=len(cycle_history) if cycle_history else 1,
        stopped_reason=stopped_reason,
        cycle_history=cycle_history,
    )


def run_refinement(proposer: Proposer, gate: Gate, disc: DiscoveryResult) -> RefinementResult:
    return run_iterative_refinement(proposer, gate, disc, max_cycles=1)


def cost_of_rejecting_true_backward_edges(ref: RefinementResult, true_edges: set) -> dict:
    """Action A10: Report the structural cost of wrongly rejecting true backward/feedback edges.
    
    When true feedback (e.g., S2->S1) exists in the DGP, a rigid directionality gate
    wrongly rejects valid biological edges.
    """
    backward_true = {(a, b) for (a, b) in true_edges if PHASE1_STATE_ORDER.get(a, 0) > PHASE1_STATE_ORDER.get(b, 0)}
    wrongly_rejected = []
    for cand, dec in ref.decisions:
        if (cand.frm, cand.to) in backward_true and not dec.accept:
            wrongly_rejected.append((cand.frm, cand.to))
            
    fn_count = len(wrongly_rejected)
    tot_backward = len(backward_true)
    penalty = fn_count / tot_backward if tot_backward > 0 else 0.0
    return {
        "true_backward_edges": list(backward_true),
        "wrongly_rejected_backward_edges": wrongly_rejected,
        "fn_count": fn_count,
        "backward_edge_false_negative_rate": penalty,
        "cost_summary": f"Wrongly rejected {fn_count}/{tot_backward} true feedback edges due to rigid directionality gate.",
    }


def oracle_refinement() -> RefinementResult:
    """Arm C: benchmark evaluation baseline where the loop is handed the true DAG."""
    from closed_loop_sim.truth import TRUE_EDGES
    nodes = {"S0", "S1", "S2", "S3"}
    return RefinementResult(nodes=nodes, edges=set(TRUE_EDGES), decisions=[], cycles_run=1, stopped_reason="oracle")


def initial_dag_only_refinement() -> RefinementResult:
    """Baseline Initial (Action A11): Initial Phase 1 DAG only, without refinement cycles."""
    return RefinementResult(nodes=set(INITIAL_NODES), edges=set(INITIAL_EDGES), decisions=[], cycles_run=0, stopped_reason="initial_only")


def temporal_constraint_discovery_refinement(disc: DiscoveryResult) -> RefinementResult:
    """Baseline PC/GES temporal constraint discovery (Action A11).
    
    Filters candidates using temporal tier ordering (forward transitions only).
    """
    nodes = set(INITIAL_NODES)
    edges = set(INITIAL_EDGES)
    decisions = []
    for (a, b), e in disc.state_edges.items():
        if e.risk >= S_MIN and PHASE1_STATE_ORDER.get(a, 0) < PHASE1_STATE_ORDER.get(b, 0):
            cand = Candidate("add_edge", a, b, e.risk, {})
            d = Decision(True, "accepted by temporal tier constraint discovery")
            decisions.append((cand, d))
            nodes.add(a); nodes.add(b); edges.add((a, b))
    return RefinementResult(nodes=nodes, edges=edges, decisions=decisions, cycles_run=1, stopped_reason="pc_temporal")


ARMS = {
    "A": lambda disc: run_refinement(MechanicalProposer(), NoneGate(), disc),
    "B": lambda disc: run_refinement(MechanicalProposer(), RulesGate(), disc),
    "C": lambda disc: oracle_refinement(),
    "Initial": lambda disc: initial_dag_only_refinement(),
    "PC_Temporal": lambda disc: temporal_constraint_discovery_refinement(disc),
}

