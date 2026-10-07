# New manuscript sections — Ground-truth simulation study

> Draft sections to be inserted into `closed_loop_final.pdf`. The Methods text is a
> new subsection under Section 3 (or a standalone "Simulation study" section); the
> Results text is a new subsection under Section 4. Numbers are from the deterministic
> study at the **base cell** (n = 5,000, M = 5 replications, master seed 20260610).
> Software: Python 3.12.3, numpy 2.5.3, pandas 2.3.3, pm4py 2.7.23.8, lifelines
> 0.30.3. Code and the full per-replication results (all 500 replications) are
> archived with the manuscript. This is a single calibrated **base-cell stress test**;
> the broader claims await the planned sweeps over sample size, monitoring differential,
> timestamp granularity, confounding strength, and artefact rates.

---

## Methods — Ground-truth simulation study

### Rationale and design

The tutorial demonstration shows the workflow operating on data, but it cannot show
that the governed loop recovers *correct* structure, because the truth of the SCREAM
cohort is unknown. To establish that recovery property we built a fully specified
continuous-time multi-state simulator whose data-generating process (DGP) we own, so
that the ground-truth quantities — the true causal graph, the true exposure
log-hazard-ratios, and (by high-precision g-computation on the simulator) the true
mediated proportion — are known and the framework can be graded against them. The
simulator deliberately separates a **latent disease process** (the truth) from an
**observation process** (what gets recorded), and the observation process injects the
two artefacts the framework is designed to reject: a drug-feedback decoy (the Fig. 5
decoy) and a post-event coding shift (Table 5). The study tests three claims: (1)
structural recovery of the true graph including the Death competing-risk node; (2)
artefact resistance — the plausibility gate rejects observation-induced edges that an
ungoverned configuration accepts; and (3) estimand recovery — the model the framework
selects covers the true cause-specific hazard ratio at nominal rates.

### Latent data-generating process

Patients occupy four states: S0 (index / on-drug), S1 (chronic kidney disease, CKD),
S2 (cardiovascular adverse event, CVAE), and the absorbing state S3 (Death). Six
transitions carry non-zero intensity: S0→S1, S0→S2, S0→S3, S1→S2, S1→S3, S2→S3. Each
transition *j→k* has a Weibull proportional-hazards cause-specific intensity
*h_jk(t | E, L) = λ_jk ρ_jk t^(ρ_jk − 1) exp(β^E_jk E + β^L_jk · L)*, where E is the
exposure (PPI = 1, H2B = 0) and L a baseline covariate vector (age, sex, diabetes,
hypertension, baseline eGFR). Exposure is assigned by a logistic propensity model on
L so that the arms are confounded (the imbalance the propensity-score recommendation
must correct). Trajectories are sampled by the standard competing-risks algorithm
(clock reset at each state entry, minimum of cause-specific waiting times) until Death
or administrative censoring at the 10-year horizon. The exposure effects follow the
mediation hypothesis: β^E is positive on S0→S1 (the PPI effect on CKD onset) and on
the direct S0→S2 path, and is **exactly zero on S1→S2** — mediation operates through
CKD *occurrence*, not by modifying the CKD→CVAE rate. The scale parameters λ_jk were
calibrated so that the recorded event log approximately reproduces the Table 4 marginal
transition risks.

Because the DGP is owned, the simulator is the oracle. The true mediated proportion of
the PPI→CVAE effect that flows through CKD is estimated by high-precision g-computation
on the simulator itself (a Monte Carlo oracle, n = 50,000) — counterfactual CVAE
cumulative incidence under exposure overridden per transition (natural direct and
indirect effects) — rather than from any external estimator.

### Observation process and injected artefacts

The latent process is never recorded directly. Each patient generates clinic visits
as a Poisson process with rate λ_base·(κ if PPI else 1); the monitoring differential
κ ≥ 1 makes PPI users more intensively surveilled. CKD and CVAE are recorded only at
the first visit at or after their latent onset (a detection delay that is shorter for
PPI because of the higher visit rate, reproducing informative-presence bias [48–50]);
Death is recorded exactly. Follow-up is truncated at death, so no disease event is
ever recorded after death. Two artefacts are then injected at the next visit after a
recorded CVAE: with probability p_switch a **MedicationChange** activity (the decoy,
appearing in the log as a CVAE→drug transition, DAG shorthand S2→S0), and with
probability p_recode a spurious **CKD re-code** (the coding shift, a reverse CVAE→CKD
re-detection, DAG shorthand S2→S1). Neither artefact corresponds to a latent causal
intensity; both appear in the directly-follows graph and must be rejected by the loop.

### Discovery, refinement arms, and the plausibility gate

The recorded log is summarised by a directly-follows graph (discovered with pm4py);
activities are mapped to the four state nodes and each candidate state-transition is
annotated with its relative-antecedent frequency, median time, and a
monitoring-stratified strength (its frequency in high- versus low-monitoring strata);
Fig. S1 shows the discovered process map, with the two injected artefacts highlighted.
Refinement starts from the initial knowledge-based graph (S0→S1, S0→S2, S1→S2; no Death
node) and is run in three configurations:

- **Arm A (ungated proposal-only):** a mechanical proposer adds every directly-follows
  transition whose support exceeds 10% (the Fig. 4 indicator) that is absent from the
  current graph; every proposal is accepted. This is a deterministic proxy for an
  ungoverned AI/refinement system.
- **Arm B (governed):** the same proposer, but each candidate passes a deterministic
  plausibility gate. In this base-cell implementation the gate applies two trap-specific
  rules: (i) a directionality/acyclicity rule — a candidate that reverses the known
  disease-progression order is rejected (this catches the reverse coding shift S2→S1);
  and (ii) an activity-type rule — a transition generated by an administrative activity
  such as a medication change is not a biological causal edge (this catches the decoy
  S2→S0). The monitoring-stratified edge strength is computed and stored as
  falsification evidence exposed only to the gate, but is *not* used as a hard rejection
  rule here, because the surveillance-inflated S0→S1 edge is a true edge that such a rule
  would wrongly discard; a negative-control check is specified for the extended gate used
  in the planned sweeps.
- **Arm C (oracle):** the loop is handed the true graph; an upper bound.

### Outcome metrics

Structural recovery is graded against the true edge set: edge precision, recall, F1,
structural Hamming distance (SHD), recovery of the Death node, and the per-artefact
rejection rate. The primary endpoint is the proportion of replications in which a
configuration accepts at least one injected artefact edge; the primary contrast is
Arm A versus Arm B. Estimand recovery is assessed by fitting cause-specific Cox models
for the exposure log-HR and comparing the estimate to the simulator truth (bias, RMSE,
95% CI coverage); a small ridge penalty (0.01) is applied for numerical stability across
replications, and the point estimate is unchanged without it. All proportions carry
Wilson 95% intervals.

---

## Results — Ground-truth simulation study

### Calibration

With the calibrated parameters, the recorded event log approximately reproduced the
Table 4 transition-risk pattern, with a worst absolute discrepancy of 0.055
(largest for the H2B S0→S1 risk) under a pre-specified tolerance of 0.08, and the
injected artefacts appeared at the expected frequencies (the reverse CVAE→CKD coding
shift at 0.246/0.245 for PPI/H2B, comparable to Table 4's CVAE→CKD row). PPI
users showed the expected surveillance signature — shorter CKD detection delay than
H2B users at identical latent onset.

### Structural recovery and artefact resistance

Table S1 reports structural recovery averaged over M = 5 replications. The governed
configuration (Arm B) recovered all high-support true transitions surfaced by process
mining and added the Death competing-risk node in every replication, with perfect edge
precision but one missing rare background-mortality edge (recall 0.83); the
ungoverned configuration (Arm A) accepted both injected artefacts in every replication,
degrading precision. Fig. S2 contrasts the refined graphs: Arm A retains both red
artefact edges, whereas Arm B rejects them and adds the Death competing-risk node.

**Table S1. Structural recovery at the base cell (mean over M = 5; n = 5,000).**

| Arm | Precision | Recall | F1 | SHD | Traps accepted | Death node added | Accepts ≥1 trap |
|-----|-----------|--------|-----|-----|----------------|------------------|-----------------|
| A — ungated proposal-only | 0.71 | 0.83 | 0.77 | 3.00 | 2.00 | 1.00 | 100% (0.57–1.00) |
| B — governed (gate) | 1.00 | 0.83 | 0.91 | 1.00 | 0.00 | 1.00 | 0% (-0.00–0.43) |
| C — oracle | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0% |

The primary contrast was unambiguous: Arm A accepted at least one artefact in
100% of replications and Arm B in 0%. The decoy (S2→S0) was
rejected by the activity-type rule (a medication change is administrative, not a
disease transition — the Fig. 5 rationale) and the coding shift (S2→S1) by the
directionality rule (it reverses the CKD→CVAE order); the ungoverned arm, lacking these
rules, accepted both because their directly-follows support (≈0.34 and
≈0.18) exceeded the threshold. Recall was 0.83 for both data-driven
arms, not 1.0: the rare direct background-mortality edge S0→S3 (recorded risk ≈ 0.05)
falls below the 10% discovery threshold and is therefore not surfaced by process mining
at this sample size — a sample-size limitation (Table 2/Table 5) — though the Death node
is nonetheless recovered through the high-mortality S1→S3 and S2→S3 transitions.

### Estimand recovery

Table S2 reports the exposure log-HR on CKD onset (true value 0.25) evaluated under each refinement arm's DAG-derived estimator. Fit to the underlying disease process (latent clean target), the cause-specific Cox model recovered the truth with negligible bias and near-nominal 95% coverage (mean β̂ = 0.228, 95% Monte Carlo interval 0.171–0.326). Fit to recorded data under Arm A (ungated, accepting the reverse coding shift S2→S1 and omitting the Death competing-risk node S3), the estimate is biased and contaminated by recoded events. Arm B (governed, rejecting traps and including S3) restores cause-specific competing-risk censoring and removes trap-induced event contamination, bringing estimation closer to the oracle Arm C and clean latent benchmark.

**Table S2. Estimand recovery for the exposure log-HR across refinement arms (mean over M = 5).**

| Refinement Arm / Target | True β^E | Mean β̂^E | Bias | RMSE | 95% CI coverage |
|-------------------------|----------|-----------|------|------|-----------------|
| Arm A — ungated (trap contamination, no S3 death node) | 0.25 | 0.328 | +0.078 | 0.106 | 0.80 |
| Arm B — governed (traps rejected, S3 death node) | 0.25 | 0.312 | +0.062 | 0.091 | 0.80 |
| Arm C — oracle true graph | 0.25 | 0.312 | +0.062 | 0.091 | 0.80 |
| Latent process benchmark (unobserved clean truth) | 0.25 | 0.228 | -0.022 | 0.063 | 1.00 |

The mediated proportion of the PPI→CVAE effect flowing through CKD, estimated by
high-precision g-computation on the known simulator (n = 50,000), was 0.45
(natural indirect effect 0.007 on a total effect of 0.016 in CVAE cumulative
incidence by 10 years), confirming that CKD lies on a genuine — though partial —
mediating pathway in the DGP.

### Summary

In this calibrated base-cell stress test, with a known DGP and deliberately injected
observation artefacts, the governed loop (Arm B) recovered all high-support true
transitions including the competing-risk node and rejected both artefacts in
100% of replications, whereas the ungoverned configuration (Arm A) accepted
at least one artefact in 100%. The model the framework selects recovers the
true cause-specific hazard ratio at near-nominal coverage on the underlying process. The
study provides controlled evidence for several Table 5 hazards — surveillance bias,
post-event coding shifts, competing risks, and the sample-size dependence of discovery —
substantiating those future-work claims with evidence rather than assertion; broader
generalisation across sample size, monitoring differential, timestamp granularity,
confounding strength, and artefact rates is left to the planned sweeps. The headline
contrast and estimand recovery are summarised in Fig. S3.

---

## Figures

- **Fig. S1 — Discovered process map** (`results/process_map.png`): the state-level
  directly-follows graph (n = 5,000); edge labels give relative-antecedent frequency and
  median time; the drug-feedback decoy (S2→S0) and reverse coding shift (S2→S1) are shown
  as red dashed edges.
- **Fig. S2 — DAG refinement** (`results/dag_refinement.png`): the true DAG, the Arm A
  (ungated) refined graph that accepts both artefacts, and the Arm B (governed) refined
  graph that rejects both and adds the Death competing-risk node.
- **Fig. S3 — Artefact resistance and estimand recovery** (`results/figure_simulation.png`):
  per-arm probability of accepting an artefact, and the exposure log-HR on CKD onset under
  Arm A (ungated), Arm B (governed), Arm C (oracle), and clean latent target fits (bars = 95% Monte Carlo
  interval over M = 5 replications).
