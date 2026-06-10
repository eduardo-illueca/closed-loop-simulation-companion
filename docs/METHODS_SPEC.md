# Ground-Truth Simulation Study: Specification and Implementation Plan

Validation study for the closed-loop knowledge-based causal model refinement framework.
Purpose: demonstrate, under a known data-generating process, that the governed human-AI loop recovers correct causal structure and rejects observation-induced artifacts, and that the structural refinement improves downstream estimand recovery.

---

## 1. What the simulation must demonstrate

The framework's novel claim is not that process mining finds temporal patterns (established), but that the governed loop (a) recovers correct structure including the competing-risk node, (b) rejects edges that are artifacts of the observation process rather than the disease process, and (c) yields a refined DAG whose implied statistical model recovers known causal quantities. The simulation is therefore built so that the truth is fixed and so that deliberate traps are present that a naive AI-only acceptance step would fail.

Three claims to support with quantitative results:

1. Structural recovery. The loop recovers the true node and edge set, including adding Death as a competing risk and confirming the CKD mediating path, while rejecting a reverse CKD edge and a drug-feedback edge.
2. Artifact resistance. The human/plausibility gate rejects observation-induced edges (surveillance-driven and coding-shift-driven) that an AI-proposal-only configuration accepts.
3. Estimand recovery. After the loop selects competing-risk and mediation models, the estimates cover the known true cause-specific hazards and the known mediated proportion at nominal rates.

---

## 2. Data-generating process (DGP)

### 2.1 States and allowed transitions

Continuous-time multi-state model with an absorbing death state. States:

- S0: index / on-drug (entry state)
- S1: CKD
- S2: CVAE
- S3: Death (absorbing)

True causal transitions (these have non-zero intensities):

| From | To | Meaning |
|------|----|---------|
| S0 | S1 | exposure and confounders drive CKD onset (exposure effect on mediator) |
| S0 | S2 | direct exposure effect on CVAE not through CKD |
| S0 | S3 | background mortality |
| S1 | S2 | CKD to CVAE (the mediating path) |
| S1 | S3 | CKD to Death (competing risk) |
| S2 | S3 | CVAE to Death (competing risk) |

Transitions that are NOT in the true DAG and must be rejected by the loop:

| From | To | Why it appears in the log |
|------|----|--------------------------|
| S2 | S1 | post-CVAE coding shift creates spurious CKD re-detection (Table 5: post-event coding shifts) |
| S2 | S0 | medication change after a cardiovascular event, administrative not causal (the Fig. 5 decoy) |

> **v1.1 clarification (event-log vocabulary).** These two traps are *event-log / DAG-level artifacts*, not biological returns to a latent state. S0 is the latent index/on-drug entry state; patients do not biologically re-enter it. In the recorded log the decoy is a **MedicationChange** activity after CVAE (DAG decoy: CVAE→Drug-use, Fig. 5), and the coding shift is a spurious **CKD re-code** activity after CVAE (reverse CKD re-detection). pm4py discovery operates on activities, never on latent states. See Addendum §A.1 for the full activity vocabulary.

### 2.2 Confounders and exposure assignment

Baseline covariate vector L per patient: age, sex, diabetes, hypertension, baseline eGFR. Draw from distributions calibrated to the SCREAM-derived ranges used in the tutorial dataset.

Exposure E (PPI = 1, H2B = 0) assigned by a logistic propensity model:

logit P(E = 1 | L) = alpha_0 + alpha_L . L

with alpha_L set so that PPI and H2B groups are imbalanced on age, baseline eGFR, and comorbidity. This gives the propensity-score balancing recommendation a real imbalance to correct and lets confounding bias appear if the adjustment set is wrong.

### 2.3 Transition intensities

For each allowed transition j to k use a proportional-hazards form with a Weibull baseline:

h_{jk}(t | E, L) = lambda_{jk} * rho_{jk} * t^(rho_{jk} - 1) * exp(beta^E_{jk} * E + beta^L_{jk} . L)

Parameter intent (sign and rough magnitude fixed in a config file, calibrated so marginal transition rates approximate Table 4 of [44]):

| Transition | beta^E (exposure effect) | Notes |
|-----------|--------------------------|-------|
| S0→S1 | positive, moderate | true PPI effect on CKD onset |
| S0→S2 | positive, small | true direct effect |
| S1→S2 | zero | mediation operates through CKD occurrence, not through modifying the CKD->CVAE rate |
| S1→S3 | small | competing mortality from CKD |
| S2→S3 | small | competing mortality from CVAE |
| S0→S3 | near zero | background mortality |

Simulate event times by the standard competing-risks algorithm: from the current state, draw the next-event time for each outgoing transition from its cause-specific hazard, take the minimum, move to that state, repeat until Death or administrative censoring at the study horizon T_max.

### 2.4 Ground-truth quantities

Because the DGP is fully specified, the following are known exactly and serve as targets:

- The true DAG (nodes, edges, edge directions).
- True cause-specific hazard ratios for E on each transition (the beta^E values).
- True mediated proportion of the total PPI effect on CVAE that flows through CKD, computed by g-computation on the DGP: simulate counterfactual cumulative incidence of CVAE under E=1 and E=0, and under interventions that hold the mediator path fixed. The simulator itself is the oracle for this quantity.
- True cumulative incidence functions for CVAE and Death by exposure arm.

---

## 3. Observation model (the part that creates artifacts)

The latent disease process from Section 2 is separated from what gets recorded. This separation is the mechanism that produces the biases the loop must withstand.

### 3.1 Visit / measurement process

Per patient, generate clinic visits as a Poisson process with rate lambda_visit(E), where PPI users are monitored more intensively:

lambda_visit = lambda_base * (kappa if E == 1 else 1)

with kappa >= 1 the monitoring-intensity differential (a swept parameter).

### 3.2 Detection rules

- CKD (S1): recorded only when an eGFR measurement at a visit falls at or below the threshold. Recorded onset time = first visit at or after true latent onset. Detection delay = waiting time to the next visit, which is shorter for PPI because of higher visit rate. This inflates the apparent S0->S1 risk and shortens its apparent median time for PPI, reproducing informative-presence bias ([48]-[50]).
- CVAE (S2): recorded at the visit at or after onset, same delay mechanism.
- Death (S3): recorded exactly (administrative source, no detection delay). The asymmetry between exactly-recorded death and visit-gated disease states is itself a realistic source of distortion.

### 3.3 Injected artifacts

1. Decoy drug-feedback edge (S2->S0). After a recorded CVAE, with probability p_switch emit a medication-change event. In the event log this produces a CVAE-then-drug transition that the discovery step will surface and the loop must reject as administrative.
2. Post-event coding shift (S2->S1). After a recorded CVAE, with probability p_recode emit a spurious CKD code at a subsequent visit, producing reverse S2->S1 transitions in the log despite no causal reverse intensity.

---

## 4. Experimental arms

The "human" adjudication is operationalized as a reproducible rule set so the study is fully automated and replicable. This formalized plausibility gate is itself a contribution, and the manuscript should state plainly that it is a programmatic proxy for expert adjudication, distinct from the separate inter-rater study already named in future work.

- Arm A, ungated proposal-only (a deterministic proxy for an ungoverned AI/refinement system; there is no LLM in the deterministic study, so the earlier name "AI-proposal-only" is retired): accept every candidate refinement whose support frequency exceeds a threshold s_min (Addendum §A.3). No plausibility or sensitivity check.
- Arm B, full framework: AI proposal plus the plausibility gate plus a sensitivity check. The gate encodes (i) directionality and acyclicity constraints, (ii) a rule that an edge whose apparent strength attenuates substantially after adjusting for monitoring intensity is flagged as a candidate artifact, and (iii) a negative-control check. This is the configuration under test.
- Arm C, oracle (optional upper bound): the loop is given the true DAG. Used only to bound achievable estimand recovery.

The headline result is the contrast A versus B on artifact rejection and estimand bias.

---

## 5. Outcome metrics

### 5.1 Structural recovery (per replication, averaged)

- Edge precision, recall, F1 against the true edge set.
- Structural Hamming distance between recovered and true DAG.
- Indicator: was the Death competing-risk node added.
- False-edge rejection rate for each injected trap (S2->S0 and S2->S1), reported separately for Arm A and Arm B.

### 5.2 Estimand recovery (Monte Carlo over replications)

- Bias and root mean squared error of the estimated E effect on S0->S1, naive versus monitoring-adjusted, to quantify how much surveillance bias the framework removes.
- 95% confidence interval coverage of the true cause-specific hazard ratios.
- Bias, RMSE, and CI coverage of the estimated mediated proportion against the DGP truth.
- Cumulative incidence calibration for CVAE and Death by arm.

A correctly behaving framework shows Arm B recovering structure and covering the true estimands at near-nominal rates, while Arm A accepts at least one artifact and shows inflated bias.

---

## 6. Robustness sweeps

Each cell run with M Monte Carlo replications (target M = 500 to 1000).

| Factor | Levels | Maps to manuscript limitation |
|--------|--------|------------------------------|
| Sample size n | 1k, 5k, 50k, 300k | sample-size requirement (Table 2, Phase 2) |
| Monitoring differential kappa | 1.0, 1.5, 2.0, 3.0 | surveillance bias (Table 5) |
| Timestamp granularity | exact, daily, monthly, yearly | temporal granularity limits (Table 5) |
| Confounding strength alpha_L | low, moderate, strong | adjustment-set sensitivity |
| Coding-shift / decoy rates | off, low, high | post-event coding shifts (Table 5) |

Reporting where the framework degrades (small n, high kappa, coarse granularity) is more credible than reporting only success regions, and it directly substantiates the Table 5 future-work claims with evidence rather than assertion.

---

## 7. Implementation: Python primary vs R

### 7.1 Recommendation

Python-primary, with an optional R bridge for one cross-check. Rationale: the DGP is a custom continuous-time simulator that is straightforward in NumPy and gives full control of the truth; the process mining step is Python-native through PM4Py, which is the same engine the HealthProcessAI tool wraps, so the validation runs through your own stack; and because you own the DGP, the true mediated proportion comes directly from the simulator rather than from an external estimator, which removes the main reason people reach for R's mediation packages.

The one place R is more mature is the Fine-Gray subdistribution model. Handle this in one of three ways, in order of preference:
1. Fit cause-specific Cox models in Python (lifelines), which are sufficient for etiologic transition-intensity targets and recover the beta^E values directly.
2. Implement the Fine-Gray subdistribution estimator in Python by constructing subdistribution risk sets with the standard censoring-weight construction and fitting a weighted Cox model, for the cumulative-incidence target.
3. Optional cross-check only: call R via rpy2 to `cmprsk::crr` or `riskRegression::FGR` to confirm the Python subdistribution estimates agree. This keeps R out of the critical path and uses it purely as an independent check.

If the team prefers R-primary instead, the equivalent stack is simsurv or a hand-coded Gillespie simulator, mstate for the multistate structure, survival and riskRegression for cause-specific and Fine-Gray fits, and CMAverse for mediation, with bupaR for discovery. That path has the strongest off-the-shelf statistics but does not share an engine with the deployed tool.

### 7.2 Python stack

| Concern | Library |
|---------|---------|
| DGP, RNG, hazards | numpy, scipy.stats |
| Data handling | pandas |
| Process discovery (DFG) | pm4py |
| Cause-specific hazards | lifelines (CoxPHFitter on cause-specific event indicators) |
| Subdistribution (Fine-Gray) | weighted Cox in lifelines, or scikit-survival, optional rpy2 cross-check |
| Mediation truth and estimate | simulation-based g-computation, implemented directly on the simulator and on fitted transition models |
| Config | pydantic or a YAML schema |
| Orchestration / sweeps | joblib or multiprocessing, one config per cell |
| Reporting | matplotlib, plus a results dataframe exported to CSV |

### 7.3 Module layout

```
closed_loop_sim/
  config.py            # parameter dataclasses / schema, defaults, sweep grids
  dgp.py               # simulate_population(L, E), simulate_trajectory(), hazards
  observation.py       # visit process, detection delays, decoy + coding-shift injection
  eventlog.py          # assemble patient-level timestamped event log (case, activity, time, group)
  discovery.py         # DFG construction via pm4py, transition frequencies and median times
  refinement.py        # candidate-proposal engine + plausibility gate (Arms A and B rule sets)
  estimation.py        # cause-specific Cox, Fine-Gray weights, g-computation mediation
  truth.py             # closed-form / simulation oracle for true DAG, HRs, mediated proportion
  metrics.py           # structural metrics, false-edge rejection, bias/RMSE/coverage
  experiment.py        # run an arm, run a cell, run the full sweep with M replications
  cli.py               # entry point: run a named experiment from a config
  tests/               # unit tests: hazard sampling, detection delay, gate rules
notebooks/
  01_single_run_walkthrough.ipynb
  02_arm_A_vs_B.ipynb
  03_sweeps_and_figures.ipynb
```

### 7.4 Reproducibility

- Single master seed, per-replication seeds derived deterministically (numpy SeedSequence) and logged.
- Every run reads a versioned config file; the config is written into the output directory alongside results.
- Pin versions in a lockfile; record pm4py and lifelines versions in the results metadata.
- Archive code and the generated tutorial-scale dataset with a Zenodo DOI (this also satisfies the BMC availability-of-data requirement in the revision checklist, item D4).

---

## 8. Work plan

| Phase | Tasks | Deliverable |
|-------|-------|-------------|
| 1. DGP and oracle | implement hazards, trajectory sampler, truth.py; verify marginal rates match Table 4 within tolerance | calibrated simulator + a truth report |
| 2. Observation model | visit process, detection delays, decoy and coding-shift injection; confirm artifacts appear in the log | event-log generator |
| 3. Loop integration | wire DFG discovery, proposal engine, Arms A and B gates | runnable single-replication pipeline |
| 4. Estimation | cause-specific Cox, Fine-Gray, g-computation mediation; validate against oracle on large n | estimation module |
| 5. Metrics and a single cell | structural + estimand metrics, run base scenario with M replications | Arm A vs B base-case result |
| 6. Sweeps | run the grid in Section 6 | results dataframe + figures |
| 7. Write-up | methods paragraph, results subsection, figures, limitations tie-in | manuscript-ready text and exhibits |

A minimal first milestone (Phases 1 to 5 at one cell: one confounder, n approx 5000, both traps on, base kappa) is enough to establish the central A-versus-B result. The sweeps extend it.

---

## 9. Mapping to the manuscript

- New Methods subsection under Section 4 or a standalone "Simulation study" section describing the DGP, observation model, arms, and metrics.
- New Results subsection: structural recovery table, the Arm A vs Arm B artifact-rejection contrast, and estimand-coverage results, plus the sweep figures.
- Table 5 future-work rows become partially evidenced rather than purely speculative, since the sweeps probe surveillance bias, temporal granularity, and competing risks directly.
- Reinforces the checklist item E1 and converts the paper from a workflow description into a workflow with a demonstrated recovery property, which is the bar a methods reviewer at BMC-MRM (or JBI) expects.

---

## Addendum A (v1.1, 2026-06-10): review-driven clarifications

These items tighten the spec after two technical reviews. They clarify, they do not change, the DGP or the latent truth. The companion LLM-extension design lives in `docs/superpowers/specs/2026-06-10-llm-closed-loop-simulation-design.md`.

### A.1 Event-log activity vocabulary

Latent states (Section 2) are never directly observed. The event log records **activities**; pm4py discovery and the DFG operate on activities only.

| Latent state | Recorded event-log activity |
|--------------|-----------------------------|
| S0 index / on-drug | `IndexDrug` (entry) |
| S1 CKD | `CKD_recorded` (eGFR ≤ threshold at a visit) |
| S2 CVAE | `CVAE_recorded` |
| S3 Death | `Death` (exact, administrative) |
| — (artifact) | `MedicationChange` / `DrugSwitch` (post-CVAE, prob. p_switch) |
| — (artifact) | `CKD_recode` (spurious post-CVAE CKD code, prob. p_recode) |

The two traps are therefore activity transitions: `CVAE_recorded → MedicationChange` (decoy; DAG interpretation CVAE→Drug-use, Fig. 5) and `CVAE_recorded → CKD_recode` (coding shift; reverse CKD re-detection). The labels "S2→S0" and "S2→S1" remain as DAG-level shorthand only.

### A.2 Arm naming

Arm A = **ungated proposal-only** (mechanical proposer + auto-accept; a deterministic proxy for an ungoverned AI/refinement system). Arm B = **gated** (mechanical proposer + plausibility gate). Arm C = oracle. The earlier "AI-proposal-only" label is retired because the deterministic study contains no LLM.

### A.3 Gate thresholds (pre-registered; calibrate on a null-edge DGP)

All thresholds are fixed in config **before any run**; where possible they are calibrated on a null-edge DGP variant (true intensity = 0, artifacts on) so they are not tuned to the result.

| Component | Specification | Default |
|-----------|---------------|---------|
| Proposal support threshold s_min | a candidate is only proposed / NoneGate-accepted if its DFG relative-antecedent frequency exceeds s_min | 0.10 (matches Fig. 4's >10% transition-risk indicator) |
| Monitoring strata | monitoring intensity = #visits in follow-up, dichotomized at the cohort median; adjust via stratification or IPW on visit rate | median split |
| Attenuation test τ_att | reject as surveillance artifact if monitoring-adjusted log-strength attenuates by ≥ τ_att vs crude, or the adjusted 95% CI crosses the null | τ_att = 50%; α (false-edge acceptance) = 0.05 |
| Negative-control failure | a designated edge/outcome with known null (HR = 1) "fails" if its 95% CI excludes 1 in the artifact-consistent direction | 95% CI excludes 1 |
| Acyclicity / protected direction | known directions stored as a directed prior partial order (Death absorbing → no out-edges; S0 entry; CKD precedes CVAE on the mediating path); reject cycles or reversals | as stored |

### A.4 Trap-specific gate rules

A single attenuation test does not catch every trap; the gate is a set of typed rules.

| Trap (event-log) | DAG interpretation | Gate signal that rejects it |
|------------------|--------------------|-----------------------------|
| `CVAE_recorded → CKD_recode` (S2→S1) | reverse CKD re-detection | reverse-known-direction (acyclicity / protected order) + post-event coding-artifact flag + attenuation / negative-control |
| `CVAE_recorded → MedicationChange` (S2→S0) | CVAE→Drug-use feedback (Fig. 5) | **activity-type rule**: `MedicationChange` is an administrative post-outcome management activity, not eligible as a biological causal edge |
| `S0 → S1` surveillance inflation | inflated CKD onset under higher monitoring | monitoring-stratified attenuation ≥ τ_att |
| any false general edge | — | negative-control failure / low plausibility |

The ungoverned arm (NoneGate) has none of these rules, so it accepts the frequent traps — preserving the A-vs-B contrast.

### A.5 Primary and secondary endpoints

**Primary endpoint:** the proportion of replications in which the recovered DAG accepts ≥1 injected artifact edge. **Primary contrast:** Arm A vs Arm B (difference in proportions, Newcombe/Wilson 95% CI). **Secondary:** per-trap false-edge rejection (each trap separately, with the proposed-vs-never-proposed split), edge precision/recall/F1, structural Hamming distance, Death-competing-risk-node-recovery indicator, and estimand bias / RMSE / 95% CI coverage. Pre-registered success: Arm B rejects both traps in ≥95% of replications at near-nominal (93–97%) estimand coverage; Arm A accepts ≥1 trap in ≥80%.

### A.6 Estimand → model mapping

Keep the targets distinct; do **not** claim Fine–Gray recovers cause-specific hazard ratios.

| Estimand | Model | Truth source |
|----------|-------|--------------|
| Cause-specific HR (β^E per transition) | cause-specific Cox (lifelines) | DGP β^E values |
| Cumulative incidence (CVAE, Death) by arm | Fine–Gray subdistribution (weighted Cox) | DGP CIF |
| Mediated proportion (PPI→CVAE via CKD) | simulation g-computation | DGP oracle |

### A.7 CKD–CVAE interpretation caveat

The S2→S1 trap is false **by construction in the simulation** — a deliberately injected post-event coding artifact representing reverse CKD re-detection. It is **not** a claim that CVAE cannot biologically affect kidney function: reverse cardiorenal effects are plausible, and a non-significant between-group CVAE→CKD signal is not evidence of absence (manuscript §5.2). Write-ups must phrase the trap as an injected coding artifact, not a biological impossibility.
