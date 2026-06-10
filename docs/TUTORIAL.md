# Tutorial: validating the closed-loop framework under a known truth

This walkthrough builds the study one layer at a time. By the end you will have
generated data with injected artefacts, discovered the process map, run the governed and
ungoverned refinement arms, and reproduced the headline result that the **gate rejects the
artefacts an ungoverned configuration accepts**.

Run the snippets in a Python session or notebook from the repo root. Each one is a few
seconds. For the interactive version, open `notebooks/closed_loop_simulation.ipynb`.

```bash
pip install -r requirements.txt    # or: pip install -e ".[dev]"
```

---

## The big idea

The framework's novel claim is not that process mining finds temporal patterns (that is
established) but that a **governed** loop recovers correct causal structure while rejecting
edges that are artefacts of the *observation* process rather than the *disease* process. To
prove that, we own the data-generating process, so the truth is fixed and we can plant traps.

```
 LATENT disease process   ──►  the true causal DAG  (dgp.py, truth.py)
        │  separation that makes the study valid
 OBSERVATION process      ──►  visits, detection delays, INJECTED artefacts (observation.py)
        │
 event log ─► process map ─► refine (Arm A | B | C) ─► estimate ─► grade vs truth
```

Disease states: **S0** index/on-drug → **S1** CKD → **S2** CVAE → **S3** Death (absorbing).
Two trap edges live only in the *recorded* log:

- `S2 → S0` — a **drug-feedback decoy**: a medication change after a cardiovascular event
  (administrative, not causal — the manuscript's Fig. 5 decoy).
- `S2 → S1` — a **reverse coding shift**: a spurious CKD re-code after a cardiovascular event.

---

## Step 1 — The latent truth (DGP)

Each patient walks a continuous-time multi-state model with Weibull proportional-hazards
cause-specific intensities. Exposure (PPI = 1, H2B = 0) is assigned by a confounded
propensity model.

```python
from closed_loop_sim.config import base_cell
from closed_loop_sim.dgp import simulate_population

cfg = base_cell()
pop = simulate_population(2000, cfg, seed=1)
print(pop.E.mean())            # ~0.69 PPI prevalence
print(pop.histories[0])        # e.g. [('S0', 0.0), ('S1', 3.1), ('S2', 5.4), ('S3', 6.0)]
```

The exposure effects encode the mediation hypothesis: `beta_E` is positive on `S0->S1` (PPI
raises CKD onset) and **exactly zero on `S1->S2`** — mediation operates through CKD
*occurrence*, not by changing the CKD→CVAE rate.

```python
print(cfg.dgp.transitions["S0->S1"].beta_E)   # 0.25
print(cfg.dgp.transitions["S1->S2"].beta_E)   # 0.0
```

## Step 2 — The oracle

Because we own the DGP, the simulator is the oracle. The true edge set and exposure
log-HRs are known, and the mediated proportion is computed by g-computation on the
simulator itself.

```python
from closed_loop_sim.truth import TRUE_EDGES, TRAP_EDGES, mediated_proportion

print(sorted(TRUE_EDGES))   # the 6 true transitions
print(sorted(TRAP_EDGES))   # [('S2', 'S0'), ('S2', 'S1')] -- must be rejected

med = mediated_proportion(cfg, n=8000, seed=4, horizon=cfg.dgp.t_max)
print(round(med["mediated_proportion"], 2))   # ~0.45 of the PPI->CVAE effect flows via CKD
```

## Step 3 — The observation process and the artefacts

Now we record the latent process through visits. PPI users are monitored more (`kappa`),
so their disease is detected sooner — the surveillance bias. Follow-up is truncated at
death (no event is ever recorded after death), and the two artefacts are injected after a
recorded CVAE.

```python
from closed_loop_sim.eventlog import build_event_log_data

data = build_event_log_data(6000, cfg, seed=1)
log = data.log
print(log.head())
acts = set(log["activity"].unique())
print("MedicationChange" in acts, "CKD_recode" in acts)   # True True -- artefacts present
```

The surveillance signature: at identical latent onset, PPI's CKD is recorded earlier.

```python
ckd = log[log.activity == "CKD_recorded"]
print(ckd[ckd.group == "PPI"].time.median(), ckd[ckd.group == "H2B"].time.median())
# PPI median < H2B median
```

## Step 4 — Discovery: the process map

The recorded log is summarised as a directly-follows graph (pm4py) and mapped to the four
state nodes. Each candidate transition carries its frequency, median time, and
monitoring-stratified strength.

```python
from closed_loop_sim.discovery import discover

disc = discover(log, data.visit_counts)
for (a, b), e in sorted(disc.state_edges.items(), key=lambda x: -x[1].risk):
    print(f"{a}->{b}  risk={e.risk:.2f}  median={e.median_time:.1f}y  src={e.source_activities}")
# S2->S0 (decoy, ~0.34) and S2->S1 (coding shift, ~0.18) appear alongside the true edges
```

Render the process map (saved to `results/process_map.png`):

```python
from closed_loop_sim.visualize import process_map
process_map(disc)
```

## Step 5 — Refinement: the three arms

Refinement starts from the initial knowledge graph `{S0->S1, S0->S2, S1->S2}` (no Death
node) and proposes the transitions the process map reveals. The arms differ only in the
**gate**:

- **Arm A (ungated):** accept every proposal above the 10% support threshold.
- **Arm B (governed):** a plausibility gate rejects backward edges (catches the reverse
  coding shift `S2->S1`) and administrative-activity edges (catches the decoy `S2->S0`).
- **Arm C (oracle):** handed the true DAG.

```python
from closed_loop_sim.refinement import ARMS
from closed_loop_sim.truth import TRAP_EDGES

for arm, run in ARMS.items():
    res = run(disc)
    traps = sorted(set(res.edges) & TRAP_EDGES)
    print(arm, "traps accepted:", traps, "| Death node:", "S3" in res.nodes)
# A traps accepted: [('S2','S0'), ('S2','S1')]   <- ungated accepts both
# B traps accepted: []                            <- governed rejects both
# C traps accepted: []
```

Render the True / A / B DAGs (saved to `results/dag_refinement.png`):

```python
from closed_loop_sim.visualize import dag_refinement
dag_refinement({arm: set(ARMS[arm](disc).edges) for arm in "ABC"})
```

## Step 6 — Metrics: structural recovery

```python
from closed_loop_sim.metrics import structural_metrics

for arm in "ABC":
    m = structural_metrics(ARMS[arm](disc))
    print(arm, f"precision={m['precision']:.2f} recall={m['recall']:.2f} "
              f"SHD={m['shd']} traps={m['n_traps_accepted']}")
# A precision=0.71 ... traps=2     B precision=1.00 ... traps=0     C precision=1.00 ... traps=0
```

Arm B's recall is 0.83 (not 1.0): the rare direct `S0->S3` background-mortality edge (~5%)
is below the 10% process-mining threshold — an honest sample-size limitation. The Death node
is still recovered through the high-mortality `S1->S3` and `S2->S3` transitions.

## Step 7 — Estimation: the surveillance-bias story

The framework selects cause-specific Cox models. Fit to the *latent* process the model
recovers the truth; fit to the *recorded* times it is inflated by surveillance bias; and a
naive visit-count adjustment **over-corrects** (collider bias) — which is why the framework
treats surveillance bias as a structural diagnostic, not a covariate adjustment.

```python
from closed_loop_sim.estimation import estimate

e = estimate(data, cfg.dgp.t_max)
for k in ["s0s1_latent", "s0s1_naive", "s0s1_adjusted"]:
    print(k, "beta_E =", round(e[k]["beta_E"], 3))   # true = 0.25
# s0s1_latent ~0.25   s0s1_naive ~0.33   s0s1_adjusted ~0.64
```

## Step 8 — The full study

Aggregate over many replications and write the archived results + figures.

```python
from closed_loop_sim.experiment import run_cell

res = run_cell(cfg, M=50, with_estimation=True)
print("Arm A accepts a trap in", res["structural"]["A"]["accept_any_trap_rate"])  # 1.0
print("Arm B accepts a trap in", res["structural"]["B"]["accept_any_trap_rate"])  # 0.0
print("latent coverage:", res["estimation"]["s0s1_latent"]["coverage"])           # ~0.95
```

Or from the command line (writes `results/base_cell_results.json`, `replications.csv`):

```bash
python -m closed_loop_sim.run_study 500
python -m closed_loop_sim.finalize    # fills docs/manuscript_sections.md numbers + figure
python -m closed_loop_sim.visualize   # process map + DAG refinement figures
```

---

## Extending the study

Everything is driven by `base_cell()` in `config.py`. To explore the robustness sweeps
described in the manuscript, vary one factor and re-run:

```python
cfg = base_cell()
cfg.obs.kappa = 3.0          # stronger surveillance differential
cfg.n = 50000                # larger sample -> the rare S0->S3 edge clears the threshold
cfg.obs.p_switch = 0.0       # turn the decoy off
res = run_cell(cfg, M=50)
```

The planned grid sweeps sample size {1k, 5k, 50k, 300k}, monitoring differential
{1, 1.5, 2, 3}, timestamp granularity, confounding strength, and artefact rates.

## How this maps to the manuscript

- **Methods / Results sections:** `docs/manuscript_sections.md` (filled with the M = 500
  numbers, with Fig. S1–S3 callouts).
- **Full specification:** `docs/METHODS_SPEC.md`.
- **Tables 4/5, Fig. 5 decoy, refs [44], [48]–[50]:** the calibration targets, injected
  artefacts, and surveillance-bias mechanism reproduce those manuscript elements directly.
