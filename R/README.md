# R port — ground-truth simulation

A faithful R translation of the Python `closed_loop_sim` package, for statisticians who work
in R. It reproduces the same headline result (Arm A accepts both injected artefacts, Arm B
rejects both) and the same estimand-recovery story, using the cause-specific Cox models from
the **`survival`** package.

## Dependencies

Base R (≥ 4.0) and **`survival`** only — no niche packages. The continuous-time DGP and the
directly-follows graph are hand-coded, so nothing else is required.

```r
install.packages("survival")   # if not already installed
```

(Optional, for richer alternatives mentioned in the manuscript: `mstate`, `simsurv`,
`riskRegression`, `bupaR`. The port does not need them.)

## Quickstart

```r
setwd("R")            # this directory
source("closed_loop_sim.R")

cfg  <- base_cell()
data <- build_event_log(5000, cfg, seed = 1)
disc <- discover(data$log, data$visit_counts)

for (a in c("A", "B", "C"))
  cat(a, "traps:", length(intersect(run_arm(a, disc)$edges, TRAP_KEYS)), "\n")
# A traps: 2   B traps: 0   C traps: 0
```

From the shell:

```bash
Rscript test_closed_loop_sim.R     # 12 base-R tests
Rscript run_study.R 100            # run the study -> ../results_R/ (summary, csv, figures)
```

Render the tutorial:

```r
rmarkdown::render("tutorial.Rmd")   # needs the 'rmarkdown' package
```

## Files

| File | Role |
|------|------|
| `closed_loop_sim.R` | the whole package: config, DGP, oracle, observation, event log, discovery, refinement (arms A/B/C), estimation (`survival::coxph`), metrics, experiment, base-R visualisation |
| `run_study.R` | run the base cell and write `../results_R/` |
| `test_closed_loop_sim.R` | base-R test suite (12 checks) |
| `tutorial.Rmd` | R Markdown walkthrough mirroring the Python tutorial |

## Function map (mirrors the Python modules)

```
base_cell()                 config.py        parameters + tuned base cell
simulate_population()       dgp.py           Weibull-PH trajectory sampler
TRUE_EDGES / mediated_proportion()  truth.py oracle: edges, g-computation mediated proportion
build_event_log()           observation/eventlog   visits, detection, artefact injection
discover()                  discovery.py     state-level DFG + monitoring evidence
run_arm("A"|"B"|"C")        refinement.py    proposer + none/rules gate; oracle
structural_metrics()        metrics.py       precision/recall/F1/SHD, trap rejection
estimate()                  estimation.py    cause-specific Cox: latent/recorded/adjusted
run_cell()                  experiment.py    aggregate over replications
process_map(), dag_refinement()  visualize.py   figures
```

## Notes

- **Reproducibility:** `set.seed()` drives every population/log build; `run_cell` derives
  per-replication seeds from a master seed.
- **The mediated-proportion oracle is Monte Carlo** — the total effect is a small difference
  of cumulative-incidence functions, so use a large `n` (e.g. 50,000, as in `run_study.R`) for
  a stable estimate.
- R is slower than the Python implementation; `run_study.R` defaults to `M = 100`. Results
  match the Python port up to Monte Carlo error.
