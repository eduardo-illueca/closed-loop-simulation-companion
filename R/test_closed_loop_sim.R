# Basic tests for the R port (base R; no testthat dependency).
#   Rscript test_closed_loop_sim.R
source("closed_loop_sim.R")

ok <- 0L; fail <- 0L
check <- function(cond, msg) {
  if (isTRUE(cond)) { ok <<- ok + 1L; cat("PASS:", msg, "\n") }
  else { fail <<- fail + 1L; cat("FAIL:", msg, "\n") }
}

set.seed(0)
cfg <- base_cell()

# hazard sampler: exponential mean (rho=1) ~ 1/lam
draws <- replicate(50000, sample_transition_time(0.5, 1.0, 0.0))
check(abs(mean(draws) - 2.0) < 0.06, "Weibull(rho=1) mean ~ 1/lam")

# config: S1->S2 exposure effect is exactly zero
check(true_beta_E(cfg, "S1->S2") == 0.0, "S1->S2 beta_E is exactly 0")

# event log: both artefacts present; no event after death
data <- build_event_log(4000, cfg, 1L)
acts <- unique(data$log$activity)
check("MedicationChange" %in% acts && "CKD_recode" %in% acts, "both artefacts appear in log")
after_death <- FALSE
for (cs in split(data$log, data$log$case)) {
  if ("Death" %in% cs$activity) {
    dt <- min(cs$time[cs$activity == "Death"])
    if (any(cs$time > dt)) after_death <- TRUE
  }
}
check(!after_death, "no activity recorded after death")

# surveillance bias: PPI CKD recorded earlier than H2B
ckd <- data$log[data$log$activity == "CKD_recorded", ]
mp <- median(ckd$time[ckd$group == "PPI"]); mh <- median(ckd$time[ckd$group == "H2B"])
check(mp < mh, "PPI CKD recorded earlier than H2B (surveillance bias)")

# discovery: both traps surface as candidates
disc <- discover(data$log, data$visit_counts)
check("S2->S0" %in% names(disc) && "S2->S1" %in% names(disc), "both traps in the DFG")

# refinement: A accepts both traps, B rejects both, B adds Death node
mA <- structural_metrics(run_arm("A", disc))
mB <- structural_metrics(run_arm("B", disc))
mC <- structural_metrics(run_arm("C", disc))
check(mA$n_traps == 2, "Arm A accepts both traps")
check(mB$n_traps == 0 && mB$death_node, "Arm B rejects both traps and adds Death node")
check(mB$precision == 1.0, "Arm B has perfect precision")
check(mC$f1 == 1.0, "Arm C (oracle) is perfect")

# estimand: latent Cox recovers ~0.25; visit adjustment over-corrects upward
e <- estimate(data, cfg$t_max)
check(abs(e$s0s1_latent$beta - 0.25) < 0.10, "latent Cox recovers true beta_E ~ 0.25")
check(e$s0s1_adjusted$beta > e$s0s1_naive$beta, "visit-count adjustment over-corrects (collider)")

cat(sprintf("\n%d passed, %d failed\n", ok, fail))
if (fail > 0) quit(status = 1)
