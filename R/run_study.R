# Run the base-cell study in R and write results to ../results_R/.
#   Rscript run_study.R [M]          (default M = 100; R is slower than Python)
#
# Writes: summary.txt, replications.csv, process_map.png, dag_refinement.png.

source("closed_loop_sim.R")

args <- commandArgs(trailingOnly = TRUE)
M <- if (length(args) >= 1) as.integer(args[1]) else 100L
OUT <- "../results_R"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
cfg <- base_cell()

message(sprintf("[1/4] base cell: M=%d, n=%d ...", M, cfg$n))
set.seed(20260610L)
seeds <- sample.int(.Machine$integer.max, M)
reps <- lapply(seeds, function(s) run_replication(cfg, s, with_estimation = TRUE))

# per-replication CSV (all M)
rows <- do.call(rbind, lapply(reps, function(r) do.call(rbind, lapply(c("A", "B", "C"), function(a) {
  s <- r$structural[[a]]
  data.frame(seed = r$seed, arm = a, precision = s$precision, recall = s$recall,
             f1 = s$f1, shd = s$shd, n_traps = s$n_traps,
             accepts_any_trap = s$accepts_any_trap,
             beta_s0s1_latent = r$estimation$s0s1_latent$beta,
             beta_s0s1_naive = r$estimation$s0s1_naive$beta,
             beta_s0s1_adjusted = r$estimation$s0s1_adjusted$beta,
             beta_s1s2 = r$estimation$s1s2$beta)
}))))
write.csv(rows, file.path(OUT, "replications.csv"), row.names = FALSE)

# aggregate
agg <- function(a, f) sapply(reps, function(r) f(r$structural[[a]]))
struct <- lapply(c("A", "B", "C"), function(a) {
  k <- sum(agg(a, function(m) m$accepts_any_trap))
  list(precision = mean(agg(a, function(m) m$precision)),
       recall = mean(agg(a, function(m) m$recall)),
       f1 = mean(agg(a, function(m) m$f1)), shd = mean(agg(a, function(m) m$shd)),
       n_traps = mean(agg(a, function(m) m$n_traps)),
       any_trap = k / M, death = mean(agg(a, function(m) m$death_node)))
})
names(struct) <- c("A", "B", "C")

# estimand recovery
est_summary <- function(key, truth) {
  betas <- sapply(reps, function(r) r$estimation[[key]]$beta)
  cover <- mean(mapply(function(r) { e <- r$estimation[[key]]
    !is.na(e$lo) && e$lo <= truth && truth <= e$hi }, reps))
  c(mean = mean(betas, na.rm = TRUE), bias = mean(betas, na.rm = TRUE) - truth,
    rmse = sqrt(mean((betas - truth)^2, na.rm = TRUE)), coverage = cover)
}
t01 <- true_beta_E(cfg, "S0->S1"); t12 <- true_beta_E(cfg, "S1->S2")

message("[2/4] mediation oracle (n=50000) ...")
mp <- mediated_proportion(cfg, 50000L, 999L, cfg$t_max)

message("[3/4] figures ...")
data1 <- build_event_log(cfg$n, cfg, 1L)
disc1 <- discover(data1$log, data1$visit_counts)
process_map(disc1, file.path(OUT, "process_map.png"))
dag_refinement(list(A = run_arm("A", disc1)$edges, B = run_arm("B", disc1)$edges),
               file.path(OUT, "dag_refinement.png"))

message("[4/4] summary ...")
con <- file(file.path(OUT, "summary.txt"), "w")
wl <- function(...) writeLines(sprintf(...), con)
wl("Ground-truth simulation (R) -- base cell, M=%d, n=%d, R %s, survival %s",
   M, cfg$n, getRversion(), as.character(packageVersion("survival")))
wl("")
wl("STRUCTURAL RECOVERY (mean over M)")
wl("%-3s %8s %7s %6s %6s %7s %9s %7s", "arm", "prec", "recall", "f1", "shd", "traps", "anytrap", "death")
for (a in c("A", "B", "C")) { s <- struct[[a]]
  wl("%-3s %8.3f %7.3f %6.2f %6.2f %7.2f %8.0f%% %6.0f%%",
     a, s$precision, s$recall, s$f1, s$shd, s$n_traps, 100 * s$any_trap, 100 * s$death) }
wl("")
wl("ESTIMAND RECOVERY (true S0S1=%.2f, S1S2=%.2f)", t01, t12)
wl("%-16s %8s %8s %7s %9s", "target", "mean", "bias", "rmse", "coverage")
for (kt in list(c("s0s1_latent", t01), c("s0s1_naive", t01), c("s0s1_adjusted", t01), c("s1s2", t12))) {
  r <- est_summary(kt[[1]], as.numeric(kt[[2]]))
  wl("%-16s %8.3f %+8.3f %7.3f %8.2f", kt[[1]], r["mean"], r["bias"], r["rmse"], r["coverage"]) }
wl("")
wl("MEDIATED PROPORTION (g-computation oracle, n=50000): %.3f (indirect %.4f / total %.4f)",
   mp$mediated_proportion, mp$indirect, mp$total)
close(con)

cat(readLines(file.path(OUT, "summary.txt")), sep = "\n")
message("\nDONE -> ", OUT, "/")
