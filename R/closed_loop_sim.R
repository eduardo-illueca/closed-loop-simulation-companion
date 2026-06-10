# Closed-Loop Causal Refinement — Ground-Truth Simulation (R port)
#
# Faithful R translation of the Python `closed_loop_sim` package. Depends only on
# base R and the `survival` package (cause-specific Cox), so it runs anywhere a
# statistician already has R. Source this file, then see run_study.R / tutorial.Rmd.
#
#   source("closed_loop_sim.R")
#
# States: S0 index/on-drug -> S1 CKD -> S2 CVAE -> S3 Death (absorbing).
# Trap edges (recorded but not causal): S2->S0 decoy, S2->S1 reverse coding shift.

suppressMessages(library(survival))

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
COVARIATE_NAMES <- c("age", "sex", "diabetes", "hypertension", "egfr")

base_cell <- function() {
  tp <- function(lam, rho, beta_E, beta_L) list(lam = lam, rho = rho, beta_E = beta_E, beta_L = beta_L)
  list(
    seed = 20260610L, n = 5000L, t_max = 10,
    covariates = list(age_mean = 70, age_sd = 10, sex_female_prob = 0.5,
                      diabetes_prob = 0.25, hypertension_prob = 0.55,
                      egfr_mean = 75, egfr_sd = 15),
    # alpha0 sets ~69% PPI; alpha creates the confounding imbalance to be corrected.
    propensity = list(alpha0 = 0.8, alpha = c(0.4, 0.0, 0.5, 0.3, -0.4)),
    obs = list(lambda_base = 1.0, kappa = 2.0, p_switch = 0.4, p_recode = 0.3),
    transitions = list(
      "S0->S1" = tp(0.0140, 1.20, 0.25, c(0.30, 0, 0.35, 0.20, -0.45)),
      "S0->S2" = tp(0.0038, 1.10, 0.20, c(0.35, 0, 0.30, 0.30, -0.10)),
      "S0->S3" = tp(0.0032, 1.05, 0.05, c(0.55, 0, 0.20, 0.15, -0.15)),
      "S1->S2" = tp(0.0285, 1.15, 0.00, c(0.25, 0, 0.30, 0.30, -0.20)),   # beta_E exactly 0
      "S1->S3" = tp(0.0207, 1.10, 0.10, c(0.50, 0, 0.25, 0.15, -0.25)),
      "S2->S3" = tp(0.1304, 0.95, 0.10, c(0.40, 0, 0.20, 0.10, -0.10))
    )
  )
}

# ---------------------------------------------------------------------------
# 2. DGP: latent truth
# ---------------------------------------------------------------------------
TRUE_TRANSITIONS <- list(S0 = c("S1", "S2", "S3"), S1 = c("S2", "S3"), S2 = "S3", S3 = character(0))

# Inverse-CDF draw from a Weibull-PH cause-specific hazard (clock reset at entry).
sample_transition_time <- function(lam, rho, linpred) {
  a <- lam * exp(linpred)
  (-log(runif(1)) / a)^(1 / rho)
}

draw_covariates <- function(n, cov) {
  cbind(age = rnorm(n, cov$age_mean, cov$age_sd),
        sex = as.numeric(runif(n) < cov$sex_female_prob),
        diabetes = as.numeric(runif(n) < cov$diabetes_prob),
        hypertension = as.numeric(runif(n) < cov$hypertension_prob),
        egfr = rnorm(n, cov$egfr_mean, cov$egfr_sd))
}

standardize <- function(L) {
  mu <- colMeans(L); sds <- apply(L, 2, sd); sds[sds == 0] <- 1
  sweep(sweep(L, 2, mu), 2, sds, "/")
}

assign_exposure <- function(L_std, prop) {
  logit <- prop$alpha0 + as.numeric(L_std %*% prop$alpha)
  as.integer(runif(length(logit)) < 1 / (1 + exp(-logit)))
}

# E may be a scalar (constant exposure) or a function(transition) -> 0/1.
simulate_trajectory <- function(L_std_row, E, cfg) {
  e_func <- if (is.function(E)) E else function(tr) as.integer(E)
  state <- "S0"; t <- 0
  states <- "S0"; times <- 0
  while (state != "S3" && t < cfg$t_max) {
    outs <- TRUE_TRANSITIONS[[state]]
    if (length(outs) == 0) break
    best_k <- NA; best_dt <- Inf
    for (k in outs) {
      tp <- cfg$transitions[[paste0(state, "->", k)]]
      lp <- tp$beta_E * e_func(paste0(state, "->", k)) + sum(tp$beta_L * L_std_row)
      dt <- sample_transition_time(tp$lam, tp$rho, lp)
      if (dt < best_dt) { best_dt <- dt; best_k <- k }
    }
    if (t + best_dt > cfg$t_max) break  # administrative censoring
    t <- t + best_dt; state <- best_k
    states <- c(states, state); times <- c(times, t)
  }
  list(states = states, times = times)
}

simulate_population <- function(n, cfg, seed) {
  set.seed(seed)
  L <- draw_covariates(n, cfg$covariates)
  L_std <- standardize(L)
  E <- assign_exposure(L_std, cfg$propensity)
  histories <- vector("list", n)
  for (i in seq_len(n)) histories[[i]] <- simulate_trajectory(L_std[i, ], E[i], cfg)
  list(L = L, L_std = L_std, E = E, histories = histories)
}

# ---------------------------------------------------------------------------
# 3. Truth / oracle
# ---------------------------------------------------------------------------
TRUE_EDGES <- list(c("S0","S1"), c("S0","S2"), c("S0","S3"),
                   c("S1","S2"), c("S1","S3"), c("S2","S3"))
TRAP_EDGES <- list(c("S2","S0"), c("S2","S1"))
STATE_ORDER <- c(S0 = 0, S1 = 1, S2 = 2, S3 = 3)
.edge_key <- function(e) paste0(e[1], "->", e[2])
TRUE_KEYS <- vapply(TRUE_EDGES, .edge_key, "")
TRAP_KEYS <- vapply(TRAP_EDGES, .edge_key, "")

true_beta_E <- function(cfg, transition) cfg$transitions[[transition]]$beta_E

marginal_transition_rates <- function(histories, E) {
  arm_rates <- function(idx) {
    origin <- c(S0 = 0, S1 = 0, S2 = 0)
    num <- setNames(integer(length(TRUE_KEYS)), TRUE_KEYS)
    for (i in idx) {
      st <- histories[[i]]$states
      for (s in c("S0","S1","S2")) if (s %in% st) origin[s] <- origin[s] + 1
      if (length(st) > 1) for (j in seq_len(length(st) - 1)) {
        key <- paste0(st[j], "->", st[j + 1]); if (key %in% TRUE_KEYS) num[key] <- num[key] + 1
      }
    }
    sapply(TRUE_KEYS, function(k) { o <- origin[sub("->.*", "", k)]; if (o > 0) num[k] / o else 0 })
  }
  list(PPI = arm_rates(which(E == 1)), H2B = arm_rates(which(E == 0)))
}

.cvae_cif <- function(cfg, n, seed, e_direct, e_med, horizon) {
  set.seed(seed)
  L <- draw_covariates(n, cfg$covariates); L_std <- standardize(L)
  invisible(assign_exposure(L_std, cfg$propensity))  # keep stream aligned
  e_func <- function(tr) if (tr == "S0->S1") e_med else e_direct
  hits <- 0
  for (i in seq_len(n)) {
    h <- simulate_trajectory(L_std[i, ], e_func, cfg)
    if (any(h$states == "S2" & h$times <= horizon)) hits <- hits + 1
  }
  hits / n
}

mediated_proportion <- function(cfg, n, seed, horizon) {
  cif11 <- .cvae_cif(cfg, n, seed, 1, 1, horizon)
  cif10 <- .cvae_cif(cfg, n, seed, 1, 0, horizon)
  cif00 <- .cvae_cif(cfg, n, seed, 0, 0, horizon)
  total <- cif11 - cif00; indirect <- cif11 - cif10
  list(mediated_proportion = if (total != 0) indirect / total else NA,
       total = total, indirect = indirect, direct = cif10 - cif00)
}

# ---------------------------------------------------------------------------
# 4. Observation process (manufactures the biases)
# ---------------------------------------------------------------------------
ACTIVITY_ORDER <- c(IndexDrug = 0, CKD_recorded = 1, CVAE_recorded = 2,
                    MedicationChange = 3, CKD_recode = 4, Death = 999)

generate_visits <- function(E, obs, horizon) {
  rate <- obs$lambda_base * (if (E == 1) obs$kappa else 1)
  v <- numeric(0); t <- 0
  repeat { t <- t + rexp(1, rate); if (t >= horizon) break; v <- c(v, t) }
  v
}

# returns data.frame(activity, time); CKD/CVAE only if detected before death.
observe_trajectory <- function(history, E, visits) {
  onset <- setNames(history$times, history$states)
  death <- if ("S3" %in% history$states) onset[["S3"]] else NA
  act <- "IndexDrug"; tm <- 0
  for (sa in list(c("S1","CKD_recorded"), c("S2","CVAE_recorded"))) {
    s <- sa[1]; a <- sa[2]
    if (s %in% names(onset)) {
      cand <- visits[visits >= onset[[s]]]
      if (length(cand)) {
        tt <- cand[1]
        if (is.na(death) || tt < death) { act <- c(act, a); tm <- c(tm, tt) }
      }
    }
  }
  if (!is.na(death)) { act <- c(act, "Death"); tm <- c(tm, death) }
  ord <- order(tm, ACTIVITY_ORDER[act])
  data.frame(activity = act[ord], time = tm[ord], stringsAsFactors = FALSE)
}

inject_artifacts <- function(events, obs, visits) {
  cvae <- events$time[events$activity == "CVAE_recorded"]
  if (length(cvae) == 0) return(events)
  nv <- visits[visits > cvae[1]]
  if (length(nv) == 0) return(events)
  nx <- nv[1]
  if (runif(1) < obs$p_switch) events <- rbind(events, data.frame(activity = "MedicationChange", time = nx))
  if (runif(1) < obs$p_recode) events <- rbind(events, data.frame(activity = "CKD_recode", time = nx))
  events[order(events$time, ACTIVITY_ORDER[events$activity]), ]
}

# ---------------------------------------------------------------------------
# 5. Event log
# ---------------------------------------------------------------------------
build_event_log <- function(n, cfg, seed) {
  set.seed(seed)
  L <- draw_covariates(n, cfg$covariates); L_std <- standardize(L)
  E <- assign_exposure(L_std, cfg$propensity)
  cases <- acts <- grps <- character(0); tms <- numeric(0)
  visit_counts <- integer(n); histories <- vector("list", n)
  for (i in seq_len(n)) {
    h <- simulate_trajectory(L_std[i, ], E[i], cfg); histories[[i]] <- h
    death <- if ("S3" %in% h$states) h$times[h$states == "S3"][1] else NA
    fu <- if (is.na(death)) cfg$t_max else min(cfg$t_max, death)
    visits <- generate_visits(E[i], cfg$obs, fu); visit_counts[i] <- length(visits)
    ev <- inject_artifacts(observe_trajectory(h, E[i], visits), cfg$obs, visits)
    cases <- c(cases, rep(i, nrow(ev))); acts <- c(acts, ev$activity)
    tms <- c(tms, ev$time); grps <- c(grps, rep(if (E[i] == 1) "PPI" else "H2B", nrow(ev)))
  }
  log <- data.frame(case = as.integer(cases), activity = acts, time = tms,
                    group = grps, stringsAsFactors = FALSE)
  log <- log[order(log$case, log$time, ACTIVITY_ORDER[log$activity]), ]
  list(log = log, L = L, L_std = L_std, E = E, visit_counts = visit_counts, histories = histories)
}

# ---------------------------------------------------------------------------
# 6. Discovery: state-level directly-follows graph + monitoring evidence
# ---------------------------------------------------------------------------
ACTIVITY_TO_NODE <- c(IndexDrug = "S0", CKD_recorded = "S1", CVAE_recorded = "S2",
                      Death = "S3", MedicationChange = "S0", CKD_recode = "S1")
ADMIN_ACTIVITIES <- c("MedicationChange")

discover <- function(log, visit_counts) {
  log <- log[order(log$case, log$time, ACTIVITY_ORDER[log$activity]), ]
  mon_median <- median(visit_counts)
  high_mon <- which(visit_counts > mon_median)
  by_case <- split(log, log$case)
  origin_cases <- list(); pair_cases <- list(); pair_delays <- list(); pair_src <- list()
  for (cn in names(by_case)) {
    sub <- by_case[[cn]]; case <- as.integer(cn)
    nodes <- unique(ACTIVITY_TO_NODE[sub$activity])
    for (nd in nodes) origin_cases[[nd]] <- c(origin_cases[[nd]], case)
    acts <- sub$activity; tms <- sub$time
    if (length(acts) > 1) for (j in seq_len(length(acts) - 1)) {
      a <- ACTIVITY_TO_NODE[acts[j]]; b <- ACTIVITY_TO_NODE[acts[j + 1]]
      if (a == b) next
      key <- paste0(a, "->", b)
      pair_cases[[key]] <- union(pair_cases[[key]], case)
      pair_delays[[key]] <- c(pair_delays[[key]], tms[j + 1] - tms[j])
      pair_src[[key]] <- union(pair_src[[key]], acts[j + 1])
    }
  }
  edges <- list()
  for (key in names(pair_cases)) {
    a <- sub("->.*", "", key); cases <- pair_cases[[key]]
    oc <- origin_cases[[a]]; denom <- length(oc)
    hi <- intersect(oc, high_mon); lo <- setdiff(oc, high_mon)
    rh <- if (length(hi)) length(intersect(cases, high_mon)) / length(hi) else 0
    rl <- if (length(lo)) length(setdiff(cases, high_mon)) / length(lo) else 0
    edges[[key]] <- list(from = a, to = sub(".*->", "", key),
                         risk = length(cases) / denom, count = length(cases),
                         median_time = median(pair_delays[[key]]),
                         src = pair_src[[key]],
                         attenuation = if (rh > 0) (rh - rl) / rh else 0)
  }
  edges
}

# ---------------------------------------------------------------------------
# 7. Refinement: proposer + gates; arms A / B / C
# ---------------------------------------------------------------------------
INITIAL_NODES <- c("S0", "S1", "S2")
INITIAL_EDGES <- c("S0->S1", "S0->S2", "S1->S2")
S_MIN <- 0.10

none_gate <- function(cand) list(accept = cand$risk >= S_MIN, why = "ungoverned: support frequency")

rules_gate <- function(cand) {
  if (STATE_ORDER[cand$from] > STATE_ORDER[cand$to])
    return(list(accept = FALSE, why = "protected-direction (backward) violation"))
  if (length(intersect(cand$src, ADMIN_ACTIVITIES)) > 0)
    return(list(accept = FALSE, why = "administrative activity, not a causal edge"))
  list(accept = TRUE, why = "forward, plausible transition")
}

run_refinement <- function(gate, disc) {
  nodes <- INITIAL_NODES; edges <- INITIAL_EDGES; decisions <- list()
  for (key in names(disc)) {
    e <- disc[[key]]
    if (e$risk < S_MIN || key %in% edges) next
    cand <- list(from = e$from, to = e$to, risk = e$risk, src = e$src)
    d <- gate(cand); decisions[[key]] <- d
    if (d$accept) { nodes <- union(nodes, c(e$from, e$to)); edges <- union(edges, key) }
  }
  list(nodes = nodes, edges = edges, decisions = decisions)
}

oracle_refinement <- function() list(nodes = c("S0","S1","S2","S3"), edges = TRUE_KEYS, decisions = list())

run_arm <- function(arm, disc) {
  if (arm == "A") run_refinement(none_gate, disc)
  else if (arm == "B") run_refinement(rules_gate, disc)
  else oracle_refinement()
}

# ---------------------------------------------------------------------------
# 8. Metrics
# ---------------------------------------------------------------------------
structural_metrics <- function(res) {
  rec <- res$edges
  tp <- length(intersect(rec, TRUE_KEYS)); fp <- length(setdiff(rec, TRUE_KEYS))
  fn <- length(setdiff(TRUE_KEYS, rec))
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall <- if (tp + fn > 0) tp / (tp + fn) else 0
  f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  traps <- intersect(rec, TRAP_KEYS)
  list(precision = precision, recall = recall, f1 = f1, shd = fp + fn,
       death_node = "S3" %in% res$nodes, n_traps = length(traps),
       accepts_any_trap = length(traps) >= 1,
       accepted_S2_S0 = "S2->S0" %in% rec, accepted_S2_S1 = "S2->S1" %in% rec)
}

# ---------------------------------------------------------------------------
# 9. Estimation: cause-specific Cox (survival::coxph)
# ---------------------------------------------------------------------------
.zscore <- function(x) { s <- sd(x); if (s == 0) s <- 1; (x - mean(x)) / s }

.fit_betaE <- function(df, covs) {
  f <- as.formula(paste("Surv(duration, event) ~ E +", paste(covs, collapse = " + ")))
  fit <- tryCatch(coxph(f, data = df), error = function(e) NULL)
  if (is.null(fit)) return(list(beta = NA, lo = NA, hi = NA))
  ci <- suppressMessages(confint(fit))
  list(beta = unname(coef(fit)["E"]), lo = ci["E", 1], hi = ci["E", 2])
}

.first_time <- function(log, activity) {
  s <- log[log$activity == activity, ]
  tapply(s$time, s$case, min)
}

estimate <- function(data, t_max) {
  n <- length(data$E); log <- data$log
  ckd <- .first_time(log, "CKD_recorded"); cvae <- .first_time(log, "CVAE_recorded")
  death <- .first_time(log, "Death")
  covdf <- as.data.frame(apply(data$L, 2, .zscore)); names(covdf) <- COVARIATE_NAMES

  # S0->S1 recorded
  dur <- numeric(n); ev <- integer(n)
  for (i in seq_len(n)) {
    ci <- ckd[as.character(i)]
    if (!is.na(ci)) { dur[i] <- ci; ev[i] <- 1L } else {
      di <- death[as.character(i)]; dur[i] <- if (!is.na(di)) di else t_max
    }
  }
  rec <- cbind(data.frame(duration = pmax(dur, 1e-6), event = ev, E = data$E), covdf)

  # S0->S1 latent (true onset)
  durL <- numeric(n); evL <- integer(n)
  for (i in seq_len(n)) {
    st <- data$histories[[i]]$states; tt <- data$histories[[i]]$times
    if ("S1" %in% st) { durL[i] <- tt[st == "S1"][1]; evL[i] <- 1L }
    else durL[i] <- if ("S3" %in% st) tt[st == "S3"][1] else t_max
  }
  lat <- cbind(data.frame(duration = pmax(durL, 1e-6), event = evL, E = data$E), covdf)

  # visit-adjusted (recorded + visit count)
  adj <- rec; adj$visits <- .zscore(as.numeric(data$visit_counts))

  # S1->S2 among CKD patients (clock reset)
  idx <- as.integer(names(ckd))
  rows <- lapply(idx, function(i) {
    ct <- ckd[as.character(i)]; vt <- cvae[as.character(i)]; dt <- death[as.character(i)]
    if (!is.na(vt) && vt > ct) c(vt - ct, 1) else { end <- if (!is.na(dt)) dt else t_max; c(max(end - ct, 1e-6), 0) }
  })
  s12 <- data.frame(duration = sapply(rows, `[`, 1), event = sapply(rows, `[`, 2), E = data$E[idx])
  for (cn in COVARIATE_NAMES) s12[[cn]] <- .zscore(data$L[idx, cn])

  list(s0s1_latent = .fit_betaE(lat, COVARIATE_NAMES),
       s0s1_naive = .fit_betaE(rec, COVARIATE_NAMES),
       s0s1_adjusted = .fit_betaE(adj, c(COVARIATE_NAMES, "visits")),
       s1s2 = .fit_betaE(s12, COVARIATE_NAMES))
}

# ---------------------------------------------------------------------------
# 10. Experiment: run replications of a cell
# ---------------------------------------------------------------------------
run_replication <- function(cfg, seed, with_estimation = TRUE) {
  data <- build_event_log(cfg$n, cfg, seed)
  disc <- discover(data$log, data$visit_counts)
  structural <- lapply(c("A","B","C"), function(a) structural_metrics(run_arm(a, disc)))
  names(structural) <- c("A","B","C")
  out <- list(seed = seed, structural = structural)
  if (with_estimation) out$estimation <- estimate(data, cfg$t_max)
  out
}

run_cell <- function(cfg, M, master_seed = 20260610L, with_estimation = TRUE) {
  set.seed(master_seed); seeds <- sample.int(.Machine$integer.max, M)
  reps <- lapply(seeds, function(s) run_replication(cfg, s, with_estimation))
  agg <- list(M = M, n = cfg$n, structural = list())
  for (a in c("A","B","C")) {
    g <- function(f) sapply(reps, function(r) f(r$structural[[a]]))
    k <- sum(g(function(m) m$accepts_any_trap))
    agg$structural[[a]] <- list(
      precision = mean(g(function(m) m$precision)), recall = mean(g(function(m) m$recall)),
      f1 = mean(g(function(m) m$f1)), shd = mean(g(function(m) m$shd)),
      n_traps = mean(g(function(m) m$n_traps)),
      accept_any_trap_rate = k / M, death_node_rate = mean(g(function(m) m$death_node)),
      reject_S2_S0 = mean(g(function(m) !m$accepted_S2_S0)),
      reject_S2_S1 = mean(g(function(m) !m$accepted_S2_S1)))
  }
  if (with_estimation) {
    t01 <- true_beta_E(cfg, "S0->S1"); t12 <- true_beta_E(cfg, "S1->S2")
    est <- list(truth_S0S1 = t01, truth_S1S2 = t12)
    for (kt in list(c("s0s1_latent", t01), c("s0s1_naive", t01), c("s0s1_adjusted", t01), c("s1s2", t12))) {
      key <- kt[[1]]; truth <- as.numeric(kt[[2]])
      betas <- sapply(reps, function(r) r$estimation[[key]]$beta)
      cover <- mean(mapply(function(r) {
        e <- r$estimation[[key]]; !is.na(e$lo) && e$lo <= truth && truth <= e$hi }, reps))
      est[[key]] <- list(mean_beta = mean(betas, na.rm = TRUE),
                         bias = mean(betas, na.rm = TRUE) - truth,
                         rmse = sqrt(mean((betas - truth)^2, na.rm = TRUE)), coverage = cover)
    }
    agg$estimation <- est
  }
  agg
}

# ---------------------------------------------------------------------------
# 11. Visualization (base R) — process map + DAG refinement
# ---------------------------------------------------------------------------
.POS <- list(S0 = c(0, 0), S1 = c(1.6, 1.05), S2 = c(3.2, 0), S3 = c(4.8, 1.05))
.LAB <- c(S0 = "Drug\n(index)", S1 = "CKD", S2 = "CVAE", S3 = "Death")
.NCOL <- c(S0 = "#3498db", S1 = "#9b59b6", S2 = "#1abc9c", S3 = "#e74c3c")
.RAD <- c("S0->S3" = 0.30, "S2->S0" = -0.38, "S2->S1" = -0.32)

.draw_node <- function(s) {
  p <- .POS[[s]]
  symbols(p[1], p[2], circles = 0.34, inches = FALSE, bg = .NCOL[s], fg = "black", add = TRUE)
  text(p[1], p[2], .LAB[s], col = "white", font = 2, cex = 0.8)
}

.draw_edge <- function(a, b, col, lty = 1, lwd = 2, label = NULL) {
  from <- .POS[[a]]; to <- .POS[[b]]; rad <- .RAD[paste0(a, "->", b)]
  if (is.na(rad)) rad <- 0
  d <- to - from; L <- sqrt(sum(d^2)); u <- d / L; perp <- c(-u[2], u[1])
  p0 <- from + u * 0.36; p2 <- to - u * 0.36
  ctrl <- (p0 + p2) / 2 + perp * rad * L
  tt <- seq(0, 1, length.out = 40)
  bez <- sapply(tt, function(t) (1 - t)^2 * p0 + 2 * (1 - t) * t * ctrl + t^2 * p2)
  lines(bez[1, ], bez[2, ], col = col, lty = lty, lwd = lwd)
  arrows(bez[1, 38], bez[2, 38], bez[1, 40], bez[2, 40], col = col, length = 0.12, lwd = lwd)
  if (!is.null(label)) {
    m <- bez[, 20]
    text(m[1] + 0.12 * sign(rad + 0.01) * perp[1], m[2] + 0.12 * sign(rad + 0.01) * perp[2],
         label, cex = 0.68, col = col)
  }
}

.canvas <- function(title) {
  plot(NA, xlim = c(-0.9, 5.7), ylim = c(-0.9, 1.9), axes = FALSE, xlab = "", ylab = "", main = title, asp = 1)
}

process_map <- function(disc, file = "../results_R/process_map.png") {
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  png(file, width = 1100, height = 650, res = 110)
  .canvas("Discovered process map (state-level DFG)")
  for (key in names(disc)) {
    e <- disc[[key]]; is_trap <- key %in% TRAP_KEYS
    col <- if (is_trap) "#c0392b" else "#34495e"
    .draw_edge(e$from, e$to, col, lty = if (is_trap) 2 else 1,
               lwd = 1.2 + 3.5 * e$risk, label = sprintf("%d%%", round(100 * e$risk)))
  }
  for (s in names(.POS)) .draw_node(s)
  legend("bottom", legend = c("recorded transition", "injected artefact (trap)"),
         col = c("#34495e", "#c0392b"), lty = c(1, 2), lwd = 2, horiz = TRUE, bty = "n")
  dev.off(); message("wrote ", file)
}

dag_refinement <- function(refined, file = "../results_R/dag_refinement.png") {
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  png(file, width = 1500, height = 540, res = 110); par(mfrow = c(1, 3))
  panel <- function(edges, title) {
    .canvas(title)
    nodes <- unique(unlist(strsplit(edges, "->")))
    for (key in edges) {
      ab <- strsplit(key, "->")[[1]]; is_trap <- key %in% TRAP_KEYS
      .draw_edge(ab[1], ab[2], if (is_trap) "#c0392b" else "#27ae60", lty = if (is_trap) 2 else 1, lwd = 2)
    }
    for (s in nodes) .draw_node(s)
  }
  panel(TRUE_KEYS, "True DAG (oracle)")
  panel(refined$A, sprintf("Arm A — ungated: accepts %d trap(s)", length(intersect(refined$A, TRAP_KEYS))))
  panel(refined$B, "Arm B — governed: rejects both traps")
  dev.off(); message("wrote ", file)
}
