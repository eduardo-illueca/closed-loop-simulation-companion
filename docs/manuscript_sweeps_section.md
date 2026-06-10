# New manuscript text — Robustness sweeps (extends the simulation study)

> Adds to the simulation Methods and Results (after the base-cell sections). Numbers are
> from one-factor-at-a-time sweeps from the base cell; see `results/sweeps.csv` and Fig. S4
> (`results/figure_sweeps.png`).

---

## Methods — Robustness sweeps

To probe where the framework degrades, we swept five factors one at a time from the base
cell, running M = 8–80 replications per level (more at smaller sample sizes): **sample size**
n ∈ {1k, 5k, 20k, 100k}; **monitoring differential** κ ∈ {1, 1.5, 2, 3}; **timestamp
granularity** ∈ {exact, daily, monthly, yearly}; **confounding strength** (the propensity
coefficients scaled ×0.5 / ×1 / ×1.5); and **artefact rates** (p_switch, p_recode) ∈
{off, low, high}. Structural recovery and artefact rejection were recorded in every cell; the
sample-size and monitoring sweeps additionally fit the cause-specific Cox estimand. The
timestamp-granularity factor rounds recorded event times to the stated grid and breaks
same-bin ties at random, modelling the analyst's loss of true temporal precedence (Table 5).

## Results — Robustness sweeps

The governed loop's artefact rejection was **robust across every condition**: Arm B rejected
both injected artefacts in 100% of replications in all 18 cells, and added the Death node in
every replication; whenever artefacts were present, Arm A accepted at least one in 100% of
replications. The sweeps localise *where* the observation process bites (Fig. S4):

| Factor | Finding |
|--------|---------|
| **Sample size** | Structural recovery is sample-size–invariant (Arm B recall 0.83 throughout, because the 10% support threshold is on transition *risk*, not count); what improves is **estimand precision** — latent-process 95% CI coverage 0.95 → 1.00 and RMSE 0.169 → 0.012 as n grows from 1k to 100k. |
| **Monitoring differential κ** | The recorded estimand's **surveillance bias grows monotonically with monitoring**: the naive cause-specific log-HR bias rises from +0.01 (κ = 1) to +0.11 (κ = 3), while the latent-process estimate stays unbiased (≈ 0). Discovered artefact frequencies also rise with κ (decoy 0.30 → 0.35). |
| **Timestamp granularity** | The fraction of order-ambiguous (tied-timestamp) transitions rises from 0.03 (exact) to 0.12 (yearly), and coarsening lowers the *discovered* decoy frequency (0.33 → 0.19) as event orderings are lost — yet the governed gate stays robust (Arm B precision 1.00) because its rules key on node identity, not the recorded timestamp. |
| **Confounding strength** | Structural recovery is unaffected (Arm A precision 0.71, Arm B precision 1.00 at low/moderate/strong confounding): the DAG-identified adjustment set absorbs the confounding. |
| **Artefact rates** | The Arm A vs Arm B contrast exists only when artefacts are present — with artefacts off both arms are clean (Arm A precision 1.00), and the contrast magnitude scales with the artefact rate (Arm A precision 1.00 → 0.79 → 0.72 as rates go off → low → high). |

Two points are worth stating plainly. First, the sweeps substantiate the relevant **Table 5**
hazards with evidence: surveillance bias (κ), temporal-granularity limits, post-event coding
shifts and competing risks are all exercised directly, and the framework's behaviour under
each is quantified rather than asserted. Second, the framework is **not** uniformly robust:
the recorded *estimand* degrades with monitoring intensity and small samples, which is exactly
why the framework pairs structural refinement with explicit identification and sensitivity
analysis rather than treating the recovered DAG as self-sufficient.

**Fig. S4 — Robustness sweeps** (`results/figure_sweeps.png`): (a) estimand coverage and RMSE
vs sample size; (b) discovered artefact risks and per-arm acceptance vs κ; (c) ambiguous-
transition rate vs timestamp granularity; (d) per-arm precision and Arm A artefact acceptance
vs artefact rate; (e) precision vs confounding strength; (f) recorded (naive) vs latent
estimand bias vs κ.
