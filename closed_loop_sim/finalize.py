"""Fill the manuscript-section template from results JSON and render a figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "results" / "base_cell_results.json").read_text())


def pct(x):
    return f"{100 * x:.0f}%"


def ci(pair):
    return f"{pair[0]:.2f}–{pair[1]:.2f}"


def fill() -> dict:
    m = RES["meta"]
    cell = RES["cell"]
    s = cell["structural"]
    e = cell["estimation"]
    cal = RES["observed_calibration"]
    med = RES["mediation_oracle"]
    edges = RES["representative_state_edges"]

    def block(arm):
        x = s[arm]
        return {
            f"{arm}_prec": f"{x['precision_mean']:.2f}", f"{arm}_rec": f"{x['recall_mean']:.2f}",
            f"{arm}_f1": f"{x['f1_mean']:.2f}", f"{arm}_shd": f"{x['shd_mean']:.2f}",
            f"{arm}_traps": f"{x['n_traps_accepted_mean']:.2f}",
            f"{arm}_death": f"{x['death_node_rate']:.2f}",
            f"{arm}_anytrap": pct(x['accept_any_trap_rate']),
            f"{arm}_anytrap_ci": ci(x['accept_any_trap_ci']),
            f"{arm}_anytrap_pct": pct(x['accept_any_trap_rate']),
        }

    def est(key, prefix):
        x = e[key]
        return {f"{prefix}_mean": f"{x['mean_beta']:.3f}", f"{prefix}_bias": f"{x['bias']:+.3f}",
                f"{prefix}_rmse": f"{x['rmse']:.3f}", f"{prefix}_cov": f"{x['coverage']:.2f}"}

    vals = {
        "M": str(m["M"]), "py": m["python"], "np": m["numpy"], "pd": m["pandas"],
        "pm4py": m["pm4py"], "lifelines": m["lifelines"],
        "cal_gap": f"{cal['worst_gap']:.3f}",
        "cs_ppi": f"{cal['coding_shift']['PPI']:.3f}", "cs_h2b": f"{cal['coding_shift']['H2B']:.3f}",
        "decoy_risk": f"{edges['S2->S0']['risk']:.2f}", "recode_risk": f"{edges['S2->S1']['risk']:.2f}",
        "B_reject_pct": pct(1 - s["B"]["accept_any_trap_rate"]),
        "med_prop": f"{med['mediated_proportion']:.2f}", "med_ind": f"{med['indirect']:.3f}",
        "med_tot": f"{med['total']:.3f}",
        "lat_p2": f"{e['s0s1_latent']['p2_5']:.3f}", "lat_p97": f"{e['s0s1_latent']['p97_5']:.3f}",
    }
    for arm in ("A", "B", "C"):
        vals.update(block(arm))
    vals.update(est("s0s1_latent", "lat"))
    vals.update(est("s0s1_naive", "naive"))
    vals.update(est("s0s1_adjusted", "adj"))
    vals.update(est("s1s2", "s12"))
    return vals


def render_doc(vals: dict) -> None:
    src = (ROOT / "docs" / "manuscript_simulation_sections.md").read_text()
    for k, v in vals.items():
        src = src.replace("{{" + k + "}}", str(v))
    out = ROOT / "docs" / "manuscript_simulation_sections_FINAL.md"
    out.write_text(src)
    leftover = [tok for tok in vals] and [s for s in src.split("{{")[1:]]
    print("wrote", out, "| unfilled placeholders:", len(src.split("{{")) - 1)


def render_figure() -> None:
    s = RES["cell"]["structural"]
    e = RES["cell"]["estimation"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))

    arms = ["A\n(ungated)", "B\n(governed)", "C\n(oracle)"]
    rates = [s[a]["accept_any_trap_rate"] for a in ("A", "B", "C")]
    bars = ax[0].bar(arms, rates, color=["#c0392b", "#27ae60", "#7f8c8d"])
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("P(accepts ≥1 injected artefact)")
    ax[0].set_title("Artefact resistance")
    for b, r in zip(bars, rates):
        ax[0].text(b.get_x() + b.get_width() / 2, r + 0.02, pct(r), ha="center", fontsize=10)

    keys = ["s0s1_latent", "s0s1_naive", "s0s1_adjusted"]
    labels = ["latent\n(clean)", "recorded\n(naive)", "visit-count\nadjusted\n(not rec.)"]
    means = np.array([e[k]["mean_beta"] for k in keys])
    lo = np.array([means[i] - e[k]["p2_5"] for i, k in enumerate(keys)])
    hi = np.array([e[k]["p97_5"] - means[i] for i, k in enumerate(keys)])
    x = np.arange(len(keys))
    ax[1].axhline(0.25, color="k", ls="--", lw=1, label="true β = 0.25")
    ax[1].errorbar(x, means, yerr=[lo, hi], fmt="o", ms=8, capsize=5, lw=1.5,
                   ecolor="#7f8c8d", zorder=3,
                   mfc="none", mec="k", mew=0)  # markers coloured below
    ax[1].scatter(x, means, color=["#27ae60", "#e67e22", "#c0392b"], s=80, zorder=4)
    for i in range(len(keys)):
        ax[1].annotate(f"{means[i]:.2f}", (x[i], means[i]), textcoords="offset points",
                       xytext=(10, 0), fontsize=10)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("estimated exposure log-HR on CKD onset")
    ax[1].set_title("Estimand recovery (S0→S1); bars = 95% MC interval")
    ax[1].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = ROOT / "results" / "figure_simulation.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    vals = fill()
    render_doc(vals)
    render_figure()
    print("\nkey numbers:")
    for k in ("cal_gap", "A_anytrap", "B_anytrap", "lat_cov", "naive_cov", "med_prop"):
        print(f"  {k} = {vals[k]}")
