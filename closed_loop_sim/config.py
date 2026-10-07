"""Configuration models and base-cell defaults.

Review fixes folded in: pydantic ``default_factory`` for nested defaults and
field validation constraints (probabilities in [0,1], rates > 0, kappa >= 1).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

COVARIATE_NAMES = ["age", "sex", "diabetes", "hypertension", "egfr"]


class CovariateConfig(BaseModel):
    age_mean: float = 70.0
    age_sd: float = Field(10.0, gt=0)
    sex_female_prob: float = Field(0.5, ge=0, le=1)
    diabetes_prob: float = Field(0.25, ge=0, le=1)
    hypertension_prob: float = Field(0.55, ge=0, le=1)
    egfr_mean: float = 75.0
    egfr_sd: float = Field(15.0, gt=0)


class PropensityConfig(BaseModel):
    """logit P(E=1|L) = alpha0 + alpha . L_standardized.

    alpha0 sets the marginal PPI prevalence; the per-covariate alpha create the
    *confounding* imbalance the propensity-score recommendation must correct.
    The study targets that confounding structure and the transition-risk pattern,
    NOT the manuscript's exact 92.8% / 7.2% marginal split (a tunable, not central
    to the recovery claims); alpha0=0.8 gives ~69% PPI for adequate H2B-arm power.
    """
    alpha0: float = 0.8
    alpha: list[float] = Field(default_factory=lambda: [0.4, 0.0, 0.5, 0.3, -0.4])


class TransitionParams(BaseModel):
    lam: float = Field(..., gt=0)   # Weibull scale rate lambda_jk
    rho: float = Field(..., gt=0)   # Weibull shape rho_jk
    beta_E: float                   # exposure log-HR
    beta_L: list[float] = Field(default_factory=lambda: [0.0] * len(COVARIATE_NAMES))


class DGPConfig(BaseModel):
    t_max: float = Field(10.0, gt=0)   # study horizon, years
    covariates: CovariateConfig = Field(default_factory=CovariateConfig)
    propensity: PropensityConfig = Field(default_factory=PropensityConfig)
    transitions: dict[str, TransitionParams]
    challenge_scenario: str | None = None  # Action A10: 'feedback_cvae_ckd', 'latent_common_cause', etc.


# Recorded-timestamp granularity: step in years (None = exact, no rounding).
GRANULARITY_STEP = {"exact": None, "daily": 1.0 / 365.25, "monthly": 1.0 / 12.0, "yearly": 1.0}


class ObservationConfig(BaseModel):
    lambda_base: float = Field(1.0, gt=0)   # H2B visits/year
    kappa: float = Field(2.0, ge=1)         # PPI monitoring multiplier
    p_switch: float = Field(0.4, ge=0, le=1)  # decoy: MedicationChange after CVAE
    p_recode: float = Field(0.3, ge=0, le=1)  # coding shift: CKD_recode after CVAE
    # Recorded-timestamp granularity. "exact" keeps the reproducible deterministic
    # ordering; coarser levels round timestamps and break same-bin ties RANDOMLY,
    # modelling the analyst's loss of true temporal precedence (Table 5).
    time_granularity: str = Field("exact", pattern="^(exact|daily|monthly|yearly)$")
    challenge_scenario: str | None = None  # Action A10: 'outcome_dependent_visits', 'held_out_artifact'
    p_held_out: float = Field(0.0, ge=0, le=1)  # held-out artifact rate (Lab_Reassay)



class SimConfig(BaseModel):
    seed: int = 20260610
    n: int = Field(5000, gt=0)
    dgp: DGPConfig
    obs: ObservationConfig = Field(default_factory=ObservationConfig)


def base_cell() -> SimConfig:
    """The minimal cell: one confounder set, n=5000, both traps on, base kappa.

    True beta_E follows spec section 2.3 (S1->S2 is exactly zero: mediation operates
    through CKD *occurrence*, not by modifying the CKD->CVAE rate). lam/rho are tuned
    against the OBSERVED event-log calibration gate, not latent rates.
    """
    transitions = {
        # lam tuned against the OBSERVED event-log Table-4 gate (worst gap ~0.05).
        "S0->S1": TransitionParams(lam=0.0140, rho=1.20, beta_E=0.25,
                                   beta_L=[0.30, 0.0, 0.35, 0.20, -0.45]),
        "S0->S2": TransitionParams(lam=0.0038, rho=1.10, beta_E=0.20,
                                   beta_L=[0.35, 0.0, 0.30, 0.30, -0.10]),
        "S0->S3": TransitionParams(lam=0.0032, rho=1.05, beta_E=0.05,
                                   beta_L=[0.55, 0.0, 0.20, 0.15, -0.15]),
        "S1->S2": TransitionParams(lam=0.0285, rho=1.15, beta_E=0.0,
                                   beta_L=[0.25, 0.0, 0.30, 0.30, -0.20]),
        "S1->S3": TransitionParams(lam=0.0207, rho=1.10, beta_E=0.10,
                                   beta_L=[0.50, 0.0, 0.25, 0.15, -0.25]),
        "S2->S3": TransitionParams(lam=0.1304, rho=0.95, beta_E=0.10,
                                   beta_L=[0.40, 0.0, 0.20, 0.10, -0.10]),
    }
    return SimConfig(dgp=DGPConfig(transitions=transitions))
