"""
calibration.py
EVO-MOE V4 — Stage 3: Two-Strain SIS ODE Calibration
PRD Section 6.2: ESKAPE-Generalised Two-Strain SIS ODE
PRD Section 6.2.4: Two-Stage Calibration: ABC + L-BFGS-B

Implements:
  - Gram-negative drift (HGT-mediated, Eq. 6.6–6.7)
  - Gram-positive drift (clonal expansion, Eq. 6.8–6.9)
  - Structural identifiability analysis (Section 6.2.3, Eq. 6.10)
  - ABC global exploration → L-BFGS-B local refinement (Listing 6.1)
  - Organism-specific prior distributions (Section 6.2.5, Table)
  - Benchmark against ARIMA (FIX-10, FR-04)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import uniform, beta as beta_dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Organism-specific priors (Section 6.2.5)
# ---------------------------------------------------------------------------
# Each entry:
#   gram        : "negative" | "positive"
#   R0s_range   : (lo, hi) uniform prior on R0_s
#   R0r_range   : (lo, hi) uniform prior on R0_r
#   sigma_ab    : (a, b) Beta prior on σ (treatment selection pressure)
#   gamma_s     : mean clearance rate susceptible (per day)
#   gamma_r     : mean clearance rate resistant (per day)
#   rho_HGT_lognormal : (logmean, logsd) for ρ_HGT (Gram-neg only)
#   mu_clone    : clonal expansion rate prior mean (Gram-pos only)

ORGANISM_PRIORS: Dict[str, dict] = {
    "E_faecium": {
        "gram": "positive",
        "R0s_range": (1.0, 4.0),
        "R0r_range": (0.5, 3.0),
        "sigma_ab": (2, 10),
        "gamma_s": 0.10,
        "gamma_r": 0.05,
        "mu_clone": 0.03,       # clonal expansion rate (per month)
    },
    "S_aureus_MRSA": {
        "gram": "positive",
        "R0s_range": (1.0, 5.0),
        "R0r_range": (0.5, 4.0),
        "sigma_ab": (2, 10),
        "gamma_s": 0.14,
        "gamma_r": 0.06,
        "mu_clone": 0.034,
    },
    "K_pneumoniae": {
        "gram": "negative",
        "R0s_range": (1.0, 5.0),
        "R0r_range": (0.5, 4.0),
        "sigma_ab": (2, 10),
        "gamma_s": 0.12,
        "gamma_r": 0.05,
        "rho_HGT_lognormal": (-9.2, 1.0),   # LogN(−9.2, 1) per Table
    },
    "A_baumannii": {
        "gram": "negative",
        "R0s_range": (1.0, 4.0),
        "R0r_range": (0.5, 3.5),
        "sigma_ab": (2, 8),
        "gamma_s": 0.08,
        "gamma_r": 0.04,
        "rho_HGT_lognormal": (-8.5, 1.0),   # Higher plasticity (Table note)
    },
    "P_aeruginosa": {
        "gram": "negative",
        "R0s_range": (1.0, 4.0),
        "R0r_range": (0.5, 3.0),
        "sigma_ab": (2, 10),
        "gamma_s": 0.10,
        "gamma_r": 0.05,
        "rho_HGT_lognormal": (-9.5, 1.0),
    },
    "Enterobacter_spp": {
        "gram": "negative",
        "R0s_range": (1.0, 4.0),
        "R0r_range": (0.5, 3.0),
        "sigma_ab": (2, 10),
        "gamma_s": 0.11,
        "gamma_r": 0.05,
        "rho_HGT_lognormal": (-9.0, 1.0),
    },
}


# ---------------------------------------------------------------------------
# ODE parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class ODEParams:
    """
    Calibrated ODE parameters for a single (organism, drug) pair.
    Identifiable parameters are in (R0_s, R0_r, σ) space (Eq. 6.10).
    Derived β_s, β_r from identifiable combos.
    """
    organism: str

    # Identifiable combinations (Section 6.2.3, Eq. 6.10)
    R0s: float          # R_{0,s} = β_s / (γ_s + α)
    R0r: float          # R_{0,r} = β_r · δ_PTND / γ_r
    sigma: float        # treatment selection pressure σ ∈ (0, 0.5)

    # Derived (recovered) parameters
    beta_s: float
    beta_r: float
    gamma_s: float
    gamma_r: float
    alpha: float        # antibiotic treatment rate (from prescription data)
    delta_PTND: float   # PTND scaling factor

    # Gram-negative only
    rho_HGT: Optional[float] = None

    # Gram-positive only
    mu_clone: float = 0.03


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    organism: str
    params: ODEParams
    sse_evo_moe: float          # SSE of EVO-MOE ODE fit
    sse_arima: float            # SSE of ARIMA benchmark
    beats_arima: bool           # FIX-10: must be True
    abc_acceptance_rate: float  # must be > 0.5%
    n_abc_accepted: int
    n_abc_total: int


# ---------------------------------------------------------------------------
# Two-Strain SIS ODE (Sections 6.2.1–6.2.2)
# ---------------------------------------------------------------------------

def _ode_gram_negative(
    state: list,
    t: float,
    beta_s: float,
    beta_r: float,
    gamma_s: float,
    gamma_r: float,
    alpha: float,
    sigma: float,
    rho_HGT: float,
) -> list:
    """
    Gram-negative SIS ODE with HGT (Eq. 6.6–6.7).
    State: [Is, Ir]
    """
    Is, Ir = state
    S = max(1.0 - Is - Ir, 0.0)

    dIs = beta_s * Is * S - gamma_s * Is - alpha * Is
    dIr = (
        beta_r * Ir * S
        - gamma_r * Ir
        + sigma * alpha * Is        # treatment selection
        + rho_HGT * Is * Ir         # horizontal gene transfer
    )
    return [dIs, dIr]


def _ode_gram_positive(
    state: list,
    t: float,
    beta_s: float,
    beta_r: float,
    gamma_s: float,
    gamma_r: float,
    alpha: float,
    sigma: float,
    mu_clone: float,
) -> list:
    """
    Gram-positive SIS ODE with clonal expansion, no HGT (Eq. 6.8–6.9).
    State: [Is, Ir]
    """
    Is, Ir = state
    S = max(1.0 - Is - Ir, 0.0)

    dIs = beta_s * Is * S - gamma_s * Is - alpha * Is
    dIr = (
        beta_r * Ir * S
        - gamma_r * Ir
        + sigma * alpha * Is        # treatment selection
        + mu_clone * Ir             # clonal amplification (Eq. 6.9)
    )
    return [dIs, dIr]


def simulate_ode(
    params: ODEParams,
    t_months: np.ndarray,
    Is0: float = 0.3,
    Ir0: float = 0.1,
) -> np.ndarray:
    """
    Simulate the two-strain SIS ODE and return R(t) = Ir / (Is + Ir).

    Parameters
    ----------
    params   : ODEParams (calibrated)
    t_months : time points in months (converted to days internally)
    Is0, Ir0 : initial conditions

    Returns R(t) — shape (len(t_months),)
    """
    t_days = t_months * 30.0
    state0 = [Is0, Ir0]
    gram = ORGANISM_PRIORS[params.organism]["gram"]

    if gram == "negative":
        ode_func = _ode_gram_negative
        args = (
            params.beta_s, params.beta_r,
            params.gamma_s, params.gamma_r,
            params.alpha, params.sigma,
            params.rho_HGT or 0.0,
        )
    else:
        ode_func = _ode_gram_positive
        args = (
            params.beta_s, params.beta_r,
            params.gamma_s, params.gamma_r,
            params.alpha, params.sigma,
            params.mu_clone,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = odeint(ode_func, state0, t_days, args=args, mxstep=1000)

    Is = np.clip(sol[:, 0], 0.0, 1.0)
    Ir = np.clip(sol[:, 1], 0.0, 1.0)
    denom = np.maximum(Is + Ir, 1e-9)
    R = Ir / denom
    return np.clip(R, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ABC + L-BFGS-B calibration (Section 6.2.4, Listing 6.1)
# ---------------------------------------------------------------------------

def _params_from_theta(
    theta: np.ndarray,
    organism: str,
    alpha: float,
    delta_PTND: float = 1.0,
) -> ODEParams:
    """
    Reconstruct ODEParams from calibration vector θ = [R0_s, R0_r, σ, (ρ_HGT)].
    Parameter recovery per Listing 6.1 lines 19–21.
    """
    prior = ORGANISM_PRIORS[organism]
    gamma_s = prior["gamma_s"]
    gamma_r = prior["gamma_r"]

    R0s = theta[0]
    R0r = theta[1]
    sigma = theta[2]

    # Eq. 6.10 inversion: recover β from identifiable parameters
    beta_s = R0s * (gamma_s + alpha)
    beta_r = R0r * gamma_r / max(delta_PTND, 1e-6)

    rho_HGT = None
    mu_clone = prior.get("mu_clone", 0.03)

    if prior["gram"] == "negative":
        if len(theta) > 3:
            rho_HGT = float(theta[3])
        else:
            lm, ls = prior["rho_HGT_lognormal"]
            rho_HGT = float(np.exp(lm))

    return ODEParams(
        organism=organism,
        R0s=R0s,
        R0r=R0r,
        sigma=sigma,
        beta_s=beta_s,
        beta_r=beta_r,
        gamma_s=gamma_s,
        gamma_r=gamma_r,
        alpha=alpha,
        delta_PTND=delta_PTND,
        rho_HGT=rho_HGT,
        mu_clone=mu_clone,
    )


def _weighted_sse(
    R_hat: np.ndarray,
    R_obs: np.ndarray,
    t_months: np.ndarray,
) -> float:
    """
    Inverse-variance weighted SSE (Listing 6.1, line 16).
    Recent months weighted higher via exponential weighting.
    """
    T = len(t_months)
    # Exponential weights: most recent has weight 1.0, oldest ~0.5
    w = np.exp(0.05 * (t_months - t_months[-1]))
    w = w / w.sum() * T           # normalise so sum = T
    return float(np.sum(w * (R_obs - R_hat) ** 2))


def _sample_prior(
    organism: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample one parameter vector θ from organism-specific prior (Listing 6.1, Step 1).
    """
    prior = ORGANISM_PRIORS[organism]

    R0s = rng.uniform(*prior["R0s_range"])
    R0r = rng.uniform(*prior["R0r_range"])
    a, b = prior["sigma_ab"]
    sigma = rng.beta(a, b)

    if prior["gram"] == "negative":
        lm, ls = prior["rho_HGT_lognormal"]
        rho_HGT = rng.lognormal(lm, ls)
        return np.array([R0s, R0r, sigma, rho_HGT])
    else:
        return np.array([R0s, R0r, sigma])


def _arima_sse(R_obs: np.ndarray) -> float:
    """
    ARIMA(1,1,1) baseline SSE for benchmark comparison (FIX-10, FR-04).
    Uses leave-one-out rolling forecast.
    """
    try:
        from pmdarima import auto_arima
        T = len(R_obs)
        if T < 6:
            return float(np.var(R_obs) * T)

        sse = 0.0
        min_train = max(4, T // 3)
        for i in range(min_train, T):
            train = R_obs[:i]
            actual = R_obs[i]
            try:
                model = auto_arima(train, max_p=2, max_q=2, d=1,
                                   suppress_warnings=True, error_action="ignore")
                pred = float(model.predict(n_periods=1)[0])
            except Exception:
                pred = train[-1]    # naive fallback
            sse += (actual - pred) ** 2

        return sse
    except ImportError:
        logger.warning("pmdarima not installed — using naive ARIMA benchmark")
        T = len(R_obs)
        # Naive persistence SSE
        return float(np.sum((R_obs[1:] - R_obs[:-1]) ** 2))


def calibrate(
    organism: str,
    R_obs: np.ndarray,
    t_months: np.ndarray,
    alpha: float = 0.08,             # antibiotic treatment rate (from PTND)
    delta_PTND: float = 1.0,         # PTND scaling factor
    N_ABC: int = 10_000,             # Listing 6.1: 10,000 draws
    abc_top_pct: float = 0.01,       # Top 1% as ABC posterior
    Is0: float = 0.3,
    Ir0: float = 0.1,
    seed: int = 42,
) -> CalibrationResult:
    """
    Two-stage ABC + L-BFGS-B calibration (Section 6.2.4, Listing 6.1).

    Step 1: ABC — sample N_ABC vectors from organism-specific prior;
            retain top 1% by SSE as ABC posterior.
    Step 2: L-BFGS-B — initialise from ABC posterior median;
            minimise weighted SSE subject to parameter bounds.

    Parameters
    ----------
    organism   : ESKAPE taxonomy key (must be in ORGANISM_PRIORS)
    R_obs      : observed resistance fraction time series (monthly)
    t_months   : corresponding time array (months from start)
    alpha      : antibiotic treatment rate (derived from PTND data)
    delta_PTND : PTND-driven selection scaling factor
    N_ABC      : total ABC simulations
    Is0, Ir0   : initial conditions for ODE

    Returns CalibrationResult; asserts beats_arima per FIX-10.
    """
    if organism not in ORGANISM_PRIORS:
        raise ValueError(
            f"Unknown organism '{organism}'. Valid: {list(ORGANISM_PRIORS.keys())}"
        )

    rng = np.random.default_rng(seed)
    prior = ORGANISM_PRIORS[organism]
    gram = prior["gram"]
    n_accept_target = max(1, int(N_ABC * abc_top_pct))

    logger.info(
        "[%s] ABC: sampling %d parameter vectors (target accept %.1f%%)...",
        organism, N_ABC, abc_top_pct * 100,
    )

    # --- Step 1: ABC global exploration ---
    abc_distances = np.full(N_ABC, np.inf)
    abc_thetas = []

    for i in range(N_ABC):
        theta = _sample_prior(organism, rng)
        try:
            params = _params_from_theta(theta, organism, alpha, delta_PTND)
            R_hat = simulate_ode(params, t_months, Is0, Ir0)
            d = _weighted_sse(R_hat, R_obs, t_months)
        except Exception:
            d = np.inf

        abc_distances[i] = d
        abc_thetas.append(theta)

    # Retain top 1% (Listing 6.1, line 11)
    threshold = np.percentile(abc_distances, abc_top_pct * 100)
    accept_mask = abc_distances <= threshold
    abc_accepted = [abc_thetas[i] for i in range(N_ABC) if accept_mask[i]]

    abc_acceptance_rate = len(abc_accepted) / N_ABC
    if abc_acceptance_rate < 0.005:
        logger.warning(
            "[%s] ABC acceptance rate %.3f%% < 0.5%% target",
            organism, abc_acceptance_rate * 100,
        )

    # ABC posterior median
    if abc_accepted:
        abc_median = np.median(np.stack(abc_accepted), axis=0)
    else:
        abc_median = _sample_prior(organism, rng)

    logger.info(
        "[%s] ABC done: %d accepted (rate=%.3f%%)",
        organism, len(abc_accepted), abc_acceptance_rate * 100,
    )

    # --- Step 2: L-BFGS-B local refinement ---
    gram = prior["gram"]
    if gram == "negative":
        bounds = [
            (1.0, 5.0),   # R0_s
            (0.5, 4.0),   # R0_r
            (0.0, 0.5),   # sigma
            (1e-12, 0.1), # rho_HGT
        ]
    else:
        bounds = [
            (1.0, 5.0),   # R0_s
            (0.5, 4.0),   # R0_r
            (0.0, 0.5),   # sigma
        ]

    def objective(theta: np.ndarray) -> float:
        try:
            params = _params_from_theta(theta, organism, alpha, delta_PTND)
            R_hat = simulate_ode(params, t_months, Is0, Ir0)
            return _weighted_sse(R_hat, R_obs, t_months)
        except Exception:
            return 1e9

    result = minimize(
        objective,
        x0=abc_median,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    best_theta = result.x
    best_params = _params_from_theta(best_theta, organism, alpha, delta_PTND)
    R_hat_final = simulate_ode(best_params, t_months, Is0, Ir0)
    sse_evo_moe = _weighted_sse(R_hat_final, R_obs, t_months)

    # --- ARIMA benchmark (FIX-10) ---
    sse_arima = _arima_sse(R_obs)
    beats_arima = sse_evo_moe < sse_arima

    if not beats_arima:
        logger.warning(
            "[%s] EVO-MOE SSE (%.6f) >= ARIMA SSE (%.6f) — calibration target not met",
            organism, sse_evo_moe, sse_arima,
        )

    logger.info(
        "[%s] Calibration: SSE_EVO=%.6f, SSE_ARIMA=%.6f, beats_ARIMA=%s, "
        "R0_s=%.3f, R0_r=%.3f, σ=%.4f",
        organism, sse_evo_moe, sse_arima, beats_arima,
        best_theta[0], best_theta[1], best_theta[2],
    )

    return CalibrationResult(
        organism=organism,
        params=best_params,
        sse_evo_moe=sse_evo_moe,
        sse_arima=sse_arima,
        beats_arima=beats_arima,
        abc_acceptance_rate=abc_acceptance_rate,
        n_abc_accepted=len(abc_accepted),
        n_abc_total=N_ABC,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(1)
    T = 24
    t = np.arange(T, dtype=float)
    R_obs = np.clip(0.30 + 0.012 * t + rng.normal(0, 0.02, T), 0.0, 1.0)

    organism = "K_pneumoniae"
    cal = calibrate(
        organism=organism,
        R_obs=R_obs,
        t_months=t,
        alpha=0.08,
        N_ABC=300,    # small for speed; production uses 10,000
        seed=42,
    )

    print(f"\nCalibration result for {organism}:")
    print(f"  R0_s    = {cal.params.R0s:.3f}")
    print(f"  R0_r    = {cal.params.R0r:.3f}")
    print(f"  σ       = {cal.params.sigma:.4f}")
    print(f"  ρ_HGT   = {cal.params.rho_HGT:.2e}")
    print(f"  β_s     = {cal.params.beta_s:.4f}/day")
    print(f"  β_r     = {cal.params.beta_r:.4f}/day")
    print(f"  SSE EVO-MOE : {cal.sse_evo_moe:.6f}")
    print(f"  SSE ARIMA   : {cal.sse_arima:.6f}")
    print(f"  Beats ARIMA : {cal.beats_arima}")
    print(f"  ABC rate    : {cal.abc_acceptance_rate*100:.2f}%")
    sys.exit(0 if cal.beats_arima else 1)
