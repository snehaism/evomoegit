"""
edge_laplace.py
EVO-MOE V4 — C2-Edge Inference Path (Laplace Approximation)
PRD Section 5.3: Purpose, Algorithm, and Hardware Constraints

Implements:
  Step 1 — Mode finding via L-BFGS-B (Eq. 5.9)
  Step 2 — Hessian computation (Eq. 5.10)
  Step 3 — Gaussian posterior approximation N(β̂, H⁻¹) (Eq. 5.11)
  Step 4 — Posterior predictive samples (S = 2,000)

Hard constraints (PRD Section 5.3.1):
  ≤ 8 GB RAM; runtime < 10 min; < 60 sec forecast-only; fully offline; no GPU.

NOTE: The C2-Edge path is NOT an approximation of the cloud model for
convenience — it is the correct engineering decision for the hardware
constraint. Edge output feeds C1 dashboard. Cloud output feeds global
prior, C4, and C5. These MUST NOT be conflated in any interface.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logit  # expit = sigmoid = inv_logit
from scipy.stats import norm

from features import FeatureMatrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (Section 5.3)
# ---------------------------------------------------------------------------

N_POSTERIOR_SAMPLES = 2_000          # Step 4: posterior predictive draws
MIN_ISOLATES_WARNING = 30            # Laplace accuracy warning threshold
BRIER_MAX = 0.25                     # Section 7.3.3 Eq. 7.9 normalisation


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LaplaceResult:
    """
    Laplace approximation posterior — atomic unit for federated sync
    (maps to OrganismPosterior in PosteriorSummary protobuf).
    """
    organism: str
    drug: str
    ward: str
    antibiotic_class: str

    # MAP estimate
    beta_hat: np.ndarray             # shape (P,) — logit-scale coefficients
    intercept_hat: float             # µ̂ — organism-drug intercept (logit)

    # Posterior covariance (Hessian inverse)
    cov_beta: np.ndarray             # shape (P, P)
    intercept_variance: float        # marginal variance of intercept

    # Marginal posterior statistics (for PosteriorSummary protobuf)
    posterior_means: np.ndarray      # shape (P,) — same as beta_hat
    posterior_variances: np.ndarray  # shape (P,) — diagonal of cov_beta
    feature_names: List[str]

    # Posterior predictive samples (shape: N_POSTERIOR_SAMPLES × T)
    p_samples: np.ndarray

    # Calibration metrics
    brier_score: float
    coverage_95: float               # empirical CI coverage
    effective_sample_size: float     # ESS from Laplace
    calibration_score: float         # Q_k component (1 - Brier/Brier_max)

    # Flags
    n_isolates: int
    low_data_warning: bool
    precision_flag: str              # "" | "Insufficient local data (<30 isolates)..."

    # Convergence
    converged: bool
    n_iterations: int
    final_loss: float


# ---------------------------------------------------------------------------
# Negative log-posterior (objective for L-BFGS-B, Eq. 5.9)
# ---------------------------------------------------------------------------

def _neg_log_posterior(
    params: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    N: np.ndarray,
    mu_global: float,
    sigma_global: float,
    prior_cov_inv: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Returns (neg_log_posterior, gradient) for L-BFGS-B.

    params[0]   = intercept µ
    params[1:]  = beta (P,)

    Loss = -Σ_t [Y_t * η_t - N_t * log(1 + exp(η_t))]
           + 0.5*(µ - µ_global)²/σ_global²
           + 0.5 * beta' * Σ_prior_inv * beta
    """
    T, P = X.shape
    mu = params[0]
    beta = params[1:]

    eta = mu + X @ beta           # (T,)
    p = expit(eta)                 # inv_logit

    # Binomial log-likelihood (stable)
    mask = N > 0
    ll = np.sum(
        Y[mask] * eta[mask]
        - N[mask] * np.log1p(np.exp(-eta[mask]))  # log(1 + exp(-η)) = -log σ(η)
        + (Y[mask] - N[mask]) * eta[mask] * (eta[mask] < 0)  # numerical stability
    )
    # Simpler stable form
    ll = np.sum(
        np.where(
            mask,
            Y * np.log(np.clip(p, 1e-12, 1.0)) + (N - Y) * np.log(np.clip(1 - p, 1e-12, 1.0)),
            0.0,
        )
    )

    # Prior terms
    prior_mu = 0.5 * ((mu - mu_global) ** 2) / (sigma_global ** 2)
    prior_beta = 0.5 * beta @ prior_cov_inv @ beta

    loss = -ll + prior_mu + prior_beta

    # Gradient
    residual = p - Y / np.clip(N, 1, None)       # (T,); avoids division by zero
    residual = np.where(mask, (N * p - Y), 0.0)

    grad_mu = np.sum(residual) + (mu - mu_global) / sigma_global ** 2
    grad_beta = X.T @ residual + prior_cov_inv @ beta

    grad = np.concatenate([[grad_mu], grad_beta])
    return float(loss), grad


# ---------------------------------------------------------------------------
# Hessian computation (Eq. 5.10)
# ---------------------------------------------------------------------------

def _compute_hessian(
    params: np.ndarray,
    X: np.ndarray,
    N: np.ndarray,
    sigma_global: float,
    prior_cov_inv: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    H = -∇² log p(β, Y | n)    at β = β̂    [Eq. 5.10]

    Shape: (P+1, P+1) — first row/col is intercept.
    """
    P = X.shape[1]
    mu = params[0]
    beta = params[1:]

    eta = mu + X @ beta
    p = expit(eta)
    W = np.where(mask, N * p * (1.0 - p), 0.0)    # diagonal weight matrix

    # Hessian of negative log-likelihood
    # H_beta = X' W X
    # H_mu_mu = Σ W_t
    # H_mu_beta = Σ W_t * x_t

    XW = X * W[:, np.newaxis]           # (T, P)
    H_ll = np.zeros((P + 1, P + 1))
    H_ll[0, 0] = np.sum(W)
    H_ll[0, 1:] = W @ X
    H_ll[1:, 0] = W @ X
    H_ll[1:, 1:] = X.T @ XW

    # Prior Hessian
    H_prior = np.zeros((P + 1, P + 1))
    H_prior[0, 0] = 1.0 / sigma_global ** 2
    H_prior[1:, 1:] = prior_cov_inv

    return H_ll + H_prior


# ---------------------------------------------------------------------------
# Main Laplace approximation (Sections 5.3.2)
# ---------------------------------------------------------------------------

def run_laplace(
    fm: FeatureMatrix,
    mu_global: float = 0.0,
    sigma_global: float = 1.5,
    horseshoe_scale: float = 0.5,    # τ₀ default when global prior unavailable
    seed: int = 42,
) -> LaplaceResult:
    """
    C2-Edge Laplace approximation for a single (organism, drug, ward) cell.

    Steps per Section 5.3.2:
      1. Mode finding with L-BFGS-B (Eq. 5.9)
      2. Hessian at MAP (Eq. 5.10)
      3. Gaussian posterior N(β̂, H⁻¹) (Eq. 5.11)
      4. S = 2,000 posterior predictive samples

    Parameters
    ----------
    fm            : FeatureMatrix from Stage 1
    mu_global     : global prior mean (logit scale) from prior registry
    sigma_global  : global prior SD (logit scale) from prior registry
    horseshoe_scale : τ₀ prior scale for beta coefficients
    seed          : RNG seed (FIX-12)
    """
    rng = np.random.default_rng(seed)
    X = fm.X.astype(float)
    Y = fm.R_obs * fm.n_obs              # reconstruct counts
    Y = np.round(Y).astype(int)
    N = fm.n_obs.astype(int)
    T, P = X.shape
    mask = N > 0

    low_data = int(fm.n_obs.sum()) < MIN_ISOLATES_WARNING
    precision_flag = ""
    if low_data:
        precision_flag = (
            f"Insufficient local data (<{MIN_ISOLATES_WARNING} isolates). "
            "Forecast based on regional prior. Uncertainty may be underestimated."
        )
        logger.warning(
            "[%s/%s/%s] %s", fm.organism, fm.drug, fm.ward, precision_flag
        )
        # Per Section 5.3.2: double variance for low-data case
        sigma_global_eff = sigma_global * 2.0
    else:
        sigma_global_eff = sigma_global

    # --- Prior covariance for beta (diagonal horseshoe-inspired prior) ---
    # τ₀² * I as a simplified edge-compatible prior (full horseshoe in cloud Stan)
    prior_var_beta = horseshoe_scale ** 2
    prior_cov_inv = np.eye(P) / prior_var_beta

    # --- Step 1: L-BFGS-B mode finding (Eq. 5.9) ---
    # Initialise from global prior mean
    params0 = np.zeros(P + 1)
    params0[0] = mu_global

    result = minimize(
        fun=_neg_log_posterior,
        x0=params0,
        args=(X, Y, N, mu_global, sigma_global_eff, prior_cov_inv),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )

    converged = result.success or (result.fun < 1e6)
    if not converged:
        logger.warning(
            "[%s/%s/%s] L-BFGS-B did not converge: %s",
            fm.organism, fm.drug, fm.ward, result.message,
        )

    params_hat = result.x
    mu_hat = float(params_hat[0])
    beta_hat = params_hat[1:]

    # --- Step 2: Hessian at MAP (Eq. 5.10) ---
    H = _compute_hessian(params_hat, X, N, sigma_global_eff, prior_cov_inv, mask)

    # --- Step 3: Gaussian posterior approximation N(β̂, H⁻¹) (Eq. 5.11) ---
    try:
        c, low = cho_factor(H, lower=False)
        cov_full = cho_solve((c, low), np.eye(P + 1))
    except np.linalg.LinAlgError:
        # Regularise Hessian if singular
        logger.warning("[%s/%s/%s] Hessian near-singular — adding ridge", fm.organism, fm.drug, fm.ward)
        H_reg = H + np.eye(P + 1) * 1e-4
        cov_full = np.linalg.solve(H_reg, np.eye(P + 1))

    intercept_variance = float(cov_full[0, 0])
    cov_beta = cov_full[1:, 1:]
    posterior_variances = np.diag(cov_beta)

    # Effective sample size (approximate from Fisher information trace)
    ess = float(np.sum(np.diag(np.linalg.inv(cov_beta))) ** -1 * T)
    ess = max(1.0, min(ess, float(T)))

    # --- Step 4: Posterior predictive samples (Section 5.3.2) ---
    # Draw S = 2,000 samples from N(β̂, H⁻¹)
    L_cov = np.linalg.cholesky(cov_full + np.eye(P + 1) * 1e-9)
    z_samples = rng.standard_normal((N_POSTERIOR_SAMPLES, P + 1))
    param_samples = params_hat + z_samples @ L_cov.T   # shape (S, P+1)

    mu_samples = param_samples[:, 0]          # (S,)
    beta_samples = param_samples[:, 1:]       # (S, P)

    # Posterior predictive p̂ (S, T)
    eta_samples = mu_samples[:, np.newaxis] + beta_samples @ X.T  # (S, T)
    p_samples = expit(eta_samples)                                  # (S, T)

    # --- Calibration metrics ---
    p_mean = p_samples.mean(axis=0)    # (T,) posterior mean prediction
    R_obs = fm.R_obs

    # Brier score: mean squared error on resistance fraction
    brier = float(np.mean((p_mean[mask] - R_obs[mask]) ** 2))

    # 95% credible interval coverage
    ci_lo = np.percentile(p_samples, 2.5, axis=0)
    ci_hi = np.percentile(p_samples, 97.5, axis=0)
    coverage = float(np.mean(
        (R_obs[mask] >= ci_lo[mask]) & (R_obs[mask] <= ci_hi[mask])
    ))

    # Quality score calibration component (Eq. 7.9)
    calibration_score = 1.0 - brier / BRIER_MAX

    logger.info(
        "[%s/%s/%s] Laplace done: Brier=%.4f, CI_cov=%.3f, ESS=%.1f, converged=%s",
        fm.organism, fm.drug, fm.ward, brier, coverage, ess, converged,
    )

    return LaplaceResult(
        organism=fm.organism,
        drug=fm.drug,
        ward=fm.ward,
        antibiotic_class=fm.antibiotic_class,
        beta_hat=beta_hat,
        intercept_hat=mu_hat,
        cov_beta=cov_beta,
        intercept_variance=intercept_variance,
        posterior_means=beta_hat.copy(),
        posterior_variances=posterior_variances,
        feature_names=fm.feature_names,
        p_samples=p_samples,
        brier_score=brier,
        coverage_95=coverage,
        effective_sample_size=ess,
        calibration_score=calibration_score,
        n_isolates=int(fm.n_obs.sum()),
        low_data_warning=low_data,
        precision_flag=precision_flag,
        converged=converged,
        n_iterations=result.nit,
        final_loss=float(result.fun),
    )


# ---------------------------------------------------------------------------
# Posterior predictive forecast helper
# ---------------------------------------------------------------------------

def forecast_laplace(
    laplace: LaplaceResult,
    X_future: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate posterior predictive resistance fractions for future feature matrix.

    Parameters
    ----------
    laplace  : LaplaceResult from run_laplace()
    X_future : (T_future, P) future feature matrix
    seed     : RNG seed

    Returns
    -------
    median_R, ci_lo_95, ci_hi_95 — each shape (T_future,)
    """
    rng = np.random.default_rng(seed)
    P = len(laplace.beta_hat)
    T_fut = X_future.shape[0]

    cov_full = np.zeros((P + 1, P + 1))
    cov_full[0, 0] = laplace.intercept_variance
    cov_full[1:, 1:] = laplace.cov_beta

    L = np.linalg.cholesky(cov_full + np.eye(P + 1) * 1e-9)
    z = rng.standard_normal((N_POSTERIOR_SAMPLES, P + 1))
    params_hat = np.concatenate([[laplace.intercept_hat], laplace.beta_hat])
    param_samples = params_hat + z @ L.T

    mu_s = param_samples[:, 0]
    beta_s = param_samples[:, 1:]

    eta = mu_s[:, np.newaxis] + beta_s @ X_future.T   # (S, T_fut)
    p = expit(eta)

    median_R = np.median(p, axis=0)
    ci_lo = np.percentile(p, 2.5, axis=0)
    ci_hi = np.percentile(p, 97.5, axis=0)

    return median_R, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# PosteriorSummary builder (for federated gRPC payload)
# ---------------------------------------------------------------------------

def build_posterior_summary_payload(
    laplace: LaplaceResult,
    node_id: str,
    prior_version: str,
    n_isolates_since_sync: int,
    model_binary_version: str,
    country: str,
    who_region: str,
    resistance_fraction_current: float,
    resistance_fraction_6mo_ago: float,
    mic_log2_mean: float,
    mic_p90: float,
) -> dict:
    """
    Build the PosteriorSummary payload dict that maps 1:1 to the protobuf schema
    (aggregation.proto OrganismPosterior fields).

    Caller uses this to populate the gRPC PosteriorSummary message.
    """
    from datetime import datetime, timezone

    return {
        "node_id": node_id,
        "prior_version_used": prior_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_isolates_since_last_sync": n_isolates_since_sync,
        "model_binary_version": model_binary_version,
        "organism_posteriors": [
            {
                "organism": laplace.organism,
                "antibiotic_class": laplace.antibiotic_class,
                "specific_drug": laplace.drug,
                "ward": laplace.ward,
                "country": country,
                "who_region": who_region,
                # Parallel arrays — logit-scale posterior (proto fields 9–11)
                "feature_names": laplace.feature_names,
                "posterior_means": [f"{v:.8f}" for v in laplace.posterior_means],
                "posterior_variances": [f"{v:.8f}" for v in laplace.posterior_variances],
                "intercept_mean": laplace.intercept_hat,
                "intercept_variance": laplace.intercept_variance,
                "effective_sample_size": laplace.effective_sample_size,
                "calibration_score": laplace.calibration_score,
                # Resistance summaries
                "resistance_fraction_current": resistance_fraction_current,
                "resistance_fraction_6mo_ago": resistance_fraction_6mo_ago,
                "mic_log2_mean": mic_log2_mean,
                "mic_p90": mic_p90,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from features import build_feature_matrix, validate_glass

    rng = np.random.default_rng(0)
    T = 24
    dates = __import__("pandas").date_range("2023-01-01", periods=T, freq="MS")

    records = []
    for d in dates:
        for i in range(40):
            records.append({
                "isolate_id": f"ISO_{d.strftime('%Y%m')}_{i:03d}",
                "collection_date": d,
                "ward": "ICU",
                "specimen_type": "blood",
                "organism_eucast": "Klebsiella pneumoniae",
                "antibiotic": "meropenem",
                "sir_result": rng.choice(["R", "S"], p=[0.60, 0.40]),
                "mic_value": float(rng.choice([0.125, 0.25, 0.5, 1.0, 4.0, 16.0])),
            })

    import pandas as pd
    df = pd.DataFrame(records)
    val = validate_glass(df)
    fm = build_feature_matrix(
        df_isolates=val.valid_records,
        df_pharmacy=None, df_icu=None, df_policy=None,
        organism="K_pneumoniae", drug="meropenem",
        ward="ICU", antibiotic_class="carbapenem",
    )

    lap = run_laplace(fm, mu_global=0.0, sigma_global=1.5, seed=42)
    print(f"\nLaplace result:")
    print(f"  Intercept:  {lap.intercept_hat:.4f} (var={lap.intercept_variance:.4f})")
    print(f"  Brier:      {lap.brier_score:.4f}")
    print(f"  CI cov:     {lap.coverage_95:.3f}")
    print(f"  ESS:        {lap.effective_sample_size:.1f}")
    print(f"  Converged:  {lap.converged} (iters={lap.n_iterations})")
    print(f"  Low data:   {lap.low_data_warning}")
    print(f"  Cal score:  {lap.calibration_score:.4f}")
    sys.exit(0)
