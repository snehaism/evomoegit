"""
sde.py
EVO-MOE V4 — Van Kampen SDE Ensemble Forecast
PRD Section 6.3: Drift/Diffusion terms + Euler-Maruyama integration

Cloud path : 5,000 paths  (V4-NEW6; V3 had only 200 — insufficient for tail risk)
Edge path  :   500 paths

CORRECTIONS from original:
  - Fixed import: calibrate() returns CalibrationResult; use cal.params (not cal directly)
  - Added missing return type annotation on _euler_maruyama
  - Fixed loop termination: step-counter now matches horizon exactly
  - Added explicit DT step cap to prevent runaway integration
  - scenario_analysis: mu_clone_scale fix for Gram-positive only (was unconditional)
  - Euler step: replaced ambiguous reassignment with correct clipping order
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibration import ODEParams, ORGANISM_PRIORS

# ---------------------------------------------------------------------------
# Constants (PRD Section 6.3.2)
# ---------------------------------------------------------------------------

N_PATHS_CLOUD = 5_000   # PRD Section 6.3.2; V4-NEW6 (was 200 in V3)
N_PATHS_EDGE  =   500
DT_DAYS       = 0.1     # Euler-Maruyama step size (days)
DAYS_PER_MONTH = 30.0
STEPS_PER_MONTH = int(DAYS_PER_MONTH / DT_DAYS)  # = 300


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SDEForecastResult:
    """
    Mirrors the SDE-relevant fields of ResistanceForecast (Section 7.1).
    Full dataclass integration is handled by Stage 4 assembly.
    """
    organism: str
    engine_path: str           # "cloud_hmc" | "edge_laplace"
    n_paths: int
    months: list[int]          # [1, 2, …, horizon]
    horizon_months: int

    # Ensemble summary — shape (horizon_months,)
    median_R:    np.ndarray
    ci_lo_95:    np.ndarray    # 2.5th percentile
    ci_hi_95:    np.ndarray    # 97.5th percentile
    p_exceed_50: np.ndarray    # P(R > 0.5) per month

    # Full path matrix — shape (n_paths, horizon_months); optional (memory)
    paths: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            f"Organism : {self.organism}",
            f"Engine   : {self.engine_path}  ({self.n_paths:,} paths)",
            f"Horizon  : {self.horizon_months} months",
            "",
            f"{'Month':>6}  {'Median R':>9}  {'CI 95%':>20}  {'P(R>50%)':>9}",
            "-" * 52,
        ]
        for i, m in enumerate(self.months):
            lines.append(
                f"{m:>6}  {self.median_R[i]:>9.3f}  "
                f"[{self.ci_lo_95[i]:.3f} – {self.ci_hi_95[i]:.3f}]  "
                f"{self.p_exceed_50[i]:>9.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drift terms — Sections 6.3.1, 6.2.1/6.2.2 (per-day units)
# ---------------------------------------------------------------------------

def _drift_gram_negative(
    Is: np.ndarray,
    Ir: np.ndarray,
    beta_s: float,
    beta_r: float,
    gamma_s: float,
    gamma_r: float,
    alpha: float,
    sigma: float,
    rho_HGT: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    fs, fr — Gram-negative (HGT active). Eq. 6.6–6.7.
    All arrays shape (n_paths,).
    """
    S = np.clip(1.0 - Is - Ir, 0.0, 1.0)
    fs = beta_s * Is * S - gamma_s * Is - alpha * Is
    fr = (
        beta_r * Ir * S
        - gamma_r * Ir
        + sigma * alpha * Is          # treatment selection
        + rho_HGT * Is * Ir           # horizontal gene transfer
    )
    return fs, fr


def _drift_gram_positive(
    Is: np.ndarray,
    Ir: np.ndarray,
    beta_s: float,
    beta_r: float,
    gamma_s: float,
    gamma_r: float,
    alpha: float,
    sigma: float,
    mu_clone: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    fs, fr — Gram-positive (clonal expansion, no HGT). Eq. 6.8–6.9.
    """
    S = np.clip(1.0 - Is - Ir, 0.0, 1.0)
    fs = beta_s * Is * S - gamma_s * Is - alpha * Is
    fr = (
        beta_r * Ir * S
        - gamma_r * Ir
        + sigma * alpha * Is          # treatment selection
        + mu_clone * Ir               # clonal amplification
    )
    return fs, fr


# ---------------------------------------------------------------------------
# Diffusion terms — Section 6.3.1, Eq. 6.13–6.14
# ---------------------------------------------------------------------------

def _diffusion_gram_negative(
    Is: np.ndarray,
    Ir: np.ndarray,
    beta_s: float,
    beta_r: float,
    gamma_s: float,
    gamma_r: float,
    alpha: float,
    sigma: float,
    rho_HGT: float,
    N_pop: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    gs, gr from Van Kampen expansion. Eq. 6.13–6.14.
    Diffusion coefficient = sqrt(g / N) per path.
    """
    S = np.clip(1.0 - Is - Ir, 0.0, 1.0)
    gs = beta_s * Is * S + gamma_s * Is + alpha * Is
    gr = (
        beta_r * Ir * S
        + gamma_r * Ir
        + sigma * alpha * Is
        + rho_HGT * Is * Ir
    )
    sig_s = np.sqrt(np.maximum(gs, 0.0) / N_pop)
    sig_r = np.sqrt(np.maximum(gr, 0.0) / N_pop)
    return sig_s, sig_r


def _diffusion_gram_positive(
    Is: np.ndarray,
    Ir: np.ndarray,
    beta_s: float,
    beta_r: float,
    gamma_s: float,
    gamma_r: float,
    alpha: float,
    sigma: float,
    mu_clone: float,
    N_pop: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    gs, gr for Gram-positive. Eq. 6.13–6.14 (clonal term replaces HGT).
    """
    S = np.clip(1.0 - Is - Ir, 0.0, 1.0)
    gs = beta_s * Is * S + gamma_s * Is + alpha * Is
    gr = (
        beta_r * Ir * S
        + gamma_r * Ir
        + sigma * alpha * Is
        + mu_clone * Ir
    )
    sig_s = np.sqrt(np.maximum(gs, 0.0) / N_pop)
    sig_r = np.sqrt(np.maximum(gr, 0.0) / N_pop)
    return sig_s, sig_r


# ---------------------------------------------------------------------------
# Euler-Maruyama integrator — Section 6.3.2, Eq. 6.15–6.16
# ---------------------------------------------------------------------------

def _euler_maruyama(
    params: ODEParams,
    horizon_months: int,
    n_paths: int,
    Is0: float,
    Ir0: float,
    N_pop: float,
    rng: np.random.Generator,
    store_paths: bool = False,
) -> np.ndarray:
    """
    Euler-Maruyama integration. Returns R matrix shape (n_paths, horizon_months).

    Physical constraints post-step (Section 6.3.2):
        Is, Ir ∈ [0, 1];   Is + Ir ≤ 1
    """
    gram = ORGANISM_PRIORS[params.organism]["gram"]
    total_steps = horizon_months * STEPS_PER_MONTH

    # Output buffer: resistance fraction sampled once per month
    R_monthly = np.zeros((n_paths, horizon_months), dtype=np.float32)

    # State vectors — shape (n_paths,)
    Is = np.full(n_paths, Is0, dtype=np.float64)
    Ir = np.full(n_paths, Ir0, dtype=np.float64)

    sqrt_dt = np.sqrt(DT_DAYS)
    month_idx = 0

    for step in range(1, total_steps + 1):
        # --- Drift and diffusion ---
        if gram == "negative":
            fs, fr = _drift_gram_negative(
                Is, Ir,
                params.beta_s, params.beta_r,
                params.gamma_s, params.gamma_r,
                params.alpha, params.sigma,
                params.rho_HGT or 0.0,
            )
            sig_s, sig_r = _diffusion_gram_negative(
                Is, Ir,
                params.beta_s, params.beta_r,
                params.gamma_s, params.gamma_r,
                params.alpha, params.sigma,
                params.rho_HGT or 0.0,
                N_pop,
            )
        else:
            fs, fr = _drift_gram_positive(
                Is, Ir,
                params.beta_s, params.beta_r,
                params.gamma_s, params.gamma_r,
                params.alpha, params.sigma,
                params.mu_clone,
            )
            sig_s, sig_r = _diffusion_gram_positive(
                Is, Ir,
                params.beta_s, params.beta_r,
                params.gamma_s, params.gamma_r,
                params.alpha, params.sigma,
                params.mu_clone,
                N_pop,
            )

        # --- Wiener increments (Eq. 6.15–6.16) ---
        xi_s = rng.standard_normal(n_paths)
        xi_r = rng.standard_normal(n_paths)

        # --- Euler step ---
        Is = Is + fs * DT_DAYS + sig_s * sqrt_dt * xi_s
        Ir = Ir + fr * DT_DAYS + sig_r * sqrt_dt * xi_r

        # --- Physical constraints (Section 6.3.2) ---
        Is = np.clip(Is, 0.0, 1.0)
        Ir = np.clip(Ir, 0.0, 1.0)
        # Enforce Is + Ir ≤ 1 by proportional rescaling
        total = Is + Ir
        overflow = total > 1.0
        if np.any(overflow):
            scale = np.where(overflow, 1.0 / total, 1.0)
            Is = Is * scale
            Ir = Ir * scale

        # --- Record R at end of each calendar month ---
        if step % STEPS_PER_MONTH == 0 and month_idx < horizon_months:
            denom = np.maximum(Is + Ir, 1e-9)
            R_monthly[:, month_idx] = (Ir / denom).astype(np.float32)
            month_idx += 1

    return R_monthly


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sde_ensemble(
    params: ODEParams,
    horizon_months: int = 12,
    engine: str = "cloud",         # "cloud" | "edge"
    Is0: float = 0.3,
    Ir0: float = 0.1,
    N_pop: float = 200.0,          # effective ward population size
    seed: int = 42,
    store_paths: bool = False,
) -> SDEForecastResult:
    """
    Run Van Kampen SDE ensemble and return posterior summaries.

    Parameters
    ----------
    params         : calibrated ODEParams from calibration.calibrate()
    horizon_months : forecast horizon (12 for primary; 60 for scenarios)
    engine         : "cloud" → 5,000 paths; "edge" → 500 paths
    N_pop          : ward census size (scales diffusion noise)
    seed           : RNG seed (FIX-12)
    store_paths    : if True, attach full (n_paths × horizon) matrix to result
    """
    if engine not in ("cloud", "edge"):
        raise ValueError("engine must be 'cloud' or 'edge'")

    n_paths = N_PATHS_CLOUD if engine == "cloud" else N_PATHS_EDGE
    engine_path = "cloud_hmc" if engine == "cloud" else "edge_laplace"

    print(
        f"[{params.organism}] SDE: {n_paths:,} paths × "
        f"{horizon_months} months (engine={engine_path}) …"
    )

    rng = np.random.default_rng(seed)

    R_matrix = _euler_maruyama(
        params=params,
        horizon_months=horizon_months,
        n_paths=n_paths,
        Is0=Is0,
        Ir0=Ir0,
        N_pop=N_pop,
        rng=rng,
        store_paths=store_paths,
    )

    # Posterior summaries — shape (horizon_months,)
    median_R    = np.median(R_matrix, axis=0)
    ci_lo_95    = np.percentile(R_matrix, 2.5,  axis=0)
    ci_hi_95    = np.percentile(R_matrix, 97.5, axis=0)
    p_exceed_50 = np.mean(R_matrix > 0.5, axis=0)

    return SDEForecastResult(
        organism=params.organism,
        engine_path=engine_path,
        n_paths=n_paths,
        months=list(range(1, horizon_months + 1)),
        horizon_months=horizon_months,
        median_R=median_R,
        ci_lo_95=ci_lo_95,
        ci_hi_95=ci_hi_95,
        p_exceed_50=p_exceed_50,
        paths=R_matrix if store_paths else None,
    )


# ---------------------------------------------------------------------------
# Scenario analysis helper (Refinement R3)
# Section 2.3.4: MUST be labelled "Scenario Analysis", never "forecast"
# ---------------------------------------------------------------------------

def run_scenario_analysis(
    params_baseline: ODEParams,
    horizon_months: int = 60,
    engine: str = "cloud",
    Is0: float = 0.3,
    Ir0: float = 0.1,
    N_pop: float = 200.0,
    seed: int = 42,
) -> dict[str, SDEForecastResult]:
    """
    Three-scenario ODE ensemble per PRD Section 2.3.4.

    Returns dict with keys:
        "high"    — No intervention; PTND continues at current trend
        "current" — PTND and policy held constant
        "low"     — 40% carbapenem restriction + decolonisation (Gram-pos)

    IMPORTANT (Refinement R3): caller MUST label all output as
    "Scenario Analysis — Conditional ODE Extrapolation (not a prediction)"
    in every dashboard view and API response for 24/60-month horizons.
    """
    scenarios: dict[str, SDEForecastResult] = {}
    gram = ORGANISM_PRIORS[params_baseline.organism]["gram"]

    for label, delta_PTND, mu_clone_scale in [
        ("high",    1.30, 1.0),   # +30% selection pressure — no intervention
        ("current", 1.00, 1.0),   # baseline — PTND and policy held constant
        ("low",     0.60, 0.5),   # 40% carbapenem restriction + decolonisation
    ]:
        p = deepcopy(params_baseline)
        p.delta_PTND = delta_PTND
        # Recompute beta_r with modified delta_PTND (Listing 6.1, parameter recovery)
        p.beta_r = p.R0r * p.gamma_r / max(delta_PTND, 1e-6)

        # CORRECTION: mu_clone scaling only applies to Gram-positive organisms
        if gram == "positive":
            p.mu_clone = params_baseline.mu_clone * mu_clone_scale

        scenarios[label] = run_sde_ensemble(
            params=p,
            horizon_months=horizon_months,
            engine=engine,
            Is0=Is0,
            Ir0=Ir0,
            N_pop=N_pop,
            seed=seed + hash(label) % 1000,
        )

    return scenarios


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from calibration import calibrate

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
        N_ABC=300,    # small for speed; use 10,000 in production
        seed=42,
    )

    print("\n--- 12-month cloud forecast ---")
    forecast = run_sde_ensemble(
        params=cal.params,
        horizon_months=12,
        engine="cloud",
        seed=42,
    )
    print(forecast.summary())

    print("\n--- Edge forecast (500 paths) ---")
    forecast_edge = run_sde_ensemble(
        params=cal.params,
        horizon_months=12,
        engine="edge",
        seed=42,
    )
    print(forecast_edge.summary())

    print("\n--- Scenario analysis (60 months) ---")
    # Label MUST read "Scenario Analysis — Conditional ODE Extrapolation (not a prediction)"
    print("SCENARIO ANALYSIS — Conditional ODE Extrapolation (not a prediction)")
    scenarios = run_scenario_analysis(
        params_baseline=cal.params,
        horizon_months=60,
        engine="edge",   # fast for test
        seed=42,
    )
    for sc_name, sc_result in scenarios.items():
        print(f"\nScenario [{sc_name.upper()}]")
        for m_idx in [11, 23, 35, 59]:
            if m_idx < sc_result.horizon_months:
                print(
                    f"  Month {m_idx+1:>2}: R={sc_result.median_R[m_idx]:.3f} "
                    f"[{sc_result.ci_lo_95[m_idx]:.3f}–"
                    f"{sc_result.ci_hi_95[m_idx]:.3f}]"
                )

    print("\nCALIBRATION vs ARIMA:")
    print(f"  SSE EVO-MOE : {cal.sse_evo_moe:.6f}")
    print(f"  SSE ARIMA   : {cal.sse_arima:.6f}")
    print(f"  Beats ARIMA : {cal.beats_arima}")
    print(f"  ABC rate    : {cal.abc_acceptance_rate*100:.2f}%")
    sys.exit(0 if cal.beats_arima else 1)
