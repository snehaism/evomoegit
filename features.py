"""
features.py
EVO-MOE V4 — Stage 1: Input Preparation and Feature Engineering
PRD Section 4.1–4.2

Implements:
  - GLASS validation pipeline (Section 4.1.3)
  - Feature 1: MIC Distribution (Section 4.2.1)
  - Feature 2: ICU Severity Index (Section 4.2.2)
  - Feature 3: Antibiotic Sales Pressure PTND (Section 4.2.3)
  - Feature 4: Policy Stringency Index (Section 4.2.4)
  - Per-feature lag optimisation (Section 4.2.5)

ESKAPE organisms: E. faecium, S. aureus MRSA, K. pneumoniae,
                  A. baumannii, P. aeruginosa, Enterobacter spp.
Antibiotic classes: carbapenem, cephalosporin_3g4g, fluoroquinolone,
                    glycopeptide, colistin
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ESKAPE taxonomy map (Section 4.1.2)
# ---------------------------------------------------------------------------

ESKAPE_TAXONOMY: Dict[str, str] = {
    # K. pneumoniae group
    "klebsiella pneumoniae": "K_pneumoniae",
    "k. pneumoniae": "K_pneumoniae",
    "klebsiella variicola": "K_pneumoniae",
    # A. baumannii group
    "acinetobacter baumannii": "A_baumannii",
    "a. baumannii": "A_baumannii",
    "acinetobacter baumannii complex": "A_baumannii",
    # P. aeruginosa
    "pseudomonas aeruginosa": "P_aeruginosa",
    "p. aeruginosa": "P_aeruginosa",
    # Enterobacter spp.
    "enterobacter cloacae": "Enterobacter_spp",
    "enterobacter cloacae complex": "Enterobacter_spp",
    "enterobacter aerogenes": "Enterobacter_spp",
    "enterobacter spp": "Enterobacter_spp",
    # E. faecium
    "enterococcus faecium": "E_faecium",
    "e. faecium": "E_faecium",
    # S. aureus MRSA
    "staphylococcus aureus": "S_aureus_MRSA",
    "s. aureus": "S_aureus_MRSA",
    "mrsa": "S_aureus_MRSA",
}

ESKAPE_ORGANISMS = list(set(ESKAPE_TAXONOMY.values()))

ANTIBIOTIC_CLASSES = [
    "carbapenem",
    "cephalosporin_3g4g",
    "fluoroquinolone",
    "glycopeptide",
    "colistin",
]

# ECOFF breakpoints (mg/L) per organism-drug pair (EUCAST 2024 reference)
ECOFF_TABLE: Dict[Tuple[str, str], float] = {
    ("K_pneumoniae", "meropenem"): 0.125,
    ("K_pneumoniae", "imipenem"): 1.0,
    ("K_pneumoniae", "ertapenem"): 0.25,
    ("A_baumannii", "meropenem"): 0.5,
    ("A_baumannii", "imipenem"): 2.0,
    ("P_aeruginosa", "meropenem"): 0.5,
    ("P_aeruginosa", "cefepime"): 1.0,
    ("Enterobacter_spp", "cefepime"): 1.0,
    ("Enterobacter_spp", "ceftriaxone"): 0.5,
    ("S_aureus_MRSA", "vancomycin"): 2.0,
    ("E_faecium", "vancomycin"): 4.0,
}

VALID_SIR = {"S", "I", "R", "SDD"}
LAG_MAX_MONTHS = 12
MIN_CROSS_CORR = 0.15          # Section 4.2.5: log warning if |ρ| < 0.15


# ---------------------------------------------------------------------------
# GLASS Validation (Section 4.1.3)
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid_records: pd.DataFrame
    rejected_count: int
    flagged_unknown_org: int
    rejection_reasons: Dict[str, int]


def validate_glass(df: pd.DataFrame) -> ValidationResult:
    """
    Apply WHO GLASS validation rules to a raw isolate DataFrame.

    Required columns:
        isolate_id, collection_date, ward, specimen_type,
        organism_eucast, antibiotic, sir_result

    Optional columns:
        mic_value (mg/L)
    """
    required = [
        "isolate_id", "collection_date", "ward",
        "specimen_type", "organism_eucast", "antibiotic", "sir_result",
    ]

    rejection_reasons: Dict[str, int] = {}
    flagged_unknown_org = 0

    # --- Check required fields ---
    missing_field_mask = df[required].isnull().any(axis=1)
    n_missing = missing_field_mask.sum()
    if n_missing:
        rejection_reasons["missing_required_field"] = int(n_missing)
    df = df[~missing_field_mask].copy()

    # --- Parse dates ---
    df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
    bad_date = df["collection_date"].isnull()
    if bad_date.sum():
        rejection_reasons["unparseable_date"] = int(bad_date.sum())
    df = df[~bad_date].copy()

    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    date_lo = today - pd.DateOffset(years=10)
    out_of_range = (df["collection_date"] > today) | (df["collection_date"] < date_lo)
    if out_of_range.sum():
        rejection_reasons["date_out_of_range"] = int(out_of_range.sum())
    df = df[~out_of_range].copy()

    # --- Organism mapping ---
    df["organism_key"] = (
        df["organism_eucast"].str.strip().str.lower().map(ESKAPE_TAXONOMY)
    )
    unknown_org = df["organism_key"].isnull()
    flagged_unknown_org = int(unknown_org.sum())
    # Keep but flag; they may still have valid MIC data worth logging
    df.loc[unknown_org, "organism_key"] = "UNKNOWN"

    # --- SIR validation ---
    bad_sir = ~df["sir_result"].str.upper().isin(VALID_SIR)
    if bad_sir.sum():
        rejection_reasons["invalid_sir_result"] = int(bad_sir.sum())
    df = df[~bad_sir].copy()

    # --- MIC plausibility check ---
    if "mic_value" in df.columns:
        mic_numeric = pd.to_numeric(df["mic_value"], errors="coerce")
        implausible_mic = mic_numeric.notna() & (
            (mic_numeric < 0.001) | (mic_numeric > 2048)
        )
        if implausible_mic.sum():
            rejection_reasons["implausible_mic"] = int(implausible_mic.sum())
        df = df[~implausible_mic].copy()
        df["mic_value"] = pd.to_numeric(df["mic_value"], errors="coerce")

    # --- Deduplication: same patient × same organism × within 30 days → keep first ---
    if "patient_id" in df.columns:
        df = df.sort_values("collection_date")
        df = df.drop_duplicates(
            subset=["patient_id", "organism_key", pd.Grouper(freq="30D")],
            keep="first",
        ) if False else df  # fallback: use date-floor approach
        df["date_bucket"] = (
            df["collection_date"] - pd.to_datetime("2000-01-01")
        ).dt.days // 30
        df = df.drop_duplicates(
            subset=["patient_id", "organism_key", "date_bucket"], keep="first"
        ).drop(columns=["date_bucket"])

    total_rejected = sum(rejection_reasons.values())
    logger.info(
        "GLASS validation: %d accepted, %d rejected, %d unknown_org flagged",
        len(df), total_rejected, flagged_unknown_org,
    )

    return ValidationResult(
        valid_records=df.reset_index(drop=True),
        rejected_count=total_rejected,
        flagged_unknown_org=flagged_unknown_org,
        rejection_reasons=rejection_reasons,
    )


# ---------------------------------------------------------------------------
# Feature 1 — MIC Distribution (Section 4.2.1, Eq. 4.1–4.5)
# ---------------------------------------------------------------------------

@dataclass
class MICFeatures:
    """Feature 1 output per (organism, drug, ward, month) cell."""
    mic_log2_mean: float        # Eq. 4.1
    mic_p90: float              # Eq. 4.2
    f_ecoff: float              # Eq. 4.3 — fraction above ECOFF
    mic_entropy: float          # Eq. 4.4  H_MIC
    delta_mic_3mo: float        # Eq. 4.5 — rolling 3-month trend
    n_isolates: int
    mic_absent: bool            # True when only SIR available


def compute_mic_features(
    df: pd.DataFrame,
    organism: str,
    drug: str,
    ward: str,
    ecoff: Optional[float] = None,
) -> pd.Series:
    """
    Compute monthly MIC feature set for a given (organism, drug, ward) slice.

    DataFrame must have columns: collection_date, mic_value (optional), sir_result.
    Returns a Series indexed by period (monthly) with MICFeatures columns.
    """
    sub = df[
        (df["organism_key"] == organism)
        & (df["antibiotic"].str.lower() == drug.lower())
        & (df["ward"] == ward)
    ].copy()

    if sub.empty:
        return pd.Series(dtype=float)

    sub["period"] = sub["collection_date"].dt.to_period("M")
    ecoff_val = ecoff or ECOFF_TABLE.get((organism, drug.lower()), None)

    records = []
    for period, grp in sub.groupby("period"):
        n = len(grp)
        mic_absent = "mic_value" not in grp.columns or grp["mic_value"].isnull().all()

        if not mic_absent:
            mic_vals = grp["mic_value"].dropna()
            log2_mic = np.log2(mic_vals.clip(lower=1e-6))
            mic_log2_mean = float(log2_mic.mean())
            mic_p90 = float(np.percentile(log2_mic, 90))

            if ecoff_val is not None:
                f_ecoff = float((mic_vals > ecoff_val).mean())
            else:
                f_ecoff = float((grp["sir_result"] == "R").mean())

            # Shannon entropy on log2-MIC histogram (Eq. 4.4)
            hist, _ = np.histogram(log2_mic, bins=10)
            p = hist / hist.sum()
            p = p[p > 0]
            mic_entropy = float(-np.sum(p * np.log2(p)))
        else:
            # SIR-only fallback: use R fraction as f_ecoff proxy
            f_ecoff = float((grp["sir_result"] == "R").mean())
            mic_log2_mean = float("nan")
            mic_p90 = float("nan")
            mic_entropy = float("nan")

        records.append({
            "period": period,
            "mic_log2_mean": mic_log2_mean,
            "mic_p90": mic_p90,
            "f_ecoff": f_ecoff,
            "mic_entropy": mic_entropy,
            "n_isolates": n,
            "mic_absent": mic_absent,
        })

    result = pd.DataFrame(records).set_index("period")

    # Eq. 4.5: rolling 3-month trend of log2 MIC mean
    result["delta_mic_3mo"] = (
        result["mic_log2_mean"] - result["mic_log2_mean"].shift(3)
    ) / 3.0

    if len(result) < 30 and not result["mic_log2_mean"].isnull().all():
        logger.warning(
            "n < 30 isolates for %s/%s/%s — Laplace may underestimate tail probability",
            organism, drug, ward,
        )

    return result


# ---------------------------------------------------------------------------
# Feature 2 — ICU Severity Index (Section 4.2.2, Eq. 4.6)
# ---------------------------------------------------------------------------

def compute_icu_severity_index(
    df_icu: pd.DataFrame,
    w_occ: float = 0.5,
    w_sofa: float = 0.3,
    w_mv: float = 0.2,
) -> pd.Series:
    """
    ISI(t) = w_occ · Occ_ICU(t) + w_SOFA · SOFA(t) + w_MV · frac_MV(t)

    df_icu columns: period (monthly), icu_occupancy_rate, sofa_mean (optional),
                    mv_fraction (optional)
    Returns Series indexed by period.
    """
    df = df_icu.copy()

    if "sofa_mean" not in df.columns or df["sofa_mean"].isnull().all():
        logger.warning(
            "SOFA data unavailable — ISI degrading to occupancy-only (precision reduced)"
        )
        df["sofa_mean"] = 0.0
        w_occ_adj, w_sofa_adj, w_mv_adj = 1.0, 0.0, 0.0
    else:
        w_occ_adj, w_sofa_adj, w_mv_adj = w_occ, w_sofa, w_mv

    if "mv_fraction" not in df.columns or df["mv_fraction"].isnull().all():
        df["mv_fraction"] = 0.0
        if w_mv_adj > 0:
            w_occ_adj += w_mv_adj
            w_mv_adj = 0.0

    isi = (
        w_occ_adj * df["icu_occupancy_rate"]
        + w_sofa_adj * df["sofa_mean"]
        + w_mv_adj * df["mv_fraction"]
    )
    return isi.rename("icu_severity_index")


# ---------------------------------------------------------------------------
# Feature 3 — PTND Antibiotic Sales Pressure (Section 4.2.3, Eq. 4.7)
# ---------------------------------------------------------------------------

def compute_ptnd(
    df_pharmacy: pd.DataFrame,
    antibiotic_class: str,
) -> pd.Series:
    """
    PTND(t) = DDD(t)_drug / PatientDays(t)_ward × 1000   [Eq. 4.7]

    df_pharmacy columns: period, ddd_total (sum of DDD for class),
                         patient_days_ward
    """
    CLASS_DRUGS: Dict[str, List[str]] = {
        "carbapenem": ["meropenem", "imipenem", "ertapenem"],
        "cephalosporin_3g4g": ["ceftriaxone", "cefotaxime", "ceftazidime", "cefepime"],
        "fluoroquinolone": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
        "glycopeptide": ["vancomycin", "teicoplanin"],
        "colistin": ["colistin", "polymyxin_b"],
    }
    drugs = CLASS_DRUGS.get(antibiotic_class, [])
    if not drugs:
        raise ValueError(f"Unknown antibiotic class: {antibiotic_class}")

    mask = df_pharmacy["antibiotic"].str.lower().isin(drugs)
    agg = (
        df_pharmacy[mask]
        .groupby("period")
        .agg(ddd_total=("ddd", "sum"), patient_days=("patient_days_ward", "first"))
    )
    ptnd = (agg["ddd_total"] / agg["patient_days"].clip(lower=1)) * 1000
    return ptnd.rename(f"ptnd_{antibiotic_class}")


# ---------------------------------------------------------------------------
# Feature 4 — Policy Stringency Index (Section 4.2.4, Eq. 4.8)
# ---------------------------------------------------------------------------

POLICY_TYPES = [
    "prescription_restriction",
    "formulary_change",
    "otc_enforcement",
    "isolation_policy",
    "decolonisation_protocol",
]

POLICY_WEIGHTS: Dict[str, float] = {
    "prescription_restriction": 0.30,
    "formulary_change": 0.25,
    "otc_enforcement": 0.20,
    "isolation_policy": 0.15,
    "decolonisation_protocol": 0.10,
}


def compute_psi(
    df_policy: pd.DataFrame,
    optimal_lags: Optional[Dict[str, int]] = None,
) -> pd.Series:
    """
    PSI(t) = Σ_j w_j · d_j(t - ℓ*_j)    [Eq. 4.8]

    df_policy columns: period + one binary column per policy type.
    optimal_lags: dict {policy_name: lag_months} from lag_optimisation().
    """
    psi = pd.Series(0.0, index=df_policy["period"], dtype=float)
    lags = optimal_lags or {p: 0 for p in POLICY_TYPES}

    for policy, weight in POLICY_WEIGHTS.items():
        if policy not in df_policy.columns:
            continue
        lag = lags.get(policy, 0)
        signal = df_policy.set_index("period")[policy].shift(lag, freq="M")
        psi = psi.add(weight * signal, fill_value=0.0)

    return psi.rename("policy_stringency_index")


# ---------------------------------------------------------------------------
# Per-Feature Lag Optimisation (Section 4.2.5, Eq. 4.9)
# ---------------------------------------------------------------------------

def optimise_lags(
    feature_series: Dict[str, pd.Series],
    R_obs: pd.Series,
    lag_max: int = LAG_MAX_MONTHS,
) -> Dict[str, int]:
    """
    ℓ*_i = argmax_{ℓ ∈ [0, lag_max]} |ρ(X_i(t-ℓ), R(t))|    [Eq. 4.9]

    If |ρ| < 0.15, the feature is included with ℓ* = 0 and a logged warning.

    Parameters
    ----------
    feature_series : dict of feature_name → monthly Series (same index as R_obs)
    R_obs          : observed resistance fraction time series
    lag_max        : maximum lag to search (months)

    Returns
    -------
    dict of feature_name → optimal lag (months)
    """
    optimal_lags: Dict[str, int] = {}
    R = R_obs.dropna()

    for name, series in feature_series.items():
        best_lag = 0
        best_rho = 0.0

        for lag in range(0, lag_max + 1):
            shifted = series.shift(lag)
            common = R.index.intersection(shifted.dropna().index)
            if len(common) < 6:
                continue
            try:
                rho = float(np.corrcoef(
                    R.loc[common].values,
                    shifted.loc[common].values
                )[0, 1])
            except Exception:
                rho = 0.0

            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_lag = lag

        if abs(best_rho) < MIN_CROSS_CORR:
            logger.warning(
                "Feature '%s': |ρ|=%.3f < %.2f threshold — included with lag=0",
                name, best_rho, MIN_CROSS_CORR,
            )
            best_lag = 0

        optimal_lags[name] = best_lag
        logger.info("Feature '%s': optimal lag=%d months (|ρ|=%.3f)", name, best_lag, best_rho)

    return optimal_lags


# ---------------------------------------------------------------------------
# Feature matrix assembler
# ---------------------------------------------------------------------------

@dataclass
class FeatureMatrix:
    """
    Complete feature matrix for one (organism, drug, ward) combination,
    ready for Stage 2 Bayesian GLM.
    """
    organism: str
    drug: str
    ward: str
    antibiotic_class: str

    X: np.ndarray                   # shape (T, n_features)
    feature_names: List[str]
    R_obs: np.ndarray               # observed resistance fractions shape (T,)
    n_obs: np.ndarray               # isolate counts per month shape (T,)
    periods: List[str]              # ISO period strings

    optimal_lags: Dict[str, int] = field(default_factory=dict)
    precision_flags: List[str] = field(default_factory=list)


def build_feature_matrix(
    df_isolates: pd.DataFrame,
    df_pharmacy: Optional[pd.DataFrame],
    df_icu: Optional[pd.DataFrame],
    df_policy: Optional[pd.DataFrame],
    organism: str,
    drug: str,
    ward: str,
    antibiotic_class: str,
    ecoff: Optional[float] = None,
    lag_max: int = LAG_MAX_MONTHS,
) -> FeatureMatrix:
    """
    Assemble the complete 7-feature matrix for Stage 2 Bayesian GLM.

    Feature order (matches Section 4.2):
        0: mic_log2_mean
        1: mic_p90
        2: f_ecoff
        3: mic_entropy
        4: icu_severity_index
        5: ptnd_{antibiotic_class}
        6: policy_stringency_index
    """
    precision_flags: List[str] = []

    # --- Resistance fractions from isolate data ---
    sub = df_isolates[
        (df_isolates["organism_key"] == organism)
        & (df_isolates["antibiotic"].str.lower() == drug.lower())
        & (df_isolates["ward"] == ward)
    ].copy()
    sub["period"] = sub["collection_date"].dt.to_period("M")

    monthly = sub.groupby("period").agg(
        n_resistant=("sir_result", lambda s: (s == "R").sum()),
        n_total=("sir_result", "count"),
    )
    monthly["R_obs"] = monthly["n_resistant"] / monthly["n_total"].clip(lower=1)

    periods_index = monthly.index

    # --- Feature 1: MIC ---
    mic_df = compute_mic_features(df_isolates, organism, drug, ward, ecoff)
    mic_cols = ["mic_log2_mean", "mic_p90", "f_ecoff", "mic_entropy"]

    # --- Feature 2: ICU ---
    if df_icu is not None:
        isi = compute_icu_severity_index(df_icu)
    else:
        isi = pd.Series(0.0, index=periods_index, name="icu_severity_index")
        precision_flags.append("icu_severity_index: occupancy-only fallback (SOFA/MV unavailable)")

    # --- Feature 3: PTND ---
    if df_pharmacy is not None:
        ptnd = compute_ptnd(df_pharmacy, antibiotic_class)
    else:
        ptnd = pd.Series(0.0, index=periods_index, name=f"ptnd_{antibiotic_class}")
        precision_flags.append(f"ptnd_{antibiotic_class}: unavailable — zero-filled")

    # --- Feature 4: PSI ---
    if df_policy is not None:
        feature_raw = {
            "mic_log2_mean": mic_df.get("mic_log2_mean", pd.Series(dtype=float)),
            "isi": isi,
            "ptnd": ptnd,
        }
        lags = optimise_lags(feature_raw, monthly["R_obs"], lag_max=lag_max)
        psi = compute_psi(df_policy, optimal_lags=lags)
    else:
        psi = pd.Series(0.0, index=periods_index, name="policy_stringency_index")
        lags = {}
        precision_flags.append("policy_stringency_index: no policy data — zero-filled")

    # --- Align all features to the same monthly index ---
    feature_dict: Dict[str, pd.Series] = {}
    for col in mic_cols:
        if col in mic_df.columns:
            feature_dict[col] = mic_df[col].reindex(periods_index)
        else:
            feature_dict[col] = pd.Series(float("nan"), index=periods_index)

    feature_dict["icu_severity_index"] = isi.reindex(periods_index, fill_value=0.0)
    feature_dict[f"ptnd_{antibiotic_class}"] = ptnd.reindex(periods_index, fill_value=0.0)
    feature_dict["policy_stringency_index"] = psi.reindex(periods_index, fill_value=0.0)

    feature_names = list(feature_dict.keys())
    X = np.column_stack([feature_dict[f].fillna(0.0).values for f in feature_names])
    R_obs = monthly["R_obs"].values
    n_obs = monthly["n_total"].values

    if len(periods_index) == 0:
        raise ValueError(
            f"No data for organism={organism}, drug={drug}, ward={ward}"
        )

    return FeatureMatrix(
        organism=organism,
        drug=drug,
        ward=ward,
        antibiotic_class=antibiotic_class,
        X=X,
        feature_names=feature_names,
        R_obs=R_obs,
        n_obs=n_obs,
        periods=[str(p) for p in periods_index],
        optimal_lags=lags,
        precision_flags=precision_flags,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(42)
    T = 24  # 24 months
    dates = pd.date_range("2023-01-01", periods=T, freq="MS")

    # Synthetic isolate dataframe
    n_isolates = 50
    records = []
    for d in dates:
        for i in range(n_isolates):
            sir = rng.choice(["R", "S"], p=[0.65, 0.35])
            records.append({
                "isolate_id": f"ISO_{d.strftime('%Y%m')}_{i:03d}",
                "collection_date": d + pd.Timedelta(days=int(rng.integers(0, 28))),
                "ward": "ICU",
                "specimen_type": "blood",
                "organism_eucast": "Klebsiella pneumoniae",
                "antibiotic": "meropenem",
                "sir_result": sir,
                "mic_value": float(rng.choice([0.064, 0.125, 0.25, 0.5, 1.0, 2.0, 8.0, 32.0])),
                "patient_id": f"PT_{i:04d}",
            })

    df_raw = pd.DataFrame(records)
    val = validate_glass(df_raw)
    print(f"GLASS validation: {len(val.valid_records)} accepted, "
          f"{val.rejected_count} rejected, "
          f"{val.flagged_unknown_org} unknown_org")
    print(f"Rejection breakdown: {val.rejection_reasons}")

    fm = build_feature_matrix(
        df_isolates=val.valid_records,
        df_pharmacy=None,
        df_icu=None,
        df_policy=None,
        organism="K_pneumoniae",
        drug="meropenem",
        ward="ICU",
        antibiotic_class="carbapenem",
    )

    print(f"\nFeature matrix: shape={fm.X.shape}")
    print(f"Features: {fm.feature_names}")
    print(f"R_obs (first 6 months): {fm.R_obs[:6].round(3)}")
    print(f"Precision flags: {fm.precision_flags}")
    sys.exit(0)
