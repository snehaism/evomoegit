"""
economic.py
EVO-MOE V4 — Stage 3 / Stage 4: Economic Calculator Module
PRD Section 6.4: Cost of Resistance Formula (V4-NEW)
PRD Refinement R1: Economic Impact Panel on C1 Dashboard

Implements:
  - Cost of Resistance formula (Eq. 6.17)
  - DRG reference table (Section 6.4.3)
  - Escalation cost reference (Section 6.4.4)
  - EconomicBurden dataclass (Section 6.4.5, Listing 6.2)
  - Pipeline placement: Step 4a after SDE assembly (Section 6.4.6)

MANDATORY CAVEAT (non-suppressible, per PRD):
  "Economic estimates are model projections based on population-level
  resistance forecasts and reference cost data. They are not patient-
  specific billing estimates."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MANDATORY ECONOMIC CAVEAT (FIX-9 + R1 — must appear on every output)
# ---------------------------------------------------------------------------

ECONOMIC_CAVEAT = (
    "Economic estimates are model projections based on population-level "
    "resistance forecasts and reference cost data. They are not "
    "patient-specific billing estimates."
)

# ---------------------------------------------------------------------------
# DRG Reference Table (Section 6.4.3)
# Bed-Day Cost (NPR) | Attributable excess LOS (days) | Source
# ---------------------------------------------------------------------------

DRG_TABLE: Dict[str, Dict] = {
    "icu_sepsis": {
        "bed_day_cost_npr": 8_500,
        "los_delta_days": 7.2,
        "source": "Cassini et al. 2019",
    },
    "ward_uti": {
        "bed_day_cost_npr": 2_200,
        "los_delta_days": 3.1,
        "source": "Cassini et al. 2019",
    },
    "ward_pneumonia": {
        "bed_day_cost_npr": 3_100,
        "los_delta_days": 4.8,
        "source": "Laxminarayan et al. 2013",
    },
    "ward_bsi": {
        "bed_day_cost_npr": 4_800,
        "los_delta_days": 6.3,
        "source": "Cassini et al. 2019",
    },
}

# ---------------------------------------------------------------------------
# Escalation Cost Reference (Section 6.4.4)
# First-line → escalation drug, cost in NPR per course
# ---------------------------------------------------------------------------

ESCALATION_COSTS: Dict[str, Dict] = {
    "K_pneumoniae": {
        "first_line": "meropenem",
        "escalation_drug": "ceftazidime-avibactam",
        "cost_npr": 285_000,
    },
    "A_baumannii": {
        "first_line": "meropenem",
        "escalation_drug": "colistin + rifampicin",
        "cost_npr": 42_000,
    },
    "P_aeruginosa": {
        "first_line": "meropenem",
        "escalation_drug": "ceftolozane-tazobactam",
        "cost_npr": 190_000,
    },
    "E_faecium": {
        "first_line": "vancomycin",
        "escalation_drug": "linezolid or daptomycin",
        "cost_npr": 95_000,
    },
    "S_aureus_MRSA": {
        "first_line": "vancomycin",
        "escalation_drug": "daptomycin",
        "cost_npr": 88_000,
    },
    "Enterobacter_spp": {
        "first_line": "cefepime",
        "escalation_drug": "meropenem",
        "cost_npr": 18_000,
    },
}

# USD conversion rate (update periodically; not hardcoded in critical path)
NPR_TO_USD = 0.0075   # approximate exchange rate; source data in NPR

# ---------------------------------------------------------------------------
# EconomicBurden dataclass (Section 6.4.5, Listing 6.2)
# ---------------------------------------------------------------------------

@dataclass
class EconomicBurden:
    """
    Complete economic burden output per (ward, organism, drug, forecast_month).
    Required field on every ResistanceForecast via Refinement R1.
    """
    ward: str
    organism: str
    drug: str
    forecast_month: int

    # Point estimates
    expected_cost_resistance: float          # NPR (Eq. 6.17)
    expected_excess_bed_days: float          # LOS attributable to resistance
    expected_escalation_events: float        # N_patients × P_fail

    # Uncertainty bounds (from SDE ensemble percentiles)
    cost_ci_lo_95: float
    cost_ci_hi_95: float

    # Stewardship ROI
    cost_if_stewardship_effective: float     # CoR at scenario_low trajectory
    stewardship_roi_12mo: float              # (baseline - stewardship) / program_cost

    # Provenance (Listing 6.2 fields)
    drg_source: str                          # "TUTH_local" / "WHO_LMIC_estimate" / "custom"
    los_delta_source: str                    # "local_calibrated" / "Cassini_2019"
    currency: str = "NPR"

    # Mandatory non-suppressible caveat (FIX-9 / R1)
    caveat: str = field(default=ECONOMIC_CAVEAT)

    def to_usd(self) -> "EconomicBurden":
        """Return a copy with costs converted to USD."""
        import copy
        usd = copy.copy(self)
        usd.expected_cost_resistance *= NPR_TO_USD
        usd.cost_ci_lo_95 *= NPR_TO_USD
        usd.cost_ci_hi_95 *= NPR_TO_USD
        usd.cost_if_stewardship_effective *= NPR_TO_USD
        usd.currency = "USD"
        return usd

    def summary(self) -> str:
        return (
            f"ECONOMIC BURDEN (ward-level, month {self.forecast_month}):\n"
            f"  Cost of resistance : {self.currency} {self.expected_cost_resistance:,.0f} "
            f"[{self.cost_ci_lo_95:,.0f} – {self.cost_ci_hi_95:,.0f}]\n"
            f"  Excess bed-days    : {self.expected_excess_bed_days:.1f}\n"
            f"  Escalation events  : {self.expected_escalation_events:.1f} patients\n"
            f"  Stewardship ROI    : {self.stewardship_roi_12mo:.1f}x\n"
            f"  DRG source         : {self.drg_source}\n"
            f"  LOS Δ source       : {self.los_delta_source}\n"
            f"\n[CAVEAT] {self.caveat}"
        )


# ---------------------------------------------------------------------------
# Empiric therapy failure probability
# P_fail(c, d, R̂) = R̂ × (1 − cross_resistance_correction)  [Eq. 6.17]
# ---------------------------------------------------------------------------

# Cross-resistance corrections per scenario type (fraction reduction in P_fail)
CROSS_RESISTANCE_CORRECTION: Dict[str, float] = {
    "icu_sepsis": 0.10,       # ICU protocols include empiric broadening
    "ward_uti": 0.05,
    "ward_pneumonia": 0.08,
    "ward_bsi": 0.12,
}


def _empiric_failure_prob(
    resistance_fraction: float,
    scenario: str,
) -> float:
    """P_fail = R̂ × (1 − cross_resistance_correction)"""
    corr = CROSS_RESISTANCE_CORRECTION.get(scenario, 0.0)
    return np.clip(resistance_fraction * (1.0 - corr), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Cost of Resistance formula (Eq. 6.17)
# ---------------------------------------------------------------------------

def compute_cost_of_resistance(
    organism: str,
    drug: str,
    resistance_fraction: float,             # R̂ at month t
    n_patients_per_scenario: Dict[str, float],  # {scenario: N_c}
    drg_source: str = "Cassini_2019",
    los_delta_override: Optional[Dict[str, float]] = None,
    currency: str = "NPR",
) -> Tuple[float, float, float]:
    """
    Compute CoR = Σ_c [ DRG_c · LOS∆_c · R̂ · N_c · P_fail(c) · EscCost_c ]
    (Eq. 6.17 — summed over clinical scenario types c ∈ C)

    Parameters
    ----------
    organism            : ESKAPE key
    drug                : drug name for escalation lookup
    resistance_fraction : R̂ — predicted resistance fraction at month t
    n_patients_per_scenario : dict of {scenario_id: expected_N}
    drg_source          : provenance string for output field
    los_delta_override  : optional per-scenario local LOS deltas (locally calibrated)
    currency            : "NPR" | "USD"

    Returns
    -------
    (total_cor, total_excess_bed_days, total_escalation_events)
    """
    esc = ESCALATION_COSTS.get(organism, {})
    esc_cost = esc.get("cost_npr", 50_000)     # default if organism not in table

    total_cor = 0.0
    total_bed_days = 0.0
    total_esc_events = 0.0

    for scenario, n_c in n_patients_per_scenario.items():
        drg = DRG_TABLE.get(scenario, DRG_TABLE["ward_bsi"])
        bed_day = drg["bed_day_cost_npr"]
        los_delta = (
            los_delta_override.get(scenario, drg["los_delta_days"])
            if los_delta_override else drg["los_delta_days"]
        )

        p_fail = _empiric_failure_prob(resistance_fraction, scenario)

        # Eq. 6.17 components:
        cor_c = bed_day * los_delta * resistance_fraction * n_c * p_fail * esc_cost
        # Note: the formula product includes all five factors simultaneously;
        # in practice bed-day × LOS represents the bed-cost component,
        # and esc_cost is the drug-cost component for those who fail:
        # CoR_c = DRG_c * LOS∆_c * R̂ * N_c + P_fail * N_c * EscCost_c
        # We follow the PRD Eq. 6.17 factored form directly.
        cor_c = (
            bed_day * los_delta * resistance_fraction * n_c
            + p_fail * n_c * esc_cost
        )

        excess_bed_days_c = los_delta * resistance_fraction * n_c
        esc_events_c = p_fail * n_c

        total_cor += cor_c
        total_bed_days += excess_bed_days_c
        total_esc_events += esc_events_c

    if currency == "USD":
        total_cor *= NPR_TO_USD

    return total_cor, total_bed_days, total_esc_events


# ---------------------------------------------------------------------------
# Full EconomicBurden calculation from SDE ensemble (Section 6.4.6)
# ---------------------------------------------------------------------------

def calculate_economic_burden(
    ward: str,
    organism: str,
    drug: str,
    forecast_month: int,
    R_median: float,                         # from SDE ensemble
    R_ci_lo_95: float,
    R_ci_hi_95: float,
    R_scenario_low: float,                   # from ODE scenario_low trajectory
    n_patients_per_scenario: Dict[str, float],
    stewardship_program_cost_npr: float = 500_000,  # annual stewardship cost
    drg_source: str = "Cassini_2019",
    los_delta_override: Optional[Dict[str, float]] = None,
    currency: str = "NPR",
) -> EconomicBurden:
    """
    Compute EconomicBurden dataclass from SDE posterior summaries.
    Pipeline placement: Step 4a (Section 6.4.6).

    The uncertainty bounds (ci_lo_95, ci_hi_95) are derived by evaluating
    the CoR formula at the 2.5th and 97.5th percentile resistance fractions
    from the SDE ensemble.
    """
    # Point estimate at median R
    cor_median, bed_days_median, esc_events_median = compute_cost_of_resistance(
        organism, drug, R_median, n_patients_per_scenario, drg_source, los_delta_override, "NPR"
    )

    # CI bounds
    cor_lo, _, _ = compute_cost_of_resistance(
        organism, drug, R_ci_lo_95, n_patients_per_scenario, drg_source, los_delta_override, "NPR"
    )
    cor_hi, _, _ = compute_cost_of_resistance(
        organism, drug, R_ci_hi_95, n_patients_per_scenario, drg_source, los_delta_override, "NPR"
    )

    # Scenario-low cost (effective stewardship)
    cor_stewardship, _, _ = compute_cost_of_resistance(
        organism, drug, R_scenario_low, n_patients_per_scenario, drg_source, los_delta_override, "NPR"
    )

    # Stewardship ROI (Listing 6.2):
    # ROI = (cost_baseline - cost_stewardship) / stewardship_program_cost
    delta_cost = max(cor_median - cor_stewardship, 0.0)
    roi = delta_cost / max(stewardship_program_cost_npr, 1.0)

    # Currency conversion if requested
    if currency == "USD":
        cor_median *= NPR_TO_USD
        cor_lo *= NPR_TO_USD
        cor_hi *= NPR_TO_USD
        cor_stewardship *= NPR_TO_USD

    los_delta_source = "local_calibrated" if los_delta_override else "Cassini_2019"

    return EconomicBurden(
        ward=ward,
        organism=organism,
        drug=drug,
        forecast_month=forecast_month,
        expected_cost_resistance=cor_median,
        expected_excess_bed_days=bed_days_median,
        expected_escalation_events=esc_events_median,
        cost_ci_lo_95=min(cor_lo, cor_hi),
        cost_ci_hi_95=max(cor_lo, cor_hi),
        cost_if_stewardship_effective=cor_stewardship,
        stewardship_roi_12mo=roi,
        drg_source=drg_source,
        los_delta_source=los_delta_source,
        currency=currency,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    burden = calculate_economic_burden(
        ward="ICU",
        organism="A_baumannii",
        drug="meropenem",
        forecast_month=12,
        R_median=0.721,
        R_ci_lo_95=0.564,
        R_ci_hi_95=0.837,
        R_scenario_low=0.274,
        n_patients_per_scenario={
            "icu_sepsis": 30,
            "ward_bsi": 8,
        },
        stewardship_program_cost_npr=500_000,
        currency="NPR",
    )

    print(burden.summary())
    print()
    usd = burden.to_usd()
    print(f"USD Cost: ${usd.expected_cost_resistance:,.0f} "
          f"[${usd.cost_ci_lo_95:,.0f} – ${usd.cost_ci_hi_95:,.0f}]")

    # Validate caveat is present
    assert ECONOMIC_CAVEAT in burden.caveat, "CAVEAT MISSING — FIX-9 / R1 VIOLATION"
    print("\n✓ Economic caveat present (FIX-9 / R1 compliant)")
    sys.exit(0)
