"""Fault-local pore-pressure and rupture-barrier calculations.

The Jelly Ball model describes a proposed global forcing pattern.  This module
keeps that forcing separate from the local fault response documented at the
Gofar transform fault:

* a positive pore-pressure perturbation lowers effective normal stress and can
  favor nucleation or aseismic slip;
* rapid dilatancy or drainage can lower pore pressure during slip, raise
  effective normal stress, and arrest rupture at a damaged structural barrier.

No global earthquake-rate multiplier is inferred from the Gofar studies.
"""

from __future__ import annotations


SCIENCE_PAPER_DOI = "10.1126/science.ady6190"
COMPANION_MODEL_DOI = "10.1029/2025GL119319"
COMPANION_DATA_DOI = "10.5281/zenodo.17067488"


def effective_normal_stress_change_mpa(pore_pressure_change_mpa: float) -> float:
    """Return delta effective normal stress using sigma_eff = sigma_n - p."""
    return -float(pore_pressure_change_mpa)


def strengthening_from_pressure_drop(
    pore_pressure_drop_mpa: float,
    baseline_effective_normal_stress_mpa: float,
) -> dict:
    """Quantify the direct effective-stress increment from a pressure drop.

    This deliberately stops short of converting strengthening into earthquake
    probability or magnitude.  That transfer function is not constrained by
    the Gofar papers.
    """
    drop = float(pore_pressure_drop_mpa)
    baseline = float(baseline_effective_normal_stress_mpa)
    if drop < 0:
        raise ValueError("pore_pressure_drop_mpa must be non-negative")
    if baseline <= 0:
        raise ValueError("baseline_effective_normal_stress_mpa must be positive")

    fraction = drop / baseline
    return {
        "pore_pressure_drop_mpa": round(drop, 3),
        "effective_normal_stress_increase_mpa": round(drop, 3),
        "baseline_effective_normal_stress_mpa": round(baseline, 3),
        "strengthening_fraction": round(fraction, 4),
        "strengthening_pct": round(fraction * 100, 1),
        "rupture_rate_multiplier": None,
        "rate_multiplier_status": "not constrained by the cited studies",
    }


def pore_pressure_response(pore_pressure_change_mpa: float) -> dict:
    """Describe the sign-aware nucleation response to a local pressure change."""
    delta_p = float(pore_pressure_change_mpa)
    delta_sigma = effective_normal_stress_change_mpa(delta_p)
    if delta_sigma < 0:
        tendency = "promoting"
        interpretation = "effective normal stress is reduced"
    elif delta_sigma > 0:
        tendency = "inhibiting"
        interpretation = "effective normal stress is increased"
    else:
        tendency = "neutral"
        interpretation = "no effective-normal-stress change"

    return {
        "pore_pressure_change_mpa": round(delta_p, 9),
        "effective_normal_stress_change_mpa": round(delta_sigma, 9),
        "nucleation_tendency": tendency,
        "interpretation": interpretation,
        "rupture_propagation": "requires a calibrated local barrier profile",
    }


def gofar_reference_payload() -> dict:
    """Return the source-labeled Gofar reference scenario used by the UI."""
    strengthening = strengthening_from_pressure_drop(15.0, 50.0)
    return {
        "site": "Gofar transform fault, East Pacific Rise",
        "scope": "fault-local reference; not a global calibration",
        "tectonic_loading_mm_yr": 140,
        "large_event_recurrence_years": [5, 6],
        "structural_barrier": {
            "geometry": "multistrand fault with transtensional stepovers",
            "stepover_offset_m": [100, 400],
            "observed_role": "repeatedly arrests approximately M6 ruptures",
        },
        "interseismic_cycle": {
            "mechanism": "compaction and sealing raise pore pressure",
            "expected_response": "recurrent swarms and aseismic deformation",
        },
        "dynamic_cycle": {
            "mechanism": "dilatancy or transient drainage lowers pore pressure",
            **strengthening,
            "calibration": (
                "15 MPa drop is from the companion conceptual simulation; "
                "it is not a direct measurement or a universal Gofar value"
            ),
        },
        "sources": {
            "rupture_barriers": f"https://doi.org/{SCIENCE_PAPER_DOI}",
            "hydromechanical_model": f"https://doi.org/{COMPANION_MODEL_DOI}",
            "simulation_archive": f"https://doi.org/{COMPANION_DATA_DOI}",
        },
    }


def jellyball_hydromechanics_payload() -> dict:
    """Return the model-boundary statement included in Jelly Ball responses."""
    return {
        "model_change": "fault-local nucleation and rupture propagation are separated",
        "global_zone_ratios_modified": False,
        "reason": (
            "the Gofar studies constrain a local barrier mechanism but do not "
            "estimate a transferable global earthquake-rate multiplier"
        ),
        "sign_convention": "delta sigma_eff = -delta pore pressure",
        "gofar_reference": gofar_reference_payload(),
    }
