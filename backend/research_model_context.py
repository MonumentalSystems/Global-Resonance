"""Source-audited 2025-2026 research context for operational models.

This module intentionally separates observations that justify a monitoring
action from results that justify a numerical probability update.  None of the
papers represented here estimates a transferable solar-to-seismic coefficient
or a present-day geomagnetic-reversal probability.
"""

from __future__ import annotations

from typing import Iterable, Mapping


CASCADIA_NSAF_DOI = "10.1130/GES02857.1"
MENDOCINO_SLAB_DOI = "10.1126/science.aeb2407"
SOLAR_HELICITY_DOI = "10.3847/2041-8213/ae6cf8"
SOLAR_WIND_ENSEMBLE_DOI = "10.1029/2025SW004823"
TORSIONAL_OSCILLATION_DOI = "10.1038/s41598-025-34336-1"
SOLAR_ROSSBY_MODE_DOI = "10.1038/s41550-026-02794-w"
INNER_CORE_DOI = "10.1038/s41561-025-01642-2"
CORE_FLOW_DOI = "10.1029/2025GC012475"
EARLY_DYNAMO_DOI = "10.1038/s41586-025-09334-y"
IGRF14_DOI = "10.1186/s40623-025-02360-0"
IGRF14_DATA_DOI = "10.5281/zenodo.14218973"

# A deliberately broad screening box for the Cascadia megathrust.  A candidate
# match still requires authoritative event classification; the rectangle is
# not a fault-surface model.
CASCADIA_SCREEN = {
    "lat_min": 39.5,
    "lat_max": 51.5,
    "lon_min": -132.5,
    "lon_max": -122.0,
    "max_depth_km": 70.0,
    "min_magnitude": 8.0,
}

# Published USGS locations for the 27 LFE families used to map the isolated
# southern-Cascadia/Mendocino zone.  These are observations supporting the
# captured-fragment interpretation, not vertices of a slab surface.
MENDOCINO_LFE_FAMILY_COORDINATES = [
    [-123.597624, 40.063867],
    [-123.632715, 40.069832],
    [-123.684090, 40.036072],
    [-123.619751, 40.070557],
    [-123.593148, 40.063883],
    [-123.593563, 40.063208],
    [-123.630452, 40.070020],
    [-123.709562, 40.088265],
    [-123.621134, 40.069922],
    [-123.627124, 40.069836],
    [-123.675838, 40.040125],
    [-123.626107, 40.068445],
    [-123.623511, 40.069963],
    [-123.668848, 40.036499],
    [-123.632642, 40.069820],
    [-123.599935, 40.064124],
    [-123.620215, 40.070052],
    [-123.610710, 40.067786],
    [-123.686711, 40.072070],
    [-123.601424, 40.064315],
    [-123.606934, 40.027307],
    [-123.644653, 40.073580],
    [-123.556055, 40.053434],
    [-123.601245, 40.064563],
    [-123.619857, 40.069836],
    [-123.712524, 40.072095],
    [-123.532186, 40.036267],
]

MENDOCINO_LFE_FAMILY_DEPTHS_KM = [
    28.189,
    27.623,
    22.192,
    28.199,
    28.578,
    28.223,
    27.747,
    24.216,
    28.134,
    27.834,
    22.449,
    27.775,
    27.960,
    23.295,
    27.698,
    28.170,
    28.126,
    28.209,
    25.676,
    28.162,
    23.712,
    27.497,
    27.045,
    28.189,
    28.164,
    24.127,
    25.939,
]


def is_great_cascadia_candidate(event: Mapping) -> bool:
    """Screen a ComCat-like event for human confirmation as a great Cascadia event.

    Magnitude 8 is an implementation convention for a "great" earthquake, not
    a threshold estimated by Goldfinger et al.  The result must not be treated
    as an automated fault attribution.
    """

    try:
        magnitude = float(event["mag"])
        latitude = float(event["lat"])
        longitude = float(event["lon"])
        depth = float(event.get("depth", 0.0))
    except (KeyError, TypeError, ValueError):
        return False

    return (
        magnitude >= CASCADIA_SCREEN["min_magnitude"]
        and CASCADIA_SCREEN["lat_min"] <= latitude <= CASCADIA_SCREEN["lat_max"]
        and CASCADIA_SCREEN["lon_min"] <= longitude <= CASCADIA_SCREEN["lon_max"]
        and 0.0 <= depth <= CASCADIA_SCREEN["max_depth_km"]
    )


def cascadia_nsaf_advisories(events: Iterable[Mapping]) -> list[dict]:
    """Return monitoring-only compound-hazard advisories for candidate events."""

    advisories = []
    for event in events:
        if not is_great_cascadia_candidate(event):
            continue
        advisories.append(
            {
                "active": True,
                "level": "MONITORING ESCALATION",
                "trigger_event_id": event.get("id", ""),
                "trigger_candidate": "great Cascadia megathrust earthquake",
                "target": "northern San Andreas fault",
                "action": (
                    "seek authoritative USGS fault attribution and monitor the "
                    "northern San Andreas as a compound-hazard scenario"
                ),
                "probability": None,
                "timing_window": None,
                "model_boundary": (
                    "paleoseismic partial synchronization does not supply a "
                    "real-time conditional probability or a calibrated alert window"
                ),
                "screening_note": (
                    "geographic/depth/magnitude screening is an implementation "
                    "convention and requires agency confirmation"
                ),
                "source": f"https://doi.org/{CASCADIA_NSAF_DOI}",
            }
        )
    return advisories


def research_model_context() -> dict:
    """Return the source and decision boundary for each incorporated result."""

    return {
        "updated_at": "2026-08-07",
        "cascadia_northern_san_andreas": {
            "finding": (
                "10 of 18 southern-Cascadia event beds are temporally paired with "
                "northern San Andreas beds; eight doublets support possible stress "
                "triggering and partial synchronization, commonly Cascadia first"
            ),
            "dating_boundary": (
                "median paired age difference is about 60 years, comparable to age "
                "uncertainty; inferred lags range from minutes to decades"
            ),
            "operational_change": (
                "add monitoring escalation after a candidate great Cascadia event"
            ),
            "probability_change": False,
            "source": f"https://doi.org/{CASCADIA_NSAF_DOI}",
        },
        "mendocino_triple_junction_geometry": {
            "finding": (
                "dipping strike-slip low-frequency earthquakes support a former "
                "Farallon (Pioneer) slab fragment captured by the Pacific plate and "
                "translating north beneath westernmost North America"
            ),
            "operational_change": (
                "represent the fragment as regional geometry/context extending the "
                "slab interface, not as a probability multiplier"
            ),
            "probability_change": False,
            "source": f"https://doi.org/{MENDOCINO_SLAB_DOI}",
            "data": "https://doi.org/10.5066/P1TCKK7G",
            "feature": {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": MENDOCINO_LFE_FAMILY_COORDINATES,
                },
                "properties": {
                    "name": "southern Cascadia isolated LFE family locations",
                    "family_ids": list(range(1, 28)),
                    "depths_km": MENDOCINO_LFE_FAMILY_DEPTHS_KM,
                    "feature_role": (
                        "observations supporting the captured-fragment interpretation"
                    ),
                    "geometry_boundary": (
                        "family hypocenters are not a slab surface, fault polygon, "
                        "or rupture forecast"
                    ),
                    "source": "https://doi.org/10.5066/P1TCKK7G",
                    "source_revision_note": (
                        "USGS notes that family locations may be updated"
                    ),
                },
            },
        },
        "solar_helicity": {
            "finding": (
                "a 24-hour SHARP time-series study found a nonlinear R-value gate "
                "controlled by total and absolute current-helicity interaction"
            ),
            "operational_change": (
                "expose the helicity-interaction proxy for validation; do not alter "
                "the production flare score without temporal recalibration"
            ),
            "probability_change": False,
            "source": f"https://doi.org/{SOLAR_HELICITY_DOI}",
        },
        "geomagnetic_field": {
            "finding": "IGRF-14 supersedes IGRF-13 for the 2025-2030 main field",
            "operational_change": "use IGRF-14 degree-one coefficients and secular variation",
            "warning_change": False,
            "sources": [
                f"https://doi.org/{IGRF14_DOI}",
                f"https://doi.org/{IGRF14_DATA_DOI}",
            ],
        },
        "inner_core": {
            "finding": (
                "repeating-earthquake waveforms support both differential rotation "
                "and localized shallow-inner-core deformation"
            ),
            "operational_change": (
                "exclude inner-core rotation/deformation from reversal and short-term "
                "surface-hazard warning scores"
            ),
            "warning_change": False,
            "source": f"https://doi.org/{INNER_CORE_DOI}",
        },
        "reviewed_not_retuned": [
            {
                "topic": "ambient solar-wind ensemble warning",
                "decision": (
                    "supports probabilistic ensemble architecture, but excludes "
                    "CMEs and does not provide transferable weights for this monitor"
                ),
                "source": f"https://doi.org/{SOLAR_WIND_ENSEMBLE_DOI}",
            },
            {
                "topic": "solar dynamo and global Rossby modes",
                "decision": (
                    "update physical interpretation only; neither study supplies a "
                    "validated live predictor for the repository's alert scores"
                ),
                "sources": [
                    f"https://doi.org/{TORSIONAL_OSCILLATION_DOI}",
                    f"https://doi.org/{SOLAR_ROSSBY_MODE_DOI}",
                ],
            },
            {
                "topic": "core-surface flow and early-Earth dynamo",
                "decision": (
                    "reinforces inversion uncertainty and the absence of an "
                    "inner-core countdown; no present-day hazard coefficient added"
                ),
                "sources": [
                    f"https://doi.org/{CORE_FLOW_DOI}",
                    f"https://doi.org/{EARLY_DYNAMO_DOI}",
                ],
            },
        ],
        "global_seismic_zone_ratios_modified": False,
    }
