#!/usr/bin/env python3
"""Source-audited status of geomagnetic-reversal forecasting.

This replaces a legacy extrapolation that presented a 500-2,000 year range as
a best estimate.  Neither IGRF secular variation nor observed inner-core
rotation/deformation provides a defensible reversal countdown.
"""

try:
    from research_model_context import IGRF14_DOI, INNER_CORE_DOI
except ImportError:  # supports `python -m backend.next_reversal`
    from backend.research_model_context import IGRF14_DOI, INNER_CORE_DOI


def reversal_forecast_status() -> dict:
    return {
        "forecast_available": False,
        "next_reversal_date": None,
        "reason": (
            "geomagnetic reversals are aperiodic and the 2025-2030 IGRF secular-"
            "variation forecast cannot be extrapolated into a reversal date"
        ),
        "inner_core_warning_input": False,
        "inner_core_boundary": (
            "annual-scale differential rotation and localized deformation are "
            "deep-Earth observations, not established reversal precursors"
        ),
        "appropriate_inputs": [
            "IGRF-14 and subsequent IAGA field models",
            "ground-observatory and satellite magnetic measurements",
            "paleomagnetic reversal statistics with explicit uncertainty",
        ],
        "sources": [
            f"https://doi.org/{IGRF14_DOI}",
            f"https://doi.org/{INNER_CORE_DOI}",
        ],
    }


if __name__ == "__main__":
    status = reversal_forecast_status()
    print("GEOMAGNETIC REVERSAL FORECAST STATUS")
    print("Forecast available:", status["forecast_available"])
    print("Next reversal date:", status["next_reversal_date"])
    print("Reason:", status["reason"])
    print("Inner-core motion used in warnings:", status["inner_core_warning_input"])
    print("Boundary:", status["inner_core_boundary"])
