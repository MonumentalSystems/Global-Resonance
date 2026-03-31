#!/usr/bin/env python3
"""
Download Swarm satellite data for STEVE/Anti-STEVE events.
Uses VirES API (https://vires.services/).

Requires: pip install viresclient
First time: run `viresclient set_token https://vires.services/ows`
and paste your VirES access token.
"""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
DATA_DIR = Path(__file__).parent / "data"

print("Swarm Data Download")
print("=" * 50)

try:
    from viresclient import SwarmRequest
    print("viresclient installed OK")
except ImportError:
    print("Need: pip install viresclient")
    print("Then: viresclient set_token https://vires.services/ows")
    sys.exit(1)

# STEVE events to get Swarm data for
events = [
    # (start, end, label)
    ("2015-09-07T04:00:00", "2015-09-07T08:00:00", "STEVE_SuperDARN1"),
    ("2017-03-01T04:00:00", "2017-03-01T08:00:00", "STEVE_Maimaga"),
    ("2018-04-10T04:00:00", "2018-04-10T08:00:00", "STEVE_TREx"),
    ("2024-10-11T08:00:00", "2024-10-11T16:00:00", "STEVE_SouthAustralia"),
]

print(f"\nWill download Swarm EFI data for {len(events)} STEVE events")
print("Products: Langmuir Probe (Ne, Te) + Ion Drift (Vx, Vy, Vz)")

for start, end, label in events:
    cache = DATA_DIR / f"swarm_{label}.csv"
    if cache.exists():
        print(f"\n  Already cached: {cache.name}")
        continue

    print(f"\n  Requesting: {label} ({start} to {end})...")
    try:
        request = SwarmRequest("https://vires.services/ows")

        # Langmuir Probe data (electron density and temperature)
        request.set_collection("SW_OPER_EFIA_LP_1B")
        request.set_products(
            measurements=["Ne", "Te", "Vs"],
            auxiliaries=["QDLat", "QDLon", "MLT"],
        )

        data = request.get_between(start, end)
        df = data.as_dataframe()

        if len(df) > 0:
            df.to_csv(cache)
            print(f"  Saved: {cache.name} ({len(df)} records)")
        else:
            print(f"  No data for this period")

    except Exception as e:
        print(f"  FAILED: {e}")
        if "token" in str(e).lower() or "auth" in str(e).lower():
            print("  Need to set VirES token:")
            print("  1. Register at https://vires.services/")
            print("  2. Run: viresclient set_token https://vires.services/ows")
            break

print("\nDone.")
print("Note: VirES also provides FAC (Field-Aligned Currents) and")
print("AEBS (Auroral Electrojet Boundary) products which could")
print("directly identify the KT boundaries during STEVE events.")
