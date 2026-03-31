#!/usr/bin/env python3
"""
Pull near-real-time Swarm satellite magnetic field data.
Looks for CME compression signatures and ULF pulsations.
"""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

def fetch_swarm_mag(hours_back=24):
    """Pull Swarm Alpha MAG data for the last N hours."""
    from viresclient import SwarmRequest

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)

    print(f"Requesting Swarm Alpha MAG-L: {start.strftime('%Y-%m-%dT%H:%M')} to {end.strftime('%Y-%m-%dT%H:%M')} UTC")

    request = SwarmRequest("https://vires.services/ows")

    # MAG-L (1 Hz magnetic field, low-rate)
    request.set_collection("SW_OPER_MAGA_LR_1B")
    request.set_products(
        measurements=["F", "B_NEC"],  # F=total field, B_NEC=North/East/Center
        auxiliaries=["QDLat", "QDLon", "MLT", "OrbitNumber"],
        sampling_step="PT10S",  # 10-second sampling to keep data manageable
    )

    data = request.get_between(
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        end.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    df = data.as_dataframe()
    return df


def fetch_swarm_fac(hours_back=6):
    """Pull Swarm Field-Aligned Currents (sensitive to CME compression)."""
    from viresclient import SwarmRequest

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)

    print(f"Requesting Swarm Alpha FAC: {start.strftime('%Y-%m-%dT%H:%M')} to {end.strftime('%Y-%m-%dT%H:%M')} UTC")

    request = SwarmRequest("https://vires.services/ows")

    request.set_collection("SW_OPER_FACATMS_2F")
    request.set_products(
        measurements=["FAC", "IRC"],
        auxiliaries=["QDLat", "QDLon", "MLT"],
    )

    data = request.get_between(
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        end.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    df = data.as_dataframe()
    return df


def analyze_mag(df):
    """Analyze magnetic field for CME signatures."""
    if df is None or len(df) == 0:
        print("  No MAG data available")
        return

    print(f"\n  Records: {len(df)}")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")

    # Total field F
    f_mean = df['F'].mean()
    f_std = df['F'].std()
    f_max = df['F'].max()
    f_min = df['F'].min()
    print(f"\n  Total field F:")
    print(f"    Mean: {f_mean:.1f} nT")
    print(f"    Std:  {f_std:.1f} nT")
    print(f"    Range: {f_min:.1f} - {f_max:.1f} nT")

    # Look for sudden changes (>100 nT in 1 minute = compression)
    df_sorted = df.sort_index()
    f_diff = df_sorted['F'].diff()
    big_jumps = f_diff[f_diff.abs() > 50]
    if len(big_jumps) > 0:
        print(f"\n  !! SUDDEN FIELD CHANGES (|dF| > 50 nT):")
        for ts, val in big_jumps.items():
            print(f"    {ts}: dF = {val:+.1f} nT")
    else:
        print(f"\n  No sudden field changes detected (all |dF| < 50 nT)")

    # B_NEC components
    if 'B_NEC' in df.columns:
        # B_NEC is a 3-element array [N, E, C]
        try:
            bnec = np.stack(df['B_NEC'].values)
            bn, be, bc = bnec[:,0], bnec[:,1], bnec[:,2]
            print(f"\n  B_NEC components:")
            print(f"    B_N: {bn.mean():.1f} +/- {bn.std():.1f} nT")
            print(f"    B_E: {be.mean():.1f} +/- {be.std():.1f} nT")
            print(f"    B_C: {bc.mean():.1f} +/- {bc.std():.1f} nT")

            # ULF: look for oscillations in the Pc3-5 band (2-600s period)
            # With 10s sampling, we can see periods 20s-600s
            # Standard deviation over rolling 5-minute window
            window = 30  # 30 samples = 5 minutes at 10s cadence
            if len(bn) > window:
                bn_rolling_std = np.array([bn[max(0,i-window):i].std()
                                          for i in range(window, len(bn))])
                ulf_max = bn_rolling_std.max()
                ulf_mean = bn_rolling_std.mean()
                print(f"\n  ULF activity (5-min rolling std of B_N):")
                print(f"    Mean: {ulf_mean:.2f} nT")
                print(f"    Max:  {ulf_max:.2f} nT")
                if ulf_max > 20:
                    print(f"    !! ELEVATED ULF - possible Pc4-5 pulsations")
                elif ulf_max > 10:
                    print(f"    Moderate ULF activity")
                else:
                    print(f"    Quiet ULF conditions")
        except Exception as e:
            print(f"  Could not parse B_NEC: {e}")

    # Orbit info
    if 'QDLat' in df.columns:
        auroral_passes = df[df['QDLat'].abs() > 60]
        print(f"\n  Auroral zone passes (|QDLat| > 60): {len(auroral_passes)} samples")

    # Save
    outfile = OUT_DIR / "swarm_live_mag.csv"
    df.to_csv(outfile)
    print(f"\n  Saved to {outfile}")


def analyze_fac(df):
    """Analyze Field-Aligned Currents for storm signatures."""
    if df is None or len(df) == 0:
        print("  No FAC data available")
        return

    print(f"\n  FAC records: {len(df)}")
    if 'FAC' in df.columns:
        fac_mean = df['FAC'].mean()
        fac_max = df['FAC'].abs().max()
        print(f"  FAC mean: {fac_mean:.3f} uA/m2")
        print(f"  FAC |max|: {fac_max:.3f} uA/m2")

        if fac_max > 10:
            print(f"  !! EXTREME FAC - major storm conditions")
        elif fac_max > 3:
            print(f"  !! Elevated FAC - substorm or CME compression")
        else:
            print(f"  Normal FAC levels")

    outfile = OUT_DIR / "swarm_live_fac.csv"
    df.to_csv(outfile)
    print(f"  Saved to {outfile}")


def main():
    print("=" * 60)
    print("SWARM LIVE DATA PULL")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    print("\n--- Magnetic Field (last 24h, 10s cadence) ---")
    try:
        mag_df = fetch_swarm_mag(hours_back=24)
        analyze_mag(mag_df)
    except Exception as e:
        print(f"  MAG fetch failed: {e}")
        mag_df = None

    print("\n--- Field-Aligned Currents (last 6h) ---")
    try:
        fac_df = fetch_swarm_fac(hours_back=6)
        analyze_fac(fac_df)
    except Exception as e:
        print(f"  FAC fetch failed: {e}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
  CME (1689 km/s, X1.4) predicted arrival: Mar 31 15:07Z +/- 7h
  Current solar wind: ~410 km/s (CME has NOT arrived yet)

  WHAT TO LOOK FOR IN SWARM DATA:
  1. Sudden F increase >100 nT = magnetopause compression
  2. B_N rapid oscillation = Pc5 pulsations (CME shock)
  3. FAC spike >3 uA/m2 = substorm onset
  4. ULF rolling std >20 nT = Schumann cavity perturbation

  The Schumann f1 shift correlates with magnetopause standoff:
    f1 ~ c / (2*pi*R_earth) * sqrt(1 + h/R_earth)
  where h = ionosphere height. CME compression lowers h,
  RAISING f1. Swarm at 450km sees this directly.
    """)


if __name__ == '__main__':
    main()
