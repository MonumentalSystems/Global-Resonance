#!/usr/bin/env python3
"""
Real-time CME impact prediction — March 31, 2026

Situation:
- X1.4 flare: 2026-03-30 03:19Z peak, source S27E45 (AR 14405)
- CME: 1689 km/s, half-angle 46°, predicted shock arrival 2026-03-31 ~15:07Z
- Vanuatu M7.3: 2026-03-28, depth 121 km (already occurred)
- Kp currently quiet (2.0-2.7), CME hasn't arrived yet

Framework prediction (Paper XXV "Jelly Ball"):
  Grade-0 (hour 0, flare SID): suppression at subsolar — ALREADY HAPPENED at 03:19Z Mar 30
  Grade-4 (hour +18): ionospheric relaxation peak — ~21:00Z Mar 30 — CHECK
  Grade-2 (hour +36-42): CME mechanical impact — ~15:00Z Mar 31 — INCOMING

The subsolar point at CME arrival (~15:07Z Mar 31) determines WHERE the
seismic enhancement/suppression pattern centers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

# === Constants from the framework ===
J_C = 2 / np.pi  # KT critical stiffness
EARTH_RADIUS_KM = 6371.0

def subsolar_point(dt_utc):
    """Calculate subsolar latitude and longitude at a given UTC time.

    Latitude: solar declination (varies ±23.44° with season)
    Longitude: depends on UTC hour (the sun is overhead at local noon)
    """
    day_of_year = dt_utc.timetuple().tm_yday
    # Solar declination (approximate)
    declination = 23.44 * np.sin(np.radians((360/365) * (day_of_year - 81)))
    # Subsolar longitude: at 12:00 UTC, subsolar is at 0°E
    # Each hour before noon = +15°E, each hour after = -15°E
    hour_fraction = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600
    subsolar_lon = (12 - hour_fraction) * 15  # degrees East
    if subsolar_lon > 180:
        subsolar_lon -= 360
    if subsolar_lon < -180:
        subsolar_lon += 360
    return declination, subsolar_lon

def angular_distance(lat1, lon1, lat2, lon2):
    """Great-circle angular distance in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

def jelly_ball_prediction(subsolar_lat, subsolar_lon, target_lat, target_lon,
                           flare_class='X1.4', cme_speed=1689):
    """Predict seismic response at a target location given CME impact geometry.

    Returns dict with:
    - angular_distance: degrees from subsolar point
    - zone: 'eye', 'wavefront', 'flank', 'antipodal'
    - rate_ratio: predicted earthquake rate / background rate
    - timing: hours after CME arrival for peak effect
    """
    ang_dist = angular_distance(subsolar_lat, subsolar_lon, target_lat, target_lon)

    # Jelly Ball spatial structure (from Paper XXV empirical calibration)
    if ang_dist < 30:
        zone = 'eye (suppression)'
        ratio = 0.85
    elif ang_dist < 60:
        zone = 'inner wavefront'
        ratio = 1.05
    elif ang_dist < 100:
        zone = 'wavefront (peak enhancement)'
        ratio = 1.36 if 'X' in flare_class else 1.18
    elif ang_dist < 140:
        zone = 'outer flank'
        ratio = 1.10
    elif ang_dist < 165:
        zone = 'far flank'
        ratio = 1.05
    else:
        zone = 'antipodal (reconvergence)'
        ratio = 1.16

    # CME speed affects timing: faster CME = earlier mechanical arrival
    # Already accounted for by DONKI prediction (15:07Z)
    # Post-arrival mechanical coupling takes ~6-12 hours to propagate through crust
    timing_hours = 6 + (ang_dist / 180) * 12  # 6h at subsolar, 18h at antipode

    return {
        'angular_distance_deg': round(ang_dist, 1),
        'zone': zone,
        'rate_ratio': ratio,
        'peak_hours_after_arrival': round(timing_hours, 1),
    }


def predict_schumann_response(cme_speed, bz_current, kp_current):
    """Predict Schumann resonance changes from incoming CME.

    The Schumann resonances (7.83, 14.3, 20.8, 27.3, 33.8 Hz) are
    eigenfrequencies of the Earth-ionosphere cavity. CME compression
    changes the cavity height, shifting the frequencies.

    From the KT framework:
    - CME compression raises ionosphere D-layer → cavity shrinks → freq UP
    - This corresponds to J crossing J_c from above (ordered → disordered)
    - The Schumann amplitude spikes as the cavity Q-factor drops
    - After passage, relaxation takes 18-48 hours
    """
    # Pre-arrival: quiet conditions, normal Schumann
    schumann_base = [7.83, 14.3, 20.8, 27.3, 33.8]  # Hz

    # CME compression factor (empirical: ~0.1-1% shift per 100 km/s above 400)
    compression = max(0, (cme_speed - 400) / 100) * 0.003

    # Bz effect: southward IMF (Bz < 0) couples more strongly
    bz_factor = 1.0 + max(0, -bz_current) * 0.01

    predictions = {
        'pre_arrival': {
            'frequencies_hz': schumann_base,
            'amplitude': 'normal',
            'kp_expected': kp_current,
        },
        'arrival_hour_0': {
            'frequencies_hz': [f * (1 + compression * bz_factor) for f in schumann_base],
            'amplitude': 'spike (2-5x normal)',
            'kp_expected': min(9, kp_current + 4),  # Sudden impulse
            'schumann_shift_percent': round(compression * bz_factor * 100, 2),
        },
        'hour_6_12': {
            'frequencies_hz': [f * (1 + compression * 0.5) for f in schumann_base],
            'amplitude': 'elevated (1.5-3x)',
            'kp_expected': min(9, kp_current + 3),
        },
        'hour_18_relaxation': {
            'frequencies_hz': [f * (1 + compression * 0.1) for f in schumann_base],
            'amplitude': 'returning to normal',
            'kp_expected': max(1, kp_current + 1),
            'note': 'Grade-4 pseudoscalar relaxation window — peak seismic enhancement',
        },
        'hour_36_48': {
            'frequencies_hz': schumann_base,  # back to normal
            'amplitude': 'normal',
            'kp_expected': kp_current,
        },
    }
    return predictions


def main():
    print("=" * 70)
    print("CME IMPACT PREDICTION — March 31, 2026")
    print("=" * 70)

    # === CURRENT SITUATION ===
    flare_peak = datetime(2026, 3, 30, 3, 19)  # X1.4 peak
    cme_arrival = datetime(2026, 3, 31, 15, 7)  # DONKI prediction
    now = datetime(2026, 3, 31, 16, 0)  # approximate current time
    cme_speed = 1689  # km/s

    print(f"\nX1.4 flare peak:    {flare_peak} UTC (AR 14405, S27E45)")
    print(f"CME speed:          {cme_speed} km/s (half-angle 46°)")
    print(f"Predicted arrival:  {cme_arrival} UTC")
    print(f"Hours since flare:  {(now - flare_peak).total_seconds()/3600:.1f}h")
    print(f"Hours to arrival:   {(cme_arrival - now).total_seconds()/3600:.1f}h")

    # === SUBSOLAR POINT AT CME ARRIVAL ===
    ss_lat, ss_lon = subsolar_point(cme_arrival)
    print(f"\nSubsolar point at arrival: {ss_lat:.1f}°N, {ss_lon:.1f}°E")

    # === GRADE TIMELINE ===
    print("\n" + "=" * 70)
    print("THREE-GRADE TIMELINE")
    print("=" * 70)

    grade0_time = flare_peak
    grade4_time = flare_peak + timedelta(hours=18)
    grade2_time = cme_arrival
    grade2_peak = cme_arrival + timedelta(hours=12)

    print(f"\nGrade-0 (SID suppression):  {grade0_time} UTC  [DONE] ALREADY PASSED")
    print(f"Grade-4 (iono relaxation):  {grade4_time} UTC  [DONE] ALREADY PASSED")
    print(f"Grade-2 (CME mechanical):   {grade2_time} UTC  <-- ARRIVING NOW")
    print(f"Grade-2 peak crustal:       {grade2_peak} UTC  <-- PEAK WINDOW")

    # === SPATIAL PREDICTIONS ===
    print("\n" + "=" * 70)
    print("SPATIAL PREDICTIONS (Jelly Ball Model)")
    print("=" * 70)

    targets = [
        ("Vanuatu (M7.3 aftershock zone)", -15.3, 167.5),
        ("Tokyo, Japan", 35.7, 139.7),
        ("Santiago, Chile", -33.4, -70.6),
        ("Istanbul, Turkey", 41.0, 29.0),
        ("Cascadia (Portland)", 45.5, -122.7),
        ("New Zealand", -41.3, 174.8),
        ("Indonesia (Sumatra)", -0.5, 101.5),
        ("Iran (Tabriz)", 38.1, 46.3),
        ("Alaska (Anchorage)", 61.2, -149.9),
        ("Central Italy", 42.5, 13.5),
        ("Azores", 38.7, -27.2),
        ("Iceland", 64.1, -21.9),
        ("Tonga", -21.2, -175.2),
        ("Philippines (Manila)", 14.6, 121.0),
        ("Ecuador (Quito)", -0.2, -78.5),
        ("Caribbean (Haiti)", 18.5, -72.3),
    ]

    print(f"\nSubsolar at arrival: {ss_lat:.1f}°N, {ss_lon:.1f}°E")
    print(f"{'Location':<35} {'Dist':>5} {'Zone':<30} {'Ratio':>6} {'Peak':>6}")
    print("-" * 90)

    results = []
    for name, lat, lon in targets:
        pred = jelly_ball_prediction(ss_lat, ss_lon, lat, lon, 'X1.4', cme_speed)
        results.append((name, pred))
        peak_time = cme_arrival + timedelta(hours=pred['peak_hours_after_arrival'])
        print(f"{name:<35} {pred['angular_distance_deg']:>5.1f}° "
              f"{pred['zone']:<30} {pred['rate_ratio']:>5.2f}x "
              f"+{pred['peak_hours_after_arrival']:.0f}h")

    # === HIGHEST RISK ZONES ===
    print("\n" + "=" * 70)
    print("HIGHEST RISK ZONES (rate_ratio > 1.2)")
    print("=" * 70)

    high_risk = [(n, p) for n, p in results if p['rate_ratio'] > 1.2]
    high_risk.sort(key=lambda x: x[1]['rate_ratio'], reverse=True)

    for name, pred in high_risk:
        peak_utc = cme_arrival + timedelta(hours=pred['peak_hours_after_arrival'])
        print(f"  !! {name}: {pred['rate_ratio']:.2f}x at {peak_utc.strftime('%b %d %H:%M')} UTC "
              f"({pred['angular_distance_deg']:.0f}° from subsolar)")

    # === SCHUMANN RESONANCE PREDICTION ===
    print("\n" + "=" * 70)
    print("SCHUMANN RESONANCE PREDICTION")
    print("=" * 70)

    schumann = predict_schumann_response(cme_speed, bz_current=-3, kp_current=2.0)

    for phase, data in schumann.items():
        print(f"\n  {phase}:")
        print(f"    Frequencies: {[round(f,2) for f in data['frequencies_hz']]} Hz")
        print(f"    Amplitude:   {data['amplitude']}")
        print(f"    Kp expected: {data['kp_expected']}")
        if 'schumann_shift_percent' in data:
            print(f"    Freq shift:  +{data['schumann_shift_percent']}%")
        if 'note' in data:
            print(f"    NOTE: {data['note']}")

    # === VANUATU SPECIFIC ===
    print("\n" + "=" * 70)
    print("VANUATU AFTERSHOCK PREDICTION")
    print("=" * 70)

    vanuatu_pred = jelly_ball_prediction(ss_lat, ss_lon, -15.3, 167.5, 'X1.4', cme_speed)
    print(f"\n  Vanuatu angular distance: {vanuatu_pred['angular_distance_deg']:.1f}°")
    print(f"  Zone: {vanuatu_pred['zone']}")
    print(f"  Rate ratio: {vanuatu_pred['rate_ratio']:.2f}x")

    # Omori law decay from M7.3 (Mar 28) + CME modulation
    days_since_mainshock = (now - datetime(2026, 3, 28)).total_seconds() / 86400
    omori_rate = 1.0 / (1 + days_since_mainshock)  # normalized Omori decay
    modulated_rate = omori_rate * vanuatu_pred['rate_ratio']

    print(f"\n  Days since M7.3: {days_since_mainshock:.1f}")
    print(f"  Omori decay factor: {omori_rate:.3f}")
    print(f"  CME-modulated rate: {modulated_rate:.3f} (Omori × Jelly Ball)")
    print(f"\n  Prediction: Vanuatu aftershock rate is {'ENHANCED' if vanuatu_pred['rate_ratio'] > 1.1 else 'SUPPRESSED' if vanuatu_pred['rate_ratio'] < 0.9 else 'NEAR BASELINE'}")
    print(f"  Watch window: {cme_arrival.strftime('%b %d %H:%M')} to "
          f"{(cme_arrival + timedelta(hours=24)).strftime('%b %d %H:%M')} UTC")

    # === TESTABLE PREDICTIONS ===
    print("\n" + "=" * 70)
    print("TESTABLE PREDICTIONS FOR NEXT 48 HOURS")
    print("=" * 70)

    predictions = [
        f"1. Kp spike to 5-7 around {cme_arrival.strftime('%b %d %H:%M')} UTC (CME shock arrival)",
        f"2. Schumann f1 shifts UP by ~{schumann['arrival_hour_0']['schumann_shift_percent']}% at arrival",
        "3. Schumann amplitude spikes 2-5x normal at arrival, decays over 18h",
        f"4. Global M5+ rate elevated 1.2-1.4x from {(cme_arrival+timedelta(hours=6)).strftime('%b %d %H:%M')} to {(cme_arrival+timedelta(hours=24)).strftime('%b %d %H:%M')} UTC",
        "5. Suppression zone (subsolar ±30°) shows DECREASED M4.5+ rate",
    ]
    for p in predictions:
        print(f"  {p}")

    # Sort by peak time for watch list
    print("\n" + "=" * 70)
    print("WATCH LIST (sorted by predicted peak time)")
    print("=" * 70)

    watch = [(n, p, cme_arrival + timedelta(hours=p['peak_hours_after_arrival']))
             for n, p in results if p['rate_ratio'] > 1.0]
    watch.sort(key=lambda x: x[2])

    for name, pred, peak_utc in watch:
        print(f"  {peak_utc.strftime('%b %d %H:%M')} UTC | {name:<30} | "
              f"{pred['rate_ratio']:.2f}x | {pred['angular_distance_deg']:.0f}°")


if __name__ == '__main__':
    main()
