#!/usr/bin/env python3
"""
Live CME impact monitor — tracks predictions vs reality in real time.

Polls USGS earthquakes, SWPC Kp/solar wind, and logs everything
to output/monitor_log.csv for post-hoc analysis.

Run: python live_monitor.py
  (runs once per invocation — use with loop/cron/watch for continuous monitoring)
  python live_monitor.py --loop 600   (poll every 10 minutes)
"""

import json
import csv
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
LOG_FILE = os.path.join(OUTPUT_DIR, 'monitor_log.csv')
PREDICTION_FILE = os.path.join(OUTPUT_DIR, 'prediction_scorecard.txt')

# === PREDICTIONS (from cme_prediction_20260331.py) ===
CME_ARRIVAL = datetime(2026, 3, 31, 15, 7, tzinfo=timezone.utc)
FLARE_PEAK = datetime(2026, 3, 30, 3, 19, tzinfo=timezone.utc)
SUBSOLAR_LAT = 3.6
SUBSOLAR_LON = -46.8

PREDICTIONS = {
    'kp_spike': {
        'description': 'Kp spikes to 5+ within 6h of CME arrival',
        'window_start': CME_ARRIVAL - timedelta(hours=1),
        'window_end': CME_ARRIVAL + timedelta(hours=6),
        'threshold': 5.0,
        'metric': 'kp_max',
        'confirmed': None,
    },
    'schumann_shift': {
        'description': 'Schumann f1 shifts up ~4% at arrival',
        'window_start': CME_ARRIVAL - timedelta(hours=1),
        'window_end': CME_ARRIVAL + timedelta(hours=3),
        'threshold': 0.02,  # 2% minimum shift to confirm
        'metric': 'schumann_f1_shift',
        'confirmed': None,
        'note': 'Requires external Schumann data source',
    },
    'wavefront_enhancement': {
        'description': 'M5+ rate elevated in 60-100 deg band (Europe/Cascadia/Alaska/Iran)',
        'window_start': CME_ARRIVAL + timedelta(hours=6),
        'window_end': CME_ARRIVAL + timedelta(hours=24),
        'threshold': 1.2,  # 1.2x background
        'metric': 'eq_rate_60_100',
        'confirmed': None,
    },
    'subsolar_suppression': {
        'description': 'M4.5+ rate suppressed within 30 deg of subsolar',
        'window_start': CME_ARRIVAL,
        'window_end': CME_ARRIVAL + timedelta(hours=24),
        'threshold': 0.9,  # below 0.9x background
        'metric': 'eq_rate_0_30',
        'confirmed': None,
    },
    'global_m5_elevated': {
        'description': 'Global M5+ count elevated vs 30-day baseline',
        'window_start': CME_ARRIVAL + timedelta(hours=6),
        'window_end': CME_ARRIVAL + timedelta(hours=48),
        'threshold': 1.2,
        'metric': 'global_m5_ratio',
        'confirmed': None,
    },
}

# High-risk targets from the Jelly Ball model
HIGH_RISK_TARGETS = [
    ('Iceland', 64.1, -21.9, 63.1, 1.36),
    ('Central Italy', 42.5, 13.5, 65.9, 1.36),
    ('Istanbul', 41.0, 29.0, 76.9, 1.36),
    ('Cascadia', 45.5, -122.7, 77.6, 1.36),
    ('Iran', 38.1, 46.3, 90.2, 1.36),
    ('Alaska', 61.2, -149.9, 93.1, 1.36),
]

import math

def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = [math.radians(x) for x in [lat1, lon1, lat2, lon2]]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def fetch_json(url, timeout=15):
    """Fetch JSON from URL with error handling."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'GR-Monitor/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None


def fetch_kp():
    """Get current Kp from SWPC."""
    data = fetch_json('https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json')
    if not data or len(data) < 2:
        return None, None
    # Last entry is most recent; skip header row
    latest = data[-1]
    # Format: [time_tag, Kp, Kp_fraction, a_running, station_count]
    try:
        kp_val = float(latest[1])
        ts = latest[0]
        return ts, kp_val
    except (IndexError, ValueError, KeyError, TypeError):
        return None, None


def fetch_solar_wind():
    """Get current solar wind speed and Bz from SWPC."""
    mag = fetch_json('https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json')
    plasma = fetch_json('https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json')

    bz = None
    vsw = None
    density = None

    if mag and len(mag) > 2:
        try:
            # columns: time_tag, bx, by, bz, bt, ...
            last = mag[-1]
            bz = float(last[3]) if last[3] not in (None, '', 'null') else None
        except (IndexError, ValueError):
            pass

    if plasma and len(plasma) > 2:
        try:
            last = plasma[-1]
            density = float(last[1]) if last[1] not in (None, '', 'null') else None
            vsw = float(last[2]) if last[2] not in (None, '', 'null') else None
        except (IndexError, ValueError):
            pass

    return bz, vsw, density


def fetch_recent_earthquakes(hours_back=24, min_mag=4.5):
    """Get recent earthquakes from USGS."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_back)
    url = (f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson'
           f'&starttime={start.strftime("%Y-%m-%dT%H:%M:%S")}'
           f'&minmagnitude={min_mag}&orderby=time&limit=200')
    data = fetch_json(url)
    if not data or 'features' not in data:
        return []

    eqs = []
    for f in data['features']:
        props = f['properties']
        coords = f['geometry']['coordinates']
        eqs.append({
            'mag': props.get('mag'),
            'place': props.get('place', ''),
            'time': datetime.fromtimestamp(props['time']/1000, tz=timezone.utc),
            'lon': coords[0],
            'lat': coords[1],
            'depth': coords[2],
            'ang_dist': angular_distance(SUBSOLAR_LAT, SUBSOLAR_LON, coords[1], coords[0]),
        })
    return eqs


def classify_earthquake_zones(eqs):
    """Bin earthquakes by angular distance from subsolar point."""
    zones = {
        'eye_0_30': [],      # suppression zone
        'inner_30_60': [],
        'wavefront_60_100': [],  # peak enhancement
        'outer_100_140': [],
        'far_140_165': [],
        'antipodal_165_180': [],
    }
    for eq in eqs:
        d = eq['ang_dist']
        if d < 30:
            zones['eye_0_30'].append(eq)
        elif d < 60:
            zones['inner_30_60'].append(eq)
        elif d < 100:
            zones['wavefront_60_100'].append(eq)
        elif d < 140:
            zones['outer_100_140'].append(eq)
        elif d < 165:
            zones['far_140_165'].append(eq)
        else:
            zones['antipodal_165_180'].append(eq)
    return zones


def check_high_risk_hits(eqs, radius_deg=10, min_mag=4.5):
    """Check if any earthquakes occurred near high-risk targets."""
    hits = []
    for eq in eqs:
        if eq['mag'] < min_mag:
            continue
        for name, tlat, tlon, _, predicted_ratio in HIGH_RISK_TARGETS:
            dist = angular_distance(eq['lat'], eq['lon'], tlat, tlon)
            if dist < radius_deg:
                hits.append({
                    'target': name,
                    'eq_mag': eq['mag'],
                    'eq_place': eq['place'],
                    'eq_time': eq['time'],
                    'distance_deg': round(dist, 1),
                    'predicted_ratio': predicted_ratio,
                })
    return hits


def run_monitor():
    """Single monitoring pass."""
    now = datetime.now(timezone.utc)
    hours_since_flare = (now - FLARE_PEAK).total_seconds() / 3600
    hours_since_arrival = (now - CME_ARRIVAL).total_seconds() / 3600

    print(f"\n{'='*70}")
    print(f"LIVE MONITOR -- {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*70}")
    print(f"Hours since X1.4 flare:  {hours_since_flare:.1f}h")
    print(f"Hours since CME arrival:  {hours_since_arrival:.1f}h")

    if hours_since_arrival < 0:
        print(f"CME arrival in:          {-hours_since_arrival:.1f}h")
    else:
        phase = 'ARRIVAL' if hours_since_arrival < 6 else \
                'WAVEFRONT PROPAGATION' if hours_since_arrival < 18 else \
                'RELAXATION' if hours_since_arrival < 36 else 'POST-EVENT'
        print(f"Phase:                   {phase}")

    # --- Fetch data ---
    print("\n--- Geomagnetic State ---")
    kp_ts, kp_val = fetch_kp()
    bz, vsw, density = fetch_solar_wind()

    if kp_val is not None:
        storm = 'QUIET' if kp_val < 4 else 'ACTIVE' if kp_val < 5 else \
                'MINOR STORM' if kp_val < 6 else 'MODERATE STORM' if kp_val < 7 else \
                'STRONG STORM' if kp_val < 8 else 'SEVERE STORM'
        print(f"  Kp = {kp_val:.1f} ({storm})  at {kp_ts}")
    if bz is not None:
        bz_status = 'SOUTHWARD (geoeffective)' if bz < -5 else \
                    'weakly south' if bz < 0 else 'northward (quiet)'
        print(f"  Bz = {bz:.1f} nT ({bz_status})")
    if vsw is not None:
        sw_status = 'FAST (CME?)' if vsw > 600 else 'elevated' if vsw > 450 else 'normal'
        print(f"  V_sw = {vsw:.0f} km/s ({sw_status})")
    if density is not None:
        print(f"  Density = {density:.1f} /cm3")

    # --- Earthquake data ---
    print("\n--- Seismicity (last 24h, M4.5+) ---")
    eqs = fetch_recent_earthquakes(hours_back=24, min_mag=4.5)
    eqs_m5 = [e for e in eqs if e['mag'] >= 5.0]

    print(f"  Total M4.5+: {len(eqs)}")
    print(f"  Total M5.0+: {len(eqs_m5)}")

    zones = classify_earthquake_zones(eqs)
    print(f"\n  Zone distribution (from subsolar {SUBSOLAR_LAT:.1f}N, {SUBSOLAR_LON:.1f}E):")
    for zone_name, zone_eqs in zones.items():
        tag = ' ** SUPPRESSION' if 'eye' in zone_name and len(zone_eqs) == 0 else \
              ' ** ENHANCED' if 'wavefront' in zone_name and len(zone_eqs) > 3 else ''
        m5_in_zone = sum(1 for e in zone_eqs if e['mag'] >= 5.0)
        print(f"    {zone_name:>20}: {len(zone_eqs):3d} eqs (M5+: {m5_in_zone}){tag}")

    # --- High-risk target check ---
    hits = check_high_risk_hits(eqs, radius_deg=15, min_mag=4.5)
    if hits:
        print(f"\n  !! HIGH-RISK TARGET HITS:")
        for h in hits:
            print(f"    {h['target']}: M{h['eq_mag']:.1f} at {h['eq_time'].strftime('%H:%M')}Z "
                  f"({h['eq_place']}, {h['distance_deg']}deg away)")
    else:
        print(f"\n  No high-risk target hits in last 24h")

    # --- Notable events ---
    print(f"\n--- Notable Events (M5.5+, last 24h) ---")
    big_eqs = sorted([e for e in eqs if e['mag'] >= 5.5], key=lambda x: -x['mag'])
    if big_eqs:
        for eq in big_eqs[:10]:
            print(f"  M{eq['mag']:.1f} | {eq['time'].strftime('%b %d %H:%M')}Z | "
                  f"{eq['place']} | {eq['ang_dist']:.0f}deg from subsolar | "
                  f"depth {eq['depth']:.0f}km")
    else:
        print("  None")

    # --- Prediction scorecard ---
    print(f"\n--- Prediction Scorecard ---")
    for key, pred in PREDICTIONS.items():
        in_window = pred['window_start'] <= now <= pred['window_end']
        past_window = now > pred['window_end']
        status = 'ACTIVE' if in_window else 'PENDING' if not past_window else 'EXPIRED'

        # Auto-check predictions where we have data
        if key == 'kp_spike' and kp_val is not None:
            if kp_val >= pred['threshold']:
                pred['confirmed'] = True
                status = 'CONFIRMED'
            elif past_window:
                pred['confirmed'] = False
                status = 'NOT CONFIRMED'

        if key == 'wavefront_enhancement':
            wf_count = len(zones['wavefront_60_100'])
            total = len(eqs)
            if total > 0:
                wf_fraction = wf_count / total
                # wavefront is 60-100deg band = ~30% of sphere by solid angle
                expected_fraction = 0.30
                ratio = wf_fraction / expected_fraction if expected_fraction > 0 else 0
                if in_window and ratio >= pred['threshold']:
                    pred['confirmed'] = True
                    status = f'CONFIRMED (ratio={ratio:.2f})'

        if key == 'subsolar_suppression':
            eye_count = len(zones['eye_0_30'])
            total = len(eqs)
            if total > 0 and in_window:
                eye_fraction = eye_count / total
                expected_fraction = 0.07  # 0-30deg is ~7% of sphere
                ratio = eye_fraction / expected_fraction if expected_fraction > 0 else 0
                if ratio < pred['threshold']:
                    pred['confirmed'] = True
                    status = f'CONFIRMED (ratio={ratio:.2f})'

        print(f"  [{status:>16}] {pred['description']}")

    # --- Log to CSV ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(['timestamp', 'hours_since_flare', 'hours_since_arrival',
                           'kp', 'bz', 'vsw', 'density',
                           'eq_total_24h', 'eq_m5_24h',
                           'eq_eye', 'eq_wavefront', 'eq_antipodal',
                           'high_risk_hits'])
        writer.writerow([
            now.isoformat(),
            round(hours_since_flare, 1),
            round(hours_since_arrival, 1),
            kp_val, bz, vsw, density,
            len(eqs), len(eqs_m5),
            len(zones['eye_0_30']),
            len(zones['wavefront_60_100']),
            len(zones['antipodal_165_180']),
            len(hits),
        ])

    print(f"\n  Logged to {LOG_FILE}")
    print(f"{'='*70}")
    return {
        'kp': kp_val, 'bz': bz, 'vsw': vsw,
        'eqs': len(eqs), 'eqs_m5': len(eqs_m5),
        'zones': {k: len(v) for k, v in zones.items()},
        'hits': hits,
    }


def refresh_dashboard():
    """Regenerate the visual dashboard."""
    try:
        from dashboard import build_dashboard
        build_dashboard()
    except Exception as e:
        print(f"  [WARN] Dashboard refresh failed: {e}")


if __name__ == '__main__':
    if '--loop' in sys.argv:
        idx = sys.argv.index('--loop')
        interval = int(sys.argv[idx+1]) if idx+1 < len(sys.argv) else 600
        print(f"Monitoring every {interval}s. Ctrl+C to stop.")
        print(f"Dashboard: output/dashboard.png (auto-refreshes)")
        while True:
            try:
                run_monitor()
                refresh_dashboard()
                print(f"  Next refresh in {interval}s...")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        run_monitor()
        refresh_dashboard()
