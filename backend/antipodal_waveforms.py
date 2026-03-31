#!/usr/bin/env python3
"""
Antipodal Waveform Analysis
==============================
Download continuous seismic waveforms from IRIS FDSN for stations
near the antipodal point of major earthquakes. Look for:

1. Grade-0 (EM) precursor: any signal in the hours BEFORE the P-wave
   arrival that shouldn't be there mechanically
2. P-wave arrival timing: confirm the ~20 min travel time through Earth
3. Pre-shock noise level: is the background seismicity elevated before
   the main shock at the antipode?
4. Tidal strain: the continuous Earth tide signal visible in broadband data

Uses obspy for FDSN web service access to IRIS/GEOFON waveforms.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"


def antipodal_point(lat, lon):
    return -lat, ((lon + 180) % 360) - 180


def angular_distance_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def find_antipodal_stations(eq_lat, eq_lon, max_dist_deg=20):
    """Find IRIS stations near the antipodal point."""
    from obspy.clients.fdsn import Client
    client = Client("IRIS")

    alat, alon = antipodal_point(eq_lat, eq_lon)

    try:
        from obspy import UTCDateTime
        inventory = client.get_stations(
            latitude=alat, longitude=alon,
            maxradius=max_dist_deg,
            channel="BHZ",
            network="IU,II,AU,G,GE,CU",  # permanent global networks only
            level="station",
            startbefore=UTCDateTime(str(pd.Timestamp.now())),
            endafter=UTCDateTime("2000-01-01"),
        )
        stations = []
        for network in inventory:
            for station in network:
                dist = angular_distance_deg(alat, alon,
                                            station.latitude, station.longitude)
                stations.append({
                    "network": network.code,
                    "station": station.code,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "dist_to_antipode": dist,
                })
        return pd.DataFrame(stations).sort_values("dist_to_antipode")
    except Exception as e:
        print(f"  Station search failed: {e}")
        return pd.DataFrame()


def download_waveform(network, station, starttime, endtime):
    """Download BHZ waveform from IRIS."""
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime
    client = Client("IRIS")

    try:
        st = client.get_waveforms(
            network=network,
            station=station,
            location="*",
            channel="BHZ",
            starttime=UTCDateTime(str(starttime)),
            endtime=UTCDateTime(str(endtime)),
        )
        return st
    except Exception as e:
        print(f"    Download failed for {network}.{station}: {e}")
        return None


def analyze_event(eq_time, eq_lat, eq_lon, eq_mag, label):
    """
    For a major earthquake:
    1. Find stations near the antipodal point
    2. Download waveforms from 2h before to 2h after
    3. Look at noise levels, P-wave arrival, and any precursors
    """
    from obspy import UTCDateTime

    print(f"\n{'='*60}")
    print(f"  {label}: M{eq_mag} at ({eq_lat:.1f}, {eq_lon:.1f})")
    print(f"  Time: {eq_time}")
    print(f"{'='*60}")

    alat, alon = antipodal_point(eq_lat, eq_lon)
    print(f"  Antipodal point: ({alat:.1f}, {alon:.1f})")

    # Find stations
    stations = find_antipodal_stations(eq_lat, eq_lon, max_dist_deg=15)
    if len(stations) == 0:
        print("  No stations found near antipode")
        return None

    print(f"  Found {len(stations)} stations within 15 deg of antipode")
    print(f"  Closest: {stations.iloc[0]['network']}.{stations.iloc[0]['station']} "
          f"({stations.iloc[0]['dist_to_antipode']:.1f} deg)")

    # Try the closest 3 stations
    results = []
    for _, sta in stations.head(3).iterrows():
        net = sta["network"]
        stn = sta["station"]
        print(f"\n  Downloading {net}.{stn} ({sta['dist_to_antipode']:.1f} deg from antipode)...",
              flush=True)

        # Download: 2h before to 2h after
        t_start = pd.Timestamp(eq_time) - timedelta(hours=2)
        t_end = pd.Timestamp(eq_time) + timedelta(hours=2)

        st = download_waveform(net, stn, t_start, t_end)
        if st is None or len(st) == 0:
            continue

        tr = st[0]  # First trace
        print(f"    Got: {tr.stats.npts} samples at {tr.stats.sampling_rate} Hz")
        print(f"    Duration: {tr.stats.endtime - tr.stats.starttime:.0f} seconds")

        # Save raw waveform
        cache_name = f"waveform_{label}_{net}_{stn}.mseed"
        st.write(str(DATA_DIR / cache_name), format="MSEED")
        print(f"    Saved: {cache_name}")

        # Analyze: split into pre-shock and post-shock windows
        eq_utc = UTCDateTime(str(eq_time))

        # Expected P-wave arrival at antipode: ~20 minutes
        p_arrival_est = eq_utc + 20 * 60  # 20 min after origin

        # Windows
        pre_2h = tr.copy().trim(eq_utc - 7200, eq_utc - 3600)  # -2h to -1h
        pre_1h = tr.copy().trim(eq_utc - 3600, eq_utc)          # -1h to 0
        post_pw = tr.copy().trim(eq_utc, p_arrival_est)          # 0 to P arrival
        post_1h = tr.copy().trim(p_arrival_est, eq_utc + 3600)   # P arrival to +1h

        rms = lambda x: np.sqrt(np.mean(x.data.astype(float)**2)) if len(x.data) > 0 else 0

        rms_pre2h = rms(pre_2h)
        rms_pre1h = rms(pre_1h)
        rms_pre_pw = rms(post_pw)
        rms_post1h = rms(post_1h)

        print(f"\n    RMS amplitude by window:")
        print(f"      -2h to -1h (background):     {rms_pre2h:.0f}")
        print(f"      -1h to 0 (pre-shock):         {rms_pre1h:.0f}")
        print(f"      0 to +20min (pre-P arrival):  {rms_pre_pw:.0f}")
        print(f"      +20min to +1h (post-P):       {rms_post1h:.0f}")

        if rms_pre2h > 0:
            print(f"\n    Ratios to background:")
            print(f"      Pre-shock (-1h to 0) / BG: {rms_pre1h/rms_pre2h:.2f}x")
            print(f"      Pre-P (0 to +20min) / BG:  {rms_pre_pw/rms_pre2h:.2f}x")
            print(f"      Post-P / BG:               {rms_post1h/rms_pre2h:.2f}x")

            # The KEY test: is pre-shock elevated above background?
            if rms_pre1h / rms_pre2h > 1.2:
                print(f"      >>> PRE-SHOCK SIGNAL DETECTED: {rms_pre1h/rms_pre2h:.2f}x above background")

        results.append({
            "station": f"{net}.{stn}",
            "dist": sta["dist_to_antipode"],
            "rms_bg": rms_pre2h,
            "rms_pre": rms_pre1h,
            "rms_preP": rms_pre_pw,
            "rms_postP": rms_post1h,
        })

    if not results:
        return None

    rdf = pd.DataFrame(results)

    # Plot waveform for the best station
    if st is not None and len(st) > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        tr_plot = st[0].copy()
        tr_plot.filter("bandpass", freqmin=0.01, freqmax=1.0)  # filter to show long-period
        times = np.arange(tr_plot.stats.npts) / tr_plot.stats.sampling_rate / 3600  # hours
        times = times - 2  # center on earthquake time

        ax.plot(times, tr_plot.data, color="steelblue", lw=0.5)
        ax.axvline(0, color="red", lw=2, alpha=0.7, label="Main shock origin")
        ax.axvline(20/60, color="orange", lw=1.5, alpha=0.7, label="~P arrival (20 min)")
        ax.set_xlabel("Hours relative to main shock")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Antipodal Station {net}.{stn}: {label} M{eq_mag}\n"
                     f"Station is {sta['dist_to_antipode']:.1f} deg from antipode")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"waveform_{label}.png", dpi=150)
        print(f"\n  Saved: waveform_{label}.png")

    return rdf


def main():
    print("=" * 60)
    print("ANTIPODAL WAVEFORM ANALYSIS")
    print("Download seismic data at the far side of major earthquakes")
    print("=" * 60)

    # Major M8+ events with good station coverage at antipode
    events = [
        ("2011-03-11 05:46:24", 38.30, 142.37, 9.1, "Tohoku2011"),
        ("2010-02-27 06:34:11", -35.85, -72.72, 8.8, "Chile2010"),
        ("2012-04-11 08:38:36", 2.31, 93.06, 8.6, "IndianOcean2012"),
        ("2008-05-12 06:28:01", 31.00, 103.32, 7.9, "Sichuan2008"),
        ("2023-02-06 01:17:34", 37.17, 36.94, 7.8, "Turkey2023"),
    ]

    all_results = []
    for eq_time, eq_lat, eq_lon, eq_mag, label in events:
        try:
            result = analyze_event(eq_time, eq_lat, eq_lon, eq_mag, label)
            if result is not None:
                all_results.append((label, result))
        except Exception as e:
            print(f"  ERROR: {e}")

    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY: Pre-shock signal at antipodal stations")
        print(f"{'='*60}")
        for label, rdf in all_results:
            for _, r in rdf.iterrows():
                ratio = r["rms_pre"] / max(r["rms_bg"], 1)
                flag = "ELEVATED" if ratio > 1.2 else "normal"
                print(f"  {label:>20s} {r['station']:>10s}: pre/bg = {ratio:.2f}x [{flag}]")

    print("\nDone. Waveforms saved in data/ as .mseed files")
    print("Plots in output/")


if __name__ == "__main__":
    main()
