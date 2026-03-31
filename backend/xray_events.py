#!/usr/bin/env python3
"""X-ray profiles for the three key flare-earthquake events."""
import sys, os, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta, timezone

def model_flare(peak_flux, begin, peak, end, bg=1.5e-6):
    """Simple impulsive flare model: linear rise, exponential decay."""
    rise_s = (peak - begin).total_seconds()
    decay_s = (end - peak).total_seconds()
    tau = decay_s / math.log(peak_flux / max(bg * 3, 1e-7))

    def flux_at(t):
        if t < begin:
            return bg
        elif t <= peak:
            frac = (t - begin).total_seconds() / rise_s
            return bg + (peak_flux - bg) * frac
        else:
            dt = (t - peak).total_seconds()
            return bg + (peak_flux - bg) * math.exp(-dt / tau)
    return flux_at


def classify(f):
    if f >= 1e-4: return f"X{f/1e-4:.1f}"
    if f >= 1e-5: return f"M{f/1e-5:.1f}"
    if f >= 1e-6: return f"C{f/1e-6:.1f}"
    return f"B{f/1e-7:.0f}"


def print_profile(label, fl_begin, fl_peak, fl_end, peak_flux, eq_time, eq_label,
                  extra_marks=None):
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"  Flare: begin {fl_begin.strftime('%H:%M')} peak {fl_peak.strftime('%H:%M')} end {fl_end.strftime('%H:%M')}Z")
    print(f"  EQ:    {eq_time.strftime('%H:%M:%S')}Z ({eq_label})")

    dt_peak = (eq_time - fl_peak).total_seconds()
    dt_end = (eq_time - fl_end).total_seconds()
    print(f"  Delta from peak: +{dt_peak:.0f}s ({dt_peak/60:.1f} min)")
    print(f"  Delta from end:  {dt_end:+.0f}s ({dt_end/60:.1f} min)")

    if dt_peak < (fl_end - fl_peak).total_seconds():
        phase = "DURING FLARE (decay phase)"
    elif dt_end < 600:
        phase = "IMMEDIATELY POST-FLARE"
    elif dt_peak < 3600:
        phase = "EARLY GRADE-0"
    else:
        phase = "GRADE-0 -> GRADE-4 TRANSITION"
    print(f"  Phase: {phase}")

    flux_fn = model_flare(peak_flux, fl_begin, fl_peak, fl_end)

    # Print profile
    t_start = fl_begin - timedelta(minutes=5)
    t_stop = eq_time + timedelta(minutes=10)
    step = 30  # seconds

    print(f"\n  {'Time':>8} {'dt_pk':>7} {'dt_EQ':>7} {'~Flux':>10} {'Class':>6}  Profile")
    print(f"  {'-'*70}")

    t = t_start
    last_min = -999
    while t <= t_stop:
        dt_pk_s = (t - fl_peak).total_seconds()
        dt_eq_s = (t - eq_time).total_seconds()
        f = flux_fn(t)
        cls = classify(f)

        # Print control: every minute, plus near key events
        cur_min = int(dt_pk_s / 60)
        near_key = (abs(dt_pk_s) < 20 or abs(dt_eq_s) < 20 or
                    abs((t - fl_begin).total_seconds()) < 20 or
                    abs((t - fl_end).total_seconds()) < 20)

        if near_key or cur_min != last_min:
            last_min = cur_min
            bar_len = int(max(0, (math.log10(max(f, 1e-8)) + 7) * 8))
            bar = "#" * bar_len

            marker = ""
            if abs(dt_pk_s) < 20:
                marker = " <<< FLARE PEAK"
            elif abs(dt_eq_s) < 20:
                marker = f" <<< {eq_label}"
            elif abs((t - fl_begin).total_seconds()) < 20:
                marker = " (begin)"
            elif abs((t - fl_end).total_seconds()) < 20:
                marker = " (end)"

            if extra_marks:
                for mt, mlabel in extra_marks:
                    if abs((t - mt).total_seconds()) < 20:
                        marker = f" {mlabel}"

            print(f"  {t.strftime('%H:%M:%S')} {dt_pk_s/60:+6.1f}m {dt_eq_s/60:+6.1f}m  {f:.2e} {cls:>6}  {bar}{marker}")

        t += timedelta(seconds=step)

    f_at_eq = flux_fn(eq_time)
    print(f"\n  Flux at earthquake: {f_at_eq:.2e} ({classify(f_at_eq)})")
    print(f"  Ratio to peak: {f_at_eq/peak_flux*100:.1f}%")

    # EM diffusion estimate
    print(f"\n  EM diffusion time to fault depth:")
    for depth_km, sigma in [(10, 0.01), (18, 0.05), (121, 0.1)]:
        mu0 = 4 * math.pi * 1e-7
        tau_em = math.sqrt(2 * mu0 * sigma * (depth_km * 1e3) ** 2)
        print(f"    {depth_km:>4}km (sigma={sigma} S/m): {tau_em:.0f}s = {tau_em/60:.1f}min")


# ===== EVENT 1: Dec 6, 2025 =====
print_profile(
    "EVENT 1: Dec 6, 2025 -- M8.1 + M7.0 Hubbard Glacier (2.8 min)",
    fl_begin=datetime(2025, 12, 6, 20, 29, tzinfo=timezone.utc),
    fl_peak=datetime(2025, 12, 6, 20, 39, tzinfo=timezone.utc),
    fl_end=datetime(2025, 12, 6, 20, 49, tzinfo=timezone.utc),
    peak_flux=8.1e-5,
    eq_time=datetime(2025, 12, 6, 20, 41, 49, tzinfo=timezone.utc),
    eq_label="M7.0 HUBBARD",
    extra_marks=[
        (datetime(2025, 12, 6, 19, 21, tzinfo=timezone.utc), "(M1.1 precursor peak)"),
    ],
)

# ===== EVENT 2: Nov 9, 2025 =====
print_profile(
    "EVENT 2: Nov 9, 2025 -- X1.7 + M6.8 Japan Trench (29 min)",
    fl_begin=datetime(2025, 11, 9, 7, 1, tzinfo=timezone.utc),
    fl_peak=datetime(2025, 11, 9, 7, 35, tzinfo=timezone.utc),
    fl_end=datetime(2025, 11, 9, 7, 55, tzinfo=timezone.utc),
    peak_flux=1.7e-4,
    eq_time=datetime(2025, 11, 9, 8, 3, 39, tzinfo=timezone.utc),
    eq_label="M6.8 JAPAN",
)

# ===== EVENT 3: Mar 30, 2026 (from real GOES data) =====
print_profile(
    "EVENT 3: Mar 30, 2026 -- X1.5 + M7.3 Vanuatu (325 min)",
    fl_begin=datetime(2026, 3, 30, 2, 47, tzinfo=timezone.utc),
    fl_peak=datetime(2026, 3, 30, 3, 19, tzinfo=timezone.utc),
    fl_end=datetime(2026, 3, 30, 3, 44, tzinfo=timezone.utc),
    peak_flux=1.5e-4,
    eq_time=datetime(2026, 3, 30, 8, 44, 13, tzinfo=timezone.utc),
    eq_label="M7.3 VANUATU",
)

# ===== SYNTHESIS =====
print(f"\n{'='*80}")
print("SYNTHESIS: THREE COUPLING TIMESCALES")
print(f"{'='*80}")
print("""
  Timescale 1: DURING FLARE (~3 min) -- Internal EM propagation
  ---------------------------------------------------------------
  Dec 2025: M8.1 -> M7.0 Hubbard at +169s
  Mechanism: X-ray SID -> ionospheric compression -> telluric pulse
             propagates through conductive mantle as EM waveguide
  Requirements: SHALLOW fault (10km), critically stressed, wavefront zone
  Schumann: f1 shift would be SIMULTANEOUS with the flare onset
  Order parameter: J crosses J_c within seconds at the subsolar point

  Timescale 2: POST-FLARE (~30 min) -- Ionospheric current redistribution
  ---------------------------------------------------------------
  Nov 2025: X1.7 -> M6.8 Japan at +29min
  Mechanism: SID ends -> electrojet adjusts -> magnetotelluric coupling
             Current system takes ~30 min to redistribute globally
  Requirements: SHALLOW plate interface (18km), subduction zone
  Schumann: f1 would show the DECAY from peak back toward baseline
  Order parameter: J relaxing back toward J_c, still above it

  Timescale 3: GRADE-0/4 TRANSITION (~5h) -- EM diffusion to depth
  ---------------------------------------------------------------
  Mar 2026: X1.5 -> M7.3 Vanuatu at +325min
  Mechanism: Ionospheric perturbation diffuses through crust/mantle
             EM diffusion to 121km depth takes hours (sigma ~ 0.1 S/m)
  Requirements: DEEP slab fault, sustained ionospheric loading
  Schumann: f1 would have returned to BASELINE before the earthquake
  Order parameter: J crossing BACK through J_c (relaxation trigger)

  THE SCHUMANN AS ORDER PARAMETER:
  ---------------------------------------------------------------
  f1 tracks the ionosphere-Earth cavity height h:
    f1 ~ c / (2*pi*R) * sqrt(1 + h/R)
  CME/SID compresses h -> f1 UP -> J > J_c (ordered, stable)
  As ionosphere relaxes -> f1 DOWN -> J crosses J_c
  AT THE CROSSING: vortex-antivortex pairs unbind (KT transition)
  This is when the stress couples to the lithosphere most strongly

  Prediction: earthquakes should cluster when df1/dt is most NEGATIVE
  (Schumann frequency falling = ionosphere relaxing = J crossing J_c)
""")
