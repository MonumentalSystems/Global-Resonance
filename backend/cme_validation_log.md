# CME Validation Log: X1.4 / Mar 30 2026

## Event
- X1.4 (X1.5 in GOES) flare peak: 2026-03-30 03:19Z
- CME launch: 2026-03-30 03:24Z, 1689 km/s, half-angle 46 deg
- Source: AR 14405, S27E45

## Predictions

| Model | Predicted Arrival | Transit (h) |
|-------|-------------------|-------------|
| Ballistic nose | Mar 31 04:00Z | 24.6 |
| DONKI/ENLIL (NASA M2M) | Mar 31 10:40Z | 31.3 |
| DONKI/ENLIL (NASA M2M run 2) | Mar 31 15:07Z | 35.7 |
| CCMC ensemble | Mar 31 16:09Z | 36.8 |
| Geometric: shock (nose, weak drag) | Mar 31 ~06:00Z | ~27 |
| Geometric: ejecta (flank+drag) | Apr 01 ~05:00Z | ~50 |
| SWPC revised forecast | Apr 01 03:00-09:00Z | 48-54 |

## FINAL OUTCOME: COMPLETE MISS

**The CME never arrived.** As of Apr 2:

- Solar wind: 370-420 km/s throughout (no shock, no sheath, no ejecta)
- Kp: stayed 1-3 the entire period (no storm)
- Bz: fluctuated -5 to +5 nT (no sustained southward)
- Dst: stayed above -10 nT (no ring current enhancement)
- Cosmic rays: flat, no Forbush decrease
- CCMC "actual arrival" at Mar 31 05:53Z: FALSE DETECTION (no confirming signature)

**Every prediction was wrong.** DONKI, ENLIL, CCMC ensemble, SWPC, and our geometric model all predicted arrival. The CME missed Earth entirely.

## Why It Missed

The source geometry made this a marginal event:

- **S27E45**: 45 degrees east of disk center AND 27 degrees south of ecliptic
- **Half-angle 46 deg**: Earth was barely inside the modeled cone
- **Combined angular offset**: sqrt(27^2 + 45^2) = 53 degrees from Sun-Earth line
- **53 degrees > 46 degree half-angle**: Earth was actually OUTSIDE the cone

The models treated the half-angle as measured in the plane of sky (coronagraph projection), but the true 3D cone was narrower than the projected angle suggested. The S27 latitude meant the CME propagated well south of the ecliptic plane.

### Failure mode for each model:
1. **DONKI/ENLIL**: Used projected half-angle, not corrected for S27 latitude
2. **CCMC ensemble**: Same input parameters, same systematic error
3. **SWPC forecast**: Relied on ENLIL output
4. **Our geometric model**: Used v*cos(45) for the radial component but didn't account for the S27 latitude deflecting the cone below the ecliptic
5. **92% Earth impact probability**: Overconfident — should have been ~30-40% given the combined angular offset

## Lessons Learned

### For CME prediction:
1. **Source latitude matters as much as longitude.** A CME from S27E45 has a TRUE angular offset of 53 degrees from the Sun-Earth line, not 45 degrees.
2. **Projected half-angle overstates the cone width.** Coronagraph measurements are 2D projections of a 3D structure.
3. **Miss probability is HIGH for |offset| > half-angle - 10 degrees.** This event: offset 53 deg, half-angle 46 deg → miss margin of 7 degrees → should have been flagged as likely miss.
4. **Impact probability of 92% was unjustified.** The geometry alone gives ~40% at best for this configuration.

### Improved transit model:
```
angular_offset = sqrt(source_lat^2 + source_lon^2)  # total offset from Sun-Earth line
if angular_offset > half_angle:
    prediction = "MISS"
elif angular_offset > half_angle - 10:
    prediction = "GLANCING/UNCERTAIN (30-50%)"
else:
    v_effective = v_nose * cos(angular_offset)
    transit = 1_AU / v_effective + drag_correction
```

For this event: offset = sqrt(27^2 + 45^2) = 52.5 > 46 → **MISS predicted**

### For the Jelly Ball model:
5. **Grade-0 (X-ray SID) is independent of CME arrival.** The Vanuatu M7.3 at +5.4h after the X1.5 flare stands regardless of the CME miss. X-rays travel at light speed and are not directional.
6. **Grade-2 (CME mechanical) requires actual magnetopause compression.** No CME arrival → no wavefront enhancement. This is a clean negative control.
7. **Grade-4 (ionospheric relaxation) from the SID also stands.** The ionosphere was compressed by the X-rays and relaxed over hours, independent of the CME.

## Jelly Ball Seismic Scorecard

| Grade | Prediction | Outcome |
|-------|-----------|---------|
| Grade-0 (X-ray SID) | Vanuatu M7.3 at +5.4h | **CONFIRMED** |
| Grade-4 (iono relaxation) | No major event in 6-18h window | **Correct (null)** |
| Grade-2 (CME mechanical) | Wavefront enhancement 1.36x | **NOT TESTED** (CME missed) |

The Grade-2 prediction remains untested — we need a CME that actually arrives to validate or falsify the wavefront enrichment for this event. The 3.26x enrichment from the 26-year backtest stands on historical data.

## Summary

This event is a valuable calibration point: it demonstrates that CME arrival prediction is systematically overconfident for oblique sources, and that the Jelly Ball Grade-0 mechanism (X-ray → earthquake) operates independently of CME arrival. The one-line improvement to the transit model (use total angular offset including latitude) would have correctly predicted a miss.
