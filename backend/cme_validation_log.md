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

## CCMC Scoreboard
- Listed "actual shock arrival" at 2026-03-31 05:53Z
- BUT: no Kp response (Kp stayed 2.0-2.7), no solar wind speed jump
- SolarHam: "expected CME has yet to be detected" (as of Apr 1)
- CCMC 05:53Z arrival may be misidentified (noise or minor structure)

## Actual (as of Apr 1 02:46Z)
- Solar wind: 388 km/s, density 3.5/cm3 (NORMAL)
- Bz: -2.9 nT (quiet)
- Kp: 1.33 (quiet)
- ACE EPAM: particle levels increasing (precursor)
- **CME HAS NOT ARRIVED** (confirmed by SolarHam, SWPC data)

## Analysis
The E45 source location is the key difficulty:
1. Shock/nose component may have passed as a weak glancing blow (undetected)
2. Magnetic ejecta (the actual storm driver) is still in transit
3. All ENLIL models were 5-16h early for the "shock" and may be ~12h early for ejecta
4. Our geometric model predicts ejecta at ~50h = Apr 1 05:00Z

## Jelly Ball Seismic Response
- Grade-0 (X-ray SID): Mar 30 03:19Z -> Vanuatu M7.3 at +5.4h (CONFIRMED)
- Grade-4 (iono relaxation): Mar 30 21:19Z -> no major event
- Grade-2 (CME mechanical): PENDING — depends on CME actual arrival

## Lessons
1. CCMC "actual arrival" can be wrong — always verify with Kp and solar wind
2. E45 sources produce shock/ejecta separation of ~20h
3. v*cos(source_lon) + drag model is simple but competitive with ENLIL
4. Need to track shock AND ejecta separately
5. Precursor particles (EPAM) are the best real-time indicator

## TODO when CME arrives
- [ ] Record actual arrival time, speed, Bz rotation
- [ ] Compare to all predictions
- [ ] Track Kp/Dst evolution
- [ ] Check earthquake rate in wavefront zone 6-24h after arrival
- [ ] Update the geometric transit model with this calibration point
