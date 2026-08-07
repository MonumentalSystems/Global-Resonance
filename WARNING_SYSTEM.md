# Warning system contract

The live warning path is:

1. `solar-monitor` polls NOAA SWPC and NASA/JSOC feeds.
2. Fresh GOES XRS observations are processed once by the Rust detector ensemble.
3. Rust exposes JSON status plus named `metrics` and `alert` SSE events.
4. FastAPI preserves upstream HTTP status and SSE framing under `/api/solar/*`.
5. The frontend renders source, observation age, degraded state, detector state, and experimental coupling indicators.

The bundled deployment builds the Rust service, binds it to `127.0.0.1:8089`, and starts it beside FastAPI. Runtime configuration mutation is disabled unless `SOLAR_MONITOR_ALLOW_CONFIG_WRITE=1` is deliberately set on a trusted internal deployment.

## Alert semantics

- A detector alert means a fresh, observed solar X-ray anomaly crossed the multi-detector threshold. It is edge-triggered and is not repeated while the same alert state remains active.
- `ELEVATED`, `ACTIVE`, and `FLARE` organize monitoring attention. They are not calibrated event probabilities.
- The Clifford-lattice criticality output remains visible as an experimental diagnostic, but is excluded from live fusion weighting, agreement, and escalation.
- Coupling scores are research indicators, not earthquake or weather forecasts.
- The Forbush pathway is not scored from flare class alone. It remains inactive until measured CME or cosmic-ray evidence is integrated.

## Freshness gates

Readiness is derived from observation timestamps, never merely from a successful HTTP poll.

| Feed | Source | Maximum age |
| --- | --- | ---: |
| X-ray | NOAA SWPC GOES XRS | 5 minutes |
| Solar wind | NOAA SWPC DSCOVR/ACE | 10 minutes |
| Electrons | NOAA SWPC GOES | 15 minutes |
| Protons | NOAA SWPC GOES | 15 minutes |
| Planetary K-index | NOAA SWPC | 4 hours |
| SHARP | NASA/JSOC | 6 hours |

Fresh XRS is required for alerting. Missing auxiliary channels produce `degraded` status rather than fabricated observations. Missing or stale XRS produces HTTP 503 from `/api/solar/health`, inhibits alerts, and clears the frontend detector readout.

## Known external dependencies

- NOAA SWPC JSON services must be reachable for XRS, particles, solar wind, and K-index.
- NASA/JSOC SHARP access is optional and may have multi-hour latency.
- The K-index feed contains a modeled Dst estimate, not Kyoto WDC Dst observations.
- No operational CME ensemble or neutron-monitor feed is currently connected to the Rust alert service.
- The warning system has unit and transport-contract validation, but it does not yet have a prospective skill evaluation on an untouched event catalog.
