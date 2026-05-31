# Solar Flare Monitoring System — Build Notes

## Goal
Extend the Symbiogenesis anomaly detection system (MonumentalSystems GitHub) to do real-time solar flare monitoring and atmospheric coupling alerting.

## Reference Implementation
- **AWS LSTM solar flare detection**: https://github.com/aws-samples/sample-sagemaker-ai-lstm-anomaly-detection-solar-flare
- Uses LSTM anomaly detection on ESA STIX multi-channel X-ray data
- 5 energy bands: 4-10, 10-15, 15-25, 25-50, 50-84 keV
- PyTorch LSTM, deployed via SageMaker
- Blog post: https://aws.amazon.com/blogs/machine-learning/build-a-solar-flare-detection-system-on-sagemaker-ai-lstm-networks-and-esa-stix-data/

## Data Sources

### Solar Flare / X-ray
- **ESA STIX (Solar Orbiter)**: FITS files from SOAR archive — https://soar.esac.esa.int/soar/
- **GOES X-ray flux**: Real-time 1-min data — https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json
- **GOES >2 MeV electron flux**: Real-time — https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json

### Solar Wind / Geomagnetic (for coupling chain)
- **DSCOVR/ACE L1 solar wind**: Real-time — https://services.swpc.noaa.gov/products/solar-wind/
- **OMNI hourly**: Historical — https://omniweb.gsfc.nasa.gov/
- **Kp/Dst real-time**: https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json

### Atmospheric Coupling Targets
- **Neutron monitors (CR flux)**: NMDB — https://www.nmdb.eu/nest/
- **Fair-weather PG**: GloCAEM/CEDA — https://data.ceda.ac.uk/badc/glocaem/data
- **FRD magnetometer (dB/dt)**: INTERMAGNET — https://imag-data.bgs.ac.uk/GIN_V1/GINServices

## Architecture Plan

### Input Channels (real-time)
1. GOES X-ray flux (M/X class flare detection)
2. GOES >2 MeV electron flux (HEEP pathway — Li et al. 2016)
3. DSCOVR solar wind (Vsw, By, Bz — Mansurov + SSC prediction)
4. Kp/Dst (geomagnetic storm onset)

### Processing (LSTM anomaly detection + physics model)
- Anomaly detection on X-ray channels (flare onset)
- CME arrival prediction from solar wind speed
- Forbush decrease prediction from flare magnitude + CME speed
- HEEP prediction from electron flux trends
- IMF By sector classification (Mansurov pathway)

### Output: Atmospheric Coupling Alert
Score each of the 5 pathways identified in Paper XXVI:

| Pathway | Input | Lag | Effect on tornadoes |
|---------|-------|-----|---------------------|
| Forbush chain | CR flux decrease | +3 to +8 days | Suppression |
| HEEP | >2 MeV electron flux | +3.7-4 days | Enhancement |
| SSC telluric | dB/dt pulse | Hours | Enhancement (if strong) |
| Mansurov | IMF By polarity | ~4-7 days | Enhancement (away) |
| Lunar tidal | Ephemeris | Deterministic | Enhancement (AMJ, near new/full) |

Combined "stressor loading" index = weighted sum of active pathways.

### Key Papers
- Li et al. 2016, CJSS 36:40-48 — HEEP -> Vostok Ez, lag 3.7-4 days
- Mironova, Tinsley & Zhou 2011, ASR 47:1867 — REF -> VAI, lag 5 days
- Tinsley & Zhou 2006 — Jz -> cloud microphysics mechanism
- Paper XXVI (this repo) — full coupling chain with Nagycenk PG confirmation

## Pick Up On GPU Machine
1. Clone the AWS sample repo above
2. Adapt the LSTM architecture for multi-source input (X-ray + electron + solar wind)
3. Use the OMNI + tornado dataset from this repo for training the coupling model
4. Deploy as real-time monitor ingesting SWPC JSON feeds
