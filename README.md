# Global Resonance

Real-time space weather and geophysical monitoring platform. Three.js globe with toggleable data layers showing solar-terrestrial coupling through the Jelly Ball model.

## Architecture

```
Global-Resonance/
  backend/          Python FastAPI — data ingestion + API endpoints
  frontend/         Three.js globe + UI
  data/             Cached datasets (620 MB, not in git)
```

## Quick Start

```bash
# Backend
cd backend
pip install fastapi uvicorn httpx numpy pandas
uvicorn server:app --reload --port 8000

# Frontend
cd frontend
npx serve .    # or any static file server
```

## Deployment (Docker / Coolify)

A single container serves the API and the built frontend together.

```bash
docker compose up --build      # → http://localhost:8000
```

In **Coolify**: New Resource → Docker Compose → point at this repo. It listens on
port **8000**. The multi-stage [Dockerfile](Dockerfile) builds the Vite frontend,
then serves it from FastAPI alongside the `/api/*` endpoints.

- **API docs (Swagger UI):** `/docs`  ·  **ReDoc:** `/redoc`  ·  **OpenAPI schema:** `/openapi.json`
- **Env vars:** `PORT` (default 8000), `NASA_API_KEY` (default `DEMO_KEY`, rate-limited — get a free key at https://api.nasa.gov), `SOLAR_MONITOR_URL` (optional external SSE source).
- **`data/` volume:** 39 of 41 endpoints fetch live and need no data dir. Two layers
  (cosmic rays, lightning) read cached files from the mounted `data/` volume.

### Streaming data

Live endpoints stream as soon as the container has internet — no setup needed.
To populate the two file-backed layers, run the refresh job (pulls live from NMDB):

```bash
python backend/refresh_data.py      # writes data/solar_wind/cosmic_rays_*.csv
```

Safe to run on a cron. The lightning NetCDF climatology is optional; `/api/lightning`
falls back to live Open-Meteo thunderstorm codes without it.

## Data Layers

All sources are **public**. "Live" = fetched at request time (5-min in-memory cache);
"Cached" = read from the `data/` volume, refreshed by `backend/refresh_data.py`.

| Layer | Source | Mode | Update | Description |
|-------|--------|------|--------|-------------|
| Earthquakes | [USGS ComCat](https://earthquake.usgs.gov/fdsnws/event/1/) | Live | Real-time | M4.5+ markers colored by Jelly Ball zone |
| Solar wind | [NOAA SWPC](https://services.swpc.noaa.gov/) | Live | 1-min | Bz, speed, density strip charts |
| GOES X-ray | [NOAA SWPC](https://services.swpc.noaa.gov/) | Live | 1-min | Flare flux + df/dt order parameter |
| Kp index | [NOAA SWPC](https://services.swpc.noaa.gov/) | Live | 3-hourly | Geomagnetic activity bar |
| Dst index | [Kyoto WDC](https://wdc.kugi.kyoto-u.ac.jp/) | Live | Hourly | Ring-current intensity |
| CME / Flares | [NASA DONKI](https://api.nasa.gov/) | Live | Event | DONKI catalog (needs `NASA_API_KEY`) |
| SDO Sun view | [NASA SDO](https://sdo.gsfc.nasa.gov/) / [SOHO](https://soho.nascom.nasa.gov/) | Live | 15-min | EUV 193A + HMI magnetogram, LASCO |
| Magnetometers | [USGS Geomag](https://geomag.usgs.gov/) | Live | 1-min | Ground station B-field |
| Seismic waveforms | [IRIS](https://service.iris.edu/) | Live | Real-time | Antipodal / foreshock analysis |
| Weather / lightning | [Open-Meteo](https://api.open-meteo.com/) | Live | Hourly | Thunderstorm codes, precipitation |
| Ocean / SST | [NOAA NCEI](https://www.ncei.noaa.gov/) / [CoastWatch](https://coastwatch.noaa.gov/) | Live | Daily | Sea-surface temperature, currents |
| Cosmic rays | [NMDB](https://www.nmdb.eu/) | Cached | Hourly | Forbush decrease (OULU/ROME/NEWK/THUL) |
| Lightning climatology | WGLC / WWLLN (NetCDF) | Cached | Static | Schumann driver (optional; Open-Meteo fallback) |
| Jelly Ball zones | Computed | — | Per-CME | Concentric rings from subsolar point |
| Subsolar / tidal | Computed | — | Continuous | Terminator, lunar phase, dF/dt overlay |
| Plate boundaries | `frontend/src/plates.json` | Static | — | Tectonic plate outlines |
