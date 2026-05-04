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

## Data Layers

| Layer | Source | Update | Description |
|-------|--------|--------|-------------|
| Earthquakes | USGS ComCat | Real-time | M4.5+ markers colored by Jelly Ball zone |
| Jelly Ball zones | Computed | Per-CME | Concentric rings from subsolar point |
| Solar wind | SWPC | 1-min | Bz, speed, density strip charts |
| GOES X-ray | SWPC | 1-min | Flare flux + df/dt order parameter |
| Kp index | SWPC | 3-hourly | Geomagnetic activity bar |
| Cosmic rays | NMDB | Hourly | Forbush decrease detection |
| Subsolar point | Computed | Continuous | Yellow star + terminator line |
| Tidal stress | Computed | Continuous | Lunar phase + dF/dt overlay |
| Plate boundaries | Static | — | Tectonic plate outlines |
| SDO Sun view | NASA SDO | 15-min | EUV 193A + HMI magnetogram |
| Swarm satellite | ESA VirES | ~6h latency | Orbital track + F field |
| Magnetometers | USGS | 1-min | Ground station B-field |
