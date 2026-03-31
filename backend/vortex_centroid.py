#!/usr/bin/env python3
"""Compute polar vortex centroids from ERA5 and correlate with magnetic pole."""
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

from polar_vortex_magnetic import get_magnetic_pole_positions

print("Loading ERA5 10hPa geopotential...")
ds = xr.open_dataset(DATA_DIR / "era5_10hpa_geopotential_monthly_NH.nc")
z = ds["z"]
lats = ds.coords["latitude"].values
lons = ds.coords["longitude"].values
time_var = "valid_time" if "valid_time" in ds.coords else "time"
times = pd.to_datetime(ds.coords[time_var].values)
print(f"  {len(times)} months, {times[0].strftime('%Y-%m')} to {times[-1].strftime('%Y-%m')}")

print("Computing vortex centroids...", flush=True)

# Pre-compute grids once
lat_60_idx = np.argmin(np.abs(lats - 60))
cos_lat = np.cos(np.radians(lats))
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
x_grid = np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))
y_grid = np.cos(np.radians(lat_grid)) * np.sin(np.radians(lon_grid))
z_grid = np.sin(np.radians(lat_grid))
area_w = cos_lat[:, np.newaxis] * np.ones((len(lats), len(lons)))

# Load all data at once into memory (avoids repeated disk reads)
z_all = z.values.squeeze()  # (time, lat, lon)
ds.close()
print(f"  Loaded {z_all.shape[0]} fields into memory", flush=True)

centroids = []
for t_idx in range(len(times)):
    z_slice = z_all[t_idx]
    if z_slice.ndim != 2:
        continue
    z_threshold = np.mean(z_slice[lat_60_idx, :])
    weights = np.maximum(z_threshold - z_slice, 0)
    combined = weights * area_w
    total = np.sum(combined)
    if total > 0:
        cx = np.sum(x_grid * combined) / total
        cy = np.sum(y_grid * combined) / total
        cz = np.sum(z_grid * combined) / total
        clat = np.degrees(np.arctan2(cz, np.sqrt(cx**2 + cy**2)))
        clon = np.degrees(np.arctan2(cy, cx))
        vortex_mask = z_slice < z_threshold
        strength = np.mean(weights[vortex_mask]) if np.any(vortex_mask) else 0
        t = times[t_idx]
        centroids.append({"date": t, "year": t.year, "month": t.month,
                          "centroid_lat": clat, "centroid_lon": clon, "strength": strength})
    if t_idx % 60 == 0:
        t = times[t_idx]
        print(f"  {t.year}-{t.month:02d}: ({clat:.1f}N, {clon:.1f}E)", flush=True)
cdf = pd.DataFrame(centroids)
cdf.to_csv(DATA_DIR / "polar_vortex_centroids.csv", index=False)
print(f"Saved: {len(cdf)} monthly centroids")

# Winter (DJF)
winter = cdf[cdf["month"].isin([12, 1, 2])].copy()
winter_yr = winter.groupby("year").agg(
    mean_lat=("centroid_lat", "mean"), mean_lon=("centroid_lon", "mean"),
    std_lon=("centroid_lon", "std")).reset_index()

poles = get_magnetic_pole_positions()
merged = pd.merge(winter_yr, poles[["year","mag_lat","mag_lon","speed_km_yr"]], on="year", how="inner")

print(f"\n=== VORTEX CENTROID vs MAGNETIC POLE ({len(merged)} winters) ===")
r_lon, p_lon = stats.pearsonr(merged["mean_lon"], merged["mag_lon"])
r_lat, p_lat = stats.pearsonr(merged["mean_lat"], merged["mag_lat"])
print(f"  Centroid lon vs pole lon: r = {r_lon:+.3f}, p = {p_lon:.4f}")
print(f"  Centroid lat vs pole lat: r = {r_lat:+.3f}, p = {p_lat:.4f}")

valid = merged.dropna(subset=["std_lon","speed_km_yr"])
if len(valid) > 5:
    r_v, p_v = stats.pearsonr(valid["std_lon"], valid["speed_km_yr"])
    print(f"  Centroid variability vs pole speed: r = {r_v:+.3f}, p = {p_v:.4f}")

# Also check: zonal wind at 60N
print("\nLoading ERA5 10hPa zonal wind...")
ds2 = xr.open_dataset(DATA_DIR / "era5_10hpa_uwind_60N_monthly.nc")
u = ds2["u"]
u_mean = u.mean(dim=["latitude","longitude"]).values.squeeze()
time_var2 = "valid_time" if "valid_time" in ds2.coords else "time"
times2 = pd.to_datetime(ds2.coords[time_var2].values)
ds2.close()

udf = pd.DataFrame({"date": times2, "u_60n": u_mean})
udf["year"] = udf["date"].dt.year
udf["month"] = udf["date"].dt.month

# Winter zonal wind (positive = westerly = strong vortex)
u_winter = udf[udf["month"].isin([12,1,2])].groupby("year").agg(u_mean=("u_60n","mean")).reset_index()
kp = pd.read_csv(DATA_DIR / "kp_daily.csv")
kp_yr = kp.groupby("year").agg(sn=("sn", lambda x: x[x>=0].mean())).reset_index()
u_merged = pd.merge(u_winter, kp_yr, on="year", how="inner")

r_u, p_u = stats.pearsonr(u_merged["u_mean"], u_merged["sn"].fillna(0))
print(f"  Winter 10hPa zonal wind vs SSN: r = {r_u:+.3f}, p = {p_u:.4f}")
print(f"  (positive = stronger vortex at solar max)")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

ax = axes[0]
ax.plot(merged["year"], merged["mean_lon"], "o-", color="steelblue", lw=2, label="Vortex centroid lon")
ax2 = ax.twinx()
ax2.plot(merged["year"], merged["mag_lon"], "s-", color="red", lw=2, label="Magnetic pole lon")
ax.set_ylabel("Vortex centroid longitude", color="steelblue")
ax2.set_ylabel("Magnetic pole longitude", color="red")
ax.set_title(f"Polar Vortex Centroid vs Magnetic Pole (winter DJF)\nr(lon) = {r_lon:+.3f}, p = {p_lon:.4f}")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")

ax = axes[1]
ax.scatter(merged["mag_lon"], merged["mean_lon"], s=40, c=merged["year"], cmap="viridis")
z = np.polyfit(merged["mag_lon"], merged["mean_lon"], 1)
xline = np.linspace(merged["mag_lon"].min(), merged["mag_lon"].max(), 100)
ax.plot(xline, np.polyval(z, xline), "r--", lw=2)
ax.set_xlabel("Magnetic pole longitude")
ax.set_ylabel("Vortex centroid longitude (winter)")
ax.set_title("Does the vortex follow the magnetic pole?")
plt.colorbar(ax.collections[0], ax=ax, label="Year")

ax = axes[2]
ax.plot(u_merged["year"], u_merged["u_mean"], "o-", color="steelblue", lw=2, label="10hPa zonal wind")
ax2 = ax.twinx()
ax2.plot(u_merged["year"], u_merged["sn"], "o-", color="orange", lw=2, alpha=0.5, label="SSN")
ax.set_ylabel("Winter mean zonal wind (m/s)")
ax2.set_ylabel("SSN", color="orange")
ax.set_title(f"Stratospheric Zonal Wind vs Solar Cycle: r = {r_u:+.3f}, p = {p_u:.4f}")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT_DIR / "vortex_centroid_vs_magnetic_pole.png", dpi=150)
print(f"\nSaved: vortex_centroid_vs_magnetic_pole.png")
