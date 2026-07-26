#!/usr/bin/env python3
"""Build and screen a county-level cancer/radon dataset.

This module deliberately keeps acquisition, normalization, and inference separate.
The resulting tables are ecological (county-level) and must not be interpreted as
individual exposure estimates or causal effects.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "cancer_soil"
OUTPUT_DIR = ROOT / "backend" / "output" / "cancer_soil"

SCP_URL = "https://www.statecancerprofiles.cancer.gov/incidencerates/index.php"
RADON_URL = "https://www.epa.gov/system/files/other-files/2024-06/radon-zones-202406.json"

STATE_FIPS = (
    "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 "
    "26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 "
    "49 50 51 53 54 55 56"
).split()

CANCER_SITES = {
    "all": "001",
    "bladder": "071",
    "brain": "076",
    "colon_rectum": "020",
    "esophagus": "017",
    "liver": "035",
    "lung": "047",
    "pancreas": "040",
    "stomach": "018",
    "thyroid": "080",
}


def _get(url: str, *, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Global-Resonance research/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _state_cancer_url(state_fips: str, cancer_code: str) -> str:
    query = urllib.parse.urlencode(
        {
            "age": "001",
            "areatype": "county",
            "cancer": cancer_code,
            "output": "1",
            "race": "00",
            "sex": "0",
            "sortOrder": "asc",
            "sortVariableName": "rate",
            "stage": "999",
            "stateFIPS": state_fips,
            "type": "incd",
            "year": "0",
        }
    )
    return f"{SCP_URL}?{query}"


def parse_cancer_csv(raw: bytes, site: str) -> pd.DataFrame:
    """Parse a State Cancer Profiles export, retaining suppression as missing."""
    text = raw.decode("iso-8859-1").replace("\r\n", "\n")
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if re.match(r"^[^,]+,FIPS,", line))
    period_match = re.search(r"(20\d{2}-20\d{2})", "\n".join(lines[:header_idx]))
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    def prefixed(prefix: str) -> str:
        matches = [str(column) for column in frame.columns if str(column).startswith(prefix)]
        if not matches:
            raise ValueError(f"State Cancer Profiles export lacks expected column prefix: {prefix}")
        return matches[0]

    geography_column = str(frame.columns[0])
    frame = frame.rename(
        columns={
            "FIPS": "fips",
            geography_column: "county_label",
            prefixed("2023 Rural-Urban Continuum Codes"): "rural_urban",
            prefixed("Age-Adjusted Incidence Rate"): "incidence_rate",
            "Average Annual Count": "annual_count",
        }
    )
    ci_columns = [column for column in frame.columns if str(column).startswith("Lower 95% Confidence Interval")]
    upper_ci_columns = [column for column in frame.columns if str(column).startswith("Upper 95% Confidence Interval")]
    if ci_columns:
        frame = frame.rename(columns={ci_columns[0]: "rate_ci_low"})
    if upper_ci_columns:
        frame = frame.rename(columns={upper_ci_columns[0]: "rate_ci_high"})
    frame["fips"] = frame["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    frame = frame[
        frame["fips"].str.match(r"^\d{5}$")
        & ~frame["fips"].str.endswith("000")
        & frame["fips"].ne("02900")
    ].copy()
    for column in ("incidence_rate", "rate_ci_low", "rate_ci_high", "annual_count"):
        frame[column] = pd.to_numeric(frame[column].astype(str).str.replace("*", "", regex=False), errors="coerce")
    frame["county"] = frame["county_label"].str.replace(r"\(\d+\)$", "", regex=True).str.strip()
    frame["state_fips"] = frame["fips"].str[:2]
    frame["site"] = site
    frame["period"] = period_match.group(1) if period_match else "unknown"
    return frame[
        [
            "fips",
            "state_fips",
            "county",
            "site",
            "period",
            "rural_urban",
            "incidence_rate",
            "rate_ci_low",
            "rate_ci_high",
            "annual_count",
        ]
    ]


def download_cancer(sites: list[str], data_dir: Path = DATA_DIR, workers: int = 4) -> pd.DataFrame:
    cache_dir = data_dir / "raw" / "cancer"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(task: tuple[str, str]) -> pd.DataFrame:
        site, state_fips = task
        path = cache_dir / f"{site}_{state_fips}.csv"
        if not path.exists():
            path.write_bytes(_get(_state_cancer_url(state_fips, CANCER_SITES[site])))
        return parse_cancer_csv(path.read_bytes(), site)

    tasks = [(site, state) for site in sites for state in STATE_FIPS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        frames = list(executor.map(fetch, tasks))
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["fips", "site"])
    normalized = data_dir / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)
    result.to_csv(normalized / "county_cancer_incidence.csv", index=False)
    return result


def _normalize_name(value: str) -> str:
    value = value.lower().replace("st.", "saint").replace("ste.", "sainte")
    value = re.sub(r"\b(county|parish|borough|census area|municipality|city and borough)\b", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def download_radon(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw_path = data_dir / "raw" / "epa_radon_zones.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raw_path.write_bytes(_get(RADON_URL))
    payload = json.loads(raw_path.read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(payload["data"])
    frame["radon_zone"] = pd.to_numeric(frame["Zone"], errors="coerce")
    frame = frame[frame["radon_zone"].isin([1, 2, 3])].copy()
    frame["county_key"] = frame["COUNTY LABEL"].map(_normalize_name)
    frame["state_key"] = frame["STATE"].map(_normalize_name)
    return frame[["county_key", "state_key", "radon_zone"]].drop_duplicates()


def join_cancer_radon(cancer: pd.DataFrame, radon: pd.DataFrame, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # State names are obtained from the EPA records themselves to avoid a second naming authority.
    state_lookup = {
        "01": "alabama", "02": "alaska", "04": "arizona", "05": "arkansas", "06": "california",
        "08": "colorado", "09": "connecticut", "10": "delaware", "11": "districtofcolumbia",
        "12": "florida", "13": "georgia", "15": "hawaii", "16": "idaho", "17": "illinois",
        "18": "indiana", "19": "iowa", "20": "kansas", "21": "kentucky", "22": "louisiana",
        "23": "maine", "24": "maryland", "25": "massachusetts", "26": "michigan", "27": "minnesota",
        "28": "mississippi", "29": "missouri", "30": "montana", "31": "nebraska", "32": "nevada",
        "33": "newhampshire", "34": "newjersey", "35": "newmexico", "36": "newyork",
        "37": "northcarolina", "38": "northdakota", "39": "ohio", "40": "oklahoma", "41": "oregon",
        "42": "pennsylvania", "44": "rhodeisland", "45": "southcarolina", "46": "southdakota",
        "47": "tennessee", "48": "texas", "49": "utah", "50": "vermont", "51": "virginia",
        "53": "washington", "54": "westvirginia", "55": "wisconsin", "56": "wyoming",
    }
    frame = cancer.copy()
    frame["county_key"] = frame["county"].map(_normalize_name)
    frame["state_key"] = frame["state_fips"].map(state_lookup)
    joined = frame.merge(radon, on=["state_key", "county_key"], how="left", validate="many_to_one")
    joined["radon_potential"] = 4 - joined["radon_zone"]  # 3=highest, 1=lowest
    out = data_dir / "normalized" / "county_cancer_radon.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.drop(columns=["county_key", "state_key"]).to_csv(out, index=False)
    return joined


def _hc3_ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    scaled = residual / np.clip(1 - leverage, 1e-8, None)
    meat = x.T @ ((scaled[:, None] ** 2) * x)
    covariance = xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.clip(np.diag(covariance), 0, None))


def _bh_adjust(pvalues: pd.Series) -> pd.Series:
    values = pvalues.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return pd.Series(result, index=pvalues.index)


def analyze(joined: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    rows = []
    for site, group in joined.groupby("site"):
        valid = group.dropna(subset=["incidence_rate", "radon_potential"]).copy()
        rho, p_spearman = stats.spearmanr(valid["radon_potential"], valid["incidence_rate"])
        state_dummies = pd.get_dummies(valid["state_fips"], drop_first=True, dtype=float)
        rural = valid["rural_urban"].eq("Rural").astype(float).rename("rural")
        design = pd.concat(
            [pd.Series(1.0, index=valid.index, name="intercept"), valid["radon_potential"], rural, state_dummies],
            axis=1,
        ).astype(float)
        beta, se = _hc3_ols(valid["incidence_rate"].to_numpy(float), design.to_numpy(float))
        z = beta[1] / se[1]
        p_adjusted_model = 2 * stats.norm.sf(abs(z))
        zone_means = valid.groupby("radon_zone")["incidence_rate"].mean()
        rows.append(
            {
                "site": site,
                "n_counties": len(valid),
                "spearman_rho": rho,
                "spearman_p": p_spearman,
                "adjusted_rate_change_per_radon_level": beta[1],
                "adjusted_hc3_se": se[1],
                "adjusted_p": p_adjusted_model,
                "zone1_high_mean": zone_means.get(1, np.nan),
                "zone2_mean": zone_means.get(2, np.nan),
                "zone3_low_mean": zone_means.get(3, np.nan),
            }
        )
    result = pd.DataFrame(rows).sort_values("site").reset_index(drop=True)
    result["adjusted_q_bh"] = _bh_adjust(result["adjusted_p"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "radon_screen.csv", index=False)
    display = result.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    markdown_rows = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    markdown_rows.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    lines = [
        "# County cancer incidence vs EPA radon-potential screen",
        "",
        "Ecological screening only. The adjusted model includes state fixed effects and the",
        "State Cancer Profiles rural/urban class; it is not a causal or individual-risk model.",
        "Positive coefficients mean higher incidence with higher radon-potential category.",
        "",
        *markdown_rows,
        "",
    ]
    (output_dir / "radon_screen.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def add_geojson(joined: pd.DataFrame, county_geojson: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """Attach all joined site values to Census county features by GEOID."""
    geo = json.loads(county_geojson.read_text(encoding="utf-8"))
    wide = joined.pivot(index="fips", columns="site", values="incidence_rate")
    radon = joined.groupby("fips", as_index=True)["radon_zone"].first()
    for feature in geo.get("features", []):
        fips = str(feature.get("properties", {}).get("GEOID", feature.get("id", ""))).zfill(5)
        props = feature.setdefault("properties", {})
        props["radon_zone"] = None if fips not in radon or pd.isna(radon[fips]) else int(radon[fips])
        if fips in wide.index:
            for site, value in wide.loc[fips].items():
                props[f"cancer_{site}"] = None if pd.isna(value) else float(value)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "county_cancer_radon.geojson"
    path.write_text(json.dumps(geo, separators=(",", ":")), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", default=",".join(CANCER_SITES), help="Comma-separated site names")
    parser.add_argument("--workers", type=int, default=4, help="Conservative parallel download count")
    parser.add_argument("--county-geojson", type=Path, help="Optional Census county GeoJSON with GEOID")
    args = parser.parse_args()
    sites = [site.strip() for site in args.sites.split(",") if site.strip()]
    unknown = sorted(set(sites) - CANCER_SITES.keys())
    if unknown:
        raise SystemExit(f"Unknown sites: {', '.join(unknown)}")
    cancer = download_cancer(sites, workers=args.workers)
    joined = join_cancer_radon(cancer, download_radon())
    result = analyze(joined)
    print(result.to_string(index=False))
    if args.county_geojson:
        print(f"GeoJSON: {add_geojson(joined, args.county_geojson)}")


if __name__ == "__main__":
    main()
