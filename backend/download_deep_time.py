#!/usr/bin/env python3
"""
Download deep time datasets for testing the subharmonic cascade.

Datasets:
1. EPICA Dome C deuterium (800kyr climate proxy)
2. LR04 benthic d18O (5.3Myr climate/ice volume)
3. GRIP 10Be (cosmic ray proxy, covers Laschamp)
4. PISO-1500 (1.5Myr paleointensity from PANGAEA)
5. Volcanic eruption database (Smithsonian GVP)
"""
import requests
import os
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "deep_time"
OUT.mkdir(parents=True, exist_ok=True)


def download(url, filename, desc=""):
    path = OUT / filename
    if path.exists():
        print(f"  [skip] {filename} (exists, {path.stat().st_size // 1024}KB)")
        return True
    print(f"  Downloading {desc or filename}...")
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "GlobalResonance/1.0"})
        if resp.status_code == 200:
            path.write_bytes(resp.content)
            print(f"  [ok] {filename} ({len(resp.content) // 1024}KB)")
            return True
        else:
            print(f"  [FAIL] HTTP {resp.status_code} for {url}")
            return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    print("=" * 70)
    print("  DOWNLOADING DEEP TIME DATASETS")
    print("=" * 70)

    # 1. EPICA Dome C (800kyr deuterium -> temperature)
    download(
        "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/epica_domec/edc3deuttemp2007.txt",
        "epica_domec_deuterium.txt",
        "EPICA Dome C deuterium 800kyr"
    )

    # 2. LR04 benthic d18O stack (5.3Myr)
    download(
        "https://lorraine-lisiecki.com/LR04stack.txt",
        "lr04_d18o_5.3myr.txt",
        "LR04 benthic d18O 5.3Myr"
    )

    # 3. GRIP cosmogenic isotopes (10Be, 36Cl)
    download(
        "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/cosmoiso/grip10be-36cl.txt",
        "grip_10be_36cl.txt",
        "GRIP 10Be/36Cl cosmogenic isotopes"
    )

    # 3b. Try alternative GRIP 10Be location
    download(
        "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/cosmoiso/",
        "grip_cosmoiso_index.html",
        "GRIP cosmoiso directory listing"
    )

    # 4. PISO-1500 from PANGAEA (1.5Myr paleointensity)
    download(
        "https://doi.pangaea.de/10.1594/PANGAEA.890724?format=textfile",
        "piso1500_paleointensity.txt",
        "PISO-1500 1.5Myr paleointensity"
    )

    # 5. Vostok ice core (420kyr, CO2 + deuterium)
    download(
        "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/deutnat.txt",
        "vostok_deuterium.txt",
        "Vostok deuterium 420kyr"
    )

    # 6. GISP2 10Be (longer Greenland record)
    download(
        "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/chem/gisp2_be10.txt",
        "gisp2_10be.txt",
        "GISP2 10Be cosmic ray proxy"
    )

    # 7. Volcanic explosivity index (Smithsonian GVP)
    download(
        "https://volcano.si.edu/downloads/GVP_Eruption_Results.csv",
        "gvp_eruptions.csv",
        "Smithsonian GVP eruption database"
    )

    # 8. NGRIP d18O (123kyr, annual to 60ka)
    download(
        "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/ngrip/ngrip-d18o-50yr.txt",
        "ngrip_d18o_50yr.txt",
        "NGRIP d18O 123kyr"
    )

    # List what we got
    print(f"\n{'='*70}")
    print(f"  DOWNLOADED FILES:")
    print(f"{'='*70}")
    for f in sorted(OUT.glob("*")):
        if f.is_file():
            print(f"  {f.name:40s} {f.stat().st_size // 1024:6d} KB")


if __name__ == "__main__":
    main()
