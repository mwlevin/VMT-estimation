"""
Approach 1 - HPMS Urban VMT Add-Back
======================================
The simplified statewide network collapses each major metro into a single
super-node, so all intra-metro VMT is missing from the assignment results.
This script adds back the missing urban VMT using observed values from
MnDOT HPMS / DVMT (Daily Vehicle Miles Travelled) reports.

Method:
    1. Load the simplified-network county VMT (county_vmt.csv produced by
       run_traffic_assignment.py).
    2. For each city/metro super-node, look up the historical observed
       VMT for the counties that the metro covers (from the URBAN_VMT
       table below -- fill in MnDOT HPMS numbers per year).
    3. Add the urban VMT to the corresponding county totals.
    4. Write the augmented county_vmt to a new CSV.

Notes:
    * URBAN_VMT below is keyed by (year, metro_name) and gives a dict of
      {county_name: daily_vmt} representing the urban portion of that
      county's VMT that the simplified network is missing. You can get
      these from the MnDOT HPMS county DVMT reports - typically the
      "urban" or "MPO" share for the relevant counties.
    * The script does NOT subtract anything from the simplified totals --
      it only ADDS urban shares back. The assumption is that the simplified
      network already represents all inter-city/rural VMT correctly, so
      what's missing is the intra-metro driving.

Usage:
    python vmt_approach1_hpms_addback.py --year 2024
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


# ---------------------------------------------------------------------------
# HPMS / DVMT urban VMT lookup table
# ---------------------------------------------------------------------------
# Fill in observed daily VMT (or AADT * length sums) per county for the
# urban / metro portion that gets collapsed into a super-node in the
# simplified network. Source: MnDOT HPMS county DVMT reports.
#
# Structure:
#   URBAN_VMT[year][metro_name] = {county_name: vmt_value}
#
# County names must match the NAME field in mn_counties.shp exactly.
# The counties listed for each metro should correspond to the bounding
# box used in simplify_network_cities.py (CITY_BOUNDARIES).
#
# TODO: replace the None placeholders with values from MnDOT reports.
# ---------------------------------------------------------------------------

URBAN_VMT: dict = {
    "2017": {
        "TwinCities_Metro": {
            # 7-county Twin Cities metro
            "Hennepin": None,
            "Ramsey": None,
            "Anoka": None,
            "Dakota": None,
            "Washington": None,
            "Carver": None,
            "Scott": None,
        },
        "Rochester": {"Olmsted": None},
        "Duluth": {"St. Louis": None},
        "St_Cloud": {"Stearns": None, "Sherburne": None, "Benton": None},
        "Mankato": {"Blue Earth": None, "Nicollet": None},
        "Moorhead": {"Clay": None},
    },
    "2019": {
        "TwinCities_Metro": {
            "Hennepin": None,
            "Ramsey": None,
            "Anoka": None,
            "Dakota": None,
            "Washington": None,
            "Carver": None,
            "Scott": None,
        },
        "Rochester": {"Olmsted": None},
        "Duluth": {"St. Louis": None},
        "St_Cloud": {"Stearns": None, "Sherburne": None, "Benton": None},
        "Mankato": {"Blue Earth": None, "Nicollet": None},
        "Moorhead": {"Clay": None},
    },
    "2024": {
        "TwinCities_Metro": {
            "Hennepin": None,
            "Ramsey": None,
            "Anoka": None,
            "Dakota": None,
            "Washington": None,
            "Carver": None,
            "Scott": None,
        },
        "Rochester": {"Olmsted": None},
        "Duluth": {"St. Louis": None},
        "St_Cloud": {"Stearns": None, "Sherburne": None, "Benton": None},
        "Mankato": {"Blue Earth": None, "Nicollet": None},
        "Moorhead": {"Clay": None},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_county_vmt(csv_path: str) -> dict:
    """Read county_vmt.csv into {county_name: vmt}."""
    out = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["county"]] = float(row["vmt"])
    return out


def aggregate_urban_addback(year: str) -> dict:
    """Sum the URBAN_VMT contributions across all metros into a single
    {county: addback_vmt} dict. Skips entries that are still None."""
    if year not in URBAN_VMT:
        raise KeyError(f"No URBAN_VMT entries for year {year}")
    total = {}
    missing = []
    for metro, county_map in URBAN_VMT[year].items():
        for county, vmt in county_map.items():
            if vmt is None:
                missing.append((metro, county))
                continue
            total[county] = total.get(county, 0.0) + vmt
    return total, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", "-y", required=True, choices=["2017", "2019", "2024"])
    ap.add_argument(
        "--simplified-vmt",
        default=None,
        help="Path to county_vmt.csv produced by run_traffic_assignment.py "
        "(default: NetworkFiles{year}_WithZones_Simplified/county_vmt.csv)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Where to write the augmented CSV "
        "(default: same folder as input, named county_vmt_with_hpms_addback.csv)",
    )
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.simplified_vmt:
        in_csv = args.simplified_vmt
    else:
        in_csv = os.path.join(
            repo_root,
            f"NetworkFiles{args.year}_WithZones_Simplified",
            "county_vmt.csv",
        )

    if not os.path.exists(in_csv):
        print(f"ERROR: input not found: {in_csv}")
        sys.exit(1)

    out_csv = args.output or os.path.join(
        os.path.dirname(in_csv), "county_vmt_with_hpms_addback.csv"
    )

    print("=" * 60)
    print(f"Approach 1: HPMS urban VMT add-back ({args.year})")
    print("=" * 60)

    sim_vmt = read_county_vmt(in_csv)
    print(f"  Loaded simplified VMT for {len(sim_vmt)} counties from {in_csv}")

    addback, missing = aggregate_urban_addback(args.year)
    print(f"  Urban add-back values: {len(addback)} county entries set")
    if missing:
        print(f"  WARNING: {len(missing)} HPMS values still placeholder (None):")
        for m, c in missing:
            print(f"     - {m} / {c}")
        print("    -> rerun this script after filling URBAN_VMT in the source")

    # Combine
    augmented = dict(sim_vmt)
    for county, add in addback.items():
        augmented[county] = augmented.get(county, 0.0) + add

    total_sim = sum(sim_vmt.values())
    total_aug = sum(augmented.values())
    total_added = sum(addback.values())

    # Write output
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["county", "simplified_vmt", "hpms_urban_addback", "total_vmt", "pct_of_total"])
        rows = []
        for county in sorted(augmented):
            sv = sim_vmt.get(county, 0.0)
            av = addback.get(county, 0.0)
            tv = augmented[county]
            rows.append((county, sv, av, tv))
        rows.sort(key=lambda r: -r[3])
        for county, sv, av, tv in rows:
            pct = tv / total_aug * 100 if total_aug > 0 else 0
            w.writerow([county, f"{sv:.2f}", f"{av:.2f}", f"{tv:.2f}", f"{pct:.4f}"])

    print(f"\n  Wrote augmented county VMT -> {out_csv}")
    print(f"\n  Simplified total VMT : {total_sim:,.2f}")
    print(f"  HPMS urban add-back  : {total_added:,.2f}")
    print(f"  Augmented total VMT  : {total_aug:,.2f}")

    print("\n  Top 10 counties (augmented):")
    print(f"    {'County':<20}{'Simplified':>15}{'+HPMS':>15}{'Total':>15}")
    print("    " + "-" * 65)
    for county, sv, av, tv in rows[:10]:
        print(f"    {county:<20}{sv:>15,.0f}{av:>15,.0f}{tv:>15,.0f}")


if __name__ == "__main__":
    main()
