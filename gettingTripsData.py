"""
streetlight_to_trips.py
=======================
Reads a StreetLight od_all CSV and the zone shapefile, extracts the
Wednesday 10am (10am-11am) hourly volume for every OD pair, and writes
a trips.txt in TNTP format.

The StreetLight data is already hourly (1-hour slot), so NO division is needed.

Usage:
    python gettingTripsData.py --year 2017
    python gettingTripsData.py --year 2019
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = "/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot"
ZONES_SHAPEFILE = os.path.join(BASE_DIR, "streetlight_zones 2/zones_for_streetlight.shp")

# Per-year StreetLight CSV paths
YEAR_CSV = {
    2017: os.path.join(BASE_DIR, "2018851_VMT2017/2018851_VMT2017_od_all.csv"),
    2019: os.path.join(BASE_DIR, "2018852_VMT2019/2018852_VMT2019_od_all.csv"),
    2024: os.path.join(BASE_DIR, "2018856_VMT2024/2018856_VMT2024_od_all.csv"),
}

# Output goes into the year-specific WithZones folder
YEAR_OUTPUT = {
    2017: os.path.join(BASE_DIR, "NetworkFiles2017_WithZones_Simplified/trips.txt"),
    2019: os.path.join(BASE_DIR, "NetworkFiles2019_WithZones_Simplified/trips.txt"),
    2024: os.path.join(BASE_DIR, "NetworkFiles2024_WithZones_Simplified/trips.txt"),
}


# =============================================================================
# 1. Load zone IDs from shapefile
# =============================================================================
def load_zone_ids(shp):
    print("[1] Loading zone IDs ...")
    gdf = gpd.read_file(shp)
    if "zone_id" in gdf.columns:
        ids = sorted(gdf["zone_id"].astype(int).unique())
    elif "name" in gdf.columns:
        ids = sorted(gdf["name"].str.replace("Zone_", "", regex=False).astype(int).unique())
    else:
        ids = sorted(gdf.index.tolist())
    print(f"    {len(ids)} zones  (IDs {ids[0]} … {ids[-1]})")
    return ids


# =============================================================================
# 2. Load StreetLight CSV  →  extract Wednesday 10am rows
# =============================================================================
def load_od_hourly(csv_path):
    """Return DataFrame with columns: origin_zone_id, dest_zone_id, od_volume.
    Filters for Wednesday + 10am (10am-11am).  Volume is already hourly."""
    print(f"[2] Loading StreetLight CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # normalise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if   "origin zone id"      in cl: col_map[c] = "origin_zone_id"
        elif "destination zone id" in cl: col_map[c] = "dest_zone_id"
        elif "day type"            in cl: col_map[c] = "day_type"
        elif "day part"            in cl: col_map[c] = "day_part"
        elif "o-d traffic"         in cl: col_map[c] = "od_volume"
    df = df.rename(columns=col_map)

    # keep only  Wednesday  +  10am (10am-11am)
    mask_wed  = df["day_type"].astype(str).str.contains("Wednesday", case=False, na=False)
    mask_10am = df["day_part"].astype(str).str.contains("10am", case=False, na=False)
    df = df[mask_wed & mask_10am].copy()

    df["od_volume"]        = pd.to_numeric(df["od_volume"], errors="coerce").fillna(0.0)
    df["origin_zone_id"]   = df["origin_zone_id"].astype(int)
    df["dest_zone_id"]     = df["dest_zone_id"].astype(int)

    print(f"    {len(df)} Wednesday 10am OD pairs")
    return df[["origin_zone_id", "dest_zone_id", "od_volume"]]


# =============================================================================
# 3. Build the single OD matrix  (hourly volume, no division)
# =============================================================================
def build_matrix(od_df, zone_ids):
    """N x N matrix.  Entry [i,j] = hourly volume (i->j).  Already 1-hour slot."""
    print("[3] Building OD matrix  (Wednesday 10am hourly) ...")
    n   = len(zone_ids)
    idx = {z: i for i, z in enumerate(zone_ids)}   # zone_id -> row/col index
    mat = np.zeros((n, n), dtype=np.float64)

    for _, row in od_df.iterrows():
        o, d = int(row["origin_zone_id"]), int(row["dest_zone_id"])
        if o == d or o not in idx or d not in idx:
            continue
        mat[idx[o], idx[d]] = row["od_volume"]   # already hourly, no /4

    nonzero = np.count_nonzero(mat)
    print(f"    {nonzero} non-zero OD pairs  |  total flow = {mat.sum():,.1f}")
    return mat


# =============================================================================
# 4. Write trips.txt  –  exact sample format
# =============================================================================
def write_trips(filepath, mat, zone_ids):
    """
    Format (verified byte-for-byte against the sample):
        <NUMBER OF ZONES> N
        <TOTAL OD FLOW> T.T
        <END OF METADATA>
        (blank)
        (blank)
        Origin \\t{1-based id} \\n
            {dest:5d} : {flow:8.1f}; {dest:5d} : {flow:8.1f}; ...   <- 5 entries / line
        (blank)
        Origin \\t...
        ...
        (3 trailing blank lines at EOF)

    Zone IDs in the file are 1-based  (shapefile 0-based + 1).
    """
    print(f"[4] Writing {filepath} ...")
    n              = len(zone_ids)
    zone_ids_1     = [z + 1 for z in zone_ids]   # 0-based -> 1-based
    total_flow     = float(mat.sum())

    with open(filepath, "w") as fh:
        # header
        fh.write(f"<NUMBER OF ZONES> {n}\n")
        fh.write(f"<TOTAL OD FLOW> {total_flow:.1f}\n")
        fh.write("<END OF METADATA>\n")
        fh.write("\n\n")

        # one block per origin
        for oi, oz in enumerate(zone_ids_1):
            # Filter out zero-flow entries
            entries = [(d, v) for d, v in zip(zone_ids_1, mat[oi]) if v > 0]
            
            # Skip origins with no outgoing trips
            if not entries:
                continue
                
            fh.write(f"Origin \t{oz} \n")

            # 5 entries per line, each exactly: f"{dest:5d} : {val:8.1f}; "
            for start in range(0, len(entries), 5):
                chunk = entries[start:start + 5]
                fh.write("".join(f"{d:5d} : {v:8.1f}; " for d, v in chunk) + "\n")

            fh.write("\n")   # blank line after each origin block

        # 2 extra trailing blank lines (sample has 3 blank lines at EOF total;
        # the last origin block already wrote one)
        fh.write("\n\n")

    nonzero_pairs = np.count_nonzero(mat)
    print(f"    Done.  {n} zones  |  {nonzero_pairs} non-zero OD pairs  |  total flow = {total_flow:,.1f}")


# =============================================================================
# Sample CSV generator (testing only)
# =============================================================================
def _generate_sample_csv(zone_ids, path, n_pairs=300):
    print(f"[SAMPLE] Generating synthetic CSV ...")
    np.random.seed(42)
    rows, pairs = [], set()
    while len(pairs) < n_pairs:
        o, d = int(np.random.choice(zone_ids)), int(np.random.choice(zone_ids))
        if o != d: pairs.add((o, d))
    for o, d in pairs:
        v_all  = round(np.random.exponential(120), 1)
        v_peak = round(v_all * np.random.uniform(0.15, 0.40), 1)
        for dt in ["All Days (M-Su)", "Monday (M-M)"]:
            for dp, vol in [("All Day (12am-12am)", v_all), ("Peak AM (6am-10am)", v_peak)]:
                rows.append({
                    "Origin Zone ID": o, "Origin Zone Name": f"Zone_{o}",
                    "Destination Zone ID": d, "Destination Zone Name": f"Zone_{d}",
                    "Day Type": dt, "Day Part": dp,
                    "Average Daily O-D Traffic (StL Volume)": vol,
                    "Origin Zone Is Pass-Through": "No",
                    "Origin Zone Direction (degrees)": "Null",
                    "Origin Zone is Bi-Direction": "No",
                    "Destination Zone Is Pass-Through": "No",
                    "Destination Zone Direction (degrees)": "Null",
                    "Destination Zone is Bi-Direction": "No",
                    "Data Periods": "May 01, 2024 - May 31, 2024",
                    "Mode of Travel": "All Vehicles CVD+",
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"    {len(rows)} rows -> {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Convert StreetLight OD CSV to trips.txt")
    parser.add_argument("--year", type=int, required=True, choices=list(YEAR_CSV.keys()),
                        help="Data year (2017, 2019, or 2024)")
    parser.add_argument("--zones", default=ZONES_SHAPEFILE,
                        help="Zone shapefile used by the StreetLight project")
    parser.add_argument("--output", default=None,
                        help="Output trips.txt path. Defaults to the year-specific statewide folder.")
    args = parser.parse_args()

    year = args.year
    csv_path = YEAR_CSV[year]
    output = args.output if args.output else YEAR_OUTPUT[year]

    print("=" * 60)
    print(f"  STREETLIGHT  ->  trips.txt   (Wednesday 10am, {year})")
    print("=" * 60)

    zone_ids = load_zone_ids(args.zones)

    od_df = load_od_hourly(csv_path)
    mat   = build_matrix(od_df, zone_ids)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    write_trips(output, mat, zone_ids)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
