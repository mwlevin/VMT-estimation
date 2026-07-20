"""Compare 2017 / 2019 / 2024 network topologies and assignment results.

All three years used the SAME demand (2024 OD), so any VMT differences
reflect ONLY changes in network topology, capacity, and free-flow speeds.

Outputs: network_comparison.csv  +  console summary.
"""
from __future__ import annotations
import os
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
YEARS = [2017, 2019, 2024]


def parse_network(path: Path) -> pd.DataFrame:
    """Parse TNTP-style network.txt into a DataFrame."""
    rows = []
    with open(path) as f:
        in_data = False
        for line in f:
            s = line.strip()
            if not s or s.startswith("~"):
                continue
            if s.startswith("<"):
                if "END OF METADATA" in s:
                    in_data = True
                continue
            if not in_data:
                continue
            parts = s.rstrip(";").split()
            if len(parts) < 10:
                continue
            try:
                rows.append({
                    "init": int(parts[0]),
                    "term": int(parts[1]),
                    "capacity": float(parts[2]),
                    "length": float(parts[3]),
                    "fft": float(parts[4]),
                })
            except ValueError:
                continue
    return pd.DataFrame(rows)


def parse_nodes(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("~"):
                continue
            parts = s.rstrip(";").split()
            if len(parts) < 3:
                continue
            try:
                rows.append({"node": int(parts[0]), "x": float(parts[1]), "y": float(parts[2])})
            except ValueError:
                continue
    return pd.DataFrame(rows)


def load_link_flows(year: int) -> pd.DataFrame | None:
    p = ROOT / f"NetworkFiles{year}_WithZones_Simplified" / "link_flows.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def load_county_addback(year: int) -> pd.DataFrame | None:
    p = ROOT / f"NetworkFiles{year}_WithZones_Simplified" / "county_vmt_subarea_addback.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    summaries = []
    link_sets: dict[int, set] = {}
    nets: dict[int, pd.DataFrame] = {}
    nodes: dict[int, pd.DataFrame] = {}

    for y in YEARS:
        net_path = ROOT / f"NetworkFiles{y}_Connected" / "network.txt"
        nodes_path = ROOT / f"NetworkFiles{y}_Connected" / "nodes.txt"
        if not net_path.exists():
            print(f"missing {net_path}")
            continue
        net = parse_network(net_path)
        nd = parse_nodes(nodes_path)
        nets[y] = net
        nodes[y] = nd

        # Compute geographic length per link from node coords (miles)
        coord = nd.set_index("node")[ ["x", "y"] ]
        a = coord.reindex(net["init"]).reset_index(drop=True)
        b = coord.reindex(net["term"]).reset_index(drop=True)
        # x=lon, y=lat
        net["geo_miles"] = [
            haversine_miles(a.iloc[i, 1], a.iloc[i, 0], b.iloc[i, 1], b.iloc[i, 0])
            for i in range(len(net))
        ]

        link_sets[y] = set(zip(net["init"], net["term"]))

        summary = {
            "year": y,
            "nodes": len(nd),
            "links": len(net),
            "total_lane_miles_geo": net["geo_miles"].sum(),
            "total_capacity_link_hr": net["capacity"].sum(),
            "mean_capacity": net["capacity"].mean(),
            "mean_fft_min": net["fft"].mean(),
            "mean_geo_miles": net["geo_miles"].mean(),
        }
        summaries.append(summary)

    df = pd.DataFrame(summaries).set_index("year")
    print("\n=== Network topology comparison (same 2024 demand applied to all) ===")
    print(df.round(2).to_string())

    # Pairwise link diffs
    print("\n=== Link set differences ===")
    pairs = [(2017, 2019), (2019, 2024), (2017, 2024)]
    for a, b in pairs:
        if a in link_sets and b in link_sets:
            only_a = link_sets[a] - link_sets[b]
            only_b = link_sets[b] - link_sets[a]
            common = link_sets[a] & link_sets[b]
            print(f"{a} vs {b}:  common={len(common):>7}  only_in_{a}={len(only_a):>6}  only_in_{b}={len(only_b):>6}")

    # Try to attach assignment results
    print("\n=== Assignment results (simplified network) ===")
    rows = []
    for y in YEARS:
        lf = load_link_flows(y)
        if lf is None:
            print(f"{y}: link_flows.csv not found")
            continue
        vmt_simpl = lf["vmt"].sum() if "vmt" in lf.columns else (lf["flow_real"] * lf["length"]).sum()
        vht = lf["vht"].sum() if "vht" in lf.columns else float("nan")
        loaded = (lf["flow_real"] > 0).sum()
        mean_vc = lf["volume_capacity_ratio"].mean() if "volume_capacity_ratio" in lf.columns else float("nan")
        congested = (lf["volume_capacity_ratio"] > 0.9).sum() if "volume_capacity_ratio" in lf.columns else 0
        cab = load_county_addback(y)
        total_vmt = cab["total_vmt"].sum() if cab is not None else float("nan")
        urban = cab["urban_subarea_vmt"].sum() if cab is not None else float("nan")
        rows.append({
            "year": y,
            "simplified_vmt": vmt_simpl,
            "urban_subarea_vmt": urban,
            "total_vmt": total_vmt,
            "vht": vht,
            "loaded_links": loaded,
            "mean_v_c": mean_vc,
            "congested_links_vc>0.9": congested,
        })
    if rows:
        adf = pd.DataFrame(rows).set_index("year")
        print(adf.round(2).to_string())
        adf.to_csv(ROOT / "TrafficAssigment" / "assignment_comparison.csv")

    out = ROOT / "TrafficAssigment" / "network_comparison.csv"
    df.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
