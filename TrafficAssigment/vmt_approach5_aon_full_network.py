"""
Approach 5 - Hybrid VMT Refinement
====================================
Goal: Get realistic per-link and county-level VMT by re-loading the OD
demand onto the FULL un-simplified Minnesota network using all-or-nothing
(AON) shortest-path assignment on free-flow times.

Why this works:
    * The simplified-network UE assignment gives stable inter-city flows
      but distorts urban VMT because every metro is a single super-node.
    * Reloading the same OD matrix on the full network with AON recovers
      realistic urban link routes -- no equilibrium needed because we are
      only after VMT (a sum), not flow patterns at exact equilibrium.

Pipeline:
    1. Read full connected network (network.txt + nodes.txt)
    2. Read OD trip matrix from the simplified folder
    3. Read zone -> node mapping; resolve virtual super-node zones to
       the nearest real full-network node by Haversine distance
    4. Build sparse graph; run scipy Dijkstra from every origin zone node
    5. Reconstruct shortest paths and accumulate flow on each link
    6. Per-link VMT = flow * length;  per-link results saved to CSV
    7. Geometric county clipping (same logic as run_traffic_assignment.py)
       produces a county_vmt_full.csv

Output files (written into the year's NetworkFiles{year}_Connected folder):
    link_flows_full_aon.csv
    county_vmt_full_aon.csv
    unreachable_pairs_full_aon.csv  (only if any pair has no path)

Usage:
    python vmt_approach5_aon_full_network.py --year 2024
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from math import radians, sin, cos, asin, sqrt

import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_paths(year: str, repo_root: str) -> dict:
    full_dir = os.path.join(repo_root, f"NetworkFiles{year}_Connected")
    simplified_dir = os.path.join(repo_root, f"NetworkFiles{year}_WithZones_Simplified")
    return {
        "full_network_txt": os.path.join(full_dir, "network.txt"),
        "full_nodes_txt": os.path.join(full_dir, "nodes.txt"),
        "trips_txt": os.path.join(simplified_dir, "trips.txt"),
        "zone_mapping_csv": os.path.join(simplified_dir, "zone_node_mapping.csv"),
        "county_shp": os.path.join(repo_root, "mn_counties", "mn_counties.shp"),
        "out_dir": full_dir,
    }


# ---------------------------------------------------------------------------
# Network reader
# ---------------------------------------------------------------------------

def read_full_network(network_txt: str):
    """Parse the TNTP-style network file. Returns dicts of edges keyed by id."""
    init_nodes, term_nodes, lengths, ff_times = [], [], [], []
    with open(network_txt, "r") as f:
        in_data = False
        for line in f:
            s = line.strip()
            if s.startswith("<END OF METADATA>"):
                in_data = True
                continue
            if not in_data or s.startswith("~") or s.startswith("<") or len(s) == 0:
                continue
            parts = s.replace(";", "").split()
            if len(parts) < 6:
                continue
            init_nodes.append(int(parts[0]))
            term_nodes.append(int(parts[1]))
            # parts[2] = capacity, parts[3] = length, parts[4] = free-flow time
            lengths.append(float(parts[3]))
            ff_times.append(float(parts[4]))
    return (
        np.asarray(init_nodes, dtype=np.int64),
        np.asarray(term_nodes, dtype=np.int64),
        np.asarray(lengths, dtype=np.float64),
        np.asarray(ff_times, dtype=np.float64),
    )


def read_full_nodes(nodes_txt: str) -> dict:
    coords = {}
    with open(nodes_txt, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("~") or s.startswith("<") or len(s) == 0:
                continue
            parts = s.replace(";", "").split()
            if len(parts) >= 3:
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return coords


def read_trips(trips_txt: str):
    """TNTP-style trips file; returns dict origin_zone -> {dest_zone: demand}."""
    od = {}
    cur_origin = None
    with open(trips_txt, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("<") or s.startswith("~"):
                continue
            if s.lower().startswith("origin"):
                cur_origin = int(s.split()[1])
                od[cur_origin] = {}
                continue
            if cur_origin is None:
                continue
            for chunk in s.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                left, right = chunk.split(":")
                dest = int(left.strip())
                demand = float(right.strip())
                if demand > 0:
                    od[cur_origin][dest] = od[cur_origin].get(dest, 0.0) + demand
    return od


def read_zone_mapping(csv_path: str):
    """Returns list of dicts with keys zone_id, original_node_id, x, y, is_virtual."""
    rows = []
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split(",")
            rows.append(
                {
                    "zone_id": int(parts[idx["zone_id"]]),
                    "original_node_id": int(parts[idx["original_node_id"]]),
                    "x": float(parts[idx["x"]]),
                    "y": float(parts[idx["y"]]),
                    "is_virtual": parts[idx["is_virtual"]].strip().lower() == "yes",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Virtual zone -> nearest real node
# ---------------------------------------------------------------------------

def haversine_miles(lon1, lat1, lon2, lat2):
    R = 3958.7613  # Earth radius in miles
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def resolve_zone_nodes(zones, node_coords, valid_node_ids):
    """For each zone return a node_id from the FULL network.

    Non-virtual zones: take original_node_id (verify it exists).
    Virtual zones: nearest full-network node (by Haversine) to the super-node
    centroid; needed because the super-node id "1" placeholder doesn't map
    into the connected network.
    """
    valid_set = set(valid_node_ids)
    coords_arr = np.array([node_coords[n] for n in valid_node_ids])  # (N, 2) lon,lat
    valid_arr = np.asarray(valid_node_ids, dtype=np.int64)

    zone_to_node = {}
    for z in zones:
        zid = z["zone_id"]
        if not z["is_virtual"] and z["original_node_id"] in valid_set:
            zone_to_node[zid] = z["original_node_id"]
            continue
        # Nearest by squared euclidean (fast + monotonic with great-circle for short ranges,
        # then refine with Haversine for the top-K candidates)
        dx = coords_arr[:, 0] - z["x"]
        dy = coords_arr[:, 1] - z["y"]
        approx = dx * dx + dy * dy
        k = min(50, len(approx))
        top = np.argpartition(approx, k - 1)[:k]
        best_node, best_d = None, float("inf")
        for j in top:
            nid = int(valid_arr[j])
            d = haversine_miles(z["x"], z["y"], coords_arr[j, 0], coords_arr[j, 1])
            if d < best_d:
                best_d = d
                best_node = nid
        zone_to_node[zid] = best_node
    return zone_to_node


# ---------------------------------------------------------------------------
# AON shortest-path loading (scipy)
# ---------------------------------------------------------------------------

def run_aon(init_nodes, term_nodes, ff_times, zone_to_node, od, num_nodes):
    """All-or-nothing: for every origin run Dijkstra, then walk predecessor
    pointers from each destination to accumulate flow on each link.

    Returns:
        link_flow : np.ndarray of shape (num_links,)
        unreachable : list of (origin_zone, dest_zone, demand)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    num_links = len(init_nodes)

    # Build sparse adjacency. Use 0..N-1 indices internally. Node IDs in TNTP
    # are 1-based but contiguous, so simple offset by 1.
    # Validate all referenced ids fit.
    max_id = max(int(init_nodes.max()), int(term_nodes.max()))
    if max_id > num_nodes:
        raise ValueError(
            f"Network references node id {max_id} but only {num_nodes} nodes loaded"
        )

    rows = init_nodes - 1
    cols = term_nodes - 1

    graph = csr_matrix(
        (ff_times, (rows, cols)),
        shape=(num_nodes, num_nodes),
    )

    # Build a quick lookup (init_idx, term_idx) -> link_id for path -> flow accumulation.
    # Use a dict keyed by (u, v); if there are parallel arcs, keep the cheapest one
    # so the lookup is consistent with what dijkstra actually used.
    edge_index = {}
    for lid in range(num_links):
        key = (int(rows[lid]), int(cols[lid]))
        prev = edge_index.get(key)
        if prev is None or ff_times[lid] < ff_times[prev]:
            edge_index[key] = lid

    link_flow = np.zeros(num_links, dtype=np.float64)
    unreachable = []

    origins = sorted(o for o in od if od[o])
    n_origins = len(origins)
    print(f"  Running AON over {n_origins} origins...")

    t0 = time.time()
    for k, ozone in enumerate(origins):
        onode = zone_to_node.get(ozone)
        if onode is None:
            for d, q in od[ozone].items():
                unreachable.append((ozone, d, q))
            continue

        dist, pred = dijkstra(
            graph,
            directed=True,
            indices=onode - 1,
            return_predecessors=True,
        )

        for dzone, demand in od[ozone].items():
            dnode = zone_to_node.get(dzone)
            if dnode is None or not np.isfinite(dist[dnode - 1]):
                unreachable.append((ozone, dzone, demand))
                continue
            # Walk predecessors from dest back to origin, adding demand to each link
            cur = dnode - 1
            origin_idx = onode - 1
            while cur != origin_idx:
                p = pred[cur]
                if p < 0:
                    break  # shouldn't happen for finite dist
                lid = edge_index.get((int(p), int(cur)))
                if lid is not None:
                    link_flow[lid] += demand
                cur = p

        if (k + 1) % max(1, n_origins // 20) == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (n_origins - (k + 1)) / rate if rate > 0 else 0
            print(
                f"    {k + 1:>5}/{n_origins} origins  "
                f"({elapsed:6.1f}s elapsed, ETA {eta:5.0f}s)"
            )

    return link_flow, unreachable


# ---------------------------------------------------------------------------
# County VMT (geometric clipping; identical logic to the main runner)
# ---------------------------------------------------------------------------

def county_clip_vmt(
    init_nodes,
    term_nodes,
    lengths,
    link_flow,
    node_coords,
    county_shp,
):
    import geopandas as gpd
    import pyproj
    from shapely.geometry import LineString, Point
    from shapely.ops import transform as shapely_transform
    from shapely.prepared import prep

    print("  Reading county boundaries...")
    counties = gpd.read_file(county_shp).to_crs("EPSG:4326")
    sindex = counties.sindex
    prep_geoms = [prep(g) for g in counties.geometry]
    counties_utm = counties.to_crs("EPSG:26915")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)
    project = transformer.transform

    def utm_length(geom):
        if geom.is_empty:
            return 0.0
        return shapely_transform(project, geom).length

    print("  Assigning nodes to counties...")
    node_county = {}
    cache = {}
    for nid, (lon, lat) in node_coords.items():
        key = (round(lon, 6), round(lat, 6))
        if key in cache:
            node_county[nid] = cache[key]
            continue
        pt = Point(lon, lat)
        cand = list(sindex.intersection(pt.bounds))
        cname = None
        for idx in cand:
            if prep_geoms[idx].contains(pt):
                cname = counties.iloc[idx]["NAME"]
                break
        if cname is None:
            pt_utm = shapely_transform(project, pt)
            d = counties_utm.geometry.distance(pt_utm)
            cname = counties.loc[d.idxmin(), "NAME"]
        cache[key] = cname
        node_county[nid] = cname
    print(f"    {len(node_county)} nodes assigned across {len(set(node_county.values()))} counties")

    print("  Clipping links to counties...")
    county_vmt = {}
    cross_links = 0
    n = len(init_nodes)
    report_every = max(1, n // 10)

    for i in range(n):
        if (i + 1) % report_every == 0:
            print(f"    link {i + 1:,}/{n:,}")
        flow = link_flow[i]
        if flow <= 0:
            continue
        link_vmt = flow * lengths[i]
        u = int(init_nodes[i])
        v = int(term_nodes[i])
        cu = node_county.get(u)
        cv = node_county.get(v)

        if cu and cv and cu == cv:
            county_vmt[cu] = county_vmt.get(cu, 0.0) + link_vmt
            continue

        pu = node_coords.get(u)
        pv = node_coords.get(v)
        if not pu or not pv:
            fb = cu or cv
            if fb:
                county_vmt[fb] = county_vmt.get(fb, 0.0) + link_vmt
            continue

        line = LineString([pu, pv])
        cand = list(sindex.intersection(line.bounds))
        seg = {}
        for idx in cand:
            clipped = line.intersection(counties.geometry.iloc[idx])
            if clipped.is_empty:
                continue
            m = utm_length(clipped)
            if m > 0:
                cname = counties.iloc[idx]["NAME"]
                seg[cname] = seg.get(cname, 0.0) + m
        if seg:
            tot = sum(seg.values())
            for cname, m in seg.items():
                county_vmt[cname] = county_vmt.get(cname, 0.0) + link_vmt * (m / tot)
            if len(seg) > 1:
                cross_links += 1
        else:
            fb = cu or cv
            if fb:
                county_vmt[fb] = county_vmt.get(fb, 0.0) + link_vmt
    print(f"    cross-county links: {cross_links:,}")
    return county_vmt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", "-y", required=True, choices=["2017", "2019", "2024"])
    ap.add_argument(
        "--cost",
        choices=["ff_time", "length"],
        default="ff_time",
        help="Edge weight for shortest paths (default: free-flow time)",
    )
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = resolve_paths(args.year, repo_root)

    for k in ("full_network_txt", "full_nodes_txt", "trips_txt", "zone_mapping_csv"):
        if not os.path.exists(paths[k]):
            print(f"ERROR: missing input: {paths[k]}")
            sys.exit(1)

    print("=" * 60)
    print(f"Approach 5: AON reload on full network ({args.year})")
    print("=" * 60)

    print("\n[1/5] Reading full network...")
    init_nodes, term_nodes, lengths, ff_times = read_full_network(paths["full_network_txt"])
    print(f"      {len(init_nodes):,} links")

    print("[2/5] Reading nodes...")
    node_coords = read_full_nodes(paths["full_nodes_txt"])
    print(f"      {len(node_coords):,} nodes")

    print("[3/5] Reading OD matrix and zone mapping...")
    od = read_trips(paths["trips_txt"])
    zones = read_zone_mapping(paths["zone_mapping_csv"])
    valid_ids = sorted(node_coords.keys())
    zone_to_node = resolve_zone_nodes(zones, node_coords, valid_ids)
    n_virtual = sum(1 for z in zones if z["is_virtual"])
    print(f"      {len(zones)} zones ({n_virtual} virtual super-nodes resolved to nearest real node)")
    total_demand = sum(sum(d.values()) for d in od.values())
    print(f"      Total OD demand: {total_demand:,.1f}")

    print(f"[4/5] AON shortest-path loading (cost = {args.cost})...")
    edge_cost = ff_times if args.cost == "ff_time" else lengths
    num_nodes = max(node_coords.keys())
    link_flow, unreachable = run_aon(
        init_nodes, term_nodes, edge_cost, zone_to_node, od, num_nodes
    )

    out_dir = paths["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("[5/5] Writing outputs...")

    # ---- Per-link CSV ----
    link_csv = os.path.join(out_dir, "link_flows_full_aon.csv")
    with open(link_csv, "w") as f:
        f.write("link_id,init_node,term_node,flow,length,vmt\n")
        for i in range(len(init_nodes)):
            f.write(
                f"{i},{int(init_nodes[i])},{int(term_nodes[i])},"
                f"{link_flow[i]:.4f},{lengths[i]:.6f},"
                f"{link_flow[i] * lengths[i]:.4f}\n"
            )
    print(f"      link flows -> {link_csv}")

    # ---- Unreachable ----
    if unreachable:
        un_csv = os.path.join(out_dir, "unreachable_pairs_full_aon.csv")
        lost = sum(q for _, _, q in unreachable)
        with open(un_csv, "w") as f:
            f.write("origin_zone,dest_zone,demand\n")
            for o, d, q in unreachable:
                f.write(f"{o},{d},{q:.4f}\n")
        print(
            f"      unreachable -> {un_csv}  "
            f"({len(unreachable):,} pairs, {lost:,.1f} demand lost)"
        )

    # ---- Totals ----
    total_vmt = float((link_flow * lengths).sum())
    print(f"\n   TOTAL NETWORK VMT (AON, full network): {total_vmt:,.2f}")

    # ---- County VMT (optional, requires geopandas) ----
    if os.path.exists(paths["county_shp"]):
        try:
            print("\n  Computing county-level VMT...")
            county_vmt = county_clip_vmt(
                init_nodes, term_nodes, lengths, link_flow, node_coords, paths["county_shp"]
            )
            tot = sum(county_vmt.values())
            cnty_csv = os.path.join(out_dir, "county_vmt_full_aon.csv")
            with open(cnty_csv, "w") as f:
                f.write("county,vmt,pct_of_total\n")
                for c, v in sorted(county_vmt.items(), key=lambda x: -x[1]):
                    pct = v / tot * 100 if tot > 0 else 0
                    f.write(f"{c},{v:.2f},{pct:.4f}\n")
            print(f"      county VMT -> {cnty_csv}")
            print("\n  Top 10 counties by VMT:")
            print(f"  {'County':<25}{'VMT':>15}{'%':>10}")
            print("  " + "-" * 50)
            for c, v in sorted(county_vmt.items(), key=lambda x: -x[1])[:10]:
                pct = v / tot * 100 if tot > 0 else 0
                print(f"  {c:<25}{v:>15,.0f}{pct:>9.2f}%")
        except ImportError as e:
            print(f"  Skipping county VMT: missing dependency ({e})")
        except Exception as e:
            print(f"  County VMT failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  County shapefile not found at {paths['county_shp']}, skipping clip")

    print("\nDone.")


if __name__ == "__main__":
    main()
