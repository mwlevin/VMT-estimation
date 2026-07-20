"""
Approach 5 (Sub-area variant) - Urban VMT via Sub-network AON
==============================================================
Much faster alternative to full-network AON: for each metro area, run
a tiny shortest-path assignment ONLY on the sub-network inside the city
rectangle, using boundary flows from the converged simplified-network
assignment as the OD demand.

Why it works
------------
The simplified UE assignment already gives us, for every metro super-node,
how many vehicles flow IN through each border link and how many flow OUT.
Inside the super-node, those vehicles must travel from their entry boundary
node to their exit boundary node. We don't know each individual O-D pair,
so we estimate:

    OD[in_node, out_node] = inflow[in_node] * outflow[out_node] / total_out

This is a gravity-style proportional split (same total inflow & outflow
preserved). Then we run AON on the small sub-network and accumulate
flow * length to get urban VMT.

What this captures
------------------
+ Through-trips and trips that start/end at the city boundary
- Resident-internal trips (both ends inside city) are NOT captured;
  those need to be added later via HPMS data or a separate intra-city
  trips matrix (see --internal-trips option, future).

Pipeline
--------
1. Read simplified-network nodes.txt -> identify super-nodes (orig_id=-1)
   and group them by city based on coordinates from CITY_BOUNDARIES.
2. Read link_flows.csv from the simplified run -> get converged flows
   (column flow_real, already de-scaled).
3. Find boundary links (super-node end on one side, regular node on the
   other) and their corresponding full-network boundary node IDs.
4. Read full-network network.txt and nodes.txt; extract nodes in each
   city bbox and the links among them as the sub-network.
5. For each city:
       * Build inflow/outflow vectors at boundary full-network nodes
       * Build OD via proportional split
       * Run scipy Dijkstra from each in-boundary node and accumulate
         flow on internal links. The replacement urban graph is treated as
         undirected so directional coding gaps in the full network do not
         falsely mark physical urban paths as unreachable.
       * Compute urban VMT and link-level CSV
6. Aggregate urban VMT into county totals and write a combined CSV.

Outputs (in NetworkFiles{year}_WithZones_Simplified/):
    urban_link_flows_<city>.csv     (one per metro)
    urban_vmt_summary.csv           (one row per metro)
    county_vmt_subarea_addback.csv  (county VMT with urban added back)

Usage:
    python vmt_approach5_subarea_aon.py --year 2024
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# City definitions (mirrors simplify_network_cities.py)
# ---------------------------------------------------------------------------
# Coordinates here are matched against the super-node coordinates in the
# simplified nodes.txt (lon, lat).
CITY_BOUNDARIES = {
    "TwinCities_Metro": {
        "bbox": (-93.85, 44.55, -92.75, 45.40),
        "super_node_xy": (-93.17, 44.96),
    },
    "Rochester": {
        "bbox": (-92.65, 43.85, -92.25, 44.20),
        "super_node_xy": (-92.466, 44.023),
    },
    "Duluth": {
        "bbox": (-92.45, 46.50, -91.80, 47.00),
        "super_node_xy": (-92.106, 46.786),
    },
    "St_Cloud": {
        "bbox": (-94.40, 45.40, -93.90, 45.70),
        "super_node_xy": (-94.16, 45.56),
    },
    "Mankato": {
        "bbox": (-94.10, 44.05, -93.85, 44.35),
        "super_node_xy": (-93.99, 44.17),
    },
    "Moorhead": {
        "bbox": (-96.85, 46.65, -96.40, 46.95),
        "super_node_xy": (-96.74, 46.87),
    },
}

XY_TOL = 1e-3  # tolerance for matching super-node coords to a city


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def read_simplified_nodes(path: str):
    """Return dict simplified_id -> (original_id, x, y).
    original_id may be -1 (virtual super-node placeholder), a small integer
    1..6 (city marker), or an actual full-network node id."""
    out = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("~") or s.startswith("<"):
                continue
            parts = s.replace(";", "").split()
            if len(parts) >= 4:
                try:
                    sid = int(parts[0])
                    # original_id may be written as float like "1.0"
                    orig = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                except ValueError:
                    continue
                out[sid] = (orig, x, y)
    return out


def read_link_flows(path: str):
    """Yield dicts of selected fields from link_flows.csv."""
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "init_node": int(row["init_node"]),
                    "term_node": int(row["term_node"]),
                    "flow_real": float(row["flow_real"]),
                    "length": float(row["length"]),
                }
            )
    return rows


def read_full_network(network_txt: str):
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
            lengths.append(float(parts[3]))
            ff_times.append(float(parts[4]))
    return (
        np.asarray(init_nodes, dtype=np.int64),
        np.asarray(term_nodes, dtype=np.int64),
        np.asarray(lengths, dtype=np.float64),
        np.asarray(ff_times, dtype=np.float64),
    )


def read_full_nodes(path: str):
    coords = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("~") or s.startswith("<"):
                continue
            parts = s.replace(";", "").split()
            if len(parts) >= 3:
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return coords


# ---------------------------------------------------------------------------
# Step 1: classify simplified super-nodes by city
# ---------------------------------------------------------------------------

def classify_super_nodes(simp_nodes: dict) -> dict:
    """Return dict super_simp_id -> city_name.

    A simplified node is treated as a super-node if EITHER:
      (a) its original_id is a small marker value (-1 or 1..6 indicating
          "virtual" placeholder), OR
      (b) its coordinates match a city's super_node_xy within XY_TOL.
    """
    out = {}
    for sid, (orig, x, y) in simp_nodes.items():
        is_marker = orig == -1 or (1 <= orig <= 6)
        for city, cfg in CITY_BOUNDARIES.items():
            sx, sy = cfg["super_node_xy"]
            if abs(x - sx) < XY_TOL and abs(y - sy) < XY_TOL:
                # match by coordinate (works regardless of marker convention)
                out[sid] = city
                break
        else:
            if is_marker:
                # marker but no coord match -- ignore (shouldn't happen)
                pass
    return out


# ---------------------------------------------------------------------------
# Step 2: build per-city boundary inflow/outflow at full-network nodes
# ---------------------------------------------------------------------------

def build_boundary_flows(link_rows, simp_nodes, super_to_city, full_coords=None):
    """For each city, return:
        inflow_full[full_node_id]  = vehicles ENTERING the city through this node
        outflow_full[full_node_id] = vehicles LEAVING the city through this node

    A "boundary link" is one where exactly one endpoint is a super-node.
    The non-super endpoint is mapped from simplified ID to full-network ID.

    No bbox filter is applied here -- the orig_id stored in the simplified
    nodes file is unreliable for inter-city connectors (~0% coordinate match
    with full network for non-boundary nodes). All boundary inflows are
    accepted; out-of-bbox detour segments are clipped from the urban link
    output downstream so they do not contaminate urban VMT.
    """
    inflow = {city: {} for city in CITY_BOUNDARIES}
    outflow = {city: {} for city in CITY_BOUNDARIES}
    skipped = 0

    for row in link_rows:
        u = row["init_node"]
        v = row["term_node"]
        u_super = u in super_to_city
        v_super = v in super_to_city
        if u_super and v_super:
            continue
        if not u_super and not v_super:
            continue

        if u_super:
            city = super_to_city[u]
            other = v
            full_id = simp_nodes.get(other, (None,))[0]
            if full_id is None or full_id <= 6:
                skipped += 1
                continue
            outflow[city][full_id] = outflow[city].get(full_id, 0.0) + row["flow_real"]
        else:
            city = super_to_city[v]
            other = u
            full_id = simp_nodes.get(other, (None,))[0]
            if full_id is None or full_id <= 6:
                skipped += 1
                continue
            inflow[city][full_id] = inflow[city].get(full_id, 0.0) + row["flow_real"]

    if skipped:
        print(f"  WARNING: skipped {skipped} boundary links with unmapped endpoints")
    return inflow, outflow


# ---------------------------------------------------------------------------
# Step 3: extract sub-network for a city
# ---------------------------------------------------------------------------

def extract_subnetwork(
    init_nodes, term_nodes, lengths, ff_times, full_node_coords, bbox,
    must_include=None,
):
    """Return arrays for links where BOTH endpoints lie inside the bbox
    OR are in the must_include set (used for boundary nodes that may sit
    just outside the bbox)."""
    minx, miny, maxx, maxy = bbox
    inside = {
        nid
        for nid, (x, y) in full_node_coords.items()
        if minx <= x <= maxx and miny <= y <= maxy
    }
    if must_include:
        inside.update(int(n) for n in must_include if n in full_node_coords)
    inside_arr = inside  # keep python set for membership tests
    mask = np.array(
        [(int(u) in inside_arr) and (int(v) in inside_arr) for u, v in zip(init_nodes, term_nodes)]
    )
    return (
        init_nodes[mask],
        term_nodes[mask],
        lengths[mask],
        ff_times[mask],
        inside_arr,
    )


# ---------------------------------------------------------------------------
# Step 4: AON inside sub-network with proportional OD split
# ---------------------------------------------------------------------------

def run_subarea_aon(
    sub_init, sub_term, sub_length, sub_ffcost, inside_nodes,
    inflow_map, outflow_map, full_coords, undirected=True,
):
    """Run AON shortest-path loading inside a sub-network.

    Boundary nodes that exist in the sub-network are used as origins (for
    in-boundary nodes) and destinations (for out-boundary nodes).  OD
    demand is split proportional to outflow magnitudes:

        OD[in, out] = inflow[in] * outflow[out] / total_outflow

    If ``undirected`` is True, the sub-network graph is symmetrized
    (each arc becomes a bidirectional edge); this matches the physical
    reality of urban road networks and avoids artificial unreachability
    caused by one-way directional encoding within the sub-area.

    Returns:
        link_flow : np.ndarray same length as sub_init
        diagnostics dict with totals and unreachable demand
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    if len(sub_init) == 0:
        return np.zeros(0), {
            "internal_links": 0,
            "in_boundary_used": 0,
            "out_boundary_used": 0,
            "demand_routed": 0.0,
            "demand_self_loop": 0.0,
            "demand_unreachable": 0.0,
            "vmt": 0.0,
        }

    # Restrict inflow/outflow to nodes that actually appear in the sub-network
    in_used = {n: q for n, q in inflow_map.items() if n in inside_nodes and q > 0}
    out_used = {n: q for n, q in outflow_map.items() if n in inside_nodes and q > 0}
    total_out = sum(out_used.values())

    if total_out <= 0 or not in_used:
        return np.zeros(len(sub_init)), {
            "internal_links": int(len(sub_init)),
            "in_boundary_used": len(in_used),
            "out_boundary_used": len(out_used),
            "demand_routed": 0.0,
            "demand_self_loop": 0.0,
            "demand_unreachable": 0.0,
            "vmt": 0.0,
        }

    # Reindex nodes into 0..N-1 contiguous
    nodes_list = sorted(inside_nodes)
    idx_of = {n: i for i, n in enumerate(nodes_list)}
    N = len(nodes_list)
    rows = np.array([idx_of[int(u)] for u in sub_init], dtype=np.int64)
    cols = np.array([idx_of[int(v)] for v in sub_term], dtype=np.int64)

    if undirected:
        # Symmetrize: include reverse edges with same cost (use min if duplicates)
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])
        all_cost = np.concatenate([sub_ffcost, sub_ffcost])
        graph = csr_matrix((all_cost, (all_rows, all_cols)), shape=(N, N))
    else:
        graph = csr_matrix((sub_ffcost, (rows, cols)), shape=(N, N))

    # Edge lookup for path reconstruction. We always credit the forward
    # arc when a path step matches it; if an undirected reversal was used
    # for routing, fall back to the reverse arc.
    edge_index_fwd = {}
    edge_index_rev = {}
    for lid in range(len(sub_init)):
        key_fwd = (int(rows[lid]), int(cols[lid]))
        key_rev = (int(cols[lid]), int(rows[lid]))
        prev = edge_index_fwd.get(key_fwd)
        if prev is None or sub_ffcost[lid] < sub_ffcost[prev]:
            edge_index_fwd[key_fwd] = lid
        prev = edge_index_rev.get(key_rev)
        if prev is None or sub_ffcost[lid] < sub_ffcost[prev]:
            edge_index_rev[key_rev] = lid

    link_flow = np.zeros(len(sub_init), dtype=np.float64)
    demand_routed = 0.0
    demand_self_loop = 0.0
    demand_unreachable = 0.0

    for in_node, in_qty in in_used.items():
        i_idx = idx_of[in_node]
        dist, pred = dijkstra(
            graph,
            directed=not undirected,
            indices=i_idx,
            return_predecessors=True,
        )
        for out_node, out_qty in out_used.items():
            demand = in_qty * out_qty / total_out
            if out_node == in_node:
                # vehicle enters and exits at same node -> 0 internal VMT,
                # but the demand is still accounted for
                demand_self_loop += demand
                continue
            o_idx = idx_of[out_node]
            d = dist[o_idx]
            if not np.isfinite(d):
                demand_unreachable += demand
                continue
            cur = o_idx
            broke = False
            while cur != i_idx:
                p = pred[cur]
                if p < 0:
                    demand_unreachable += demand
                    broke = True
                    break
                lid = edge_index_fwd.get((int(p), int(cur)))
                if lid is None:
                    lid = edge_index_rev.get((int(p), int(cur)))
                if lid is not None:
                    link_flow[lid] += demand
                cur = p
            if not broke:
                demand_routed += demand

    vmt = float((link_flow * sub_length).sum())
    return link_flow, {
        "internal_links": int(len(sub_init)),
        "in_boundary_used": len(in_used),
        "out_boundary_used": len(out_used),
        "demand_routed": demand_routed,
        "demand_self_loop": demand_self_loop,
        "demand_unreachable": demand_unreachable,
        "vmt": vmt,
    }


# ---------------------------------------------------------------------------
# Step 5: county allocation for the urban link flows
# ---------------------------------------------------------------------------

def allocate_to_counties(
    sub_init, sub_term, sub_length, link_flow, full_coords, county_shp,
):
    import geopandas as gpd
    import pyproj
    from shapely.geometry import LineString, Point
    from shapely.ops import transform as shapely_transform
    from shapely.prepared import prep

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

    node_county = {}
    cache = {}
    for nid in set(int(x) for x in np.concatenate([sub_init, sub_term])):
        if nid not in full_coords:
            continue
        lon, lat = full_coords[nid]
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

    county_vmt = {}
    for i in range(len(sub_init)):
        if link_flow[i] <= 0:
            continue
        link_vmt = link_flow[i] * sub_length[i]
        u = int(sub_init[i])
        v = int(sub_term[i])
        cu = node_county.get(u)
        cv = node_county.get(v)
        if cu and cv and cu == cv:
            county_vmt[cu] = county_vmt.get(cu, 0.0) + link_vmt
            continue
        pu = full_coords.get(u)
        pv = full_coords.get(v)
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
        else:
            fb = cu or cv
            if fb:
                county_vmt[fb] = county_vmt.get(fb, 0.0) + link_vmt
    return county_vmt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", "-y", required=True, choices=["2017", "2019", "2024"])
    ap.add_argument(
        "--simp-dir",
        default=None,
        help="Custom simplified/scenario folder containing link_flows.csv, nodes.txt, and county_vmt.csv. "
             "Defaults to NetworkFiles{year}_WithZones_Simplified.",
    )
    ap.add_argument(
        "--full-dir",
        default=None,
        help="Custom full connected network folder. Defaults to NetworkFiles{year}_Connected.",
    )
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.simp_dir:
        simp_dir = args.simp_dir
        if not os.path.isabs(simp_dir):
            simp_dir = os.path.join(repo_root, simp_dir)
    else:
        simp_dir = os.path.join(repo_root, f"NetworkFiles{args.year}_WithZones_Simplified")

    if args.full_dir:
        full_dir = args.full_dir
        if not os.path.isabs(full_dir):
            full_dir = os.path.join(repo_root, full_dir)
    else:
        full_dir = os.path.join(repo_root, f"NetworkFiles{args.year}_Connected")

    paths = {
        "simp_nodes": os.path.join(simp_dir, "nodes.txt"),
        "link_flows": os.path.join(simp_dir, "link_flows.csv"),
        "county_simp": os.path.join(simp_dir, "county_vmt.csv"),
        "full_network": os.path.join(full_dir, "network.txt"),
        "full_nodes": os.path.join(full_dir, "nodes.txt"),
        "county_shp": os.path.join(repo_root, "mn_counties", "mn_counties.shp"),
    }
    for k in ("simp_nodes", "link_flows", "full_network", "full_nodes"):
        if not os.path.exists(paths[k]):
            print(f"ERROR: required input missing: {paths[k]}")
            sys.exit(1)

    print("=" * 60)
    print(f"Sub-area Approach 5 (year {args.year})")
    print("=" * 60)

    print("\n[1/5] Reading simplified nodes & link flows...")
    simp_nodes = read_simplified_nodes(paths["simp_nodes"])
    link_rows = read_link_flows(paths["link_flows"])
    super_to_city = classify_super_nodes(simp_nodes)
    print(f"      simplified nodes: {len(simp_nodes)}")
    print(f"      simplified links: {len(link_rows)}")
    print(f"      super-nodes classified: {len(super_to_city)}")

    print("[2/5] Reading full network (needed for bbox check)...")
    init_nodes, term_nodes, lengths, ff_times = read_full_network(paths["full_network"])
    full_coords = read_full_nodes(paths["full_nodes"])
    print(f"      {len(init_nodes):,} links / {len(full_coords):,} nodes")

    print("[3/5] Building boundary inflows/outflows per city (bbox-filtered)...")
    inflow, outflow = build_boundary_flows(link_rows, simp_nodes, super_to_city, full_coords)
    for city in CITY_BOUNDARIES:
        ti = sum(inflow[city].values())
        to = sum(outflow[city].values())
        print(
            f"      {city:<22} in={ti:>11,.0f}  out={to:>11,.0f}  "
            f"({len(inflow[city])} in-nodes, {len(outflow[city])} out-nodes)"
        )

    # Per-city processing
    print("\n[4/5] Running sub-area AON per city (undirected full-network Dijkstra; "
          "result is post-filtered to keep only links inside city bbox)...")
    summary_rows = []
    all_link_flows = []  # list of (city, init, term, length, link_flow)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    max_id = int(max(int(init_nodes.max()), int(term_nodes.max())))
    rows_all = init_nodes - 1
    cols_all = term_nodes - 1
    graph_rows = np.concatenate([rows_all, cols_all])
    graph_cols = np.concatenate([cols_all, rows_all])
    graph_costs = np.concatenate([ff_times, ff_times])
    full_graph = csr_matrix(
        (graph_costs, (graph_rows, graph_cols)), shape=(max_id, max_id)
    )
    edge_index_fwd = {}
    edge_index_rev = {}
    for lid in range(len(init_nodes)):
        key = (int(rows_all[lid]), int(cols_all[lid]))
        prev = edge_index_fwd.get(key)
        if prev is None or ff_times[lid] < ff_times[prev]:
            edge_index_fwd[key] = lid
        rev_key = (int(cols_all[lid]), int(rows_all[lid]))
        prev = edge_index_rev.get(rev_key)
        if prev is None or ff_times[lid] < ff_times[prev]:
            edge_index_rev[rev_key] = lid

    print(
        f"      built undirected full graph: {max_id} nodes, "
        f"{len(init_nodes):,} original arcs"
    )

    # Pre-compute per-link in-bbox membership for each city, used to clip the
    # urban_link_flows output to genuinely intra-metro links. This eliminates
    # statewide detours that the full-network Dijkstra would otherwise pick up
    # when boundary nodes are scattered around the metro.
    city_link_in_bbox = {}
    for city, cfg in CITY_BOUNDARIES.items():
        bx0, by0, bx1, by1 = cfg["bbox"]
        # Buffer slightly so links touching the edge stay included
        buf = 0.05
        bx0b, by0b, bx1b, by1b = bx0 - buf, by0 - buf, bx1 + buf, by1 + buf
        u_xy = np.array([full_coords.get(int(n), (-999, -999)) for n in init_nodes])
        v_xy = np.array([full_coords.get(int(n), (-999, -999)) for n in term_nodes])
        u_in = (u_xy[:, 0] >= bx0b) & (u_xy[:, 0] <= bx1b) & (u_xy[:, 1] >= by0b) & (u_xy[:, 1] <= by1b)
        v_in = (v_xy[:, 0] >= bx0b) & (v_xy[:, 0] <= bx1b) & (v_xy[:, 1] >= by0b) & (v_xy[:, 1] <= by1b)
        city_link_in_bbox[city] = u_in & v_in
        print(f"      {city:<22} bbox-internal links: {city_link_in_bbox[city].sum():,}")

    for city, cfg in CITY_BOUNDARIES.items():
        t0 = time.time()
        in_used = {n: q for n, q in inflow[city].items() if q > 0 and n <= max_id}
        out_used = {n: q for n, q in outflow[city].items() if q > 0 and n <= max_id}
        total_out = sum(out_used.values())

        link_flow = np.zeros(len(init_nodes), dtype=np.float64)
        demand_routed = 0.0
        demand_self_loop = 0.0
        demand_unreachable = 0.0

        if total_out > 0 and in_used:
            for in_node, in_qty in in_used.items():
                i_idx = in_node - 1
                dist, pred = dijkstra(
                    full_graph, directed=False, indices=i_idx, return_predecessors=True
                )
                for out_node, out_qty in out_used.items():
                    demand = in_qty * out_qty / total_out
                    if out_node == in_node:
                        demand_self_loop += demand
                        continue
                    o_idx = out_node - 1
                    d = dist[o_idx]
                    if not np.isfinite(d):
                        demand_unreachable += demand
                        continue
                    cur = o_idx
                    broke = False
                    while cur != i_idx:
                        p = pred[cur]
                        if p < 0:
                            demand_unreachable += demand
                            broke = True
                            break
                        lid = edge_index_fwd.get((int(p), int(cur)))
                        if lid is None:
                            lid = edge_index_rev.get((int(p), int(cur)))
                        if lid is not None:
                            link_flow[lid] += demand
                        cur = p
                    if not broke:
                        demand_routed += demand

        # CLIP: keep only flows on links inside the city bbox. Out-of-bbox
        # path segments are statewide detours produced by the full-network
        # Dijkstra; they are NOT urban VMT for this city.
        in_bbox = city_link_in_bbox[city]
        link_flow_full = link_flow.copy()
        link_flow = link_flow * in_bbox
        clipped_vmt = float((link_flow_full * lengths).sum() - (link_flow * lengths).sum())

        elapsed = time.time() - t0
        # VMT contribution from this city is the flow added on links along the
        # routes -- summed across the whole network.  County clipping (next
        # step) will geographically attribute it.
        vmt = float((link_flow * lengths).sum())

        diag = {
            "city": city,
            "internal_links": len(init_nodes),
            "in_boundary_used": len(in_used),
            "out_boundary_used": len(out_used),
            "total_inflow": sum(inflow[city].values()),
            "total_outflow": sum(outflow[city].values()),
            "demand_routed": demand_routed,
            "demand_self_loop": demand_self_loop,
            "demand_unreachable": demand_unreachable,
            "vmt": vmt,
            "vmt_clipped_out_of_bbox": clipped_vmt,
            "elapsed_sec": elapsed,
        }
        summary_rows.append(diag)
        all_link_flows.append((city, init_nodes, term_nodes, lengths, link_flow))
        print(
            f"   {city:<22} VMT={vmt:>14,.0f}  "
            f"routed={demand_routed:>10,.0f}  "
            f"self-loop={demand_self_loop:>8,.0f}  "
            f"unreach={demand_unreachable:>10,.0f}  "
            f"({len(in_used)} origins, {elapsed:.1f}s)"
        )

        # Per-city link flow CSV (only nonzero)
        out_csv = os.path.join(simp_dir, f"urban_link_flows_{city}.csv")
        with open(out_csv, "w") as f:
            f.write("init_node,term_node,flow,length,vmt\n")
            for i in range(len(init_nodes)):
                if link_flow[i] > 0:
                    f.write(
                        f"{int(init_nodes[i])},{int(term_nodes[i])},"
                        f"{link_flow[i]:.4f},{lengths[i]:.6f},"
                        f"{link_flow[i]*lengths[i]:.4f}\n"
                    )

    # Summary CSV
    summary_csv = os.path.join(simp_dir, "urban_vmt_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "city", "internal_links", "in_boundary_used", "out_boundary_used",
                "total_inflow", "total_outflow", "demand_routed",
                "demand_self_loop", "demand_unreachable", "vmt", "vmt_clipped_out_of_bbox", "elapsed_sec",
            ],
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\n   summary -> {summary_csv}")

    # County allocation
    if os.path.exists(paths["county_shp"]):
        print("\n[5/5] Allocating urban link flows to counties...")
        try:
            urban_county_vmt = {}
            for city, sub_init, sub_term, sub_length, link_flow in all_link_flows:
                cv = allocate_to_counties(
                    sub_init, sub_term, sub_length, link_flow,
                    full_coords, paths["county_shp"],
                )
                for k, v in cv.items():
                    urban_county_vmt[k] = urban_county_vmt.get(k, 0.0) + v

            # Combine with simplified-network county VMT
            simp_county = {}
            if os.path.exists(paths["county_simp"]):
                with open(paths["county_simp"], "r") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        simp_county[row["county"]] = float(row["vmt"])

            # ----------------------------------------------------------------
            # Cancel out the simplified-network connector-link VMT.
            # Connector links (one endpoint = super-node) in the simplified
            # network already contribute flow*length VMT to county_vmt.csv,
            # using the super-node centroid (downtown) as the segment endpoint.
            # We are now replacing those abstract connectors with real
            # sub-area routing, so subtract their contribution to avoid
            # double-counting.
            # ----------------------------------------------------------------
            simp_coords = {sid: (xy[1], xy[2]) for sid, xy in simp_nodes.items()}
            conn_init, conn_term, conn_length, conn_flow = [], [], [], []
            for row in link_rows:
                u, v = row["init_node"], row["term_node"]
                u_is_super = u in super_to_city
                v_is_super = v in super_to_city
                # links with exactly one super-node endpoint are connectors
                if u_is_super ^ v_is_super:
                    conn_init.append(u)
                    conn_term.append(v)
                    conn_length.append(row["length"])
                    conn_flow.append(row["flow_real"])
            n_conn = len(conn_init)
            print(f"\n   Cancelling {n_conn} simplified connector links "
                  f"(super-node <-> regular node) from county VMT...")
            if n_conn > 0:
                conn_county_vmt = allocate_to_counties(
                    np.asarray(conn_init, dtype=np.int64),
                    np.asarray(conn_term, dtype=np.int64),
                    np.asarray(conn_length, dtype=np.float64),
                    np.asarray(conn_flow, dtype=np.float64),
                    simp_coords, paths["county_shp"],
                )
                conn_total = sum(conn_county_vmt.values())
                print(f"      connector VMT removed: {conn_total:,.0f} "
                      f"(across {len(conn_county_vmt)} counties)")
                for c, v in conn_county_vmt.items():
                    simp_county[c] = simp_county.get(c, 0.0) - v
            else:
                conn_county_vmt = {}

            combined = dict(simp_county)
            for c, v in urban_county_vmt.items():
                combined[c] = combined.get(c, 0.0) + v

            tot = sum(combined.values())
            out_csv = os.path.join(simp_dir, "county_vmt_subarea_addback.csv")
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "county", "simplified_vmt_raw", "connector_vmt_removed",
                    "simplified_vmt_net", "urban_subarea_vmt", "total_vmt",
                    "pct_of_total",
                ])
                rows = []
                for c in sorted(combined):
                    sv_net = simp_county.get(c, 0.0)
                    cv = conn_county_vmt.get(c, 0.0)
                    sv_raw = sv_net + cv
                    uv = urban_county_vmt.get(c, 0.0)
                    tv = combined[c]
                    rows.append((c, sv_raw, cv, sv_net, uv, tv))
                rows.sort(key=lambda x: -x[5])
                for c, sv_raw, cv, sv_net, uv, tv in rows:
                    pct = tv / tot * 100 if tot > 0 else 0
                    w.writerow([
                        c, f"{sv_raw:.2f}", f"{cv:.2f}", f"{sv_net:.2f}",
                        f"{uv:.2f}", f"{tv:.2f}", f"{pct:.4f}",
                    ])
            print(f"      county VMT -> {out_csv}")
            print(f"\n   Top 10 counties (combined):")
            print(f"   {'County':<22}{'Simp(net)':>12}{'Urban':>12}{'Total':>12}")
            print("   " + "-" * 58)
            for c, sv_raw, cv, sv_net, uv, tv in rows[:10]:
                print(f"   {c:<22}{sv_net:>12,.0f}{uv:>12,.0f}{tv:>12,.0f}")
        except ImportError as e:
            print(f"   geopandas not available, skipping county allocation: {e}")
        except Exception as e:
            print(f"   county allocation failed: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
