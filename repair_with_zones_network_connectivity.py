"""
Repair reachability in a with-zones TNTP network without long centroid fans.

This script is intended for diagnostic/scenario runs after zone centroids have
already been attached. It writes a new folder and leaves the input untouched.

Repairs:
1. Add exact reverse links for non-centroid road links that are missing one.
2. Add very short bidirectional stitch links between road components whose
   nearest nodes are within a distance threshold.

The stitch links are synthetic road-network repair links, not zone connectors.
They are meant to close tiny coordinate/splitting gaps that otherwise make the
directed assignment graph unreachable.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair a with-zones TNTP network with local graph stitches."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing nodes.txt and network.txt")
    parser.add_argument("--output-dir", required=True, help="New repaired output folder")
    parser.add_argument("--stitch-threshold-miles", type=float, default=0.25,
                        help="Maximum road-component stitch distance in miles")
    parser.add_argument("--stitch-capacity", type=float, default=99999.0)
    parser.add_argument("--stitch-speed", type=float, default=30.0)
    parser.add_argument("--stitch-link-type", type=int, default=8)
    parser.add_argument("--no-reverse-repair", action="store_true",
                        help="Do not add missing reverse links for road links")
    return parser.parse_args()


def read_nodes(path: Path):
    nodes = {}
    lines = []
    with path.open() as f:
        for line in f:
            lines.append(line)
            s = line.strip()
            if not s or s.startswith("~") or s.startswith("<"):
                continue
            parts = s.replace(";", "").split()
            if len(parts) < 4:
                continue
            try:
                node_id = int(parts[0])
                nodes[node_id] = {
                    "original": int(float(parts[1])),
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                }
            except ValueError:
                continue
    return nodes, lines


def read_network(path: Path):
    metadata = {}
    header_lines = []
    links = []
    in_data = False

    with path.open() as f:
        for line in f:
            s = line.strip()
            if not in_data:
                if "<NUMBER OF ZONES>" in line:
                    metadata["num_zones"] = int(line[line.index(">") + 1:].strip())
                elif "<NUMBER OF NODES>" in line:
                    metadata["num_nodes"] = int(line[line.index(">") + 1:].strip())
                elif "<FIRST THRU NODE>" in line:
                    metadata["first_thru_node"] = int(line[line.index(">") + 1:].strip())
                elif "<NUMBER OF LINKS>" in line:
                    metadata["num_links"] = int(line[line.index(">") + 1:].strip())
                header_lines.append(line)
                if s == "<END OF METADATA>":
                    in_data = True
                continue

            if not s or s.startswith("~") or s.startswith("<"):
                continue
            parts = s.replace(";", "").split()
            if len(parts) < 10:
                continue
            try:
                links.append({
                    "init": int(parts[0]),
                    "term": int(parts[1]),
                    "capacity": float(parts[2]),
                    "length": float(parts[3]),
                    "free_flow_time": float(parts[4]),
                    "b": float(parts[5]),
                    "power": float(parts[6]),
                    "speed": float(parts[7]),
                    "toll": float(parts[8]),
                    "link_type": int(float(parts[9])),
                })
            except ValueError:
                continue

    return metadata, header_lines, links


def add_missing_reverse_road_links(links):
    edge_set = {(l["init"], l["term"]) for l in links}
    added = []
    for link in links:
        if link["link_type"] == 9:
            continue
        rev = (link["term"], link["init"])
        if rev in edge_set:
            continue
        new_link = dict(link)
        new_link["init"], new_link["term"] = link["term"], link["init"]
        added.append(new_link)
        edge_set.add(rev)
    return added


def road_components(links, first_thru_node):
    adj = defaultdict(set)
    road_nodes = set()
    for link in links:
        if link["link_type"] == 9:
            continue
        a, b = link["init"], link["term"]
        if a < first_thru_node or b < first_thru_node:
            continue
        adj[a].add(b)
        adj[b].add(a)
        road_nodes.add(a)
        road_nodes.add(b)

    comp_id = {}
    comps = []
    for node in road_nodes:
        if node in comp_id:
            continue
        cid = len(comps)
        stack = [node]
        comp_id[node] = cid
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, ()):
                if v not in comp_id:
                    comp_id[v] = cid
                    stack.append(v)
        comps.append(comp)
    return comp_id, comps


def project_node_coords(nodes, node_ids):
    gdf = gpd.GeoDataFrame(
        {"node": node_ids},
        geometry=[Point(nodes[n]["x"], nodes[n]["y"]) for n in node_ids],
        crs="EPSG:4326",
    ).to_crs(epsg=26915)
    xy = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    return gdf, xy


def add_component_stitches(nodes, links, first_thru_node, threshold_miles, capacity, speed, link_type):
    _, comps = road_components(links, first_thru_node)
    if not comps:
        return [], {"components_before": 0}

    sizes = [len(c) for c in comps]
    largest_id = max(range(len(comps)), key=lambda i: sizes[i])
    largest_nodes = comps[largest_id]
    largest_gdf, largest_xy = project_node_coords(nodes, largest_nodes)
    largest_tree = cKDTree(largest_xy)

    threshold_m = threshold_miles * 1609.344
    stitches = []
    stitched_components = 0

    for cid, comp_nodes in enumerate(comps):
        if cid == largest_id:
            continue
        comp_gdf, comp_xy = project_node_coords(nodes, comp_nodes)
        distances, nearest_idx = largest_tree.query(comp_xy, k=1)
        best_local_idx = int(np.argmin(distances))
        best_dist_m = float(distances[best_local_idx])
        if best_dist_m > threshold_m:
            continue

        a = int(comp_gdf.iloc[best_local_idx]["node"])
        b = int(largest_gdf.iloc[int(nearest_idx[best_local_idx])]["node"])
        length_miles = max(best_dist_m / 1609.344, 0.001)
        fft = length_miles / speed * 60.0
        base = {
            "capacity": capacity,
            "length": length_miles,
            "free_flow_time": fft,
            "b": 0.0,
            "power": 1.0,
            "speed": speed,
            "toll": 0.0,
            "link_type": link_type,
        }
        stitches.append({"init": a, "term": b, **base})
        stitches.append({"init": b, "term": a, **base})
        stitched_components += 1

    return stitches, {
        "components_before": len(comps),
        "largest_before": sizes[largest_id],
        "stitched_components": stitched_components,
    }


def write_network(path: Path, metadata, links):
    with path.open("w") as f:
        f.write(f"<NUMBER OF ZONES> {metadata['num_zones']}\n")
        f.write(f"<NUMBER OF NODES> {metadata['num_nodes']}\n")
        f.write(f"<FIRST THRU NODE> {metadata['first_thru_node']}\n")
        f.write(f"<NUMBER OF LINKS> {len(links)}\n")
        f.write("<ORIGINAL HEADER>~ \tInit node \tTerm node \tCapacity \tLength \tFree Flow Time \tB\tPower\tSpeed limit \tToll \tType\t;\n")
        f.write("<END OF METADATA>\n\n")
        f.write("~\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\tspeed\ttoll\tlink_type\t;\n")
        for link in links:
            f.write(
                f"\t{link['init']}\t{link['term']}\t{link['capacity']:.2f}"
                f"\t{link['length']:.5f}\t{link['free_flow_time']:.5f}"
                f"\t{link['b']}\t{link['power']}\t{link['speed']:.2f}"
                f"\t{link['toll']}\t{link['link_type']}\t;\n"
            )


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes, node_lines = read_nodes(input_dir / "nodes.txt")
    metadata, _, links = read_network(input_dir / "network.txt")

    print("Connectivity repair")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Input links: {len(links):,}")

    new_links = list(links)
    reverse_added = []
    if not args.no_reverse_repair:
        reverse_added = add_missing_reverse_road_links(new_links)
        new_links.extend(reverse_added)
        print(f"  Added missing reverse road links: {len(reverse_added):,}")

    stitches, stitch_info = add_component_stitches(
        nodes,
        new_links,
        metadata["first_thru_node"],
        args.stitch_threshold_miles,
        args.stitch_capacity,
        args.stitch_speed,
        args.stitch_link_type,
    )
    new_links.extend(stitches)
    print(f"  Road components before stitching: {stitch_info.get('components_before', 0):,}")
    print(f"  Largest road component before stitching: {stitch_info.get('largest_before', 0):,}")
    print(f"  Stitched components: {stitch_info.get('stitched_components', 0):,}")
    print(f"  Added stitch links: {len(stitches):,}")
    print(f"  Output links: {len(new_links):,}")

    shutil.copy2(input_dir / "nodes.txt", output_dir / "nodes.txt")
    if (input_dir / "zone_node_mapping.csv").exists():
        shutil.copy2(input_dir / "zone_node_mapping.csv", output_dir / "zone_node_mapping.csv")
    if (input_dir / "trips.txt").exists():
        shutil.copy2(input_dir / "trips.txt", output_dir / "trips.txt")

    write_network(output_dir / "network.txt", metadata, new_links)
    shutil.copy2(output_dir / "network.txt", output_dir / "net.txt")
    print("  Wrote network.txt and net.txt")


if __name__ == "__main__":
    main()
