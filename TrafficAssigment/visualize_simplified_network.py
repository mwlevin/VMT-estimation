"""Plot the SIMPLIFIED network as it stands at the start of traffic
assignment, for each year. Shows:
  - all simplified links (light gray)
  - regular nodes (faint dots)
  - super-nodes (large colored markers, one color per metro)
  - metro bbox rectangles
Outputs one PNG per year + one combined three-panel figure.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection

CITY_BOUNDARIES = {
    "TwinCities_Metro": {"bbox": (-93.85, 44.55, -92.75, 45.40),
                         "super_node_xy": (-93.17, 44.96), "color": "#e41a1c"},
    "Rochester":        {"bbox": (-92.65, 43.85, -92.25, 44.20),
                         "super_node_xy": (-92.466, 44.023), "color": "#377eb8"},
    "Duluth":           {"bbox": (-92.45, 46.50, -91.80, 47.00),
                         "super_node_xy": (-92.106, 46.786), "color": "#4daf4a"},
    "St_Cloud":         {"bbox": (-94.40, 45.40, -93.90, 45.70),
                         "super_node_xy": (-94.16, 45.56), "color": "#984ea3"},
    "Mankato":          {"bbox": (-94.10, 44.05, -93.85, 44.35),
                         "super_node_xy": (-93.99, 44.17), "color": "#ff7f00"},
    "Moorhead":         {"bbox": (-96.85, 46.65, -96.40, 46.95),
                         "super_node_xy": (-96.74, 46.87), "color": "#a65628"},
}
XY_TOL = 1e-3


def read_node_coords(path):
    coords = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            nid, lon, lat = line.strip().split(",")
            coords[int(nid)] = (float(lon), float(lat))
    return coords


def read_links(path):
    init_nodes, term_nodes = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            init_nodes.append(int(row["init_node"]))
            term_nodes.append(int(row["term_node"]))
    return np.array(init_nodes), np.array(term_nodes)


def classify_super_nodes(coords):
    """Return dict simp_id -> city based on coord match to super_node_xy."""
    out = {}
    for sid, (x, y) in coords.items():
        for city, cfg in CITY_BOUNDARIES.items():
            sx, sy = cfg["super_node_xy"]
            if abs(x - sx) < XY_TOL and abs(y - sy) < XY_TOL:
                out[sid] = city
                break
    return out


def plot_year(ax, year, repo_root):
    simp_dir = os.path.join(repo_root, f"NetworkFiles{year}_WithZones_Simplified")
    coords = read_node_coords(os.path.join(simp_dir, "node_coords.csv"))
    init_n, term_n = read_links(os.path.join(simp_dir, "link_flows.csv"))
    super_to_city = classify_super_nodes(coords)

    # Build line segments
    segs = []
    super_ids = set(super_to_city.keys())
    is_super_link = np.zeros(len(init_n), dtype=bool)
    for i, (u, v) in enumerate(zip(init_n, term_n)):
        if u in coords and v in coords:
            segs.append([coords[u], coords[v]])
            if u in super_ids or v in super_ids:
                is_super_link[i] = True

    segs = np.array(segs)
    # Plot regular links
    reg = segs[~is_super_link[:len(segs)]]
    sup = segs[is_super_link[:len(segs)]]
    ax.add_collection(LineCollection(reg, colors="#bbbbbb", linewidths=0.15, zorder=1))
    ax.add_collection(LineCollection(sup, colors="#000000", linewidths=0.4,
                                     alpha=0.6, zorder=2))

    # Regular nodes (subsample for speed)
    xs = np.array([c[0] for c in coords.values()])
    ys = np.array([c[1] for c in coords.values()])
    step = max(1, len(xs) // 40000)
    ax.scatter(xs[::step], ys[::step], s=0.05, c="#888888", zorder=1.5)

    # Bboxes + super-nodes
    legend_handles = []
    for city, cfg in CITY_BOUNDARIES.items():
        bx0, by0, bx1, by1 = cfg["bbox"]
        rect = mpatches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                  fill=False, edgecolor=cfg["color"],
                                  linewidth=1.6, zorder=4)
        ax.add_patch(rect)
        # All super-nodes for this city
        sids = [s for s, c in super_to_city.items() if c == city]
        sx = [coords[s][0] for s in sids]
        sy = [coords[s][1] for s in sids]
        ax.scatter(sx, sy, s=70, c=cfg["color"], edgecolors="black",
                   linewidths=0.8, zorder=5)
        legend_handles.append(mpatches.Patch(color=cfg["color"],
                                             label=f"{city} ({len(sids)} super)"))

    ax.set_xlim(-97.5, -89.3)
    ax.set_ylim(43.4, 49.5)
    ax.set_aspect("equal")
    ax.set_title(f"{year}  |  {len(coords):,} nodes, {len(init_n):,} links, "
                 f"{len(super_to_city)} super-nodes")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
              framealpha=0.9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", default=["2017", "2019", "2024"])
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Per-year individual figures
    for y in args.years:
        fig, ax = plt.subplots(figsize=(9, 11))
        plot_year(ax, y, repo_root)
        out = os.path.join(repo_root, f"simplified_network_{y}.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")

    # Combined three-panel
    fig, axes = plt.subplots(1, len(args.years), figsize=(9 * len(args.years), 11))
    if len(args.years) == 1:
        axes = [axes]
    for ax, y in zip(axes, args.years):
        plot_year(ax, y, repo_root)
    out = os.path.join(repo_root, "simplified_network_all_years.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
