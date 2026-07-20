"""Interactive flow visualization for simplified-network assignments.

Two modes:

1) single  -- one year, color links by flow, popup shows flow/capacity/VMT.
       python visualize_flow_maps.py single --year 2024

2) diff    -- compare two years on a common-link basis.
       python visualize_flow_maps.py diff --a 2017 --b 2024
   Links shared by both years (matched by endpoint coordinates) are
   colored by delta flow (B - A) on a diverging scale.
   Links present only in B are drawn in solid green (added).
   Links present only in A are drawn in solid red dashed (removed).

Outputs HTML files at the workspace root.
"""

import argparse
import csv
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def year_dir(year: str) -> str:
    return os.path.join(REPO_ROOT, f"NetworkFiles{year}_WithZones_Simplified")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_node_coords(path: str) -> dict:
    out = {}
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["node_id"])] = (float(row["lon"]), float(row["lat"]))
    return out


def load_link_flows(path: str):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "init": int(row["init_node"]),
                "term": int(row["term_node"]),
                "flow": float(row["flow_real"]),
                "cap": float(row["capacity"]),
                "vc": float(row["volume_capacity_ratio"]),
                "len": float(row["length"]),
                "vmt": float(row["vmt"]),
                "speed": float(row["congested_speed"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def flow_color(flow: float, vmax: float):
    """Map flow to a hex color using a viridis-like 5-stop ramp on log scale."""
    import math
    if flow <= 0 or vmax <= 0:
        return "#cccccc"
    t = math.log10(max(flow, 1.0)) / math.log10(max(vmax, 10.0))
    t = max(0.0, min(1.0, t))
    # 5-stop ramp: dark blue -> teal -> green -> yellow -> red
    stops = [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 144, 141)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = int(c0[0] + (c1[0] - c0[0]) * f)
            g = int(c0[1] + (c1[1] - c0[1]) * f)
            b = int(c0[2] + (c1[2] - c0[2]) * f)
            return f"#{r:02x}{g:02x}{b:02x}"
    c = stops[-1][1]
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def diverging_color(delta: float, dmax: float):
    """Red (positive delta) <-> white <-> blue (negative delta)."""
    if dmax <= 0:
        return "#999999"
    t = max(-1.0, min(1.0, delta / dmax))
    if t >= 0:
        # white -> red
        r = 255
        g = int(255 * (1 - t))
        b = int(255 * (1 - t))
    else:
        # white -> blue
        r = int(255 * (1 + t))
        g = int(255 * (1 + t))
        b = 255
    return f"#{r:02x}{g:02x}{b:02x}"


def line_weight(flow: float, vmax: float):
    import math
    if flow <= 0:
        return 1.0
    t = math.log10(max(flow, 1.0)) / math.log10(max(vmax, 10.0))
    return 1.0 + 5.0 * max(0.0, min(1.0, t))


# ---------------------------------------------------------------------------
# Single-year map
# ---------------------------------------------------------------------------

def build_single_map(year: str, min_flow: float, max_links: int, out_html: str):
    import folium

    d = year_dir(year)
    coords = load_node_coords(os.path.join(d, "node_coords.csv"))
    links = load_link_flows(os.path.join(d, "link_flows.csv"))

    # Filter to active links with valid endpoints
    active = [
        L for L in links
        if L["flow"] >= min_flow and L["init"] in coords and L["term"] in coords
    ]
    active.sort(key=lambda L: -L["flow"])
    if max_links and len(active) > max_links:
        print(f"  capping at top {max_links} links by flow (had {len(active):,})")
        active = active[:max_links]

    if not active:
        print("  no active links to draw")
        return

    flows = [L["flow"] for L in active if L["flow"] > 0]
    vmax = max(flows) if flows else 1.0

    # Center map on mean of node coords
    lats = [coords[L["init"]][1] for L in active]
    lons = [coords[L["init"]][0] for L in active]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))

    m = folium.Map(
        location=center, zoom_start=7,
        tiles="cartodbpositron", control_scale=True,
    )
    folium.TileLayer("cartodbdark_matter", name="CartoDB Dark").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Streets",
    ).add_to(m)

    fg = folium.FeatureGroup(name=f"Links {year} (n={len(active):,})")
    # Draw lower flows first so high-flow links sit on top
    for L in sorted(active, key=lambda x: x["flow"]):
        u, v = L["init"], L["term"]
        lon1, lat1 = coords[u]
        lon2, lat2 = coords[v]
        c = flow_color(L["flow"], vmax)
        w = line_weight(L["flow"], vmax)
        popup_html = (
            f"<b>Link {u} → {v}</b><br>"
            f"Flow: <b>{L['flow']:,.0f}</b> veh<br>"
            f"Capacity: {L['cap']:,.0f}<br>"
            f"V/C: {L['vc']:.3f}<br>"
            f"Length: {L['len']:.2f} mi<br>"
            f"Speed: {L['speed']:.1f} mph<br>"
            f"VMT: {L['vmt']:,.0f}"
        )
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color=c, weight=w, opacity=0.85,
            tooltip=f"{u}→{v}  flow={L['flow']:,.0f}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg)
    fg.add_to(m)

    # Legend
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index:9999;
                background: white; padding: 10px 14px; border:1px solid #888;
                border-radius:6px; font-family:sans-serif; font-size:12px;">
      <b>{year} Link Flow (veh)</b><br>
      <span style="background:#440154;display:inline-block;width:18px;">&nbsp;</span> low<br>
      <span style="background:#3b528b;display:inline-block;width:18px;">&nbsp;</span><br>
      <span style="background:#21908d;display:inline-block;width:18px;">&nbsp;</span> mid<br>
      <span style="background:#5ec962;display:inline-block;width:18px;">&nbsp;</span><br>
      <span style="background:#fde725;display:inline-block;width:18px;">&nbsp;</span> max ({vmax:,.0f})<br>
      <span style="color:#666">Click a link for details</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(out_html)
    print(f"  wrote {out_html}  ({len(active):,} links, max flow {vmax:,.0f})")


# ---------------------------------------------------------------------------
# Diff map
# ---------------------------------------------------------------------------

def coord_key(lon: float, lat: float, prec: int = 5):
    return (round(lon, prec), round(lat, prec))


def link_key(coords: dict, u: int, v: int, prec: int = 5):
    if u not in coords or v not in coords:
        return None
    return (coord_key(*coords[u], prec), coord_key(*coords[v], prec))


def build_diff_map(year_a: str, year_b: str, min_flow: float, max_links: int, out_html: str):
    import folium

    da = year_dir(year_a)
    db = year_dir(year_b)
    coords_a = load_node_coords(os.path.join(da, "node_coords.csv"))
    coords_b = load_node_coords(os.path.join(db, "node_coords.csv"))
    links_a = load_link_flows(os.path.join(da, "link_flows.csv"))
    links_b = load_link_flows(os.path.join(db, "link_flows.csv"))

    # Build dicts keyed by coordinate pair; if duplicate, keep highest flow
    def index(links, coords):
        out = {}
        for L in links:
            k = link_key(coords, L["init"], L["term"])
            if k is None:
                continue
            prev = out.get(k)
            if prev is None or L["flow"] > prev["flow"]:
                out[k] = L
        return out

    A = index(links_a, coords_a)
    B = index(links_b, coords_b)

    keys_a = set(A)
    keys_b = set(B)
    common = keys_a & keys_b
    only_a = keys_a - keys_b      # removed in B
    only_b = keys_b - keys_a      # added in B

    print(f"  {year_a}: {len(A):,} unique links | {year_b}: {len(B):,}")
    print(f"  common: {len(common):,}  only-{year_a}: {len(only_a):,}  "
          f"only-{year_b}: {len(only_b):,}")

    # Compute deltas for common
    deltas = []
    for k in common:
        d = B[k]["flow"] - A[k]["flow"]
        if abs(d) >= min_flow or A[k]["flow"] >= min_flow or B[k]["flow"] >= min_flow:
            deltas.append((k, d))
    deltas.sort(key=lambda x: -abs(x[1]))
    if max_links and len(deltas) > max_links:
        print(f"  capping common layer to top {max_links} by |delta| (had {len(deltas):,})")
        deltas = deltas[:max_links]
    if not deltas:
        print("  no significant deltas")
    dmax = max((abs(d) for _, d in deltas), default=1.0)

    # Center map
    all_lons = [k[0][0] for k in common] or [k[0][0] for k in keys_a | keys_b]
    all_lats = [k[0][1] for k in common] or [k[0][1] for k in keys_a | keys_b]
    center = (sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons))

    m = folium.Map(location=center, zoom_start=7,
                   tiles="cartodbpositron", control_scale=True)
    folium.TileLayer("cartodbdark_matter", name="CartoDB Dark").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Streets",
    ).add_to(m)

    fg_common = folium.FeatureGroup(name=f"Common Δflow ({year_b}−{year_a})")
    for k, d in sorted(deltas, key=lambda x: abs(x[1])):
        (lon1, lat1), (lon2, lat2) = k
        a = A[k]; b = B[k]
        c = diverging_color(d, dmax)
        w = 1.0 + 5.0 * (abs(d) / dmax) ** 0.5
        popup_html = (
            f"<b>Link {a['init']}→{a['term']}  ({year_a})</b><br>"
            f"<b>Link {b['init']}→{b['term']}  ({year_b})</b><br>"
            f"Flow {year_a}: <b>{a['flow']:,.0f}</b> veh<br>"
            f"Flow {year_b}: <b>{b['flow']:,.0f}</b> veh<br>"
            f"Δ: <b>{d:+,.0f}</b><br>"
            f"Capacity {year_a}: {a['cap']:,.0f}<br>"
            f"Capacity {year_b}: {b['cap']:,.0f}<br>"
            f"Length: {b['len']:.2f} mi"
        )
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color=c, weight=w, opacity=0.9,
            tooltip=f"Δ={d:+,.0f}",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(fg_common)
    fg_common.add_to(m)

    # Added in B
    fg_added = folium.FeatureGroup(name=f"Only in {year_b} (added)")
    only_b_sorted = sorted(
        (k for k in only_b if B[k]["flow"] >= min_flow),
        key=lambda k: -B[k]["flow"],
    )[:max_links]
    for k in only_b_sorted:
        b = B[k]
        (lon1, lat1), (lon2, lat2) = k
        popup_html = (
            f"<b>NEW in {year_b}</b><br>"
            f"Link {b['init']}→{b['term']}<br>"
            f"Flow: <b>{b['flow']:,.0f}</b> veh<br>"
            f"Capacity: {b['cap']:,.0f}<br>"
            f"Length: {b['len']:.2f} mi"
        )
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color="#1a9850", weight=3, opacity=0.95,
            tooltip=f"NEW {year_b}: {b['flow']:,.0f}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg_added)
    fg_added.add_to(m)

    # Removed (only in A)
    fg_removed = folium.FeatureGroup(name=f"Only in {year_a} (removed)")
    only_a_sorted = sorted(
        (k for k in only_a if A[k]["flow"] >= min_flow),
        key=lambda k: -A[k]["flow"],
    )[:max_links]
    for k in only_a_sorted:
        a = A[k]
        (lon1, lat1), (lon2, lat2) = k
        popup_html = (
            f"<b>REMOVED after {year_a}</b><br>"
            f"Link {a['init']}→{a['term']}<br>"
            f"Flow: <b>{a['flow']:,.0f}</b> veh<br>"
            f"Capacity: {a['cap']:,.0f}<br>"
            f"Length: {a['len']:.2f} mi"
        )
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color="#b2182b", weight=3, opacity=0.85, dash_array="6,4",
            tooltip=f"REMOVED {year_a}: {a['flow']:,.0f}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg_removed)
    fg_removed.add_to(m)

    # Legend
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index:9999;
                background: white; padding: 10px 14px; border:1px solid #888;
                border-radius:6px; font-family:sans-serif; font-size:12px;
                line-height:1.4;">
      <b>{year_b} − {year_a} flow delta</b><br>
      <span style="background:#0000ff;display:inline-block;width:18px;">&nbsp;</span>
        decreased (max −{dmax:,.0f})<br>
      <span style="background:#ffffff;display:inline-block;width:18px;border:1px solid #888;">&nbsp;</span>
        ~equal<br>
      <span style="background:#ff0000;display:inline-block;width:18px;">&nbsp;</span>
        increased (max +{dmax:,.0f})<br>
      <span style="background:#1a9850;display:inline-block;width:18px;">&nbsp;</span>
        new in {year_b} ({len(only_b):,})<br>
      <span style="background:#b2182b;display:inline-block;width:18px;">&nbsp;</span>
        only in {year_a} ({len(only_a):,})<br>
      <span style="color:#666">Click a link for details</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(out_html)
    print(f"  wrote {out_html}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("single", help="One-year flow map")
    sp.add_argument("--year", "-y", required=True, choices=["2017", "2019", "2024"])
    sp.add_argument("--min-flow", type=float, default=10.0,
                    help="skip links with flow below this")
    sp.add_argument("--max-links", type=int, default=40000,
                    help="cap drawn links to top-N by flow")
    sp.add_argument("--out", default=None)

    dp = sub.add_parser("diff", help="Two-year delta map")
    dp.add_argument("--a", required=True, choices=["2017", "2019", "2024"])
    dp.add_argument("--b", required=True, choices=["2017", "2019", "2024"])
    dp.add_argument("--min-flow", type=float, default=10.0)
    dp.add_argument("--max-links", type=int, default=40000,
                    help="cap each layer (common/added/removed) to top-N by |flow|/|delta|")
    dp.add_argument("--out", default=None)

    args = ap.parse_args()

    if args.cmd == "single":
        out = args.out or os.path.join(
            REPO_ROOT, f"flow_map_{args.year}.html"
        )
        print(f"[single] year={args.year}")
        build_single_map(args.year, args.min_flow, args.max_links, out)
    else:
        if args.a == args.b:
            ap.error("--a and --b must differ")
        out = args.out or os.path.join(
            REPO_ROOT, f"flow_diff_{args.a}_vs_{args.b}.html"
        )
        print(f"[diff] {args.a} -> {args.b}")
        build_diff_map(args.a, args.b, args.min_flow, args.max_links, out)


if __name__ == "__main__":
    main()
