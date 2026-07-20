"""Build a 'real-network' single-year flow map.

Combines:
  - simplified link_flows.csv  (rural / inter-city UE flows)  -- excluding
    centroid-connector links and super-node connector links
  - urban_link_flows_<City>.csv from vmt_approach5_subarea_aon.py
    (per-link sub-area AON flows on the FULL connected network)

So instead of seeing the artificial straight-line super-node connectors and
zone-centroid connectors, you see the actual shortest-path flows on the
underlying road network inside each metro rectangle.

Outputs HTML at workspace root: realflow_map_<year>.html

Usage:
    python visualize_realflow_maps.py --year 2024
"""

import argparse
import csv
import math
import os
import re


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_simplified_node_coords(simp_dir: str) -> dict:
    out = {}
    with open(os.path.join(simp_dir, "node_coords.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["node_id"])] = (float(row["lon"]), float(row["lat"]))
    return out


def load_super_nodes(simp_dir: str) -> set:
    """Simplified-network IDs that are super-node markers (orig_id == -1)."""
    super_ids = set()
    with open(os.path.join(simp_dir, "nodes.txt")) as f:
        for line in f:
            s = line.strip()
            if s.startswith("~") or not s:
                continue
            p = s.replace(";", "").split()
            if len(p) < 4:
                continue
            try:
                nid = int(p[0])
                # original_node_id; treat -1 or 1..6 as super-node markers
                orig = float(p[1])
                if orig in (-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                    super_ids.add(nid)
            except ValueError:
                continue
    return super_ids


def load_zone_ids(simp_dir: str) -> set:
    """IDs that act as zone centroids (appear as Origin lines in trips.txt)."""
    zones = set()
    with open(os.path.join(simp_dir, "trips.txt")) as f:
        for line in f:
            m = re.match(r"\s*Origin\s+(\d+)", line)
            if m:
                zones.add(int(m.group(1)))
    return zones


def load_simplified_link_flows(simp_dir: str):
    rows = []
    with open(os.path.join(simp_dir, "link_flows.csv")) as f:
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


def load_full_node_coords(full_dir: str) -> dict:
    out = {}
    with open(os.path.join(full_dir, "nodes.txt")) as f:
        for line in f:
            s = line.strip()
            if s.startswith("~") or not s:
                continue
            p = s.replace(";", "").split()
            if len(p) < 3:
                continue
            try:
                out[int(p[0])] = (float(p[1]), float(p[2]))
            except ValueError:
                continue
    return out


def load_urban_link_flows(simp_dir: str) -> list:
    """Yield (city, init, term, flow, length) from all urban_link_flows_*.csv.

    Returns aggregated by (init,term): if the same link appears in multiple
    city files (rare), keeps the larger flow.
    """
    out = {}
    for name in os.listdir(simp_dir):
        if not name.startswith("urban_link_flows_") or not name.endswith(".csv"):
            continue
        city = name[len("urban_link_flows_"):-len(".csv")]
        with open(os.path.join(simp_dir, name)) as f:
            r = csv.DictReader(f)
            for row in r:
                u = int(row["init_node"])
                v = int(row["term_node"])
                flow = float(row["flow"])
                length = float(row["length"])
                key = (u, v)
                prev = out.get(key)
                if prev is None or flow > prev["flow"]:
                    out[key] = {
                        "city": city, "init": u, "term": v,
                        "flow": flow, "length": length,
                        "vmt": flow * length,
                    }
    return list(out.values())


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def flow_color(flow: float, vmax: float):
    if flow <= 0 or vmax <= 0:
        return "#cccccc"
    t = math.log10(max(flow, 1.0)) / math.log10(max(vmax, 10.0))
    t = max(0.0, min(1.0, t))
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


def line_weight(flow: float, vmax: float):
    if flow <= 0:
        return 1.0
    t = math.log10(max(flow, 1.0)) / math.log10(max(vmax, 10.0))
    return 1.0 + 5.0 * max(0.0, min(1.0, t))


# ---------------------------------------------------------------------------
# Build map
# ---------------------------------------------------------------------------

def build_map(year: str, min_flow: float, max_links: int, out_html: str):
    import folium

    simp_dir = os.path.join(REPO_ROOT, f"NetworkFiles{year}_WithZones_Simplified")
    full_dir = os.path.join(REPO_ROOT, f"NetworkFiles{year}_Connected")

    print(f"  loading simplified network ({simp_dir})...")
    simp_coords = load_simplified_node_coords(simp_dir)
    super_nodes = load_super_nodes(simp_dir)
    zones = load_zone_ids(simp_dir)
    simp_links = load_simplified_link_flows(simp_dir)
    print(f"    super-nodes: {len(super_nodes)}  zones: {len(zones)}  "
          f"links: {len(simp_links):,}")

    # Filter simplified links: drop centroid connectors and super-node connectors
    bad = super_nodes | zones
    inter_links = [
        L for L in simp_links
        if L["init"] not in bad and L["term"] not in bad
        and L["init"] in simp_coords and L["term"] in simp_coords
        and L["flow"] >= min_flow
    ]
    dropped = len(simp_links) - len(inter_links)
    print(f"    kept {len(inter_links):,} inter-city links "
          f"(dropped {dropped:,} connector/centroid/zero-flow)")

    print(f"  loading full network coords ({full_dir})...")
    full_coords = load_full_node_coords(full_dir)

    print("  loading urban sub-area flows...")
    urban_links = load_urban_link_flows(simp_dir)
    urban_links = [
        L for L in urban_links
        if L["flow"] >= min_flow
        and L["init"] in full_coords and L["term"] in full_coords
    ]
    print(f"    urban links with flow>={min_flow}: {len(urban_links):,}")

    # Cap each layer to top-N by flow
    inter_links.sort(key=lambda L: -L["flow"])
    urban_links.sort(key=lambda L: -L["flow"])
    if max_links and len(inter_links) > max_links:
        print(f"    capping inter-city to top {max_links}")
        inter_links = inter_links[:max_links]
    if max_links and len(urban_links) > max_links:
        print(f"    capping urban to top {max_links}")
        urban_links = urban_links[:max_links]

    all_flows = [L["flow"] for L in inter_links] + [L["flow"] for L in urban_links]
    if not all_flows:
        print("  no flows to draw")
        return
    vmax = max(all_flows)

    # Center
    lats, lons = [], []
    for L in inter_links[:1000]:
        lon, lat = simp_coords[L["init"]]
        lats.append(lat); lons.append(lon)
    for L in urban_links[:1000]:
        lon, lat = full_coords[L["init"]]
        lats.append(lat); lons.append(lon)
    center = (sum(lats) / len(lats), sum(lons) / len(lons))

    m = folium.Map(location=center, zoom_start=7,
                   tiles="cartodbpositron", control_scale=True)
    folium.TileLayer("cartodbdark_matter", name="CartoDB Dark").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Streets",
    ).add_to(m)

    # County boundaries
    county_shp = os.path.join(REPO_ROOT, "mn_counties", "mn_counties.shp")
    if os.path.exists(county_shp):
        try:
            import geopandas as gpd
            counties = gpd.read_file(county_shp).to_crs("EPSG:4326")
            name_col = "NAME" if "NAME" in counties.columns else (
                "CTY_NAME" if "CTY_NAME" in counties.columns else counties.columns[0]
            )
            fg_counties = folium.FeatureGroup(name="County boundaries", show=True)
            folium.GeoJson(
                counties[[name_col, "geometry"]].to_json(),
                name="counties",
                style_function=lambda f: {
                    "fillColor": "#000000", "fillOpacity": 0.0,
                    "color": "#444444", "weight": 1.2, "opacity": 0.7,
                },
                highlight_function=lambda f: {
                    "fillColor": "#ffaa00", "fillOpacity": 0.15,
                    "color": "#ff6600", "weight": 2.5,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=[name_col], aliases=["County:"], sticky=True,
                ),
                popup=folium.GeoJsonPopup(fields=[name_col], aliases=["County:"]),
            ).add_to(fg_counties)
            fg_counties.add_to(m)
            print(f"    added {len(counties)} county polygons")
        except Exception as e:
            print(f"    failed to add counties: {e}")

    # Inter-city (simplified UE flows on the rural backbone)
    fg_inter = folium.FeatureGroup(name=f"Inter-city UE flows ({len(inter_links):,})")
    for L in sorted(inter_links, key=lambda x: x["flow"]):
        lon1, lat1 = simp_coords[L["init"]]
        lon2, lat2 = simp_coords[L["term"]]
        c = flow_color(L["flow"], vmax)
        w = line_weight(L["flow"], vmax)
        popup_html = (
            f"<b>Inter-city link {L['init']} → {L['term']}</b><br>"
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
            tooltip=f"{L['init']}→{L['term']}  flow={L['flow']:,.0f}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg_inter)
    fg_inter.add_to(m)

    # Urban sub-area (real road network inside metro rectangles)
    fg_urban = folium.FeatureGroup(name=f"Urban sub-area AON ({len(urban_links):,})")
    for L in sorted(urban_links, key=lambda x: x["flow"]):
        lon1, lat1 = full_coords[L["init"]]
        lon2, lat2 = full_coords[L["term"]]
        c = flow_color(L["flow"], vmax)
        w = line_weight(L["flow"], vmax)
        popup_html = (
            f"<b>Urban [{L['city']}] {L['init']} → {L['term']}</b><br>"
            f"Flow: <b>{L['flow']:,.0f}</b> veh<br>"
            f"Length: {L['length']:.2f} mi<br>"
            f"VMT: {L['vmt']:,.0f}"
        )
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color=c, weight=w, opacity=0.9,
            tooltip=f"{L['city']} {L['init']}→{L['term']}  flow={L['flow']:,.0f}",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fg_urban)
    fg_urban.add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index:9999;
                background: white; padding: 10px 14px; border:1px solid #888;
                border-radius:6px; font-family:sans-serif; font-size:12px;
                line-height:1.4;">
      <b>{year} real-network flows</b><br>
      Inter-city UE + urban sub-area AON<br>
      (centroid & super-node connectors hidden)<br>
      <span style="background:#440154;display:inline-block;width:18px;">&nbsp;</span> low<br>
      <span style="background:#21908d;display:inline-block;width:18px;">&nbsp;</span> mid<br>
      <span style="background:#fde725;display:inline-block;width:18px;">&nbsp;</span>
        max ({vmax:,.0f} veh)<br>
      <span style="color:#666">Click any link for details</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    print(f"  wrote {out_html}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", "-y", required=True, choices=["2017", "2019", "2024"])
    ap.add_argument("--min-flow", type=float, default=10.0)
    ap.add_argument("--max-links", type=int, default=80000,
                    help="cap each layer (inter-city, urban) to top-N by flow")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(REPO_ROOT, f"realflow_map_{args.year}.html")
    print(f"[realflow] year={args.year}")
    build_map(args.year, args.min_flow, args.max_links, out)


if __name__ == "__main__":
    main()
