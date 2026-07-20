"""Difference map between two years using REAL-network flows.

Combines:
  - simplified link_flows (inter-city, with centroid + super-node connectors removed)
  - urban_link_flows_<City>.csv (sub-area AON on the full road network)

Matches links between two years by endpoint coordinates and renders:
  - Common links: solid line, colored by delta (B - A) on a diverging scale,
    line width scaled by |delta|.
  - Only in B (added): dashed line, colored by +flow on the diverging scale.
  - Only in A (removed): dashed line, colored by -flow on the diverging scale.

All links share the same diverging color scale.

Outputs HTML: realflow_diff_<a>_vs_<b>.html

Usage:
    python visualize_realflow_diff.py --a 2017 --b 2024
"""

import argparse
import csv
import math
import os
import re


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# IO (mirrors visualize_realflow_maps.py)
# ---------------------------------------------------------------------------

def load_simp_node_coords(simp_dir):
    out = {}
    with open(os.path.join(simp_dir, "node_coords.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["node_id"])] = (float(row["lon"]), float(row["lat"]))
    return out


def load_super_nodes(simp_dir):
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
                orig = float(p[1])
                if orig in (-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                    super_ids.add(nid)
            except ValueError:
                continue
    return super_ids


def load_zones(simp_dir):
    zones = set()
    with open(os.path.join(simp_dir, "trips.txt")) as f:
        for line in f:
            m = re.match(r"\s*Origin\s+(\d+)", line)
            if m:
                zones.add(int(m.group(1)))
    return zones


def load_simp_links(simp_dir):
    rows = []
    with open(os.path.join(simp_dir, "link_flows.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "init": int(row["init_node"]),
                "term": int(row["term_node"]),
                "flow": float(row["flow_real"]),
                "cap": float(row["capacity"]),
                "len": float(row["length"]),
            })
    return rows


def load_full_node_coords(full_dir):
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


def load_urban_links(simp_dir):
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
                    }
    return list(out.values())


# ---------------------------------------------------------------------------
# Build per-year link dict keyed by COORDINATE PAIR (so we can match across
# years even when node IDs differ).
# ---------------------------------------------------------------------------

def coord_key(lon, lat, prec=5):
    return (round(lon, prec), round(lat, prec))


def link_key_coords(p1, p2, prec=5):
    return (coord_key(*p1, prec), coord_key(*p2, prec))


def build_year_links(year, min_flow):
    simp_dir = os.path.join(REPO_ROOT, f"NetworkFiles{year}_WithZones_Simplified")
    full_dir = os.path.join(REPO_ROOT, f"NetworkFiles{year}_Connected")

    simp_coords = load_simp_node_coords(simp_dir)
    super_nodes = load_super_nodes(simp_dir)
    zones = load_zones(simp_dir)
    simp_links = load_simp_links(simp_dir)
    full_coords = load_full_node_coords(full_dir)
    urban_links = load_urban_links(simp_dir)

    bad = super_nodes | zones
    out = {}  # coord-key -> {"flow", "len", "kind", "cap"}

    # Inter-city
    for L in simp_links:
        if L["init"] in bad or L["term"] in bad:
            continue
        if L["init"] not in simp_coords or L["term"] not in simp_coords:
            continue
        if L["flow"] < min_flow:
            continue
        k = link_key_coords(simp_coords[L["init"]], simp_coords[L["term"]])
        prev = out.get(k)
        if prev is None or L["flow"] > prev["flow"]:
            out[k] = {
                "flow": L["flow"], "len": L["len"], "cap": L["cap"],
                "kind": "inter",
            }

    # Urban (sub-area)
    for L in urban_links:
        if L["init"] not in full_coords or L["term"] not in full_coords:
            continue
        if L["flow"] < min_flow:
            continue
        k = link_key_coords(full_coords[L["init"]], full_coords[L["term"]])
        prev = out.get(k)
        if prev is None or L["flow"] > prev["flow"]:
            out[k] = {
                "flow": L["flow"], "len": L["length"], "cap": 0.0,
                "kind": f"urban[{L['city']}]",
            }

    return out


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------

def diverging_color(delta, dmax):
    """Diverging blue (negative) -> yellow (zero) -> red (positive)."""
    if dmax <= 0:
        return "#ffff66"
    t = max(-1.0, min(1.0, delta / dmax))
    a = abs(t)
    if t >= 0:
        # yellow (255,255,102) -> dark red (165,0,38)
        r = int(255 + (165 - 255) * a)
        g = int(255 + (0 - 255) * a)
        b = int(102 + (38 - 102) * a)
    else:
        # yellow (255,255,102) -> dark blue (49,54,149)
        r = int(255 + (49 - 255) * a)
        g = int(255 + (54 - 255) * a)
        b = int(102 + (149 - 102) * a)
    return f"#{r:02x}{g:02x}{b:02x}"


def line_weight(mag, dmax):
    if dmax <= 0:
        return 1.5
    t = max(0.0, min(1.0, mag / dmax))
    return 1.5 + 6.0 * (t ** 0.5)


# ---------------------------------------------------------------------------
# Build map
# ---------------------------------------------------------------------------

def build_diff(year_a, year_b, min_flow, max_links, out_html):
    import folium

    print(f"  building {year_a} link set...")
    A = build_year_links(year_a, min_flow)
    print(f"    {len(A):,} unique links")
    print(f"  building {year_b} link set...")
    B = build_year_links(year_b, min_flow)
    print(f"    {len(B):,} unique links")

    keys_a = set(A); keys_b = set(B)
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    print(f"  common: {len(common):,}  only-{year_a}: {len(only_a):,}  "
          f"only-{year_b}: {len(only_b):,}")

    # Compute deltas
    deltas = []
    for k in common:
        d = B[k]["flow"] - A[k]["flow"]
        if abs(d) >= min_flow or A[k]["flow"] >= min_flow or B[k]["flow"] >= min_flow:
            deltas.append((k, d))
    deltas.sort(key=lambda x: -abs(x[1]))
    if max_links and len(deltas) > max_links:
        print(f"    capping common to top {max_links} by |Δ|")
        deltas = deltas[:max_links]

    only_a_sorted = sorted(only_a, key=lambda k: -A[k]["flow"])[:max_links]
    only_b_sorted = sorted(only_b, key=lambda k: -B[k]["flow"])[:max_links]

    # Global magnitude scale: combine common deltas + added flows + removed flows
    mags = [abs(d) for _, d in deltas]
    mags += [B[k]["flow"] for k in only_b_sorted]
    mags += [A[k]["flow"] for k in only_a_sorted]
    dmax = max(mags) if mags else 1.0
    print(f"    dmax (color scale max): {dmax:,.0f}")

    # Center
    all_keys = list(common) + list(only_a) + list(only_b)
    if not all_keys:
        print("  nothing to draw"); return
    lons = [k[0][0] for k in all_keys[:5000]]
    lats = [k[0][1] for k in all_keys[:5000]]
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
            fg_c = folium.FeatureGroup(name="County boundaries", show=True)
            folium.GeoJson(
                counties[[name_col, "geometry"]].to_json(),
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
            ).add_to(fg_c)
            fg_c.add_to(m)
        except Exception as e:
            print(f"    counties failed: {e}")

    # Common - solid lines
    fg_common = folium.FeatureGroup(name=f"Common Δ ({year_b}−{year_a})  n={len(deltas):,}")
    for k, d in sorted(deltas, key=lambda x: abs(x[1])):
        (lon1, lat1), (lon2, lat2) = k
        a = A[k]; b = B[k]
        c = diverging_color(d, dmax)
        w = line_weight(abs(d), dmax)
        popup_html = (
            f"<b>{a['kind']} → {b['kind']}</b><br>"
            f"Flow {year_a}: <b>{a['flow']:,.0f}</b> veh<br>"
            f"Flow {year_b}: <b>{b['flow']:,.0f}</b> veh<br>"
            f"Δ: <b>{d:+,.0f}</b> veh<br>"
            f"Length: {b['len']:.2f} mi"
        )
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color=c, weight=w, opacity=0.9,
            tooltip=f"Δ={d:+,.0f}",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(fg_common)
    fg_common.add_to(m)

    # Added in B - dashed bright green
    fg_added = folium.FeatureGroup(name=f"Only in {year_b} (added)  n={len(only_b_sorted):,}")
    for k in only_b_sorted:
        (lon1, lat1), (lon2, lat2) = k
        b = B[k]
        w = max(line_weight(b["flow"], dmax), 3.5)
        popup_html = (
            f"<b>NEW in {year_b} ({b['kind']})</b><br>"
            f"Did NOT exist (or had 0 flow) in {year_a}<br>"
            f"Flow: <b>{b['flow']:,.0f}</b> veh<br>"
            f"Length: {b['len']:.2f} mi"
        )
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color="#00aa00", weight=w, opacity=0.95,
            dash_array="10,6",
            tooltip=f"NEW in {year_b}: {b['flow']:,.0f}",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(fg_added)
    fg_added.add_to(m)

    # Removed (only in A) - dashed magenta
    fg_removed = folium.FeatureGroup(name=f"Only in {year_a} (removed)  n={len(only_a_sorted):,}")
    for k in only_a_sorted:
        (lon1, lat1), (lon2, lat2) = k
        a = A[k]
        w = max(line_weight(a["flow"], dmax), 3.5)
        popup_html = (
            f"<b>REMOVED after {year_a} ({a['kind']})</b><br>"
            f"Existed in {year_a}, gone in {year_b}<br>"
            f"Flow: <b>{a['flow']:,.0f}</b> veh<br>"
            f"Length: {a['len']:.2f} mi"
        )
        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color="#cc00cc", weight=w, opacity=0.95,
            dash_array="10,6",
            tooltip=f"REMOVED in {year_a}: {a['flow']:,.0f}",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(fg_removed)
    fg_removed.add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index:9999;
                background: white; padding: 10px 14px; border:1px solid #888;
                border-radius:6px; font-family:sans-serif; font-size:12px;
                line-height:1.5;">
      <b>Real-network Δ flow ({year_b} − {year_a})</b><br>
      <span style="background:#313695;display:inline-block;width:18px;">&nbsp;</span>
        big decrease (max −{dmax:,.0f})<br>
      <span style="background:#74add1;display:inline-block;width:18px;">&nbsp;</span>
        slight decrease<br>
      <span style="background:#ffff66;display:inline-block;width:18px;border:1px solid #888;">&nbsp;</span>
        ≈ unchanged<br>
      <span style="background:#fdae61;display:inline-block;width:18px;">&nbsp;</span>
        slight increase<br>
      <span style="background:#a50026;display:inline-block;width:18px;">&nbsp;</span>
        big increase (max +{dmax:,.0f})<br>
      <hr style="margin:4px 0">
      <span style="color:#00aa00;font-weight:bold;letter-spacing:1px">- - -</span>
        added in {year_b} (not in {year_a})<br>
      <span style="color:#cc00cc;font-weight:bold;letter-spacing:1px">- - -</span>
        removed after {year_a} (not in {year_b})<br>
      Connectors / super-nodes hidden.
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(out_html)
    print(f"  wrote {out_html}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, choices=["2017", "2019", "2024"])
    ap.add_argument("--b", required=True, choices=["2017", "2019", "2024"])
    ap.add_argument("--min-flow", type=float, default=10.0)
    ap.add_argument("--max-links", type=int, default=60000,
                    help="cap each layer (common, added, removed) to top-N")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.a == args.b:
        ap.error("--a and --b must differ")

    out = args.out or os.path.join(
        REPO_ROOT, f"realflow_diff_{args.a}_vs_{args.b}.html"
    )
    print(f"[realflow-diff] {args.a} -> {args.b}")
    build_diff(args.a, args.b, args.min_flow, args.max_links, out)


if __name__ == "__main__":
    main()
