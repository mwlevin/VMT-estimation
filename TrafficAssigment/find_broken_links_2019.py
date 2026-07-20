"""Identify broken/missing road links in 2019 vs 2017 and 2024.

Key insight: node IDs differ across years (node 1 has different coords each year),
so we MUST compare links by their endpoint COORDINATES, not by node IDs.

Build a folium map showing links unique to each year in the Arrowhead region
(Cook / Lake / St. Louis / Carlton counties), where the 2019 VMT spike is.
"""
from __future__ import annotations
from pathlib import Path
import folium
import geopandas as gpd
from shapely.geometry import LineString, box

ROOT = Path(__file__).resolve().parent.parent

# Arrowhead bbox: covers Cook, Lake, St. Louis (Duluth area + North Shore + BWCA edge)
BBOX = (-92.6, 46.4, -89.4, 48.3)  # lon_min, lat_min, lon_max, lat_max

YEARS = [2017, 2019, 2024]
ROUND = 5  # 5 decimal degrees ~ 1.1 m


def parse_nodes(p: Path) -> dict[int, tuple[float, float]]:
    coords = {}
    with open(p) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("~"):
                continue
            parts = s.rstrip(";").split()
            if len(parts) < 3:
                continue
            try:
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return coords


def parse_links(p: Path) -> list[tuple[int, int]]:
    out = []
    in_data = False
    with open(p) as f:
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
            if len(parts) < 2:
                continue
            try:
                out.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    return out


def build_coord_links(year: int) -> set[tuple]:
    """Return set of links keyed by sorted endpoint coordinates."""
    nd = parse_nodes(ROOT / f"NetworkFiles{year}_Connected" / "nodes.txt")
    links = parse_links(ROOT / f"NetworkFiles{year}_Connected" / "network.txt")
    out = set()
    for a, b in links:
        if a not in nd or b not in nd:
            continue
        ax, ay = nd[a]
        bx, by = nd[b]
        # bbox filter: at least one endpoint in bbox
        in_bbox = lambda x, y: BBOX[0] <= x <= BBOX[2] and BBOX[1] <= y <= BBOX[3]
        if not (in_bbox(ax, ay) or in_bbox(bx, by)):
            continue
        ka = (round(ax, ROUND), round(ay, ROUND))
        kb = (round(bx, ROUND), round(by, ROUND))
        key = tuple(sorted([ka, kb]))
        out.add(key)
    return out


def main():
    print(f"Filtering Arrowhead bbox {BBOX}...")
    sets = {y: build_coord_links(y) for y in YEARS}
    for y, s in sets.items():
        print(f"  {y}: {len(s):,} links")

    only_17 = sets[2017] - sets[2019] - sets[2024]
    only_19 = sets[2019] - sets[2017] - sets[2024]
    only_24 = sets[2024] - sets[2017] - sets[2019]
    in_17_24_not_19 = (sets[2017] & sets[2024]) - sets[2019]  # likely broken in 2019
    in_19_only_among_pair = sets[2019] - sets[2017]
    common_all = sets[2017] & sets[2019] & sets[2024]

    print(f"\nLinks unique to 2017: {len(only_17):,}")
    print(f"Links unique to 2019: {len(only_19):,}")
    print(f"Links unique to 2024: {len(only_24):,}")
    print(f"Links in 2017 AND 2024 but NOT in 2019 (lost-then-restored, suspicious): {len(in_17_24_not_19):,}")
    print(f"Links in all three years:                                                {len(common_all):,}")

    # Build map centered on Lake County / North Shore
    m = folium.Map(location=[47.4, -91.3], zoom_start=8, tiles=None)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Streets"
    ).add_to(m)

    # County overlay
    counties_path = ROOT / "mn_counties" / "mn_counties.shp"
    if counties_path.exists():
        try:
            cdf = gpd.read_file(counties_path).to_crs(4326)
            bbox_poly = box(*BBOX)
            cdf = cdf[cdf.geometry.intersects(bbox_poly)]
            name_col = next((c for c in ["CTY_NAME", "NAME", "COUNTY_NAM", "COUNTY"] if c in cdf.columns), cdf.columns[0])
            folium.GeoJson(
                cdf.__geo_interface__,
                name="Counties",
                style_function=lambda f: {"color": "#333", "weight": 2, "fill": False, "opacity": 0.7},
                tooltip=folium.GeoJsonTooltip(fields=[name_col], aliases=["County:"], sticky=True),
            ).add_to(m)
        except Exception as e:
            print(f"county overlay skipped: {e}")

    def add_layer(name, link_set, color, weight, dash=None, opacity=0.85):
        fg = folium.FeatureGroup(name=name, show=True)
        for (ka, kb) in link_set:
            folium.PolyLine(
                locations=[(ka[1], ka[0]), (kb[1], kb[0])],
                color=color, weight=weight, opacity=opacity,
                dash_array=dash,
            ).add_to(fg)
        fg.add_to(m)

    # Plot in order: common (faint), unique-each, then highlight broken-in-2019
    add_layer(f"Common (all 3 years) [{len(common_all):,}]", common_all, "#888888", 1.5, opacity=0.35)
    add_layer(f"Only 2017 [{len(only_17):,}]", only_17, "#1f77b4", 2.5)
    add_layer(f"Only 2019 [{len(only_19):,}]", only_19, "#ff7f0e", 2.5)
    add_layer(f"Only 2024 [{len(only_24):,}]", only_24, "#2ca02c", 2.5)
    add_layer(
        f"BROKEN in 2019 (in 2017+2024 but missing in 2019) [{len(in_17_24_not_19):,}]",
        in_17_24_not_19, "#d62728", 4.5, dash="8,5", opacity=0.95,
    )

    # Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white;
                padding:10px 14px; border:2px solid #444; font-family:Arial; font-size:13px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.3); max-width:340px;">
      <b>Arrowhead network changes 2017-2019-2024</b><br>
      Same 2024 demand applied to all years.<br><br>
      <span style="color:#888">━━</span> Common to all three years ({len(common_all):,})<br>
      <span style="color:#1f77b4">━━</span> Only in 2017 ({len(only_17):,})<br>
      <span style="color:#ff7f0e">━━</span> Only in 2019 ({len(only_19):,})<br>
      <span style="color:#2ca02c">━━</span> Only in 2024 ({len(only_24):,})<br>
      <span style="color:#d62728">━ ━ ━</span> <b>Broken in 2019</b><br>
      &nbsp;&nbsp;(present in 2017 AND 2024, missing in 2019): {len(in_17_24_not_19):,}<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out = ROOT / "TrafficAssigment" / "arrowhead_broken_links_2019.html"
    m.save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
