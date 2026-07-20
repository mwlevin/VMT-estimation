"""Map the 2019 urban-routing detours vs 2017.

For each year, urban_link_flows_<city>.csv lists every link that carried
flow during the sub-area Dijkstra routing. Comparing 2019 vs 2017 shows
which roads were used as detour routes only in 2019.

Focus: links carrying significant VMT in 2019 but ~0 in 2017,
geographically extending NORTH/EAST from Twin Cities through Pine -> Carlton ->
St. Louis -> Lake -> Cook County.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import folium
import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
CITIES = ["TwinCities_Metro", "Rochester", "Duluth", "St_Cloud", "Mankato", "Moorhead"]
ROUND = 5


def parse_node_coords(path: Path) -> dict[int, tuple[float, float]]:
    out = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("~") or s.startswith("<"):
                continue
            parts = s.replace(";", "").split()
            if len(parts) < 3:
                continue
            try:
                out[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return out


def load_year(year: int, coords: dict) -> pd.DataFrame:
    parts = []
    for c in CITIES:
        p = ROOT / f"NetworkFiles{year}_WithZones_Simplified" / f"urban_link_flows_{c}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["city"] = c
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # attach coords from the connected-network nodes.txt for this year
    df["x_init"] = df["init_node"].map(lambda n: coords.get(n, (None, None))[0])
    df["y_init"] = df["init_node"].map(lambda n: coords.get(n, (None, None))[1])
    df["x_term"] = df["term_node"].map(lambda n: coords.get(n, (None, None))[0])
    df["y_term"] = df["term_node"].map(lambda n: coords.get(n, (None, None))[1])
    before = len(df)
    df = df.dropna(subset=["x_init", "y_init", "x_term", "y_term"])
    if len(df) < before:
        print(f"  {year}: dropped {before - len(df):,} rows missing coords")
    return df


def build_keyed(df: pd.DataFrame) -> pd.DataFrame:
    """Key each link by sorted endpoint coords so years can be merged."""
    a = df[["x_init", "y_init"]].round(ROUND).to_numpy()
    b = df[["x_term", "y_term"]].round(ROUND).to_numpy()
    keys = []
    for i in range(len(df)):
        ka, kb = (a[i, 0], a[i, 1]), (b[i, 0], b[i, 1])
        keys.append(tuple(sorted([ka, kb])))
    df = df.copy()
    df["key"] = keys
    return df


def main():
    coords17 = parse_node_coords(ROOT / "NetworkFiles2017_Connected" / "nodes.txt")
    coords19 = parse_node_coords(ROOT / "NetworkFiles2019_Connected" / "nodes.txt")
    coords24 = parse_node_coords(ROOT / "NetworkFiles2024_Connected" / "nodes.txt")
    df17 = load_year(2017, coords17)
    df19 = load_year(2019, coords19)
    df24 = load_year(2024, coords24)
    print(f"loaded urban links 2017={len(df17):,} 2019={len(df19):,} 2024={len(df24):,}")

    df17 = build_keyed(df17)
    df19 = build_keyed(df19)
    df24 = build_keyed(df24)

    # Aggregate flow per key per year (sum because multiple cities or directions)
    flow_col = "flow" if "flow" in df19.columns else ("flow_real" if "flow_real" in df19.columns else None)
    print("flow col:", flow_col)

    agg17 = df17.groupby("key")[flow_col].sum() if flow_col else None
    agg19 = df19.groupby("key")[flow_col].sum() if flow_col else None
    agg24 = df24.groupby("key")[flow_col].sum() if flow_col else None

    # Take coords from 2019 row table (key -> first endpoint pair)
    coord_lookup = {k: (g.iloc[0]["x_init"], g.iloc[0]["y_init"], g.iloc[0]["x_term"], g.iloc[0]["y_term"])
                    for k, g in df19.groupby("key")}
    # Add any 2017-only keys
    for k, g in df17.groupby("key"):
        if k not in coord_lookup:
            coord_lookup[k] = (g.iloc[0]["x_init"], g.iloc[0]["y_init"], g.iloc[0]["x_term"], g.iloc[0]["y_term"])
    for k, g in df24.groupby("key"):
        if k not in coord_lookup:
            coord_lookup[k] = (g.iloc[0]["x_init"], g.iloc[0]["y_init"], g.iloc[0]["x_term"], g.iloc[0]["y_term"])

    # Detour: high flow in 2019, zero/tiny in 2017 AND zero/tiny in 2024
    THRESH = 50.0
    detour_keys = []
    only19_total_vmt = 0
    only19_2024_normal_total_vmt = 0
    for k in agg19.index:
        f19 = agg19.get(k, 0)
        f17 = agg17.get(k, 0) if agg17 is not None else 0
        f24 = agg24.get(k, 0) if agg24 is not None else 0
        if f19 < THRESH:
            continue
        if f17 < THRESH * 0.1 and f24 < THRESH * 0.1:
            detour_keys.append((k, f19, f17, f24))

    print(f"\nDetour links (flow>={THRESH} in 2019 but ~0 in both 2017 and 2024): {len(detour_keys):,}")
    detour_keys.sort(key=lambda t: -t[1])
    print("Top 10 detour links by 2019 flow:")
    for k, f19, f17, f24 in detour_keys[:10]:
        print(f"  flow19={f19:>8,.0f}  flow17={f17:>6,.0f}  flow24={f24:>6,.0f}  coords={k}")

    # Build map
    m = folium.Map(location=[46.5, -92.5], zoom_start=7, tiles=None)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Streets"
    ).add_to(m)

    # County overlay
    cdf = gpd.read_file(ROOT / "mn_counties" / "mn_counties.shp").to_crs(4326)
    name_col = next((c for c in ["CTY_NAME", "NAME", "COUNTY_NAM", "COUNTY"] if c in cdf.columns), cdf.columns[0])
    folium.GeoJson(
        cdf.__geo_interface__, name="Counties",
        style_function=lambda f: {"color": "#444", "weight": 1.5, "fill": False, "opacity": 0.6},
        tooltip=folium.GeoJsonTooltip(fields=[name_col], aliases=["County:"], sticky=True),
    ).add_to(m)

    # Layer 1: 2017 urban routing footprint (faint blue)
    fg17 = folium.FeatureGroup(name=f"2017 urban routing ({len(df17):,} link-records)", show=False)
    plotted = set()
    for k, g in df17.groupby("key"):
        if k in plotted:
            continue
        plotted.add(k)
        x1, y1, x2, y2 = coord_lookup[k]
        folium.PolyLine(
            locations=[(y1, x1), (y2, x2)], color="#1f77b4",
            weight=1.5, opacity=0.45,
        ).add_to(fg17)
    fg17.add_to(m)

    # Layer 2: 2024 urban routing footprint (faint green)
    fg24 = folium.FeatureGroup(name=f"2024 urban routing ({len(df24):,} link-records)", show=False)
    plotted = set()
    for k, g in df24.groupby("key"):
        if k in plotted:
            continue
        plotted.add(k)
        x1, y1, x2, y2 = coord_lookup[k]
        folium.PolyLine(
            locations=[(y1, x1), (y2, x2)], color="#2ca02c",
            weight=1.5, opacity=0.45,
        ).add_to(fg24)
    fg24.add_to(m)

    # Layer 3: 2019 urban routing footprint (orange)
    fg19 = folium.FeatureGroup(name=f"2019 urban routing ({len(df19):,} link-records)", show=False)
    plotted = set()
    for k, g in df19.groupby("key"):
        if k in plotted:
            continue
        plotted.add(k)
        x1, y1, x2, y2 = coord_lookup[k]
        folium.PolyLine(
            locations=[(y1, x1), (y2, x2)], color="#ff7f0e",
            weight=1.5, opacity=0.5,
        ).add_to(fg19)
    fg19.add_to(m)

    # Layer 4: DETOUR LINKS (red, thick) - only in 2019
    fg_d = folium.FeatureGroup(name=f"DETOUR (2019 only, flow>={THRESH:.0f}): {len(detour_keys):,} links",
                               show=True)
    if detour_keys:
        max_f = max(t[1] for t in detour_keys)
    else:
        max_f = 1.0
    for k, f19, f17, f24 in detour_keys:
        x1, y1, x2, y2 = coord_lookup[k]
        w = 2.5 + 6 * (f19 / max_f)
        folium.PolyLine(
            locations=[(y1, x1), (y2, x2)], color="#d62728",
            weight=w, opacity=0.9,
            tooltip=f"2019 flow={f19:,.0f} (2017={f17:,.0f}, 2024={f24:,.0f})",
        ).add_to(fg_d)
    fg_d.add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white;
                padding:10px 14px; border:2px solid #444; font-family:Arial; font-size:13px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.3); max-width:380px;">
      <b>Why 2019 urban VMT spiked in NE Minnesota</b><br>
      Same 2024 demand, sub-area Dijkstra routing.<br><br>
      <span style="color:#1f77b4">━</span> 2017 urban paths<br>
      <span style="color:#ff7f0e">━</span> 2019 urban paths<br>
      <span style="color:#2ca02c">━</span> 2024 urban paths<br>
      <span style="color:#d62728"><b>━━</b></span> <b>2019-only detour links</b><br>
      &nbsp;&nbsp;(flow ≥ {THRESH:.0f} in 2019, ~0 in 2017 and 2024)<br>
      &nbsp;&nbsp;count: {len(detour_keys):,}<br>
      <hr style="margin:6px 0">
      <b>Urban VMT by metro:</b><br>
      <table style="font-size:12px"><tr><th></th><th>2017</th><th>2019</th><th>2024</th></tr>
      <tr><td>TC</td><td>40.1M</td><td><b>47.9M</b></td><td>36.8M</td></tr>
      <tr><td>Mankato</td><td>7.1M</td><td><b>8.9M</b></td><td>5.0M</td></tr>
      <tr><td>St.Cloud</td><td>4.5M</td><td><b>7.2M</b></td><td>5.4M</td></tr>
      </table>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out = ROOT / "TrafficAssigment" / "detour_2019_vs_others.html"
    m.save(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
