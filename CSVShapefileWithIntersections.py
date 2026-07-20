"""
CSV + Shapefile Network Builder WITH Intersection Nodes
========================================================
This script builds a road network by:
1. Matching CSV attributes (capacity, FFS, lanes) to shapefile segments
2. Finding geometric intersections between all road segments
3. Splitting links at BOTH attribute changes AND intersections
4. Creating a fully connected network

The key improvement over the original: nodes are created at road intersections,
not just where attributes change.
"""

from pathlib import Path
import argparse
import warnings
import struct
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
from shapely.ops import split, snap, linemerge, substring
from shapely.strtree import STRtree
from collections import defaultdict
import folium

warnings.filterwarnings('ignore')

# ==========================
# USER PATHS / CONSTANTS
# ==========================

SHP_PATH = Path("/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/old_data_with_not_all_of_CSV/2024data/MN_Routes.shp")
CSV_PATH = Path("/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/new_data_CSV_with_all_Links_information/UM_DATA_REQUEST_2024.csv")
OUT_DIR = Path("/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/NetworkFiles2024_Connected_HPMS_Split")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# FUNCTIONAL CLASS FILTER
# ==========================
# Only include roads with functional class <= MAX_FUNC
# 1 = Interstate
# 2 = Principal Arterial - Freeway/Expressway
# 3 = Principal Arterial - Other
# 4 = Minor Arterial
# 5 = Major Collector
# 6 = Minor Collector  (excluded)
# 7 = Local            (excluded)
MAX_FUNC = 5

# Also exclude FAC=5 (Non-mainline connectors, no AADT, excluded from mileage)
EXCLUDE_FAC = {5}

MIN_FFS = 5.0          # mph
MIN_CAPACITY = 500.0   # veh/h
COORD_PRECISION = 5    # legacy exact-key fallback; node matching now uses SNAP_TOLERANCE
SNAP_TOLERANCE = 1.0   # meters - merge endpoints/intersections within this distance


def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build connected TNTP network from MnDOT route shapefile and HPMS CSV."
    )
    parser.add_argument("--shp", type=Path, default=SHP_PATH, help="Input MN_Routes.shp path")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Input HPMS CSV path")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output connected network folder")
    return parser.parse_args()


def read_polylinez_m_arrays(shp_path: Path):
    """Read per-vertex M-values from an ESRI PolyLineZ shapefile."""
    arrays = []
    with open(shp_path, "rb") as f:
        f.read(100)
        while True:
            rec_header = f.read(8)
            if not rec_header:
                break
            _, content_len_words = struct.unpack(">2i", rec_header)
            content = f.read(content_len_words * 2)
            if len(content) < 44:
                arrays.append([])
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type != 13:  # PolyLineZ
                arrays.append([])
                continue
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            pos = 44 + 4 * num_parts + 16 * num_points
            pos += 16 + 8 * num_points  # Z range + Z array
            if len(content) < pos + 16 + 8 * num_points:
                arrays.append([])
                continue
            pos += 16  # M range
            arrays.append(list(struct.unpack("<" + "d" * num_points, content[pos:pos + 8 * num_points])))
    return arrays


# ==========================
# ROUTE SYSTEM DEFINITIONS (same as original)
# ==========================

ROUTE_SYST_DEFAULTS = {
    1:  {'name': 'Interstate Trunk Highway',       'func': 1, 'fac': 2, 'lanes': 2, 'bffs': 70},
    2:  {'name': 'U.S. Trunk Highway',             'func': 2, 'fac': 2, 'lanes': 2, 'bffs': 65},
    3:  {'name': 'Minnesota Trunk Highway',        'func': 3, 'fac': 2, 'lanes': 1, 'bffs': 55},
    41: {'name': 'Interstate HOV',                 'func': 1, 'fac': 1, 'lanes': 1, 'bffs': 65},
    4:  {'name': 'County State-Aid Highway',       'func': 4, 'fac': 2, 'lanes': 1, 'bffs': 50},
    5:  {'name': 'Municipal State-Aid Street',     'func': 5, 'fac': 2, 'lanes': 1, 'bffs': 35},
    7:  {'name': 'County Road',                    'func': 6, 'fac': 2, 'lanes': 1, 'bffs': 45},
    8:  {'name': 'Township Road',                  'func': 7, 'fac': 2, 'lanes': 1, 'bffs': 35},
    9:  {'name': 'Unorganized Territory Road',     'func': 7, 'fac': 2, 'lanes': 1, 'bffs': 30},
    10: {'name': 'Municipal Street',               'func': 7, 'fac': 2, 'lanes': 1, 'bffs': 25},
    19: {'name': 'Frontage Road',                  'func': 7, 'fac': 2, 'lanes': 1, 'bffs': 35},
    22: {'name': 'Ramps',                          'func': 1, 'fac': 4, 'lanes': 1, 'bffs': 40},
}

DEFAULT_ROUTE_SYST = {'name': 'Unknown', 'func': 7, 'fac': 2, 'lanes': 1, 'bffs': 25}


# ==========================
# GEOMETRY HELPERS
# ==========================

def ensure_2d(geom):
    """Drop Z-coordinates if present."""
    if geom is None or geom.is_empty:
        return geom
    if getattr(geom, "has_z", False):
        if isinstance(geom, LineString):
            return LineString([(x, y) for x, y, *_ in geom.coords])
        elif isinstance(geom, MultiLineString):
            return MultiLineString(
                [LineString([(x, y) for x, y, *_ in g.coords]) for g in geom.geoms]
            )
        elif isinstance(geom, Point):
            return Point(geom.x, geom.y)
    return geom


def get_line_coords(geom):
    """Get coordinates from LineString or MultiLineString."""
    geom = ensure_2d(geom)
    if isinstance(geom, LineString):
        return list(geom.coords)
    elif isinstance(geom, MultiLineString):
        coords = []
        for line in geom.geoms:
            coords.extend(list(line.coords))
        return coords
    return []


def cumulative_distances_for_coords(coords):
    distances = [0.0]
    total = 0.0
    for i in range(1, len(coords)):
        p0 = Point(coords[i - 1][:2])
        p1 = Point(coords[i][:2])
        total += p0.distance(p1)
        distances.append(total)
    return distances


def cumulative_distances_for_geometry(geom):
    """Return cumulative distances for all vertices, respecting multipart breaks."""
    geom = ensure_2d(geom)
    distances = []
    total = 0.0
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        for i, coord in enumerate(coords):
            if i > 0:
                total += Point(coords[i - 1][:2]).distance(Point(coord[:2]))
            distances.append(total)
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            coords = list(part.coords)
            for i, coord in enumerate(coords):
                if distances and i == 0:
                    distances.append(total)
                else:
                    if i > 0:
                        total += Point(coords[i - 1][:2]).distance(Point(coord[:2]))
                    distances.append(total)
    return distances


def interpolate_measure(distances, measures, distance):
    if not distances or not measures or len(distances) != len(measures):
        return None
    if distance <= distances[0]:
        return float(measures[0])
    if distance >= distances[-1]:
        return float(measures[-1])
    idx = np.searchsorted(distances, distance, side="right") - 1
    idx = max(0, min(idx, len(distances) - 2))
    d0, d1 = distances[idx], distances[idx + 1]
    m0, m1 = float(measures[idx]), float(measures[idx + 1])
    if d1 <= d0:
        return m0
    t = (distance - d0) / (d1 - d0)
    return m0 + t * (m1 - m0)


def interpolate_distance_for_measure(distances, measures, measure):
    if not distances or not measures or len(distances) != len(measures):
        return None
    pairs = sorted(zip([float(m) for m in measures], distances), key=lambda x: x[0])
    ms = [p[0] for p in pairs]
    ds = [p[1] for p in pairs]
    if measure <= ms[0]:
        return float(ds[0])
    if measure >= ms[-1]:
        return float(ds[-1])
    idx = np.searchsorted(ms, measure, side="right") - 1
    idx = max(0, min(idx, len(ms) - 2))
    m0, m1 = ms[idx], ms[idx + 1]
    d0, d1 = float(ds[idx]), float(ds[idx + 1])
    if m1 <= m0:
        return d0
    t = (measure - m0) / (m1 - m0)
    return d0 + t * (d1 - d0)


# ==========================
# FIND INTERSECTIONS BETWEEN SEGMENTS
# ==========================

def find_all_intersections(gdf):
    """
    Find all geometric intersection points between road segments.
    Returns a dict: segment_index -> list of (distance_along_segment, Point)
    """
    info("Finding geometric intersections between road segments...")
    
    # Ensure 2D geometries
    geometries = [ensure_2d(g) for g in gdf.geometry.tolist()]
    
    # Build spatial index
    tree = STRtree(geometries)
    
    # Store split points for each segment: idx -> [(distance, point), ...]
    segment_splits = defaultdict(list)
    
    total = len(geometries)
    intersection_count = 0
    
    for i, geom_i in enumerate(geometries):
        if i % 5000 == 0:
            info(f"  Processing segment {i}/{total}...")
        
        if geom_i is None or geom_i.is_empty:
            continue
        
        # Find candidates that might intersect
        candidates = tree.query(geom_i)
        
        for j in candidates:
            if j <= i:  # Avoid duplicate checks and self-intersection
                continue
            
            geom_j = geometries[j]
            if geom_j is None or geom_j.is_empty:
                continue
            
            # Check actual intersection
            if not geom_i.intersects(geom_j):
                continue
            
            intersection = geom_i.intersection(geom_j)
            
            if intersection.is_empty:
                continue
            
            # Extract intersection points
            points = []
            if intersection.geom_type == 'Point':
                points = [intersection]
            elif intersection.geom_type == 'MultiPoint':
                points = list(intersection.geoms)
            elif intersection.geom_type == 'LineString':
                # Lines overlap - use endpoints
                coords = list(intersection.coords)
                if coords:
                    points = [Point(coords[0][:2]), Point(coords[-1][:2])]
            elif intersection.geom_type == 'MultiLineString':
                for line in intersection.geoms:
                    coords = list(line.coords)
                    if coords:
                        points.extend([Point(coords[0][:2]), Point(coords[-1][:2])])
            elif intersection.geom_type == 'GeometryCollection':
                for g in intersection.geoms:
                    if g.geom_type == 'Point':
                        points.append(g)
                    elif g.geom_type == 'MultiPoint':
                        points.extend(list(g.geoms))
            
            # Record intersection points for both segments
            for pt in points:
                pt = ensure_2d(pt)
                
                # Check if point is at endpoints (don't split at endpoints)
                coords_i = get_line_coords(geom_i)
                coords_j = get_line_coords(geom_j)
                
                # Distance along each segment
                dist_i = geom_i.project(pt)
                dist_j = geom_j.project(pt)
                
                # Only add if not at endpoints (with small tolerance)
                eps = 0.1  # 0.1 meters
                
                if eps < dist_i < geom_i.length - eps:
                    segment_splits[i].append((dist_i, pt))
                
                if eps < dist_j < geom_j.length - eps:
                    segment_splits[j].append((dist_j, pt))
                
                intersection_count += 1
    
    info(f"  Found {intersection_count} intersection points")
    info(f"  Segments needing splits: {len(segment_splits)}")
    
    return segment_splits


def split_segment_at_points(geom, split_points):
    """
    Split a LineString at given points.
    split_points: list of (distance, Point)
    Returns list of LineString segments.
    """
    geom = ensure_2d(geom)
    
    if not split_points or geom is None or geom.is_empty:
        return [geom] if geom and not geom.is_empty else []
    
    # Handle MultiLineString
    if isinstance(geom, MultiLineString):
        # Merge into single line if possible
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            geom = merged
        else:
            # Can't merge, just return original
            return [geom]
    
    # Sort split points by distance
    split_points = sorted(split_points, key=lambda x: x[0])
    
    # Get all coordinates
    coords = list(geom.coords)
    
    segments = []
    current_coords = [coords[0]]
    split_idx = 0
    
    # Walk through the line's coordinates
    cumulative_dist = 0
    
    for i in range(1, len(coords)):
        seg_start = Point(coords[i-1][:2])
        seg_end = Point(coords[i][:2])
        seg_length = seg_start.distance(seg_end)
        
        # Check if any split points fall in this segment
        while split_idx < len(split_points):
            split_dist, split_pt = split_points[split_idx]
            
            # Is this split point in the current segment?
            if cumulative_dist <= split_dist <= cumulative_dist + seg_length + 0.01:
                # Interpolate the exact position
                if seg_length > 0:
                    t = (split_dist - cumulative_dist) / seg_length
                    t = max(0, min(1, t))
                    split_x = coords[i-1][0] + t * (coords[i][0] - coords[i-1][0])
                    split_y = coords[i-1][1] + t * (coords[i][1] - coords[i-1][1])
                else:
                    split_x, split_y = coords[i-1][0], coords[i-1][1]
                
                # End current segment at split point
                current_coords.append((split_x, split_y))
                
                if len(current_coords) >= 2:
                    new_seg = LineString(current_coords)
                    if new_seg.length > 0.1:  # Min 0.1 meters
                        segments.append(new_seg)
                
                # Start new segment from split point
                current_coords = [(split_x, split_y)]
                split_idx += 1
            else:
                break
        
        current_coords.append(coords[i])
        cumulative_dist += seg_length
    
    # Add final segment
    if len(current_coords) >= 2:
        final_seg = LineString(current_coords)
        if final_seg.length > 0.1:
            segments.append(final_seg)
    
    return segments if segments else [geom]


def substring_by_distance_miles(geom, start_mi, end_mi):
    """Return the portion of a route geometry between two distance measures."""
    geom = ensure_2d(geom)
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            geom = merged
        else:
            return None

    if not isinstance(geom, LineString):
        return None

    start_dist = max(0.0, float(start_mi) * 1609.344)
    end_dist = min(float(end_mi) * 1609.344, geom.length)

    if end_dist <= start_dist + 0.1:
        return None

    part = substring(geom, start_dist, end_dist)
    if part is None or part.is_empty:
        return None
    return ensure_2d(part)


# ==========================
# HCM CLASSIFICATION + FFS/CAPACITY (same as original)
# ==========================

def get_facility_classification(row) -> str:
    """Classifies a road segment based on HCM rules."""
    try:
        fac = row['FAC']
        func = row['FUNC']
        lanes_per_dir = row['THRU_LNS']

        if fac == 4:
            return "Ramp"
        if fac == 5:
            return "Non-mainline"
        
        if func == 1:
            return "Freeway"
        if func == 2:
            return "Freeway"
        
        if lanes_per_dir >= 2:
            return "Multilane"
        if lanes_per_dir == 1:
            return "TwoLane"
            
    except Exception:
        pass
    return "TwoLane"


def calculate_ffs_capacity_row(row) -> pd.Series:
    """Calculate FFS and capacity using the older HCM-style adjustments."""
    classification = row['classification']
    num_lanes_dir = max(1, row['THRU_LNS'] if pd.notna(row['THRU_LNS']) else 1)
    func = row.get('FUNC', 7)
    data_source = row.get('data_source', 'Default')
    is_csv_source = str(data_source).startswith('CSV')

    shoulder_r = pd.to_numeric(row.get('RIGHT_SHL_WIDTH', -9), errors='coerce')
    shoulder_l = pd.to_numeric(row.get('LEFT_SHL_WIDTH', -9), errors='coerce')
    median_type = pd.to_numeric(row.get('MED_TYPE', -9), errors='coerce')
    
    route_syst = row.get('ROUTE_SYST', 99)
    try:
        route_syst = int(route_syst)
    except:
        route_syst = 99
    
    route_defaults = ROUTE_SYST_DEFAULTS.get(route_syst, DEFAULT_ROUTE_SYST)
    default_bffs = row.get('DEFAULT_BFFS', route_defaults['bffs'])

    f_A = 0.0
    f_S = 0.0
    f_LC = 0.0
    f_M = 0.0

    # Approximation used in the older builder.
    area_type = "Urban" if func <= 4 else "Rural"
    if area_type == "Urban":
        access_density = 35.0
    elif func <= 6:
        access_density = 7.0
    else:
        access_density = 5.0
    if access_density > 0:
        f_A = min(10.0, access_density / 4.0)
    
    if classification == "Freeway":
        BFFS = 75.4 if is_csv_source else max(default_bffs, 65)
        rs_missing = pd.isna(shoulder_r) or shoulder_r == -9
        c = 6.0 if rs_missing else min(6.0, shoulder_r)

        k_map = {2: 0.6, 3: 0.4, 4: 0.2}
        k = k_map.get(num_lanes_dir, 0.1)
        f_LC = k * (6.0 - c)

        FFS = BFFS - f_LC
        ffs_for_calc = max(50.0, FFS)
        c_per_lane = min(2400.0, 2200.0 + 10.0 * (ffs_for_calc - 50.0))
        Capacity = c_per_lane * num_lanes_dir

    elif classification == "Multilane":
        BFFS = 60.0 if is_csv_source else max(default_bffs, 50)
        f_M = 1.6 if median_type == 2 else 0.0

        rs_missing = pd.isna(shoulder_r) or shoulder_r == -9
        c = 6.0 if rs_missing else min(6.0, shoulder_r)

        k_map = {2: 0.6, 3: 0.4, 4: 0.2}
        k = k_map.get(num_lanes_dir, 0.1)
        f_LC = k * (6.0 - c)

        FFS = BFFS - f_LC - f_M - f_A
        ffs_for_calc = max(45.0, FFS)
        c_per_lane = min(2300.0, 1900.0 + 20.0 * (ffs_for_calc - 45.0))
        Capacity = c_per_lane * num_lanes_dir

    elif classification == "TwoLane":
        BFFS = 55.0 if is_csv_source else default_bffs

        ls_missing = pd.isna(shoulder_l) or shoulder_l == -9
        rs_missing = pd.isna(shoulder_r) or shoulder_r == -9
        if ls_missing and rs_missing:
            eff_shoulder = 6.0
        elif ls_missing:
            eff_shoulder = shoulder_r
        elif rs_missing:
            eff_shoulder = shoulder_l
        else:
            eff_shoulder = min(shoulder_l, shoulder_r)

        if eff_shoulder < 2:
            f_S = 4.2
        elif eff_shoulder < 4:
            f_S = 1.3
        else:
            f_S = 0.0

        FFS = BFFS - f_A - f_S
        c_per_lane = 1700.0
        Capacity = c_per_lane * num_lanes_dir

    elif classification == "Ramp":
        FFS = 40.0 if is_csv_source else default_bffs
        Capacity = 1800.0 * num_lanes_dir

    elif classification == "Non-mainline":
        FFS = 25.0
        Capacity = 1500.0 * num_lanes_dir

    elif classification == "Non-inventory":
        FFS = default_bffs
        Capacity = 1700.0 * num_lanes_dir

    else:
        FFS = default_bffs if data_source == 'Default' else 25.0
        Capacity = 1000.0 * num_lanes_dir

    if data_source == 'Default':
        FFS = max(FFS, default_bffs * 0.7)

    FFS = round(max(FFS, MIN_FFS), 1)
    Capacity = max(Capacity, MIN_CAPACITY)
    
    return pd.Series([FFS, Capacity, classification])


# ==========================
# MATCH CSV TO SHAPEFILE
# ==========================

def match_csv_to_shapefile_segments(roads_gdf, csv_df):
    """Match CSV attributes to shapefile segments."""
    info("Matching CSV attributes to shapefile segments...")
    
    results = []
    matched_csv = 0
    used_defaults = 0
    filled_hpms_gaps = 0
    
    csv_df = csv_df.copy()
    csv_df['BEGPT'] = pd.to_numeric(csv_df['BEGPT'], errors='coerce')
    csv_df['ENDPT'] = pd.to_numeric(csv_df['ENDPT'], errors='coerce')
    
    csv_grouped = csv_df.groupby('ROUTE_ID')
    
    def build_result(road, geom, route_id, route_syst, route_defaults, seg_begin, seg_end, csv_match, hpms_records=None):
        route_direction = route_id[-1] if route_id else 'I'
        result = {
            'geometry': geom,
            'ROUTE_ID': route_id,
            'ROUTE_SYST': route_syst,
            'route_syst_name': route_defaults['name'],
            'BEGPNT': seg_begin,
            'ENDPNT': seg_end,
            'route_direction': route_direction,
            'route_geom_miles': geom.length / 1609.344 if geom is not None and not geom.is_empty else 0.0,
            'hpms_records': hpms_records or [],
            'measure_distances': road.get('measure_distances', []),
            'measure_values': road.get('measure_values', []),
        }
        if csv_match is not None:
            result['THRU_LNS'] = csv_match.get('THRU_LNS', route_defaults['lanes'])
            result['FAC'] = csv_match.get('FAC', route_defaults['fac'])
            result['FUNC'] = csv_match.get('FUNC', route_defaults['func'])
            result['MED_TYPE'] = csv_match.get('MED_TYPE', 0)
            result['RIGHT_SHL_WIDTH'] = csv_match.get('RIGHT_SHL_WIDTH', -9)
            result['LEFT_SHL_WIDTH'] = csv_match.get('LEFT_SHL_WIDTH', -9)
            result['data_source'] = 'CSV'
            result['DEFAULT_BFFS'] = route_defaults['bffs']
        else:
            result['THRU_LNS'] = route_defaults['lanes']
            result['FUNC'] = route_defaults['func']
            result['FAC'] = route_defaults['fac']
            result['MED_TYPE'] = 0
            result['RIGHT_SHL_WIDTH'] = -9
            result['LEFT_SHL_WIDTH'] = -9
            result['data_source'] = 'Default'
            result['DEFAULT_BFFS'] = route_defaults['bffs']
        return result

    for idx, road in roads_gdf.iterrows():
        route_id = str(road['ROUTE_ID']).strip()
        route_syst = road.get('ROUTE_SYST', 99)
        
        try:
            route_syst = int(route_syst)
        except:
            route_syst = 99
        
        segment_length_miles = road.geometry.length / 1609.344
        m_values = road.get('M_VALUES', [])
        coords = get_line_coords(road.geometry)
        measure_distances = []
        measure_values = []
        if isinstance(m_values, list) and len(m_values) == len(coords) and len(m_values) >= 2:
            measure_distances = cumulative_distances_for_coords(coords)
            measure_values = [float(m) for m in m_values]
            seg_begin = min(measure_values)
            seg_end = max(measure_values)
        else:
            seg_begin = pd.to_numeric(road.get('BEGPNT'), errors='coerce')
            seg_end = pd.to_numeric(road.get('ENDPNT'), errors='coerce')
            if pd.isna(seg_begin):
                seg_begin = 0.0
            if pd.isna(seg_end):
                seg_end = seg_begin + segment_length_miles
        road = road.copy()
        road['measure_distances'] = measure_distances
        road['measure_values'] = measure_values
        
        route_defaults = ROUTE_SYST_DEFAULTS.get(route_syst, DEFAULT_ROUTE_SYST)
        
        overlapping = pd.DataFrame()
        
        if route_id in csv_grouped.groups:
            route_csv = csv_grouped.get_group(route_id)
            
            overlapping = route_csv[
                (route_csv['BEGPT'] < seg_end) &
                (route_csv['ENDPT'] > seg_begin)
            ]
            
            if not overlapping.empty:
                overlapping = overlapping.copy()
                overlapping['overlap_start'] = overlapping['BEGPT'].clip(lower=seg_begin)
                overlapping['overlap_end'] = overlapping['ENDPT'].clip(upper=seg_end)
                overlapping['overlap_amount'] = overlapping['overlap_end'] - overlapping['overlap_start']
                overlapping = overlapping[overlapping['overlap_amount'] > 0].sort_values('overlap_start')

        if not overlapping.empty:
            hpms_records = overlapping.to_dict('records')
            csv_match = overlapping.loc[overlapping['overlap_amount'].idxmax()]
            results.append(build_result(
                road, road.geometry, route_id, route_syst, route_defaults,
                seg_begin, seg_end, csv_match, hpms_records
            ))
            matched_csv += 1
        else:
            results.append(build_result(
                road, road.geometry, route_id, route_syst, route_defaults,
                seg_begin, seg_end, None, []
            ))
            used_defaults += 1
    
    info(f"  Matched to CSV: {matched_csv}")
    info(f"  Filled HPMS gaps with nearest CSV attributes: {filled_hpms_gaps}")
    info(f"  Used defaults: {used_defaults}")
    
    return gpd.GeoDataFrame(results, crs=roads_gdf.crs)


def select_hpms_record(hpms_records, local_begin, local_end):
    """Pick the HPMS record that best covers a final split link interval."""
    if not hpms_records:
        return None, False

    matches = []
    for rec in hpms_records:
        try:
            beg = float(rec['BEGPT'])
            end = float(rec['ENDPT'])
        except (KeyError, TypeError, ValueError):
            continue
        overlap = min(end, local_end) - max(beg, local_begin)
        if overlap > 0:
            matches.append((overlap, rec))

    if matches:
        return max(matches, key=lambda x: x[0])[1], False

    # Preserve topology through HPMS milepost gaps by borrowing the nearest
    # adjacent HPMS attributes.
    midpoint = (local_begin + local_end) / 2.0
    candidates = []
    for rec in hpms_records:
        try:
            beg = float(rec['BEGPT'])
            end = float(rec['ENDPT'])
        except (KeyError, TypeError, ValueError):
            continue
        distance = min(abs(beg - midpoint), abs(end - midpoint))
        candidates.append((distance, rec))

    if candidates:
        return min(candidates, key=lambda x: x[0])[1], True
    return None, False


def calculate_link_attrs_for_interval(row, local_begin, local_end):
    """Calculate FFS/capacity for a split link from its route milepost interval."""
    hpms_record, used_gap_fill = select_hpms_record(row.get('hpms_records', []), local_begin, local_end)
    if hpms_record is None:
        return {
            'THRU_LNS': row.get('THRU_LNS', 1),
            'FAC': row.get('FAC', 2),
            'FUNC': row.get('FUNC', 7),
            'MED_TYPE': row.get('MED_TYPE', 0),
            'RIGHT_SHL_WIDTH': row.get('RIGHT_SHL_WIDTH', -9),
            'LEFT_SHL_WIDTH': row.get('LEFT_SHL_WIDTH', -9),
            'DEFAULT_BFFS': row.get('DEFAULT_BFFS', ROUTE_SYST_DEFAULTS.get(row.get('ROUTE_SYST', 99), DEFAULT_ROUTE_SYST)['bffs']),
            'data_source': row.get('data_source', 'Default'),
            'classification': row.get('classification', 'Unknown'),
            'FFS': row.get('FFS', 30),
            'CAPACITY': row.get('CAPACITY', 1000),
            'used_hpms_gap_fill': False,
        }

    attr_row = pd.Series({
        'classification': None,
        'THRU_LNS': hpms_record.get('THRU_LNS', row.get('THRU_LNS', 1)),
        'FAC': hpms_record.get('FAC', row.get('FAC', 2)),
        'FUNC': hpms_record.get('FUNC', row.get('FUNC', 7)),
        'MED_TYPE': hpms_record.get('MED_TYPE', row.get('MED_TYPE', 0)),
        'RIGHT_SHL_WIDTH': hpms_record.get('RIGHT_SHL_WIDTH', row.get('RIGHT_SHL_WIDTH', -9)),
        'LEFT_SHL_WIDTH': hpms_record.get('LEFT_SHL_WIDTH', row.get('LEFT_SHL_WIDTH', -9)),
        'ROUTE_SYST': row.get('ROUTE_SYST', 99),
        'DEFAULT_BFFS': row.get('DEFAULT_BFFS', ROUTE_SYST_DEFAULTS.get(row.get('ROUTE_SYST', 99), DEFAULT_ROUTE_SYST)['bffs']),
        'data_source': 'CSV',
    })
    attr_row['classification'] = get_facility_classification(attr_row)
    ffs, capacity, classification = calculate_ffs_capacity_row(attr_row)
    return {
        'THRU_LNS': attr_row['THRU_LNS'],
        'FAC': attr_row['FAC'],
        'FUNC': attr_row['FUNC'],
        'MED_TYPE': attr_row['MED_TYPE'],
        'RIGHT_SHL_WIDTH': attr_row['RIGHT_SHL_WIDTH'],
        'LEFT_SHL_WIDTH': attr_row['LEFT_SHL_WIDTH'],
        'data_source': 'CSV_GapFill' if used_gap_fill else 'CSV',
        'classification': classification,
        'FFS': ffs,
        'CAPACITY': capacity,
        'used_hpms_gap_fill': used_gap_fill,
    }


def split_final_segment_by_hpms(row, parent_geom, seg):
    """Split an already intersection-cut segment at HPMS milepost boundaries."""
    measure_distances = row.get('measure_distances', [])
    measure_values = row.get('measure_values', [])
    if not measure_distances or not measure_values or not row.get('hpms_records', []):
        route_begin = float(row.get('BEGPNT', 0.0))
        route_end = float(row.get('ENDPNT', route_begin + row.get('route_geom_miles', 0.0)))
        return [(seg, route_begin, route_end)]

    coords = list(seg.coords)
    if len(coords) < 2:
        return []

    start_pt = Point(coords[0][:2])
    end_pt = Point(coords[-1][:2])
    d0 = parent_geom.project(start_pt)
    d1 = parent_geom.project(end_pt)
    m0 = interpolate_measure(measure_distances, measure_values, d0)
    m1 = interpolate_measure(measure_distances, measure_values, d1)
    if m0 is None or m1 is None:
        route_begin = float(row.get('BEGPNT', 0.0))
        route_end = float(row.get('ENDPNT', route_begin + row.get('route_geom_miles', 0.0)))
        return [(seg, route_begin, route_end)]

    low_m, high_m = min(m0, m1), max(m0, m1)
    boundaries = [low_m, high_m]
    for rec in row.get('hpms_records', []):
        for key in ('BEGPT', 'ENDPT'):
            try:
                b = float(rec[key])
            except (KeyError, TypeError, ValueError):
                continue
            if low_m + 1e-6 < b < high_m - 1e-6:
                boundaries.append(b)
    boundaries = sorted(set(round(b, 8) for b in boundaries))
    if len(boundaries) <= 2:
        return [(seg, low_m, high_m)]

    base_d = min(d0, d1)
    pieces = []
    local_distances = []
    for b in boundaries:
        gd = interpolate_distance_for_measure(measure_distances, measure_values, b)
        if gd is None:
            continue
        local = gd - base_d
        local = max(0.0, min(seg.length, local))
        local_distances.append((b, local))

    local_distances = sorted(local_distances, key=lambda x: x[1])
    for (m_a, d_a), (m_b, d_b) in zip(local_distances[:-1], local_distances[1:]):
        if d_b <= d_a + 0.1:
            continue
        part = substring(seg, d_a, d_b)
        if part is not None and not part.is_empty and part.length >= 0.1:
            pieces.append((ensure_2d(part), min(m_a, m_b), max(m_a, m_b)))

    # Splitting must never change connectivity. If unusual M-values or multipart
    # geometry make the split pieces fail to cover the original segment, keep the
    # original segment and only use the best interval for attributes.
    coverage = sum(part.length for part, _, _ in pieces)
    if not pieces or coverage < seg.length - 0.5:
        return [(seg, low_m, high_m)]
    return pieces


# ==========================
# BUILD CONNECTED NETWORK
# ==========================

def build_connected_network(gdf, segment_splits):
    """
    Build network by:
    1. Splitting segments at intersection points
    2. Creating nodes at all endpoints (both original and new split points)
    3. Ensuring nodes are shared between connecting links
    """
    info("Building connected network...")
    
    # Use a meter-based spatial hash instead of exact coordinate rounding.
    # HPMS splitting can create endpoints that should be identical but differ
    # by tiny floating-point amounts; exact keys fragment the network.
    snap_tol = max(float(SNAP_TOLERANCE), 1e-9)
    snap_tol2 = snap_tol * snap_tol
    node_grid = defaultdict(list)  # grid cell -> node ids
    node_coords = {}               # node id -> (x, y)
    node_data = []
    link_data = []
    node_counter = 1
    hpms_gap_fill_links = 0
    snapped_node_reuses = 0

    def grid_key(x, y):
        return (int(np.floor(x / snap_tol)), int(np.floor(y / snap_tol)))

    def get_or_create_node(pt):
        nonlocal node_counter, snapped_node_reuses
        x, y = float(pt[0]), float(pt[1])
        gx, gy = grid_key(x, y)
        best_node = None
        best_dist2 = snap_tol2

        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for nid in node_grid.get((ix, iy), []):
                    nx, ny = node_coords[nid]
                    dist2 = (x - nx) ** 2 + (y - ny) ** 2
                    if dist2 <= best_dist2:
                        best_dist2 = dist2
                        best_node = nid

        if best_node is not None:
            if best_dist2 > 1e-12:
                snapped_node_reuses += 1
            return best_node

        nid = node_counter
        node_counter += 1
        node_coords[nid] = (x, y)
        node_grid[grid_key(x, y)].append(nid)
        node_data.append({
            'node_id': nid,
            'x': x,
            'y': y
        })
        return nid
    
    total = len(gdf)
    
    for idx, row in gdf.iterrows():
        if idx % 5000 == 0:
            info(f"  Processing segment {idx}/{total}...")
        
        geom = ensure_2d(row.geometry)
        if geom is None or geom.is_empty:
            continue
        
        # Get split points for this segment
        splits = segment_splits.get(idx, [])
        
        # Split the segment
        if splits:
            segments = split_segment_at_points(geom, splits)
        else:
            segments = [geom] if isinstance(geom, LineString) else list(geom.geoms) if isinstance(geom, MultiLineString) else []
        
        # Process each segment
        for seg in segments:
            if seg is None or seg.is_empty or seg.length < 0.1:
                continue
            
            # Handle MultiLineString that might have been returned
            if isinstance(seg, MultiLineString):
                sub_segs = list(seg.geoms)
            else:
                sub_segs = [seg]
            
            for seg in sub_segs:
                if seg is None or seg.is_empty or seg.length < 0.1:
                    continue
                
                coords = list(seg.coords)
                if len(coords) < 2:
                    continue

                for final_seg, local_begin, local_end in split_final_segment_by_hpms(row, geom, seg):
                    final_coords = list(final_seg.coords)
                    if len(final_coords) < 2:
                        continue

                    start_pt = final_coords[0][:2]
                    end_pt = final_coords[-1][:2]

                    start_node = get_or_create_node(start_pt)
                    end_node = get_or_create_node(end_pt)

                    # Skip self-loops
                    if start_node == end_node:
                        continue

                    attrs = calculate_link_attrs_for_interval(row, local_begin, local_end)
                    ffs = attrs['FFS']
                    capacity = attrs['CAPACITY']
                    classification = attrs['classification']
                    data_source = attrs['data_source']
                    route_syst_name = row.get('route_syst_name', 'Unknown')
                    if attrs.get('used_hpms_gap_fill'):
                        hpms_gap_fill_links += 1

                    route_direction = row.get('route_direction', 'I')
                    fac = attrs.get('FAC', row.get('FAC', 2))

                    if route_direction == 'D' or fac == 6:
                        link_start_node = end_node
                        link_end_node = start_node
                    else:
                        link_start_node = start_node
                        link_end_node = end_node

                    if pd.isna(ffs) or ffs <= 0:
                        ffs = MIN_FFS
                    if pd.isna(capacity) or capacity <= 0:
                        capacity = MIN_CAPACITY

                    link_length_miles = final_seg.length / 1609.344

                    if link_length_miles < 0.001:
                        continue

                    free_flow_time = (link_length_miles / ffs) * 60.0

                    link_data.append({
                        'init_node': link_start_node,
                        'term_node': link_end_node,
                        'capacity': capacity,
                        'length': link_length_miles,
                        'free_flow_time': free_flow_time,
                        'ffs': ffs,
                        'classification': classification,
                        'data_source': data_source,
                        'route_syst_name': route_syst_name,
                        'route_direction': route_direction,
                        'route_mi_beg': local_begin,
                        'route_mi_end': local_end,
                        'geometry': final_seg
                    })
    
    info(f"  Created {len(node_data)} nodes")
    info(f"  Created {len(link_data)} links")
    info(f"  Reused nearby nodes within {SNAP_TOLERANCE:g} m: {snapped_node_reuses}")
    info(f"  Links using nearest HPMS attributes for milepost gaps: {hpms_gap_fill_links}")
    
    return node_data, link_data


def check_connectivity(link_data):
    """Check network connectivity."""
    from collections import deque
    
    adj = defaultdict(set)
    for link in link_data:
        n1 = link['init_node']
        n2 = link['term_node']
        adj[n1].add(n2)
        adj[n2].add(n1)
    
    visited = set()
    components = []
    
    def bfs(start):
        q = deque([start])
        component = set()
        while q:
            node = q.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    q.append(neighbor)
        return component
    
    for node in adj.keys():
        if node not in visited:
            comp = bfs(node)
            components.append(comp)
    
    sizes = sorted([len(c) for c in components], reverse=True)
    
    info(f"\nConnectivity check:")
    info(f"  Total nodes: {len(adj)}")
    info(f"  Connected components: {len(components)}")
    if sizes:
        info(f"  Largest component: {sizes[0]} nodes ({sizes[0]/len(adj)*100:.1f}%)")
        if len(sizes) > 1:
            info(f"  Second largest: {sizes[1]} nodes")
        if len(sizes) > 5:
            info(f"  Top 5 sizes: {sizes[:5]}")
    
    return components, sizes


# ==========================
# OUTPUT FILES
# ==========================

def write_output_files(node_data, link_data, out_dir, crs):
    """Write all output files."""
    
    # Create GeoDataFrames
    nodes_gdf = gpd.GeoDataFrame(
        node_data,
        geometry=gpd.points_from_xy([n['x'] for n in node_data], [n['y'] for n in node_data]),
        crs=crs
    )
    
    links_gdf = gpd.GeoDataFrame(link_data, crs=crs)
    
    # Convert to WGS84 for output
    nodes_wgs = nodes_gdf.to_crs(epsg=4326)
    links_wgs = links_gdf.to_crs(epsg=4326)
    
    # Write nodes.txt
    info("Writing nodes.txt...")
    with open(out_dir / "nodes.txt", 'w') as f:
        f.write("~ node_id\tx\ty\t;\n")
        for idx, row in nodes_wgs.iterrows():
            f.write(f"\t{row['node_id']}\t{row.geometry.x:.8f}\t{row.geometry.y:.8f}\t;\n")
    
    # Write network.txt
    info("Writing network.txt...")
    num_nodes = len(node_data)
    num_links = len(link_data)
    
    with open(out_dir / "network.txt", 'w') as f:
        f.write(f"<NUMBER OF ZONES> {num_nodes}\n")
        f.write(f"<NUMBER OF NODES> {num_nodes}\n")
        f.write("<FIRST THRU NODE> 1\n")
        f.write(f"<NUMBER OF LINKS> {num_links}\n")
        f.write("<ORIGINAL HEADER>~ \tInit node \tTerm node \tCapacity \tLength "
                "\tFree Flow Time \tB\tPower\tSpeed limit \tToll \tType\t;\n")
        f.write("<END OF METADATA>\n\n")
        f.write("~\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower"
                "\tspeed\ttoll\tlink_type\t;\n")
        
        for link in link_data:
            b = 0.15
            power = 4
            toll = 0
            link_type = 1
            
            f.write(f"\t{link['init_node']}\t{link['term_node']}\t{link['capacity']:.0f}"
                   f"\t{link['length']:.5f}\t{link['free_flow_time']:.5f}"
                   f"\t{b}\t{power}\t{link['ffs']:.2f}\t{toll}\t{link_type}\t;\n")
    
    # Write shapefiles
    info("Writing shapefiles...")
    nodes_wgs.to_file(out_dir / "nodes.shp")
    
    links_for_shp = links_wgs.copy()
    links_for_shp = links_for_shp.rename(columns={
        'classification': 'class',
        'data_source': 'source',
        'route_syst_name': 'rd_type'
    })
    links_for_shp.to_file(out_dir / "links.shp")
    
    info(f"Output written to {out_dir}")
    
    return nodes_wgs, links_wgs


def create_visualization(nodes_gdf, links_gdf, output_path):
    """Create an HTML map visualization."""
    info("Creating visualization...")
    
    # Get center point
    bounds = links_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    
    # Sample links for visualization (too many would be slow)
    sample_size = min(10000, len(links_gdf))
    links_sample = links_gdf.sample(n=sample_size, random_state=42) if len(links_gdf) > sample_size else links_gdf
    
    def style_function(feature):
        source = feature['properties'].get('source', 'Unknown')
        if source == 'CSV':
            return {'color': 'blue', 'weight': 2, 'opacity': 0.7}
        else:
            return {'color': 'gray', 'weight': 1, 'opacity': 0.5}
    
    folium.GeoJson(
        links_sample,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['init_node', 'term_node', 'ffs', 'capacity'],
            aliases=['Start:', 'End:', 'FFS:', 'Capacity:']
        )
    ).add_to(m)
    
    m.save(str(output_path))
    info(f"Saved visualization: {output_path}")


# ==========================
# MAIN
# ==========================

def main():
    global SHP_PATH, CSV_PATH, OUT_DIR
    args = parse_args()
    SHP_PATH = args.shp
    CSV_PATH = args.csv
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    info("="*70)
    info("CSV + Shapefile Network Builder WITH Intersection Nodes")
    info("="*70)
    
    if not SHP_PATH.exists():
        raise FileNotFoundError(f"Shapefile not found: {SHP_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    
    # Load shapefile
    info(f"Loading shapefile: {SHP_PATH}")
    roads = gpd.read_file(SHP_PATH)
    m_arrays = read_polylinez_m_arrays(SHP_PATH)
    if len(m_arrays) == len(roads):
        roads["M_VALUES"] = m_arrays
        info("  Loaded route M-values from shapefile")
    else:
        warn(f"  Could not align M-values ({len(m_arrays)}) with shapefile rows ({len(roads)})")
    info(f"  Loaded {len(roads)} road segments")
    info(f"  CRS: {roads.crs}")
    
    # Clean up
    roads["ROUTE_ID"] = roads["ROUTE_ID"].astype(str).str.strip()
    roads = roads[~roads.geometry.is_empty & roads.geometry.is_valid].copy()
    
    # Project to meters for accurate calculations
    roads_proj = roads.to_crs(epsg=26915)  # UTM zone 15N for Minnesota
    info(f"  Projected to EPSG:26915 (UTM 15N)")
    
    # Load CSV
    info(f"Loading CSV: {CSV_PATH}")
    csv_df = pd.read_csv(CSV_PATH, dtype=object, keep_default_na=False, encoding='latin-1', on_bad_lines='warn')
    csv_df.columns = [c.strip().upper().replace("\ufeff", "") for c in csv_df.columns]
    csv_df["ROUTE_ID"] = csv_df["ROUTE_ID"].astype(str).str.strip()
    
    numeric_cols = [
        "BEGPT", "ENDPT", "THRU_LNS", "FUNC", "FAC", "MED_TYPE",
        "RIGHT_SHL_WIDTH", "LEFT_SHL_WIDTH"
    ]
    for col in numeric_cols:
        if col in csv_df.columns:
            csv_df[col] = pd.to_numeric(csv_df[col], errors='coerce')
    
    info(f"  Loaded {len(csv_df)} CSV records")
    
    # Match CSV to shapefile
    info("\n" + "="*70)
    info("STEP 1: Match CSV attributes to shapefile segments")
    info("="*70)
    matched_gdf = match_csv_to_shapefile_segments(roads_proj, csv_df)
    
    # Clean and calculate FFS/Capacity
    matched_gdf['THRU_LNS'] = pd.to_numeric(matched_gdf['THRU_LNS'], errors='coerce').fillna(1)
    matched_gdf['FUNC'] = pd.to_numeric(matched_gdf['FUNC'], errors='coerce').fillna(7)
    matched_gdf['FAC'] = pd.to_numeric(matched_gdf['FAC'], errors='coerce').fillna(2)
    matched_gdf['MED_TYPE'] = pd.to_numeric(matched_gdf['MED_TYPE'], errors='coerce').fillna(-9)
    matched_gdf['RIGHT_SHL_WIDTH'] = pd.to_numeric(matched_gdf['RIGHT_SHL_WIDTH'], errors='coerce').fillna(-9)
    matched_gdf['LEFT_SHL_WIDTH'] = pd.to_numeric(matched_gdf['LEFT_SHL_WIDTH'], errors='coerce').fillna(-9)
    
    # --- FILTER: Keep only functional class 1-5 (drop Minor Collector & Local) ---
    info("\n" + "="*70)
    info(f"STEP 1b: Filter by functional class (FUNC <= {MAX_FUNC}) and facility type")
    info("="*70)
    
    before_count = len(matched_gdf)
    
    # Show distribution before filtering
    func_counts = matched_gdf['FUNC'].value_counts().sort_index()
    func_names = {1: 'Interstate', 2: 'Princ. Arterial-Fwy', 3: 'Princ. Arterial-Other',
                  4: 'Minor Arterial', 5: 'Major Collector', 6: 'Minor Collector', 7: 'Local'}
    info("  Functional class distribution (before filter):")
    for fc, cnt in func_counts.items():
        keep = '✓' if fc <= MAX_FUNC else '✗'
        info(f"    {keep} FUNC {int(fc):d} ({func_names.get(int(fc), 'Unknown')}): {cnt:,} segments")
    
    fac_counts = matched_gdf['FAC'].value_counts().sort_index()
    fac_names = {1: 'One-Way', 2: 'Two-Way', 4: 'Ramp', 5: 'Non-Mainline', 6: 'Non-Inventory Dir'}
    info("  Facility type distribution (before filter):")
    for ft, cnt in fac_counts.items():
        keep = '✗' if int(ft) in EXCLUDE_FAC else '✓'
        info(f"    {keep} FAC {int(ft):d} ({fac_names.get(int(ft), 'Unknown')}): {cnt:,} segments")
    
    # Apply filters
    func_mask = matched_gdf['FUNC'] <= MAX_FUNC
    fac_mask = ~matched_gdf['FAC'].isin(EXCLUDE_FAC)
    matched_gdf = matched_gdf[func_mask & fac_mask].copy()
    matched_gdf = matched_gdf.reset_index(drop=True)  # Reset index for intersection finding
    
    after_count = len(matched_gdf)
    removed = before_count - after_count
    info(f"  Kept {after_count:,} segments ({after_count/before_count*100:.1f}%)")
    info(f"  Removed {removed:,} segments ({removed/before_count*100:.1f}%)")
    
    matched_gdf['classification'] = matched_gdf.apply(get_facility_classification, axis=1)
    matched_gdf[['FFS', 'CAPACITY', 'classification']] = matched_gdf.apply(calculate_ffs_capacity_row, axis=1)
    
    # Find intersections
    info("\n" + "="*70)
    info("STEP 2: Find geometric intersections between roads")
    info("="*70)
    segment_splits = find_all_intersections(matched_gdf)
    
    # Build connected network
    info("\n" + "="*70)
    info("STEP 3: Build connected network with intersection nodes")
    info("="*70)
    node_data, link_data = build_connected_network(matched_gdf, segment_splits)
    
    # Check connectivity
    components, sizes = check_connectivity(link_data)
    
    # Write output
    info("\n" + "="*70)
    info("STEP 4: Write output files")
    info("="*70)
    nodes_wgs, links_wgs = write_output_files(node_data, link_data, OUT_DIR, roads_proj.crs)
    
    # Create visualization
    create_visualization(nodes_wgs, links_wgs, OUT_DIR / "network_visualization.html")
    
    info("\n" + "="*70)
    info("DONE!")
    info("="*70)
    info(f"Output directory: {OUT_DIR}")
    info(f"Total nodes: {len(node_data)}")
    info(f"Total links: {len(link_data)}")
    if sizes:
        info(f"Largest connected component: {sizes[0]} nodes ({sizes[0]/len(node_data)*100:.1f}%)")


if __name__ == "__main__":
    main()
