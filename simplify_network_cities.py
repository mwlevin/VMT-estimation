"""
Network Simplifier - Collapse Urban Areas to Super Nodes
=========================================================
This script simplifies a large road network by:
1. Identifying nodes within major city boundaries
2. Removing all internal city links and nodes
3. Creating a single "super node" for each city
4. Reconnecting border links to the super node

This dramatically reduces network complexity while preserving
inter-city connectivity for statewide traffic assignment.
"""

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import unary_union
from collections import defaultdict
import folium

warnings.filterwarnings('ignore')

# ==========================
# USER PATHS / CONSTANTS
# ==========================

# Input network files (from CSVShapefileWithIntersections.py output)
INPUT_DIR = Path("/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/NetworkFiles2024_Connected_HPMS_Split")
OUTPUT_DIR = Path("/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/NetworkFiles2024_Simplified_HPMS_Split")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# CITY DEFINITIONS
# ==========================
# Define city boundaries as approximate bounding boxes or polygons
# Format: (min_lon, min_lat, max_lon, max_lat) in WGS84
# You can also use shapefiles for more precise boundaries

# Twin Cities Metro includes: Minneapolis, Saint Paul, Bloomington, Edina,
# Plymouth, Maple Grove, Brooklyn Park, Eagan, Burnsville, Eden Prairie,
# Minnetonka, St. Louis Park, Richfield, Hopkins, Golden Valley, etc.

CITY_BOUNDARIES = {
    'TwinCities_Metro': {
        # Metro area boundary covering 7-county metro
        'bbox': (-93.75, 44.70, -92.80, 45.25),
        'super_node_location': (-93.17, 44.96),    # Central location (between downtowns)
    },
    'Rochester': {
        'bbox': (-92.60, 43.90, -92.35, 44.12),
        'super_node_location': (-92.466, 44.023),    # Downtown Rochester
    },
    'Duluth': {
        'bbox': (-92.25, 46.65, -91.95, 46.88),
        'super_node_location': (-92.106, 46.786),    # Downtown Duluth
    },
    'St_Cloud': {
        'bbox': (-94.25, 45.50, -94.05, 45.62),
        'super_node_location': (-94.16, 45.56),
    },
    'Mankato': {
        'bbox': (-94.05, 44.12, -93.90, 44.22),
        'super_node_location': (-93.99, 44.17),
    },
    'Moorhead': {
        'bbox': (-96.80, 46.82, -96.68, 46.92),
        'super_node_location': (-96.74, 46.87),
    },
}

TOPOLOGY_SNAP_DECIMALS = 7  # WGS84 decimals; 7 ~= 1 cm latitude, enough for exact/coincident duplicates.


def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collapse major urban areas into super nodes."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Connected network input folder")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Simplified network output folder")
    return parser.parse_args()


# ==========================
# LOAD NETWORK
# ==========================

def load_network_from_txt(input_dir):
    """Load network from nodes.txt and network.txt files."""
    info("Loading network from text files...")
    
    # Load nodes
    nodes_file = input_dir / "nodes.txt"
    nodes = []
    with open(nodes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('~') or not line:
                continue
            parts = line.replace('\t', ' ').split()
            # Remove semicolons
            parts = [p.replace(';', '') for p in parts if p != ';']
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    x = float(parts[1])  # longitude
                    y = float(parts[2])  # latitude
                    nodes.append({'node_id': node_id, 'x': x, 'y': y})
                except ValueError:
                    continue
    
    nodes_df = pd.DataFrame(nodes)
    info(f"  Loaded {len(nodes_df)} nodes")
    
    # Load links
    network_file = input_dir / "network.txt"
    links = []
    in_data = False
    
    with open(network_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '<END OF METADATA>' in line:
                in_data = True
                continue
            if not in_data:
                continue
            if line.startswith('~') or not line:
                continue
            
            parts = line.replace('\t', ' ').split()
            parts = [p.replace(';', '') for p in parts if p != ';']
            
            if len(parts) >= 10:
                try:
                    links.append({
                        'init_node': int(parts[0]),
                        'term_node': int(parts[1]),
                        'capacity': float(parts[2]),
                        'length': float(parts[3]),
                        'free_flow_time': float(parts[4]),
                        'b': float(parts[5]),
                        'power': float(parts[6]),
                        'speed': float(parts[7]),
                        'toll': float(parts[8]),
                        'link_type': int(parts[9]) if len(parts) > 9 else 1
                    })
                except ValueError:
                    continue
    
    links_df = pd.DataFrame(links)
    info(f"  Loaded {len(links_df)} links")
    
    return nodes_df, links_df


def load_network_from_shapefiles(input_dir):
    """Load network from shapefiles."""
    info("Loading network from shapefiles...")
    
    nodes_gdf = gpd.read_file(input_dir / "nodes.shp")
    links_gdf = gpd.read_file(input_dir / "links.shp")
    
    # Convert to DataFrames with coordinates
    nodes_df = pd.DataFrame({
        'node_id': nodes_gdf['node_id'],
        'x': nodes_gdf.geometry.x,
        'y': nodes_gdf.geometry.y
    })
    
    links_df = pd.DataFrame({
        'init_node': links_gdf['init_node'],
        'term_node': links_gdf['term_node'],
        'capacity': links_gdf['capacity'],
        'length': links_gdf['length'],
        'free_flow_time': links_gdf['free_flow_t'],
        'ffs': links_gdf['ffs'],
        'b': 0.15,
        'power': 4,
        'speed': links_gdf['ffs'],
        'toll': 0,
        'link_type': 1
    })
    
    info(f"  Loaded {len(nodes_df)} nodes")
    info(f"  Loaded {len(links_df)} links")
    
    return nodes_df, links_df


# ==========================
# IDENTIFY NODES IN CITIES
# ==========================

def create_city_polygons(city_boundaries):
    """Create Shapely polygons from city boundary definitions."""
    city_polygons = {}
    
    for city_name, city_info in city_boundaries.items():
        if 'bbox' in city_info:
            min_lon, min_lat, max_lon, max_lat = city_info['bbox']
            polygon = box(min_lon, min_lat, max_lon, max_lat)
        elif 'polygon' in city_info:
            polygon = Polygon(city_info['polygon'])
        else:
            warn(f"No boundary defined for {city_name}")
            continue
        
        city_polygons[city_name] = {
            'polygon': polygon,
            'super_node_location': city_info['super_node_location']
        }
    
    return city_polygons


def identify_nodes_in_cities(nodes_df, city_polygons):
    """
    Identify which nodes fall within each city boundary.
    Returns dict: city_name -> set of node_ids
    """
    info("Identifying nodes within city boundaries...")
    
    # Create GeoDataFrame of nodes
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df['x'], nodes_df['y']),
        crs='EPSG:4326'
    )
    
    city_nodes = {}
    
    for city_name, city_info in city_polygons.items():
        polygon = city_info['polygon']
        
        # Find nodes within this city
        within_mask = nodes_gdf.geometry.within(polygon)
        nodes_in_city = set(nodes_gdf.loc[within_mask, 'node_id'].tolist())
        
        city_nodes[city_name] = nodes_in_city
        info(f"  {city_name}: {len(nodes_in_city)} nodes inside boundary")
    
    return city_nodes


# ==========================
# SIMPLIFY NETWORK
# ==========================

def simplify_network(nodes_df, links_df, city_nodes, city_polygons):
    """
    Simplify the network by collapsing cities to super nodes.
    
    For each city:
    1. Remove all internal links (both endpoints in city)
    2. Create a super node at city center
    3. Redirect border links to the super node
    4. Aggregate capacity/attributes for parallel links
    """
    info("Simplifying network...")
    
    # Create set of all city nodes
    all_city_nodes = set()
    for nodes in city_nodes.values():
        all_city_nodes.update(nodes)
    
    info(f"  Total nodes in cities: {len(all_city_nodes)}")
    
    # Create mapping: old_node_id -> new_node_id (super node)
    node_to_supernode = {}
    super_nodes = {}  # city_name -> super_node_id
    
    # Start super node IDs after the max existing node ID
    max_node_id = nodes_df['node_id'].max()
    super_node_id = max_node_id + 1
    
    # Store super node locations for distance calculation
    super_node_locations = {}  # super_node_id -> (x, y)
    
    for city_name, city_node_set in city_nodes.items():
        for node_id in city_node_set:
            node_to_supernode[node_id] = super_node_id
        super_nodes[city_name] = super_node_id
        # Store the super node location
        super_node_locations[super_node_id] = city_polygons[city_name]['super_node_location']
        info(f"  {city_name} -> Super Node {super_node_id}")
        super_node_id += 1
    
    # Create node coordinate lookup for distance calculation
    node_coords = nodes_df.set_index('node_id')[['x', 'y']].to_dict('index')
    
    # Classify links
    internal_links = []  # Both endpoints in same city - REMOVE
    border_links = []    # One endpoint in city, one outside - REDIRECT
    external_links = []  # Both endpoints outside all cities - KEEP
    
    for idx, link in links_df.iterrows():
        init_node = link['init_node']
        term_node = link['term_node']
        
        init_in_city = init_node in all_city_nodes
        term_in_city = term_node in all_city_nodes
        
        if init_in_city and term_in_city:
            # Check if same city (internal) or different cities (inter-city link)
            init_city = None
            term_city = None
            for city_name, city_node_set in city_nodes.items():
                if init_node in city_node_set:
                    init_city = city_name
                if term_node in city_node_set:
                    term_city = city_name
            
            if init_city == term_city:
                internal_links.append(idx)
            else:
                # Link between two different cities - this is important!
                border_links.append(idx)
        elif init_in_city or term_in_city:
            border_links.append(idx)
        else:
            external_links.append(idx)
    
    info(f"  Internal links (removed): {len(internal_links)}")
    info(f"  Border links (redirected): {len(border_links)}")
    info(f"  External links (kept): {len(external_links)}")
    
    # Build new links DataFrame
    new_links = []
    
    # Keep external links as-is
    for idx in external_links:
        new_links.append(links_df.loc[idx].to_dict())
    
    # Process border links - redirect city endpoints to super nodes
    # and calculate actual distance to the super node
    for idx in border_links:
        link = links_df.loc[idx].to_dict()
        
        orig_init_node = link['init_node']
        orig_term_node = link['term_node']
        
        # Redirect to super node if in city
        if orig_init_node in node_to_supernode:
            link['init_node'] = node_to_supernode[orig_init_node]
        if orig_term_node in node_to_supernode:
            link['term_node'] = node_to_supernode[orig_term_node]
        
        # Skip self-loops (both endpoints became same super node)
        if link['init_node'] == link['term_node']:
            continue
        
        new_links.append(link)
    
    new_links_df = pd.DataFrame(new_links)
    
    # Add super node coordinates to node_coords for distance calculation
    for sn_id, (x, y) in super_node_locations.items():
        node_coords[sn_id] = {'x': x, 'y': y}
    
    # Aggregate parallel links (same init_node and term_node)
    # with ACTUAL distance calculation
    info("  Aggregating parallel links with actual distance calculation...")
    
    if len(new_links_df) > 0:
        # Group by init_node and term_node
        grouped = new_links_df.groupby(['init_node', 'term_node']).agg({
            'capacity': 'sum',           # Sum capacities (parallel paths)
            'length': 'first',           # Will recalculate below
            'free_flow_time': 'first',   # Will recalculate below
            'b': 'first',
            'power': 'first',
            'speed': 'max',              # Use highest speed
            'toll': 'sum',
            'link_type': 'first'
        }).reset_index()
        
        # Recalculate ACTUAL length and free flow time for each aggregated link
        def calculate_actual_distance(row):
            """Calculate actual distance between two nodes using Haversine formula."""
            init_id = int(row['init_node'])
            term_id = int(row['term_node'])
            
            if init_id not in node_coords or term_id not in node_coords:
                return row['length'], row['free_flow_time']
            
            # Get coordinates (lon, lat in WGS84)
            x1, y1 = node_coords[init_id]['x'], node_coords[init_id]['y']
            x2, y2 = node_coords[term_id]['x'], node_coords[term_id]['y']
            
            # Haversine formula for distance in miles
            from math import radians, sin, cos, sqrt, atan2
            R = 3959  # Earth's radius in miles
            
            lat1, lat2 = radians(y1), radians(y2)
            lon1, lon2 = radians(x1), radians(x2)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            distance_miles = R * c
            
            # Calculate free flow time (minutes) = distance / speed * 60
            speed = row['speed'] if row['speed'] > 0 else 30  # Default 30 mph
            free_flow_time = (distance_miles / speed) * 60
            
            return distance_miles, free_flow_time
        
        # Apply actual distance calculation
        results = grouped.apply(calculate_actual_distance, axis=1)
        grouped['length'] = [r[0] for r in results]
        grouped['free_flow_time'] = [r[1] for r in results]
        
        new_links_df = grouped
    
    info(f"  Links after aggregation: {len(new_links_df)}")
    
    # Build new nodes DataFrame
    # Keep nodes that are not in any city
    external_nodes = nodes_df[~nodes_df['node_id'].isin(all_city_nodes)].copy()
    
    # Add super nodes
    super_node_rows = []
    for city_name, super_node_id in super_nodes.items():
        location = city_polygons[city_name]['super_node_location']
        super_node_rows.append({
            'node_id': super_node_id,
            'x': location[0],
            'y': location[1],
            'city_name': city_name
        })
    
    super_nodes_df = pd.DataFrame(super_node_rows)
    
    # Combine
    new_nodes_df = pd.concat([external_nodes, super_nodes_df], ignore_index=True)
    
    info(f"  New nodes: {len(new_nodes_df)} (was {len(nodes_df)})")
    info(f"  New links: {len(new_links_df)} (was {len(links_df)})")
    
    # Only keep nodes that are actually used in links
    used_nodes = set(new_links_df['init_node'].unique()) | set(new_links_df['term_node'].unique())
    new_nodes_df = new_nodes_df[new_nodes_df['node_id'].isin(used_nodes)].copy()
    
    info(f"  Nodes after cleanup: {len(new_nodes_df)}")

    new_nodes_df, new_links_df, merged_count = merge_coincident_nodes(
        new_nodes_df, new_links_df, super_nodes, decimals=TOPOLOGY_SNAP_DECIMALS
    )
    if merged_count:
        info(f"  Topology repair merged {merged_count} coincident nodes")
        info(f"  Nodes after topology repair: {len(new_nodes_df)}")
        info(f"  Links after topology repair: {len(new_links_df)}")
    
    return new_nodes_df, new_links_df, super_nodes


def merge_coincident_nodes(nodes_df, links_df, super_nodes, decimals=7):
    """
    Merge nodes that occupy the same rounded coordinate and rewrite links.

    HPMS splitting can leave separate node IDs at the same physical point.
    On the map they touch, but graph algorithms see different nodes. This
    repair preserves super-nodes and collapses coincident non-super nodes into
    a single representative before the final sequential IDs are written.
    """
    if nodes_df.empty or links_df.empty:
        return nodes_df, links_df, 0

    super_node_ids = set(int(v) for v in super_nodes.values())
    nodes = nodes_df.copy()
    nodes['node_id'] = nodes['node_id'].astype(int)
    nodes['_snap_x'] = nodes['x'].round(decimals)
    nodes['_snap_y'] = nodes['y'].round(decimals)

    old_to_rep = {}
    merged_count = 0

    for _, group in nodes.groupby(['_snap_x', '_snap_y'], sort=False):
        ids = [int(v) for v in group['node_id'].tolist()]
        if len(ids) == 1:
            old_to_rep[ids[0]] = ids[0]
            continue

        super_ids = [nid for nid in ids if nid in super_node_ids]
        rep = super_ids[0] if super_ids else min(ids)
        for nid in ids:
            old_to_rep[nid] = rep
        merged_count += len(ids) - 1

    if merged_count == 0:
        return nodes_df, links_df, 0

    repaired_links = links_df.copy()
    repaired_links['init_node'] = repaired_links['init_node'].astype(int).map(old_to_rep)
    repaired_links['term_node'] = repaired_links['term_node'].astype(int).map(old_to_rep)
    repaired_links = repaired_links.dropna(subset=['init_node', 'term_node']).copy()
    repaired_links['init_node'] = repaired_links['init_node'].astype(int)
    repaired_links['term_node'] = repaired_links['term_node'].astype(int)
    repaired_links = repaired_links[repaired_links['init_node'] != repaired_links['term_node']].copy()

    if not repaired_links.empty:
        repaired_links = repaired_links.groupby(['init_node', 'term_node'], as_index=False).agg({
            'capacity': 'sum',
            'length': 'min',
            'free_flow_time': 'min',
            'b': 'first',
            'power': 'first',
            'speed': 'max',
            'toll': 'sum',
            'link_type': 'first',
        })

    used_nodes = set(repaired_links['init_node'].unique()) | set(repaired_links['term_node'].unique())
    reps = set(old_to_rep.values())
    repaired_nodes = nodes[nodes['node_id'].isin(reps & used_nodes)].copy()
    repaired_nodes = repaired_nodes.drop(columns=['_snap_x', '_snap_y'])

    return repaired_nodes, repaired_links, merged_count


# ==========================
# OUTPUT FILES
# ==========================

def create_sequential_node_mapping(nodes_df, super_nodes):
    """
    Create a mapping from original node IDs to sequential IDs (1, 2, 3, ..., N).
    Super nodes are placed first to make them easy to identify.
    
    Returns:
    - old_to_new: dict mapping original node_id to new sequential id
    - new_super_nodes: dict mapping city_name to new sequential node_id
    """
    old_to_new = {}
    new_super_nodes = {}
    
    # First assign IDs to super nodes (1, 2, ..., num_super_nodes)
    new_id = 1
    for city_name, old_node_id in super_nodes.items():
        old_to_new[old_node_id] = new_id
        new_super_nodes[city_name] = new_id
        new_id += 1
    
    # Then assign IDs to remaining nodes
    super_node_ids = set(super_nodes.values())
    for _, row in nodes_df.iterrows():
        old_id = row['node_id']
        if old_id not in super_node_ids:
            old_to_new[old_id] = new_id
            new_id += 1
    
    info(f"  Created node ID mapping: {len(old_to_new)} nodes -> 1 to {new_id - 1}")
    return old_to_new, new_super_nodes


def write_output_files(nodes_df, links_df, super_nodes, output_dir):
    """Write simplified network files with sequential node IDs."""
    info("Writing output files...")
    
    # Create sequential node ID mapping
    old_to_new, new_super_nodes = create_sequential_node_mapping(nodes_df, super_nodes)
    
    num_nodes = len(nodes_df)
    num_links = len(links_df)
    num_super_nodes = len(super_nodes)
    
    # Write nodes.txt with NEW sequential IDs
    with open(output_dir / "nodes.txt", 'w') as f:
        f.write("~ node_id\tx\ty\t;\n")
        
        # Write super nodes first
        for city_name, old_node_id in super_nodes.items():
            row = nodes_df[nodes_df['node_id'] == old_node_id].iloc[0]
            new_id = old_to_new[old_node_id]
            f.write(f"\t{new_id}\t{row['x']:.8f}\t{row['y']:.8f}\t;\n")
        
        # Write remaining nodes
        super_node_ids = set(super_nodes.values())
        for _, row in nodes_df.iterrows():
            if row['node_id'] not in super_node_ids:
                new_id = old_to_new[row['node_id']]
                f.write(f"\t{new_id}\t{row['x']:.8f}\t{row['y']:.8f}\t;\n")
    
    # Write network.txt with NEW sequential node IDs
    with open(output_dir / "network.txt", 'w') as f:
        f.write(f"<NUMBER OF ZONES> {num_super_nodes}\n")  # Super nodes are zones
        f.write(f"<NUMBER OF NODES> {num_nodes}\n")
        f.write(f"<FIRST THRU NODE> {num_super_nodes + 1}\n")  # First non-zone node
        f.write(f"<NUMBER OF LINKS> {num_links}\n")
        f.write("<ORIGINAL HEADER>~ \tInit node \tTerm node \tCapacity \tLength "
                "\tFree Flow Time \tB\tPower\tSpeed limit \tToll \tType\t;\n")
        f.write("<END OF METADATA>\n\n")
        f.write("~\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower"
                "\tspeed\ttoll\tlink_type\t;\n")
        
        for _, link in links_df.iterrows():
            # Map to new sequential IDs (convert to int for dict lookup)
            init_old = int(link['init_node'])
            term_old = int(link['term_node'])
            if init_old not in old_to_new:
                print(f"WARNING: Node {init_old} not in mapping!")
                continue
            if term_old not in old_to_new:
                print(f"WARNING: Node {term_old} not in mapping!")
                continue
            new_init = old_to_new[init_old]
            new_term = old_to_new[term_old]
            f.write(f"\t{new_init}\t{new_term}\t{link['capacity']:.0f}"
                   f"\t{link['length']:.5f}\t{link['free_flow_time']:.5f}"
                   f"\t{link['b']}\t{int(link['power'])}\t{link['speed']:.2f}"
                   f"\t{link['toll']}\t{int(link['link_type'])}\t;\n")
    
    # Write super nodes info with NEW sequential IDs
    with open(output_dir / "super_nodes.txt", 'w') as f:
        f.write("city_name\tsuper_node_id\tx\ty\n")
        for city_name, old_node_id in super_nodes.items():
            row = nodes_df[nodes_df['node_id'] == old_node_id].iloc[0]
            new_id = new_super_nodes[city_name]
            f.write(f"{city_name}\t{new_id}\t{row['x']:.8f}\t{row['y']:.8f}\n")
    
    # Write node ID mapping for reference
    with open(output_dir / "node_id_mapping.csv", 'w') as f:
        f.write("new_id,original_id,x,y\n")
        for old_id, new_id in sorted(old_to_new.items(), key=lambda x: x[1]):
            row = nodes_df[nodes_df['node_id'] == old_id].iloc[0]
            f.write(f"{new_id},{old_id},{row['x']:.8f},{row['y']:.8f}\n")
    
    info(f"  Written to {output_dir}")
    info(f"  Super nodes: {new_super_nodes}")


def create_visualization(nodes_df, links_df, super_nodes, city_polygons, output_path, region='twin_cities'):
    """
    Create visualization showing simplified network.
    
    region: 'twin_cities' to show only Twin Cities area (fast), 
            'all' to show entire state (slow/large file)
    """
    info("Creating visualization...")
    
    # Define viewing regions
    regions = {
        'twin_cities': {
            'center': (44.96, -93.27),  # Minneapolis-St Paul
            'zoom': 9,
            'bbox': (-94.0, 44.5, -92.5, 45.4)  # min_lon, min_lat, max_lon, max_lat
        },
        'rochester': {
            'center': (44.02, -92.47),
            'zoom': 11,
            'bbox': (-92.8, 43.8, -92.2, 44.3)
        },
        'duluth': {
            'center': (46.79, -92.10),
            'zoom': 11,
            'bbox': (-92.4, 46.6, -91.8, 47.0)
        },
        'all': {
            'center': (46.0, -94.5),
            'zoom': 6,
            'bbox': (-97.5, 43.0, -89.0, 49.5)
        }
    }
    
    reg = regions.get(region, regions['twin_cities'])
    center_lat, center_lon = reg['center']
    min_lon, min_lat, max_lon, max_lat = reg['bbox']
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=reg['zoom'])
    
    # Draw city boundaries
    for city_name, city_info in city_polygons.items():
        polygon = city_info['polygon']
        coords = list(polygon.exterior.coords)
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color='red',
            fill=True,
            fill_opacity=0.2,
            popup=f"{city_name} (collapsed)"
        ).add_to(m)
    
    # Filter nodes to region
    node_coords = nodes_df.set_index('node_id')[['x', 'y']].to_dict('index')
    
    # Filter links to those within the region bbox
    links_in_region = []
    super_node_set = set(super_nodes.values())
    
    for _, link in links_df.iterrows():
        init_node = int(link['init_node'])
        term_node = int(link['term_node'])
        
        if init_node not in node_coords or term_node not in node_coords:
            continue
        
        start = node_coords[init_node]
        end = node_coords[term_node]
        
        # Check if either endpoint is in the region
        start_in = min_lon <= start['x'] <= max_lon and min_lat <= start['y'] <= max_lat
        end_in = min_lon <= end['x'] <= max_lon and min_lat <= end['y'] <= max_lat
        
        # Also include links connected to super nodes in the region
        is_super_link = init_node in super_node_set or term_node in super_node_set
        
        if start_in or end_in or is_super_link:
            links_in_region.append((init_node, term_node, start, end, is_super_link))
    
    info(f"  Drawing {len(links_in_region)} links in {region} region...")
    
    # Draw links
    for init_node, term_node, start, end, is_super_link in links_in_region:
        color = 'blue' if is_super_link else 'gray'
        weight = 3 if is_super_link else 1
        
        folium.PolyLine(
            locations=[(start['y'], start['x']), (end['y'], end['x'])],
            color=color,
            weight=weight,
            opacity=0.7
        ).add_to(m)
    
    # Draw super nodes
    for city_name, node_id in super_nodes.items():
        if node_id in node_coords:
            coord = node_coords[node_id]
            folium.CircleMarker(
                location=[coord['y'], coord['x']],
                radius=15,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.8,
                popup=f"{city_name} (Super Node {node_id})"
            ).add_to(m)
    
    # Try to save, handle permission errors (common with iCloud)
    try:
        m.save(str(output_path))
        info(f"  Saved visualization: {output_path}")
    except PermissionError:
        # Try alternative filename with timestamp
        import time
        alt_path = output_path.parent / f"simplified_network_{int(time.time())}.html"
        try:
            m.save(str(alt_path))
            info(f"  Saved visualization (alt): {alt_path}")
        except PermissionError:
            warn(f"  Could not save visualization due to permission error. Skipping.")


def check_connectivity(links_df):
    """Check network connectivity after simplification."""
    from collections import deque
    
    adj = defaultdict(set)
    for _, link in links_df.iterrows():
        n1 = int(link['init_node'])
        n2 = int(link['term_node'])
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
    
    info(f"\nConnectivity check (simplified network):")
    info(f"  Total nodes: {len(adj)}")
    info(f"  Connected components: {len(components)}")
    if sizes:
        info(f"  Largest component: {sizes[0]} nodes ({sizes[0]/len(adj)*100:.1f}%)")
        if len(sizes) > 1:
            info(f"  Second largest: {sizes[1]} nodes")
    
    return len(components)


# ==========================
# MAIN
# ==========================

def main():
    global INPUT_DIR, OUTPUT_DIR
    args = parse_args()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    info("=" * 70)
    info("Network Simplifier - Collapse Urban Areas to Super Nodes")
    info("=" * 70)
    
    # Check input files exist
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    
    # Try to load network from text files first, then shapefiles
    if (INPUT_DIR / "nodes.txt").exists() and (INPUT_DIR / "network.txt").exists():
        nodes_df, links_df = load_network_from_txt(INPUT_DIR)
    elif (INPUT_DIR / "nodes.shp").exists() and (INPUT_DIR / "links.shp").exists():
        nodes_df, links_df = load_network_from_shapefiles(INPUT_DIR)
    else:
        raise FileNotFoundError(f"No network files found in {INPUT_DIR}")
    
    # Create city polygons
    info("\n" + "=" * 70)
    info("STEP 1: Define city boundaries")
    info("=" * 70)
    city_polygons = create_city_polygons(CITY_BOUNDARIES)
    
    for city_name in city_polygons:
        info(f"  {city_name}: boundary defined")
    
    # Identify nodes in cities
    info("\n" + "=" * 70)
    info("STEP 2: Identify nodes within cities")
    info("=" * 70)
    city_nodes = identify_nodes_in_cities(nodes_df, city_polygons)
    
    # Simplify network
    info("\n" + "=" * 70)
    info("STEP 3: Simplify network")
    info("=" * 70)
    new_nodes_df, new_links_df, super_nodes = simplify_network(
        nodes_df, links_df, city_nodes, city_polygons
    )
    
    # Check connectivity
    num_components = check_connectivity(new_links_df)
    
    # Write output
    info("\n" + "=" * 70)
    info("STEP 4: Write output files")
    info("=" * 70)
    write_output_files(new_nodes_df, new_links_df, super_nodes, OUTPUT_DIR)
    
    # Create visualization (only Twin Cities area for speed)
    create_visualization(
        new_nodes_df, new_links_df, super_nodes, city_polygons,
        OUTPUT_DIR / "simplified_network_twincities.html",
        region='twin_cities'  # Only show Twin Cities area for faster loading
    )
    
    # Summary
    info("\n" + "=" * 70)
    info("SUMMARY")
    info("=" * 70)
    info(f"Original network: {len(nodes_df)} nodes, {len(links_df)} links")
    info(f"Simplified network: {len(new_nodes_df)} nodes, {len(new_links_df)} links")
    info(f"Reduction: {(1 - len(new_nodes_df)/len(nodes_df))*100:.1f}% nodes, "
         f"{(1 - len(new_links_df)/len(links_df))*100:.1f}% links")
    info(f"Super nodes created: {len(super_nodes)}")
    for city_name, node_id in super_nodes.items():
        info(f"  - {city_name}: Node {node_id}")
    info(f"\nOutput directory: {OUTPUT_DIR}")
    
    if num_components > 1:
        warn(f"\nNetwork has {num_components} disconnected components!")
        warn("Consider adjusting city boundaries or checking input data.")


if __name__ == "__main__":
    main()
