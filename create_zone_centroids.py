"""
Create Zone Centroids for Minnesota Network
============================================
This script:
1. Reads the zone polygons (from streetlight_zones)
2. Reads the network nodes
3. For each zone, finds a node inside the zone (or closest to centroid)
4. Reorders nodes so zone centroid nodes come first (1-1000)
5. Creates centroid connectors from each zone to nearby network nodes
6. Outputs new network files in TNTP format compatible with traffic assignment

Output:
- nodes_with_zones.txt: Node file with zone centroids first
- network_with_connectors.txt: Network file with centroid connectors
- zone_node_mapping.csv: Mapping of zone_id to original_node_id
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
import os
import argparse

# Configuration
ZONES_SHAPEFILE = "streetlight_zones 2/zones_for_streetlight.shp"
NODES_FILE = "NetworkFiles2024_Simplified_HPMS_Split/nodes.txt"
NETWORK_FILE = "NetworkFiles2024_Simplified_HPMS_Split/network.txt"
SUPER_NODES_FILE = "NetworkFiles2024_Simplified_HPMS_Split/super_nodes.txt"  # From simplify script
OUTPUT_FOLDER = "NetworkFiles2024_WithZones_Simplified_HPMS_Split"

# Centroid connector parameters
CONNECTOR_CAPACITY = 99999  # Very high capacity (no congestion)
CONNECTOR_ALPHA = 0.0       # No BPR congestion
CONNECTOR_BETA = 1.0
CONNECTOR_SPEED = 30        # mph for travel time calculation
NUM_CONNECTORS_PER_ZONE = 3  # Number of network nodes to connect each zone to
LONG_CONNECTOR_WARNING_MILES = 5.0  # Print diagnostics for unexpectedly long connectors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach StreetLight zone centroids/connectors to a simplified network."
    )
    parser.add_argument("--zones", default=ZONES_SHAPEFILE, help="Input zone shapefile")
    parser.add_argument("--nodes", default=NODES_FILE, help="Simplified nodes.txt")
    parser.add_argument("--network", default=NETWORK_FILE, help="Simplified network.txt")
    parser.add_argument("--super-nodes", default=SUPER_NODES_FILE, help="Simplified super_nodes.txt")
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER, help="Output with-zones folder")
    parser.add_argument("--num-connectors", type=int, default=NUM_CONNECTORS_PER_ZONE,
                        help="Maximum number of connector target nodes per zone.")
    parser.add_argument("--max-connector-miles", type=float, default=None,
                        help="Reject connector candidates longer than this distance.")
    parser.add_argument("--no-scc-fallback", action="store_true",
                        help="Do not add a far fallback connector to the main directed SCC.")
    parser.add_argument(
        "--zone-node-location",
        choices=["representative-node", "polygon-centroid"],
        default="representative-node",
        help="Coordinates to write for virtual zone nodes.",
    )
    return parser.parse_args()

# Collapsed city boundaries (must match simplify_network_cities.py)
# Format: (min_lon, min_lat, max_lon, max_lat) in WGS84
COLLAPSED_CITIES = {
    'TwinCities_Metro': (-93.75, 44.70, -92.80, 45.25),
    'Rochester': (-92.60, 43.90, -92.35, 44.12),
    'Duluth': (-92.25, 46.65, -91.95, 46.88),
    'St_Cloud': (-94.25, 45.50, -94.05, 45.62),
    'Mankato': (-94.05, 44.12, -93.90, 44.22),
    'Moorhead': (-96.80, 46.82, -96.68, 46.92),
}


def read_nodes(nodes_file):
    """Read nodes from the network nodes file."""
    nodes = []
    with open(nodes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('~') or not line:
                continue
            parts = line.replace(';', '').split()
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    nodes.append({'node_id': node_id, 'x': x, 'y': y})
                except ValueError:
                    continue
    return pd.DataFrame(nodes)


def read_network(network_file):
    """Read links from the network file."""
    links = []
    metadata = {}
    
    with open(network_file, 'r') as f:
        in_data = False
        for line in f:
            line_stripped = line.strip()
            
            # Parse metadata
            if '<NUMBER OF ZONES>' in line:
                metadata['num_zones'] = int(line[line.index('>') + 1:].strip())
            elif '<NUMBER OF NODES>' in line:
                metadata['num_nodes'] = int(line[line.index('>') + 1:].strip())
            elif '<NUMBER OF LINKS>' in line:
                metadata['num_links'] = int(line[line.index('>') + 1:].strip())
            elif '<FIRST THRU NODE>' in line:
                metadata['first_thru_node'] = int(line[line.index('>') + 1:].strip())
            
            if line_stripped == '<END OF METADATA>':
                in_data = True
                continue
            
            if in_data and line_stripped and not line_stripped.startswith('~'):
                parts = line_stripped.replace(';', '').split()
                if len(parts) >= 8:
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
                            'toll': float(parts[8]) if len(parts) > 8 else 0,
                            'link_type': int(parts[9]) if len(parts) > 9 else 1
                        })
                    except (ValueError, IndexError):
                        continue
    
    return pd.DataFrame(links), metadata


def read_super_nodes(super_nodes_file):
    """Read super nodes mapping from the simplification script output."""
    super_nodes = {}  # city_name -> {'node_id': id, 'x': x, 'y': y}
    
    if not os.path.exists(super_nodes_file):
        print(f"Warning: Super nodes file not found: {super_nodes_file}")
        return super_nodes
    
    with open(super_nodes_file, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                city_name = parts[0]
                node_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                super_nodes[city_name] = {'node_id': node_id, 'x': x, 'y': y}
    
    print(f"Loaded {len(super_nodes)} super nodes")
    return super_nodes


def get_collapsed_city_polygons():
    """Create Shapely polygons from collapsed city boundaries."""
    city_polygons = {}
    for city_name, bbox in COLLAPSED_CITIES.items():
        min_lon, min_lat, max_lon, max_lat = bbox
        city_polygons[city_name] = box(min_lon, min_lat, max_lon, max_lat)
    return city_polygons


def find_zone_centroids(zones_gdf, nodes_df, super_nodes=None):
    """
    For each zone, find a node inside it (or closest to its centroid).
    If a zone falls within a collapsed city boundary, assigns it the city's super node.
    Returns a mapping of zone_id -> node_id
    """
    import warnings
    warnings.filterwarnings('ignore', message='.*geographic CRS.*')
    
    # Get collapsed city polygons
    city_polygons = get_collapsed_city_polygons()
    
    # Convert nodes to GeoDataFrame
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=[Point(x, y) for x, y in zip(nodes_df['x'], nodes_df['y'])],
        crs="EPSG:4326"
    )
    
    # Project to UTM zone 15N (Minnesota) for accurate distance calculations
    nodes_gdf_proj = nodes_gdf.to_crs(epsg=26915)
    
    # Ensure zones are in same CRS
    if zones_gdf.crs != nodes_gdf.crs:
        zones_gdf = zones_gdf.to_crs(nodes_gdf.crs)
    zones_gdf_proj = zones_gdf.to_crs(epsg=26915)
    
    zone_to_node = {}
    used_nodes = set()
    zones_in_cities = {}  # Track which zones are inside collapsed cities
    
    print(f"Finding centroid nodes for {len(zones_gdf)} zones...")
    
    for idx, zone in zones_gdf.iterrows():
        # Convert 0-indexed zone_id to 1-indexed (TNTP format requires zones 1-N)
        zone_id_from_file = zone.get('zone_id', idx)
        zone_id = int(zone_id_from_file) + 1  # Convert to 1-indexed
        zone_geom = zone.geometry
        zone_centroid = zone_geom.centroid
        
        # CHECK 1: Is this zone inside a collapsed city?
        assigned_to_super_node = False
        if super_nodes:
            for city_name, city_polygon in city_polygons.items():
                # Check if zone centroid is within the collapsed city boundary
                if zone_centroid.within(city_polygon):
                    # Assign to city's super node
                    if city_name in super_nodes:
                        centroid_node_id = super_nodes[city_name]['node_id']
                        zone_to_node[zone_id] = centroid_node_id
                        if city_name not in zones_in_cities:
                            zones_in_cities[city_name] = []
                        zones_in_cities[city_name].append(zone_id)
                        assigned_to_super_node = True
                        break
        
        if assigned_to_super_node:
            continue  # Move to next zone
        
        # CHECK 2: Find a node inside the zone (normal case)
        # Find nodes inside this zone
        nodes_in_zone = nodes_gdf[nodes_gdf.geometry.within(zone_geom)]
        
        # Filter out already used nodes
        available_nodes = nodes_in_zone[~nodes_in_zone['node_id'].isin(used_nodes)]
        
        if len(available_nodes) > 0:
            # Find the node closest to the zone centroid among available nodes
            distances = available_nodes.geometry.distance(zone_centroid)
            closest_idx = distances.idxmin()
            centroid_node_id = available_nodes.loc[closest_idx, 'node_id']
        else:
            # No nodes inside zone, find closest node to centroid
            available_nodes_all = nodes_gdf[~nodes_gdf['node_id'].isin(used_nodes)]
            if len(available_nodes_all) == 0:
                print(f"Warning: No available nodes for zone {zone_id}")
                continue
            distances = available_nodes_all.geometry.distance(zone_centroid)
            closest_idx = distances.idxmin()
            centroid_node_id = available_nodes_all.loc[closest_idx, 'node_id']
        
        zone_to_node[zone_id] = centroid_node_id
        used_nodes.add(centroid_node_id)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(zones_gdf)} zones")
    
    # Print summary of zones assigned to collapsed cities
    if zones_in_cities:
        print(f"\nZones assigned to collapsed city super nodes:")
        for city_name, city_zones in zones_in_cities.items():
            print(f"  {city_name}: {len(city_zones)} zones -> Super Node {super_nodes[city_name]['node_id']}")
    
    print(f"Found centroid nodes for {len(zone_to_node)} zones")
    return zone_to_node, zones_in_cities


def find_connector_nodes(
    zones_gdf,
    nodes_df,
    zone_to_node,
    links_df,
    super_nodes=None,
    zones_in_cities=None,
    num_connectors=3,
    max_connector_miles=None,
    require_scc_fallback=True,
):
    """
    For each zone, find the nearest local thru-nodes to connect to.
    
    CRITICAL: TAPAS uses FIRST_THRU_NODE — Dijkstra only expands through
    nodes >= first_thru_node. Zone nodes (1-N) are NOT used as intermediate nodes.
    Therefore, connector targets must be thru-nodes, not other zone nodes.
    
    Older versions restricted connector targets to the directed thru-node SCC.
    That avoided some unreachable pairs but could create very long straight
    "fan" connectors when the nearest SCC node was far away. This version
    chooses geographically nearest thru-nodes first and reports long links.

    Returns dict of zone_id -> list of (node_id, length_miles)
    """
    from collections import defaultdict, deque

    # Zone nodes are now virtual nodes written separately as IDs 1..N.
    # Keep every original simplified-network node as a thru-node so the road
    # graph is not broken by turning a real road node into a non-thru zone.
    thru_nodes = set(int(x) for x in nodes_df['node_id'])

    print("Finding local connector nodes...")
    print("  Keeping all original network nodes as connector targets")
    print(f"  Candidate thru-nodes: {len(thru_nodes)}")
    if max_connector_miles is not None:
        print(f"  Maximum connector length: {max_connector_miles:.2f} mi")
    if not require_scc_fallback:
        print("  SCC fallback connectors: disabled")

    # Find the main directed thru-node SCC of the preserved road graph.
    # Each zone should have at least one connector into this SCC; otherwise
    # it may be geographically local but unreachable in directed assignment.
    fwd_thru = defaultdict(set)
    bwd_thru = defaultdict(set)
    for _, link in links_df.iterrows():
        a, b = int(link['init_node']), int(link['term_node'])
        if a in thru_nodes and b in thru_nodes:
            fwd_thru[a].add(b)
            bwd_thru[b].add(a)

    def bfs(graph, start, allowed_nodes=None):
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in visited and (allowed_nodes is None or neighbor in allowed_nodes):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    undirected_thru = defaultdict(set)
    for n in thru_nodes:
        for nb in fwd_thru.get(n, set()):
            undirected_thru[n].add(nb)
            undirected_thru[nb].add(n)

    visited = set()
    largest_und = set()
    for n in thru_nodes:
        if n in visited:
            continue
        comp = bfs(undirected_thru, n, thru_nodes)
        visited.update(comp)
        if len(comp) > len(largest_und):
            largest_und = comp

    seed = None
    for n in sorted(largest_und):
        if fwd_thru.get(n) and bwd_thru.get(n):
            seed = n
            break
    if seed is None:
        raise RuntimeError("Could not find a seed node for the main thru-node SCC.")

    thru_scc = bfs(fwd_thru, seed, thru_nodes) & bfs(bwd_thru, seed, thru_nodes)
    print(f"  Main directed thru-node SCC: {len(thru_scc)} nodes "
          f"({100 * len(thru_scc) / len(thru_nodes):.1f}% of candidates)")

    candidate_nodes = nodes_df[nodes_df['node_id'].isin(thru_nodes)].copy()
    if candidate_nodes.empty:
        raise RuntimeError("No thru-node connector candidates found.")

    # Use projected coordinates so connector length and threshold are measured
    # in real distance, not rough lon/lat degrees.
    candidate_nodes_wgs = gpd.GeoDataFrame(
        candidate_nodes,
        geometry=[Point(x, y) for x, y in zip(candidate_nodes['x'], candidate_nodes['y'])],
        crs="EPSG:4326",
    )
    candidate_nodes_proj = candidate_nodes_wgs.to_crs(epsg=26915)

    if zones_gdf.crs is None:
        zones_gdf = zones_gdf.set_crs("EPSG:4326")
    zones_proj = zones_gdf.to_crs(epsg=26915)

    node_coords = np.array([(geom.x, geom.y) for geom in candidate_nodes_proj.geometry])
    tree = cKDTree(node_coords)

    scc_candidate_nodes = candidate_nodes[candidate_nodes['node_id'].isin(thru_scc)].copy()
    if scc_candidate_nodes.empty:
        raise RuntimeError("Main thru-node SCC has no connector candidates.")
    scc_candidate_nodes_wgs = gpd.GeoDataFrame(
        scc_candidate_nodes,
        geometry=[Point(x, y) for x, y in zip(scc_candidate_nodes['x'], scc_candidate_nodes['y'])],
        crs="EPSG:4326",
    )
    scc_candidate_nodes_proj = scc_candidate_nodes_wgs.to_crs(epsg=26915)
    scc_node_coords = np.array([(geom.x, geom.y) for geom in scc_candidate_nodes_proj.geometry])
    scc_tree = cKDTree(scc_node_coords)
    
    zone_connectors = {}
    # Get set of zones inside collapsed cities (they don't need separate connectors)
    zones_in_collapsed_cities = set()
    city_zone_to_super_node = {}
    if zones_in_cities:
        for city_name, city_zones in zones_in_cities.items():
            zones_in_collapsed_cities.update(city_zones)
            if super_nodes and city_name in super_nodes:
                super_node_id = int(super_nodes[city_name]['node_id'])
                for zone_id in city_zones:
                    city_zone_to_super_node[int(zone_id)] = super_node_id
    
    print(f"Finding {num_connectors} connector nodes for {len(zones_gdf)} zones...")
    print(f"  {len(zones_in_collapsed_cities)} zones inside collapsed cities will share their city super-node")
    
    long_connectors = []
    scc_fallbacks = []
    rejected_scc_fallbacks = []
    no_connector_zones = []
    
    for idx, zone in zones_gdf.iterrows():
        # Convert 0-indexed zone_id to 1-indexed
        zone_id_from_file = zone.get('zone_id', idx)
        zone_id = int(zone_id_from_file) + 1  # Convert to 1-indexed
        if zone_id not in zone_to_node:
            continue

        if zone_id in city_zone_to_super_node:
            # The internal city network has already been collapsed to this
            # super-node. Keep this connector nearly zero-length so city zones
            # do not create long artificial fan VMT.
            zone_connectors[zone_id] = [(city_zone_to_super_node[zone_id], 0.001)]
            continue
            
        zone_centroid = zones_proj.loc[idx].geometry.centroid
        zone_xy = [zone_centroid.x, zone_centroid.y]
        
        connectors = []
        if max_connector_miles is not None:
            radius_m = max_connector_miles * 1609.344
            candidate_indices = tree.query_ball_point(zone_xy, r=radius_m)
            candidate_pairs = [
                (float(np.linalg.norm(node_coords[int(node_idx)] - zone_xy)), int(node_idx))
                for node_idx in candidate_indices
            ]
            candidate_pairs.sort(key=lambda x: x[0])
        else:
            # Find nearest local thru-nodes. Query extra candidates so skipped
            # centroid/super nodes do not reduce the requested connector count.
            k = min(len(candidate_nodes), max(num_connectors + 50, num_connectors))
            distances, indices = tree.query(zone_xy, k=k)
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)
            candidate_pairs = [(float(dist), int(node_idx)) for dist, node_idx in zip(distances, indices)]

        for dist_m, node_idx in candidate_pairs:
            node_id = int(candidate_nodes.iloc[int(node_idx)]['node_id'])
            length_miles = dist_m / 1609.344
            connectors.append((node_id, length_miles))
            if length_miles > LONG_CONNECTOR_WARNING_MILES:
                long_connectors.append((zone_id, node_id, length_miles))
            if len(connectors) >= num_connectors:
                break

        # Guarantee at least one connector into the main directed SCC.
        # If all nearest local connectors are outside it, replace the farthest
        # connector with the nearest SCC connector. This avoids widespread
        # unreachable OD demand while keeping the other connectors local.
        if require_scc_fallback and connectors and not any(node_id in thru_scc for node_id, _ in connectors):
            scc_dist, scc_idx = scc_tree.query(zone_xy, k=1)
            scc_node_id = int(scc_candidate_nodes.iloc[int(scc_idx)]['node_id'])
            scc_length_miles = float(scc_dist) / 1609.344
            if max_connector_miles is not None and scc_length_miles > max_connector_miles:
                rejected_scc_fallbacks.append((zone_id, scc_node_id, scc_length_miles))
                zone_connectors[zone_id] = connectors
                continue
            if scc_node_id not in [node_id for node_id, _ in connectors]:
                replace_idx = max(range(len(connectors)), key=lambda i: connectors[i][1])
                connectors[replace_idx] = (scc_node_id, scc_length_miles)
            scc_fallbacks.append((zone_id, scc_node_id, scc_length_miles))
            if scc_length_miles > LONG_CONNECTOR_WARNING_MILES:
                long_connectors.append((zone_id, scc_node_id, scc_length_miles))
        if not connectors:
            no_connector_zones.append(zone_id)
        
        zone_connectors[zone_id] = connectors
    
    missing = [z for z, conns in zone_connectors.items() if len(conns) < num_connectors]
    if missing:
        print(f"  WARNING: {len(missing)} zones received fewer than {num_connectors} connectors")

    if no_connector_zones:
        print(f"  WARNING: {len(no_connector_zones)} zones received zero connectors")
        print(f"  Zero-connector zone samples: {no_connector_zones[:10]}")
    
    if long_connectors:
        long_connectors.sort(key=lambda x: -x[2])
        print(f"  WARNING: {len(long_connectors)} connectors are longer than "
              f"{LONG_CONNECTOR_WARNING_MILES:.1f} miles")
        print("  Longest connector samples:")
        for zone_id, node_id, length_miles in long_connectors[:10]:
            print(f"    zone {zone_id} -> node {node_id}: {length_miles:.2f} mi")

    if scc_fallbacks:
        scc_fallbacks.sort(key=lambda x: -x[2])
        print(f"  Added SCC fallback connectors for {len(scc_fallbacks)} zones")
        print("  Longest SCC fallback samples:")
        for zone_id, node_id, length_miles in scc_fallbacks[:10]:
            print(f"    zone {zone_id} -> SCC node {node_id}: {length_miles:.2f} mi")

    if rejected_scc_fallbacks:
        rejected_scc_fallbacks.sort(key=lambda x: -x[2])
        print(f"  Rejected {len(rejected_scc_fallbacks)} SCC fallbacks over the max length")
        print("  Longest rejected SCC fallback samples:")
        for zone_id, node_id, length_miles in rejected_scc_fallbacks[:10]:
            print(f"    zone {zone_id} -> SCC node {node_id}: {length_miles:.2f} mi")
    
    return zone_connectors


def build_zone_node_coords(zones_gdf, zone_to_node, nodes_df, mode,
                           zones_in_cities=None, super_nodes=None):
    """Return {zone_id: (x, y)} for the virtual zone nodes."""
    if mode == "representative-node":
        return None

    city_zone_to_super = {}
    if zones_in_cities and super_nodes:
        for city_name, zone_ids in zones_in_cities.items():
            if city_name not in super_nodes:
                continue
            for zone_id in zone_ids:
                city_zone_to_super[int(zone_id)] = super_nodes[city_name]

    if zones_gdf.crs is None:
        zones_gdf = zones_gdf.set_crs("EPSG:4326")
    zones_wgs = zones_gdf.to_crs(epsg=4326)
    coords = {}
    for idx, zone in zones_wgs.iterrows():
        zone_id_from_file = zone.get('zone_id', idx)
        zone_id = int(zone_id_from_file) + 1
        if zone_id not in zone_to_node:
            continue
        if zone_id in city_zone_to_super:
            sn = city_zone_to_super[zone_id]
            coords[zone_id] = (sn['x'], sn['y'])
            continue
        centroid = zone.geometry.centroid
        coords[zone_id] = (centroid.x, centroid.y)
    return coords


def create_node_mapping(nodes_df, zone_to_node, zones_in_cities=None, super_nodes=None):
    """
    Create new node IDs where:
    - Nodes 1 to num_zones are virtual zone centroid nodes
    - Every original simplified-network node remains a thru-node after the zones

    This is important for TAPAS: nodes below FIRST_THRU_NODE are not expanded
    as intermediate nodes. If a real road node is reused as a zone node, that
    road node stops acting as a thru-node and can break directed connectivity.
    
    Returns:
    - old_to_new: dict mapping old node_id to new node_id
    - new_to_old: dict mapping new node_id to old node_id
    - total_nodes: max node ID assigned
    - virtual_zone_nodes: set of all zone_ids
    """
    num_zones = len(zone_to_node)

    old_to_new = {}
    new_to_old = {}

    # Step 1: Assign virtual zone node IDs 1..num_zones.
    virtual_zone_nodes = set(zone_to_node.keys())
    for zone_id in sorted(zone_to_node.keys()):
        old_node_id = int(zone_to_node[zone_id])
        new_to_old[zone_id] = old_node_id
    
    # Step 2: Assign IDs to ALL original network nodes after the virtual zones.
    next_id = num_zones + 1
    for _, row in nodes_df.iterrows():
        old_node_id = int(row['node_id'])
        old_to_new[old_node_id] = next_id
        new_to_old[next_id] = old_node_id
        next_id += 1
    
    total_nodes = next_id - 1
    
    print(f"  Virtual zone nodes: {len(virtual_zone_nodes)}")
    print(f"  Preserved thru-nodes: {len(nodes_df)}")
    print(f"  Total nodes: {total_nodes}")
    
    return old_to_new, new_to_old, total_nodes, virtual_zone_nodes


def write_output_files(nodes_df, links_df, zones_gdf, zone_to_node, zone_connectors, 
                       old_to_new, output_folder, total_nodes=None, 
                       virtual_zone_nodes=None, zones_in_cities=None, super_nodes=None,
                       zone_node_coords=None):
    """Write the new network files."""
    os.makedirs(output_folder, exist_ok=True)
    
    num_zones = len(zone_to_node)
    num_nodes = total_nodes if total_nodes else len(nodes_df)
    
    if virtual_zone_nodes is None:
        virtual_zone_nodes = set()
    
    # Calculate number of new connector links
    # ALL zones (including virtual ones) now get direct thru-node connectors
    num_connectors = sum(len(conns) * 2 for conns in zone_connectors.values())  # *2 for both directions
    num_virtual_connectors = 0  # No more super-node based virtual links
    num_links = len(links_df) + num_connectors
    
    # Write node mapping
    mapping_file = os.path.join(output_folder, 'zone_node_mapping.csv')
    with open(mapping_file, 'w') as f:
        f.write("zone_id,new_node_id,original_node_id,x,y,is_virtual,city\n")
        for zone_id in sorted(zone_to_node.keys()):
            old_node_id = int(zone_to_node[zone_id])
            new_node_id = zone_id
            node_row = nodes_df[nodes_df['node_id'] == old_node_id].iloc[0]
            x, y = zone_node_coords.get(zone_id, (node_row['x'], node_row['y'])) if zone_node_coords else (node_row['x'], node_row['y'])
            city = ''
            if zones_in_cities:
                for cn, czones in zones_in_cities.items():
                    if zone_id in czones:
                        city = cn
                        break
            f.write(f"{zone_id},{new_node_id},{old_node_id},{x},{y},yes,{city}\n")
    print(f"Written: {mapping_file}")
    
    # Write nodes file (reordered)
    nodes_file = os.path.join(output_folder, 'nodes.txt')
    with open(nodes_file, 'w') as f:
        f.write("~ new_node_id\toriginal_node_id\tx\ty\t;\n")
        
        # Write zone nodes first (1 to num_zones)
        for zone_id in sorted(zone_to_node.keys()):
            old_node_id = int(zone_to_node[zone_id])
            node_row = nodes_df[nodes_df['node_id'] == old_node_id].iloc[0]
            x, y = zone_node_coords.get(zone_id, (node_row['x'], node_row['y'])) if zone_node_coords else (node_row['x'], node_row['y'])
            f.write(f"\t{zone_id}\t{old_node_id}\t{x}\t{y}\t;\n")
        
        # Write all original network nodes as thru-nodes.
        for _, row in nodes_df.iterrows():
            old_node_id = int(row['node_id'])
            new_node_id = old_to_new[old_node_id]
            f.write(f"\t{new_node_id}\t{row['node_id']}\t{row['x']}\t{row['y']}\t;\n")
    print(f"Written: {nodes_file}")
    
    # Write network file with centroid connectors
    network_file = os.path.join(output_folder, 'network.txt')
    with open(network_file, 'w') as f:
        # Write metadata
        f.write(f"<NUMBER OF ZONES> {num_zones}\n")
        f.write(f"<NUMBER OF NODES> {num_nodes}\n")
        f.write(f"<FIRST THRU NODE> {num_zones + 1}\n")
        f.write(f"<NUMBER OF LINKS> {num_links}\n")
        f.write("<ORIGINAL HEADER>~ \tInit node \tTerm node \tCapacity \tLength \tFree Flow Time \tB\tPower\tSpeed limit \tToll \tType\t;\n")
        f.write("<END OF METADATA>\n\n")
        f.write("~\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\tspeed\ttoll\tlink_type\t;\n")
        
        # Write ALL centroid connectors (virtual zone <-> thru-node network nodes)
        for zone_id in sorted(zone_connectors.keys()):
            zone_node = zone_id
            for network_node_old, length_miles in zone_connectors[zone_id]:
                network_node = old_to_new[int(network_node_old)]
                travel_time = length_miles / CONNECTOR_SPEED * 60
                
                # Zone -> Network (outbound)
                f.write(f"\t{zone_node}\t{network_node}\t{CONNECTOR_CAPACITY}\t{length_miles:.5f}\t{travel_time:.5f}\t{CONNECTOR_ALPHA}\t{CONNECTOR_BETA}\t{CONNECTOR_SPEED}\t0\t9\t;\n")
                # Network -> Zone (inbound)
                f.write(f"\t{network_node}\t{zone_node}\t{CONNECTOR_CAPACITY}\t{length_miles:.5f}\t{travel_time:.5f}\t{CONNECTOR_ALPHA}\t{CONNECTOR_BETA}\t{CONNECTOR_SPEED}\t0\t9\t;\n")
        
        # Write original links with updated node IDs
        skipped = 0
        for _, link in links_df.iterrows():
            init_old = int(link['init_node'])
            term_old = int(link['term_node'])
            if init_old not in old_to_new:
                skipped += 1
                continue
            if term_old not in old_to_new:
                skipped += 1
                continue
            init_node = old_to_new[init_old]
            term_node = old_to_new[term_old]
            f.write(f"\t{init_node}\t{term_node}\t{link['capacity']:.2f}\t{link['length']:.5f}\t{link['free_flow_time']:.5f}\t{link['b']}\t{link['power']}\t{link['speed']:.2f}\t{link['toll']}\t{link['link_type']}\t;\n")
        if skipped:
            print(f"  WARNING: Skipped {skipped} links with unmapped nodes")
    
    print(f"Written: {network_file}")
    print(f"\nNetwork Summary:")
    print(f"  Zones: {num_zones}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Links: {num_links} (incl. {num_connectors} centroid connectors)")
    print(f"  First thru node: {num_zones + 1}")
    print(f"  Virtual zone nodes: {len(virtual_zone_nodes)} (connected directly to thru-nodes)")


def main():
    global ZONES_SHAPEFILE, NODES_FILE, NETWORK_FILE, SUPER_NODES_FILE, OUTPUT_FOLDER
    args = parse_args()
    ZONES_SHAPEFILE = args.zones
    NODES_FILE = args.nodes
    NETWORK_FILE = args.network
    SUPER_NODES_FILE = args.super_nodes
    OUTPUT_FOLDER = args.output_folder

    print("="*60)
    print("Create Zone Centroids for Minnesota Network (Simplified)")
    print("="*60)
    
    # Read zones
    print(f"\nReading zones from: {ZONES_SHAPEFILE}")
    zones_gdf = gpd.read_file(ZONES_SHAPEFILE)
    print(f"  Found {len(zones_gdf)} zones")
    
    # Read nodes
    print(f"\nReading nodes from: {NODES_FILE}")
    nodes_df = read_nodes(NODES_FILE)
    print(f"  Found {len(nodes_df)} nodes")
    
    # Read network
    print(f"\nReading network from: {NETWORK_FILE}")
    links_df, metadata = read_network(NETWORK_FILE)
    print(f"  Found {len(links_df)} links")
    
    # Read super nodes (from the network simplification)
    print(f"\nReading super nodes from: {SUPER_NODES_FILE}")
    super_nodes = read_super_nodes(SUPER_NODES_FILE)
    if super_nodes:
        print(f"  Found {len(super_nodes)} collapsed cities:")
        for city_name, sn_info in super_nodes.items():
            print(f"    - {city_name}: Node {sn_info['node_id']}")
    
    # Find zone centroids (node inside or closest to each zone)
    # Zones inside collapsed cities will be assigned to the city's super node
    print("\n" + "-"*60)
    zone_to_node, zones_in_cities = find_zone_centroids(zones_gdf, nodes_df, super_nodes)
    
    # Find connector nodes for each zone
    # Zones inside collapsed cities don't need separate connectors
    print("\n" + "-"*60)
    zone_connectors = find_connector_nodes(
        zones_gdf, nodes_df, zone_to_node, links_df, 
        super_nodes=super_nodes, 
        zones_in_cities=zones_in_cities,
        num_connectors=args.num_connectors,
        max_connector_miles=args.max_connector_miles,
        require_scc_fallback=not args.no_scc_fallback,
    )
    
    # Create node ID mapping
    print("\n" + "-"*60)
    print("Creating node ID mapping...")
    old_to_new, new_to_old, total_nodes, virtual_zone_nodes = create_node_mapping(
        nodes_df, zone_to_node, zones_in_cities=zones_in_cities, super_nodes=super_nodes
    )
    print(f"  Mapped {len(old_to_new)} nodes -> max ID: {total_nodes}")

    zone_node_coords = build_zone_node_coords(
        zones_gdf, zone_to_node, nodes_df, args.zone_node_location,
        zones_in_cities=zones_in_cities,
        super_nodes=super_nodes,
    )
    
    # Write output files
    print("\n" + "-"*60)
    print(f"Writing output files to: {OUTPUT_FOLDER}")
    write_output_files(nodes_df, links_df, zones_gdf, zone_to_node, zone_connectors, 
                       old_to_new, OUTPUT_FOLDER, total_nodes=total_nodes,
                       virtual_zone_nodes=virtual_zone_nodes,
                       zones_in_cities=zones_in_cities, super_nodes=super_nodes,
                       zone_node_coords=zone_node_coords)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nOutput files in {OUTPUT_FOLDER}/:")
    print("  - nodes.txt: Node file with zone centroids first")
    print("  - network.txt: Network with centroid connectors (TNTP format)")
    print("  - zone_node_mapping.csv: Zone to node mapping")
    
    if zones_in_cities:
        print(f"\nNote: {sum(len(z) for z in zones_in_cities.values())} zones inside collapsed cities")
        print("  These zones share their city's super node and use its network connections.")
    
    print("\nYou can now copy trips.txt to this folder and run traffic assignment!")


if __name__ == '__main__':
    main()
