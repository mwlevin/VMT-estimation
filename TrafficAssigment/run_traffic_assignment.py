"""
Traffic Assignment Runner
=========================
This script runs user equilibrium traffic assignment on networks in the data folder
using either MSA (Method of Successive Averages) or TAPAS algorithm.

Based on main_CNDP.py but simplified to just run traffic assignment.

Usage:
    python run_traffic_assignment.py [--net NETWORK] [--algorithm msa|tapas] [--iterations N]
"""

import sys
import os
import time
import argparse
import csv

# Since this file is inside TrafficAssigment folder, use current directory
traffic_assign_path = os.path.dirname(__file__)
sys.path.insert(0, traffic_assign_path)

from src import Network

# Same scaling parameters as main_CNDP.py
SCAL_FLOW = {
    'SiouxFalls': 1e-3,
    'SiouxFalls2': 1,
    'EasternMassachusetts': 1e-3,
    'BerlinMitteCenter': 1e-3,
    'Anaheim': 1e-3,
    'Barcelona': 1e-3,
    'Braess': 1,
    'HarkerFriesz': 1,
    'Minnesota': 1e-3,
    'Minnesota2024': 1e-3
}

INFLATE_TRIPS = {
    'SiouxFalls': 1,
    'SiouxFalls2': 1,
    'EasternMassachusetts': 4,
    'BerlinMitteCenter': 2,
    'Anaheim': 4,
    'Barcelona': 2,
    'Braess': 1,
    'HarkerFriesz': 0.25,
    'Minnesota': 1,
    'Minnesota2024': 1
}

# Default instance files for each network
DEFAULT_INS = {
    'SiouxFalls': 'net',
    'SiouxFalls2': 'net',
    'EasternMassachusetts': 'net',
    'BerlinMitteCenter': 'net',
    'Anaheim': 'net',
    'Barcelona': 'net',
    'Braess': 'net',
    'HarkerFriesz': 'net'
}


def _load_warm_start_link_flows(network, csv_path, scal_flow):
    """
    Initialize aggregate link flows from a previous link_flows.csv export.

    The assignment engine stores internal flows in scaled units. The exported
    CSV contains both flow_scaled and flow_real; prefer flow_scaled when present.
    """
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Warm-start link flow file not found: {csv_path}")

    by_id = {link.id: link for link in network.links}
    loaded = 0
    missing = 0

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                link_id = int(row['link_id'])
            except (KeyError, TypeError, ValueError):
                continue

            link = by_id.get(link_id)
            if link is None:
                missing += 1
                continue

            if row.get('flow_scaled') not in (None, ''):
                flow = float(row['flow_scaled'])
            elif row.get('flow_real') not in (None, ''):
                flow = float(row['flow_real']) * scal_flow
            else:
                continue

            link.x = flow
            link.xstar = 0
            loaded += 1

    print(f"  Warm-started {loaded:,} link flows from: {csv_path}")
    if missing:
        print(f"  Warning: {missing:,} warm-start rows did not match current link IDs")
    return loaded


def run_traffic_assignment(net_name, ins='net', algorithm='msa', max_iter=10, gap_threshold=1e-4, clear_cache=False, data_dir=None, warm_start_file=None, start_iteration=1, tapas_equilibrate_iter=None, tapas_old_origin_shifts=False, itapas_max_mfs_per_link=None, itapas_cost_mu=None, itapas_flow_mu=None, itapas_use_newton_shift=True, itapas_remove_cycles=True):
    """
    Run traffic assignment on a network.
    
    Parameters:
    -----------
    net_name : str
        Network name (folder in data/)
    ins : str
        Instance file name (without .txt)
    algorithm : str
        'msa' or 'tapas'
    max_iter : int
        Maximum iterations
    gap_threshold : float
        Convergence gap
    clear_cache : bool
        If True, clear cached Dijkstra results and recompute
    data_dir : str or None
        Custom directory containing net.txt, trips.txt, dijkstra_cache.
        If None, defaults to data/{net_name}/
    warm_start_file : str or None
        Optional previous link_flows.csv used to initialize aggregate link flows.
        This is intended for MSA continuation, not true TAPAS bush warm-starting.
    start_iteration : int
        First MSA iteration number. Use 4 to continue from a 3-iteration run.
    """
    
    print("="*60)
    print("Traffic Assignment Runner")
    print("="*60)
    print(f"Network: {net_name}")
    print(f"Instance: {ins}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Max iterations: {max_iter}")
    print(f"Gap threshold: {gap_threshold}")
    print()
    
    # Get scaling parameters (same as main_CNDP.py)
    scal_flow = SCAL_FLOW.get(net_name, 1)
    inflate_trips = INFLATE_TRIPS.get(net_name, 1)
    b_prop = 0.5  # Budget proportion (not used for pure traffic assignment)
    scal_time = 1.0
    
    print(f"Scale flow: {scal_flow}")
    print(f"Inflate trips: {inflate_trips}")
    print()
    
    # Create network (this reads network and trips files automatically)
    print("Loading network...")
    start_time = time.time()
    
    network = Network.Network(net_name, ins, b_prop, scal_time, scal_flow, inflate_trips, data_dir=data_dir)
    
    if clear_cache:
        print("  Clearing Dijkstra cache (--clear-cache flag)")
        import shutil
        if os.path.isdir(network._cache_dir):
            shutil.rmtree(network._cache_dir)
            print(f"  Removed cache directory: {network._cache_dir}")
    
    load_time = time.time() - start_time
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Links: {len(network.links)}")
    print(f"  Zones: {len(network.zones)}")
    print(f"  Total Demand (TD): {network.TD:.2f}")
    print(f"  Origins with demand: {len(network.origins)}")
    print(f"  Network loaded in {load_time:.2f} seconds")
    
    # Run traffic assignment
    print("\n" + "="*60)
    print(f"Running {algorithm.upper()} Traffic Assignment")
    print("="*60)
    
    # Set parameters for traffic assignment
    network.params.PRINT_TAP_ITER = True
    network.params.PRINT_TAPAS_INFO = False
    network.params.tapas_max_iter = max_iter
    network.params.msa_max_iter = max_iter
    network.params.msa_start_iteration = max(1, start_iteration)
    network.params.min_gap = gap_threshold

    # TAPAS_CHANGE_2026_06_13_E_CLI_ROLLBACK_FLAGS:
    # The old origin-loop PAS shifting behavior can be restored from the
    # command line for exact comparisons. The number of unique active-PAS
    # equilibration passes is also exposed because the TAPAS paper treats it
    # as a user parameter.
    if tapas_equilibrate_iter is not None:
        network.params.tapas_equilibrate_iter = tapas_equilibrate_iter
        network.params.itapas_equilibrate_iter = tapas_equilibrate_iter
    network.params.tapas_shift_relevant_pas_during_origin_scan = tapas_old_origin_shifts
    if itapas_max_mfs_per_link is not None:
        network.params.itapas_max_mfs_per_link = itapas_max_mfs_per_link
    if itapas_cost_mu is not None:
        network.params.itapas_pas_cost_mu = itapas_cost_mu
    if itapas_flow_mu is not None:
        network.params.itapas_pas_flow_mu = itapas_flow_mu
    network.params.itapas_use_newton_shift = itapas_use_newton_shift
    network.params.itapas_remove_cycles = itapas_remove_cycles

    if algorithm.lower() == 'tapas':
        print(f"  TAPAS active-PAS shift passes/iteration: {network.params.tapas_equilibrate_iter}")
        print(f"  TAPAS old origin-loop PAS shifts: {network.params.tapas_shift_relevant_pas_during_origin_scan}")
    elif algorithm.lower() == 'itapas':
        print(f"  iTAPAS active-PAS shift passes/iteration: {network.params.itapas_equilibrate_iter}")
        print(f"  iTAPAS cost effectiveness mu: {network.params.itapas_pas_cost_mu}")
        print(f"  iTAPAS flow effectiveness nu: {network.params.itapas_pas_flow_mu}")
        print(f"  iTAPAS max MFS attempts/link: {network.params.itapas_max_mfs_per_link}")
        print(f"  iTAPAS Newton shift: {network.params.itapas_use_newton_shift}")
        print(f"  iTAPAS cycle removal before origin scan: {network.params.itapas_remove_cycles}")

    if warm_start_file:
        if algorithm.lower() != 'msa':
            raise ValueError(
                "link_flows.csv warm start is only available for MSA because it "
                "contains aggregate link flows, not the origin-by-origin bush "
                "flows needed by TAPAS. Run TAPAS without --warm-start-link-flows."
            )
        _load_warm_start_link_flows(network, warm_start_file, scal_flow)
        print(f"  MSA continuation starts at iteration: {network.params.msa_start_iteration}")
    
    print("\nIter\tTSTT\t\t\tSPTT\t\t\tGap\t\tAEC")
    print("-"*70)
    
    start_time = time.time()
    
    # y is the link availability vector (all 1s = all links available)
    #y = [1] * len(network.links)
    
    
    if algorithm.lower() == 'msa':
        tstt = network.msa('UE')
    elif algorithm.lower() == 'tapas':
        tstt = network.tapas('UE')
    elif algorithm.lower() == 'itapas':
        tstt = network.itapas('UE')
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    assignment_time = time.time() - start_time
    
    # Calculate final results
    sptt = network.getSPTT('UE')
    gap = (tstt - sptt) / tstt if tstt > 0 else 0
    
    # Calculate VMT (Vehicle Miles Traveled) using link lengths
    # Divide by scal_flow to convert scaled flow back to real vehicles
    vmt = 0
    for link in network.links:
        real_flow = link.x / scal_flow if scal_flow > 0 else link.x
        vmt += real_flow * link.length
    
    # Calculate VHT (Vehicle Hours Traveled)
    # tt is in minutes -> convert to hours; flow back to real vehicles
    total_vht = 0
    for link in network.links:
        tt = link.getTravelTime(link.x, 'UE')
        real_flow = link.x / scal_flow if scal_flow > 0 else link.x
        total_vht += real_flow * (tt / 60.0)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Network: {net_name}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Total System Travel Time (TSTT): {tstt:,.2f}")
    print(f"Shortest Path Travel Time (SPTT): {sptt:,.2f}")
    print(f"Relative Gap: {gap:.6f}")
    print(f"Total Demand: {network.TD:,.2f}")
    print(f"Vehicle Miles Traveled (VMT): {vmt:,.2f}")
    print(f"Vehicle Hours Traveled (VHT): {total_vht:,.2f}")
    print(f"Assignment time: {assignment_time:.2f} seconds")

    output_dir = data_dir if data_dir else os.path.join('data', net_name)
    summary_csv = os.path.join(output_dir, 'assignment_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'network',
                'instance',
                'algorithm',
                'max_iterations',
                'gap_threshold',
                'final_gap',
                'stopping_status',
                'tstt',
                'sptt',
                'total_demand',
                'vmt',
                'vht',
                'assignment_time_sec',
                'data_dir',
            ],
        )
        writer.writeheader()
        writer.writerow({
            'network': net_name,
            'instance': ins,
            'algorithm': algorithm,
            'max_iterations': max_iter,
            'gap_threshold': gap_threshold,
            'final_gap': gap,
            'stopping_status': 'gap_threshold_met' if gap <= gap_threshold else 'max_iterations_or_solver_stop',
            'tstt': tstt,
            'sptt': sptt,
            'total_demand': network.TD,
            'vmt': vmt,
            'vht': total_vht,
            'assignment_time_sec': assignment_time,
            'data_dir': output_dir,
        })
    print(f"Assignment summary saved to: {summary_csv}")
    
    # Print some link flow statistics
    flows = [link.x for link in network.links]
    print(f"\nLink Flow Statistics:")
    print(f"  Min flow: {min(flows):.2f}")
    print(f"  Max flow: {max(flows):.2f}")
    print(f"  Avg flow: {sum(flows)/len(flows):.2f}")
    
    # ============================================================
    # EXPORT FINAL LINK FLOWS (for color-coded mapping later)
    # ============================================================
    link_flow_csv = os.path.join(output_dir, 'link_flows.csv')
    print(f"\nExporting final link flows to: {link_flow_csv}")
    
    with open(link_flow_csv, 'w') as f:
        f.write('link_id,init_node,term_node,flow_scaled,flow_real,capacity,'
                'volume_capacity_ratio,length,free_flow_time,travel_time,'
                'congested_speed,vmt,vht\n')
        for link in network.links:
            flow_scaled = link.x
            flow_real = flow_scaled / scal_flow if scal_flow > 0 else flow_scaled
            tt = link.getTravelTime(link.x, 'UE')  # converged travel time (minutes)
            vc = flow_scaled / link.C if link.C > 0 else 0.0
            # Congested speed = length / travel_time, with tt in minutes
            speed = (link.length / (tt / 60.0)) if tt > 0 else 0.0
            link_vmt = flow_real * link.length
            link_vht = flow_real * (tt / 60.0)
            f.write(f'{link.id},{link.start.id},{link.end.id},'
                    f'{flow_scaled:.6f},{flow_real:.4f},{link.C:.2f},'
                    f'{vc:.6f},{link.length:.6f},{link.t_ff:.6f},{tt:.6f},'
                    f'{speed:.4f},{link_vmt:.4f},{link_vht:.6f}\n')
    print(f"  Wrote {len(network.links):,} link records")
    
    # Also export node coordinates (needed to draw the map)
    node_file_src = None
    if data_dir is not None:
        for cand in ('nodes.txt', 'node.txt'):
            p = os.path.join(data_dir, cand)
            if os.path.exists(p):
                node_file_src = p
                break
    else:
        p = os.path.join('data', net_name, 'node.txt')
        if os.path.exists(p):
            node_file_src = p
    
    if node_file_src is not None:
        node_csv = os.path.join(output_dir, 'node_coords.csv')
        with open(node_file_src, 'r') as fin, open(node_csv, 'w') as fout:
            fout.write('node_id,lon,lat\n')
            for line in fin:
                line = line.strip()
                if line.startswith('~') or line.startswith('<') or len(line) == 0:
                    continue
                parts = line.replace(';', '').split()
                if len(parts) >= 4:
                    fout.write(f'{int(parts[0])},{float(parts[2])},{float(parts[3])}\n')
        print(f"  Node coordinates exported to: {node_csv}")
    
    # Report unreachable O-D pairs
    if hasattr(network, 'unreachable_pairs') and network.unreachable_pairs:
        pairs = network.unreachable_pairs
        total_lost_demand = sum(p['demand'] for p in pairs)
        # Get unique O-D pairs (same pair may appear from multiple bushes)
        unique_pairs = set((p['origin'], p['destination']) for p in pairs)
        unique_origins = set(p['origin'] for p in pairs)
        unique_dests = set(p['destination'] for p in pairs)
        
        print(f"\n{'='*60}")
        print(f"UNREACHABLE O-D PAIRS")
        print(f"{'='*60}")
        print(f"  Total unreachable pairs: {len(unique_pairs)}")
        print(f"  Unique origins affected: {len(unique_origins)}")
        print(f"  Unique destinations affected: {len(unique_dests)}")
        print(f"  Total lost demand: {total_lost_demand:,.2f}")
        print(f"  Lost demand as % of total: {total_lost_demand/network.TD*100:.2f}%" if network.TD > 0 else "")
        
        # Save detailed log to file
        log_file = os.path.join(data_dir if data_dir else os.path.join('data', net_name), 'unreachable_pairs.csv')
        with open(log_file, 'w') as f:
            f.write("origin,destination,demand\n")
            for p in pairs:
                f.write(f"{p['origin']},{p['destination']},{p['demand']:.4f}\n")
        print(f"  Detailed log saved to: {log_file}")
        
        # Print first 20 pairs as sample
        if len(unique_pairs) <= 20:
            print(f"\n  All unreachable pairs:")
        else:
            print(f"\n  First 20 unreachable pairs (of {len(unique_pairs)}):")
        for i, (o, d) in enumerate(sorted(unique_pairs)):
            if i >= 20:
                break
            print(f"    Origin {o} -> Destination {d}")
    else:
        print(f"\n  All O-D pairs are reachable.")
    
    # ============================================================
    # COUNTY-LEVEL VMT EXTRACTION  (Geometric Line Clipping)
    # ============================================================
    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point, LineString
        from shapely.ops import transform as shapely_transform
        from shapely.prepared import prep
        import pyproj
        
        # Find paths relative to the script location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        county_shp = os.path.join(base_dir, 'mn_counties', 'mn_counties.shp')
        if data_dir is not None:
            node_file = os.path.join(data_dir, 'nodes.txt')
        else:
            node_file = os.path.join('data', net_name, 'node.txt')
        
        if os.path.exists(county_shp) and os.path.exists(node_file):
            print("\n" + "="*60)
            print("COUNTY-LEVEL VMT  (Geometric Line Clipping)")
            print("="*60)
            
            # --- Step 1: Read node coordinates ---
            print("Reading node coordinates...")
            node_coords = {}  # node_id -> (lon, lat)
            with open(node_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('~') or line.startswith('<') or len(line) == 0:
                        continue
                    parts = line.replace(';', '').split()
                    if len(parts) >= 4:
                        nid = int(parts[0])
                        lon = float(parts[2])
                        lat = float(parts[3])
                        node_coords[nid] = (lon, lat)
            print(f"  Loaded coordinates for {len(node_coords)} nodes")
            
            # --- Step 2: Read county boundaries ---
            print("Reading county boundaries...")
            counties = gpd.read_file(county_shp).to_crs('EPSG:4326')
            print(f"  {len(counties)} counties loaded")
            
            # Spatial index and prepared geometries for fast lookups
            county_sindex = counties.sindex
            prep_geoms = [prep(g) for g in counties.geometry]
            
            # Projector WGS84 -> UTM 15N for accurate length measurements
            _proj = pyproj.Transformer.from_crs(
                'EPSG:4326', 'EPSG:26915', always_xy=True
            )
            project_to_utm = _proj.transform
            
            def utm_length(geom):
                """Return length of a shapely geometry in metres (UTM 15N)."""
                if geom.is_empty:
                    return 0.0
                return shapely_transform(project_to_utm, geom).length
            
            # --- Step 3: Pre-assign each node to a county ---
            print("Assigning nodes to counties...")
            node_county = {}  # node_id -> county_name
            
            # Pre-compute county geometries in UTM for accurate distance fallback
            counties_utm = counties.to_crs('EPSG:26915')
            
            def find_county_for_point(lon, lat):
                """Find which county a point belongs to (nearest as fallback)."""
                pt = Point(lon, lat)
                candidates = list(county_sindex.intersection(pt.bounds))
                for idx in candidates:
                    if prep_geoms[idx].contains(pt):
                        return counties.iloc[idx]['NAME']
                # Fallback: nearest county in projected CRS (metres)
                pt_utm = shapely_transform(project_to_utm, pt)
                dists = counties_utm.geometry.distance(pt_utm)
                return counties.loc[dists.idxmin(), 'NAME']
            
            coord_cache = {}  # (lon6, lat6) -> county_name
            for nid, (lon, lat) in node_coords.items():
                key = (round(lon, 6), round(lat, 6))
                if key not in coord_cache:
                    coord_cache[key] = find_county_for_point(lon, lat)
                node_county[nid] = coord_cache[key]
            print(f"  Assigned {len(node_county)} nodes to counties")
            
            # --- Step 4: Compute VMT per county (geometric clipping) ---
            # For each link with flow:
            #   * Fast path – both endpoints in the same county → 100 % VMT
            #   * Slow path – endpoints in different counties →
            #       build LineString, intersect with each candidate county,
            #       measure clipped length in UTM, split VMT proportionally
            print("Computing county-level VMT (geometric line clipping)...")
            county_vmt = {}   # county_name -> vmt
            total_vmt_all = 0.0
            cross_county_links = 0
            
            n_links = len(network.links)
            report_every = max(1, n_links // 10)
            
            for i, link in enumerate(network.links):
                if (i + 1) % report_every == 0:
                    print(f"  Processing link {i+1:,}/{n_links:,} ...")
                
                real_flow = link.x / scal_flow if scal_flow > 0 else link.x
                link_vmt = real_flow * link.length
                total_vmt_all += link_vmt
                
                if link_vmt < 1e-10:
                    continue
                
                start_id = link.start.id
                end_id   = link.end.id
                sc = node_county.get(start_id)
                ec = node_county.get(end_id)
                
                # ---- Fast path: both endpoints in the same county ----
                if sc and ec and sc == ec:
                    county_vmt[sc] = county_vmt.get(sc, 0.0) + link_vmt
                    continue
                
                # ---- Slow path: geometric clipping ----
                start_xy = node_coords.get(start_id)
                end_xy   = node_coords.get(end_id)
                
                if not start_xy or not end_xy:
                    # No coordinates (shouldn't happen); fallback
                    fb = sc or ec
                    if fb:
                        county_vmt[fb] = county_vmt.get(fb, 0.0) + link_vmt
                    continue
                
                line_geom = LineString([start_xy, end_xy])
                
                # Candidate counties whose bounding box overlaps the link
                candidates = list(county_sindex.intersection(line_geom.bounds))
                
                if not candidates:
                    fb = sc or ec
                    if fb:
                        county_vmt[fb] = county_vmt.get(fb, 0.0) + link_vmt
                    continue
                
                # Clip the link by each candidate county polygon
                segments = {}  # county_name -> clipped UTM length (metres)
                for idx in candidates:
                    clipped = line_geom.intersection(counties.geometry.iloc[idx])
                    if clipped.is_empty:
                        continue
                    seg_m = utm_length(clipped)
                    if seg_m > 0:
                        cname = counties.iloc[idx]['NAME']
                        segments[cname] = segments.get(cname, 0.0) + seg_m
                
                if segments:
                    total_seg = sum(segments.values())
                    for cname, seg_m in segments.items():
                        proportion = seg_m / total_seg
                        county_vmt[cname] = county_vmt.get(cname, 0.0) + link_vmt * proportion
                    if len(segments) > 1:
                        cross_county_links += 1
                else:
                    # Intersection was empty (e.g. link on exact boundary)
                    fb = sc or ec
                    if fb:
                        county_vmt[fb] = county_vmt.get(fb, 0.0) + link_vmt
            
            print(f"  Links spanning multiple counties: {cross_county_links:,}")
            
            # --- Step 5: Print & save results ---
            sorted_counties = sorted(county_vmt.items(), key=lambda x: -x[1])
            total_county_vmt = sum(county_vmt.values())
            
            print(f"\n{'County':<25} {'VMT':>15} {'% of Total':>12}")
            print("-" * 52)
            for cname, cvmt in sorted_counties:
                pct = cvmt / total_county_vmt * 100 if total_county_vmt > 0 else 0
                print(f"{cname:<25} {cvmt:>15,.2f} {pct:>11.2f}%")
            print("-" * 52)
            print(f"{'TOTAL':<25} {total_county_vmt:>15,.2f} {100.0:>11.2f}%")
            
            # Save to CSV
            csv_path = os.path.join(data_dir if data_dir else os.path.join('data', net_name), 'county_vmt.csv')
            with open(csv_path, 'w') as f:
                f.write('county,vmt,pct_of_total\n')
                for cname, cvmt in sorted_counties:
                    pct = cvmt / total_county_vmt * 100 if total_county_vmt > 0 else 0
                    f.write(f'{cname},{cvmt:.2f},{pct:.4f}\n')
            print(f"\nCounty VMT saved to: {csv_path}")
        else:
            if not os.path.exists(county_shp):
                print(f"\n  County shapefile not found at: {county_shp}")
                print("  Skipping county-level VMT. Download MN counties to mn_counties/")
            if not os.path.exists(node_file):
                print(f"\n  Node file not found at: {node_file}")
    except Exception as e:
        print(f"\n  Warning: County VMT extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'tstt': tstt,
        'sptt': sptt,
        'gap': gap,
        'total_demand': network.TD,
        'assignment_time': assignment_time,
        'network': network
    }


def main():
    parser = argparse.ArgumentParser(description='Run traffic assignment on a network')
    parser.add_argument('--net', '-n', type=str, default='Minnesota',
                        help='Network name (default: Minnesota)')
    parser.add_argument('--year', '-y', type=str, default=None,
                        choices=['2017', '2019', '2024'],
                        help='Year to run (2017, 2019, or 2024). Uses NetworkFiles{year}_WithZones_Simplified folder.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Custom folder containing network.txt/net.txt, trips.txt, and nodes.txt. Overrides --year.')
    parser.add_argument('--ins', '-i', type=str, default='net',
                        help='Instance file name without .txt (default: net)')
    parser.add_argument('--algorithm', '-a', type=str, 
                        default='msa', help='Assignment algorithm (default: msa)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Maximum iterations (default: 10)')
    parser.add_argument('--gap', '-g', type=float, default=1e-4,
                        help='Convergence gap threshold (default: 1e-4)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cached Dijkstra results and recompute from scratch')
    parser.add_argument('--warm-start-link-flows', action='store_true',
                        help='Initialize aggregate flows from link_flows.csv in the data directory and continue with MSA')
    parser.add_argument('--warm-start-file', type=str, default=None,
                        help='Custom link_flows.csv to use for aggregate MSA warm start')
    parser.add_argument('--start-iteration', type=int, default=1,
                        help='First MSA iteration number for warm-start continuation, e.g. 4 after a 3-iteration run')
    parser.add_argument('--tapas-equilibrate-iter', type=int, default=None,
                        help='TAPAS/iTAPAS active-PAS shift passes per outer iteration. Default uses Params.py.')
    parser.add_argument('--tapas-old-origin-shifts', action='store_true',
                        help='Restore the previous TAPAS behavior that shifts relevant PASs inside each origin loop.')
    parser.add_argument('--itapas-max-mfs-per-link', type=int, default=None,
                        help='Maximum MFS attempts for a problematic link in iTAPAS.')
    parser.add_argument('--itapas-cost-mu', type=float, default=None,
                        help='iTAPAS cost effectiveness factor mu. Default uses Params.py.')
    parser.add_argument('--itapas-flow-mu', type=float, default=None,
                        help='iTAPAS flow effectiveness factor nu. Default uses Params.py.')
    parser.add_argument('--itapas-no-newton', action='store_true',
                        help='Use robust bisection instead of Newton-style single-origin PAS shifts in iTAPAS.')
    parser.add_argument('--itapas-no-cycle-removal', action='store_true',
                        help='Skip the existing cycle-removal pass before each iTAPAS origin scan.')
    
    args = parser.parse_args()
    
    # Change to TrafficAssigment directory so relative paths work
    os.chdir(traffic_assign_path)
    
    # Resolve data directory from --data-dir or --year
    data_dir = None
    if args.data_dir:
        base_dir = os.path.dirname(traffic_assign_path)
        data_dir = args.data_dir
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(base_dir, data_dir)
    elif args.year:
        base_dir = os.path.dirname(traffic_assign_path)
        data_dir = os.path.join(base_dir, f'NetworkFiles{args.year}_WithZones_Simplified')
    
    if data_dir:
        if not os.path.isdir(data_dir):
            print(f"Error: Directory not found: {data_dir}")
            sys.exit(1)
        if args.ins == 'net' and not os.path.exists(os.path.join(data_dir, 'net.txt')):
            if os.path.exists(os.path.join(data_dir, 'network.txt')):
                args.ins = 'network'
                print("net.txt not found; using network.txt instead.")
            else:
                print(f"Error: Neither net.txt nor network.txt found in: {data_dir}")
                sys.exit(1)
        if not os.path.exists(os.path.join(data_dir, 'trips.txt')):
            print(f"Error: trips.txt not found in: {data_dir}")
            print(f"Run: python3 gettingTripsData.py --year {args.year}")
            sys.exit(1)
        print(f"Using data directory: {data_dir}")

    warm_start_file = args.warm_start_file
    if warm_start_file and not os.path.isabs(warm_start_file):
        base_dir = os.path.dirname(traffic_assign_path)
        warm_start_file = os.path.join(base_dir, warm_start_file)
    elif args.warm_start_link_flows:
        warm_start_base = data_dir if data_dir else os.path.join(traffic_assign_path, 'data', args.net)
        warm_start_file = os.path.join(warm_start_base, 'link_flows.csv')
    
    try:
        results = run_traffic_assignment(
            net_name=args.net,
            ins=args.ins,
            algorithm=args.algorithm,
            max_iter=args.iterations,
            gap_threshold=args.gap,
            clear_cache=args.clear_cache,
            data_dir=data_dir,
            warm_start_file=warm_start_file,
            start_iteration=args.start_iteration,
            tapas_equilibrate_iter=args.tapas_equilibrate_iter,
            tapas_old_origin_shifts=args.tapas_old_origin_shifts,
            itapas_max_mfs_per_link=args.itapas_max_mfs_per_link,
            itapas_cost_mu=args.itapas_cost_mu,
            itapas_flow_mu=args.itapas_flow_mu,
            itapas_use_newton_shift=not args.itapas_no_newton,
            itapas_remove_cycles=not args.itapas_no_cycle_removal
        )
        
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable networks in data/ folder:")
        data_path = os.path.join(traffic_assign_path, 'data')
        if os.path.exists(data_path):
            for d in os.listdir(data_path):
                if os.path.isdir(os.path.join(data_path, d)):
                    print(f"  - {d}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
