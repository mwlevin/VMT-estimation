"""Deeper analysis of the 65% unreachable demand issue"""
from collections import defaultdict, deque

net_file = '/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/TrafficAssigment/data/Minnesota/net.txt'
trips_file = '/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/TrafficAssigment/data/Minnesota/trips.txt'
nodes_file = '/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/TrafficAssigment/data/Minnesota/nodes.txt'

# Parse net.txt
adj = defaultdict(set)
in_data = False
for line in open(net_file):
    s = line.strip()
    if '<END OF METADATA>' in s:
        in_data = True
        continue
    if not in_data or not s or s.startswith('~'):
        continue
    parts = s.replace(';','').split()
    if len(parts) >= 2:
        init_n, term_n = int(parts[0]), int(parts[1])
        adj[init_n].add(term_n)
        # Note: TNTP links are directional, but for connectivity check we use undirected
        adj[term_n].add(init_n)

all_nodes = set(adj.keys())

# BFS to find components
visited = set()
components = []
node_to_comp = {}
for start in all_nodes:
    if start in visited:
        continue
    comp_id = len(components)
    comp = set()
    q = deque([start])
    while q:
        n = q.popleft()
        if n in visited:
            continue
        visited.add(n)
        comp.add(n)
        node_to_comp[n] = comp_id
        for nb in adj[n]:
            if nb not in visited:
                q.append(nb)
    components.append(comp)

main_comp = max(components, key=len)
main_comp_id = components.index(main_comp)

# Parse trips to compute demand per O-D pair
print("Parsing trips.txt...")
total_demand = 0
origin_demand = defaultdict(float)
dest_demand = defaultdict(float)
unreachable_demand = 0
reachable_demand = 0
missing_zone_demand = 0

current_origin = None
for line in open(trips_file):
    s = line.strip()
    if s.startswith('<') or not s:
        continue
    if s.startswith('Origin'):
        current_origin = int(s.split()[1])
        continue
    if current_origin is not None and ':' in s:
        # Parse "dest : demand ; dest : demand ;" format
        import re
        pairs_raw = re.findall(r'(\d+)\s*:\s*([\d.]+)', s)
        for dest_s, demand_s in pairs_raw:
            dest = int(dest_s)
            demand = float(demand_s)
            total_demand += demand
            origin_demand[current_origin] += demand
            dest_demand[dest] += demand
            
            o_in_net = current_origin in all_nodes
            d_in_net = dest in all_nodes
            
            if not o_in_net or not d_in_net:
                missing_zone_demand += demand
            elif node_to_comp.get(current_origin) != node_to_comp.get(dest):
                unreachable_demand += demand
            else:
                reachable_demand += demand

print(f"\n===== DEMAND ANALYSIS =====")
print(f"Total demand: {total_demand:.1f}")
print(f"Reachable demand (same component): {reachable_demand:.1f} ({100*reachable_demand/total_demand:.1f}%)")
print(f"Unreachable demand (diff components): {unreachable_demand:.1f} ({100*unreachable_demand/total_demand:.1f}%)")
print(f"Missing zone demand (zone not in network): {missing_zone_demand:.1f} ({100*missing_zone_demand/total_demand:.1f}%)")

# Which zones have biggest demand but are missing from network?
print(f"\n===== ZONES NOT IN NETWORK =====")
missing_zones = sorted(set(range(1,1001)) - all_nodes)
print(f"Total missing zones: {len(missing_zones)}")
for z in missing_zones:
    od = origin_demand.get(z, 0)
    dd = dest_demand.get(z, 0)
    if od + dd > 0:
        print(f"  Zone {z}: origin_demand={od:.1f}, dest_demand={dd:.1f}")

# Which zones have connectors but are in small disconnected components (not main)?
print(f"\n===== ZONES IN DISCONNECTED COMPONENTS (NOT MAIN) =====")
zones_disconnected = []
for z in range(1, 1001):
    if z in all_nodes and z not in main_comp:
        comp_id = node_to_comp[z]
        comp_size = len(components[comp_id])
        od = origin_demand.get(z, 0) + dest_demand.get(z, 0)
        zones_disconnected.append((z, comp_size, od))
if zones_disconnected:
    for z, cs, od in sorted(zones_disconnected):
        print(f"  Zone {z}: component size={cs}, total demand={od:.1f}")
else:
    print("  None - all zones in network are in the main component")

# Summary: where does the 65% unreachable come from?
print(f"\n===== SUMMARY =====")
print(f"The issue is that {len(missing_zones)} zone nodes (out of 1000) do not appear in net.txt at all.")
print(f"This accounts for {missing_zone_demand:.1f} demand ({100*missing_zone_demand/total_demand:.1f}%).")
print(f"Disconnected component demand accounts for {unreachable_demand:.1f} ({100*unreachable_demand/total_demand:.1f}%).")
print(f"Combined unreachable: {missing_zone_demand + unreachable_demand:.1f} ({100*(missing_zone_demand+unreachable_demand)/total_demand:.1f}%)")

# Check: node 996 appears in nodes.txt between nodes 8 and 10 (out of order). 
# Read nodes.txt to check node ordering
print(f"\n===== NODE 996 INVESTIGATION =====")
node_996_data = None
for line in open(nodes_file):
    if '996' in line and not line.startswith('~'):
        parts = line.strip().replace(';','').split()
        if len(parts) >= 4 and parts[0] == '996':
            node_996_data = parts
            break
if node_996_data:
    print(f"Node 996 in nodes.txt: new_id={node_996_data[0]}, orig_id={node_996_data[1]}, x={node_996_data[2]}, y={node_996_data[3]}")
    print(f"  This is the TwinCities super node (orig_id=1, x=-93.17, y=44.96)")
    print(f"  Node 996 has {len(adj[996])} neighbors in net.txt")
