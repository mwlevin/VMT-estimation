"""Quick analysis of net.txt connectivity"""
from collections import defaultdict, deque

net_file = '/Users/nanka005/Library/Mobile Documents/com~apple~CloudDocs/VMTProjectMndot/TrafficAssigment/data/Minnesota/net.txt'

connector_count = 0
zone_nodes_with_connectors = set()
all_nodes = set()
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
    if len(parts) >= 10:
        init_n, term_n = int(parts[0]), int(parts[1])
        link_type = int(float(parts[9]))
        all_nodes.add(init_n)
        all_nodes.add(term_n)
        adj[init_n].add(term_n)
        adj[term_n].add(init_n)
        if link_type == 9:
            connector_count += 1
            if init_n <= 1000: zone_nodes_with_connectors.add(init_n)
            if term_n <= 1000: zone_nodes_with_connectors.add(term_n)

print(f"Total type-9 connector links: {connector_count}")
print(f"Unique zone nodes (1-1000) with connectors: {len(zone_nodes_with_connectors)}")
print(f"Zone nodes with NO connectors: {1000 - len(zone_nodes_with_connectors)}")
print(f"Total unique nodes in links: {len(all_nodes)}")
print(f"Node IDs 1-1000 in links: {len([n for n in all_nodes if n <= 1000])}")

# Check missing zones
missing = sorted(set(range(1,1001)) - zone_nodes_with_connectors)
print(f"\nMissing zone nodes (no type-9 connectors): {len(missing)}")
if missing:
    print(f"  First 30: {missing[:30]}")

# BFS connectivity check
print("\nRunning BFS connectivity check...")
visited = set()
components = []
for start in all_nodes:
    if start in visited:
        continue
    comp = set()
    q = deque([start])
    while q:
        n = q.popleft()
        if n in visited:
            continue
        visited.add(n)
        comp.add(n)
        for nb in adj[n]:
            if nb not in visited:
                q.append(nb)
    components.append(comp)

sizes = sorted([len(c) for c in components], reverse=True)
print(f"Connected components: {len(components)}")
print(f"Top 10 component sizes: {sizes[:10]}")
if sizes:
    print(f"Largest component: {sizes[0]} nodes ({100*sizes[0]/len(all_nodes):.1f}%)")

# Find the main component
main_comp = max(components, key=len)

# How many zone nodes (1-1000) are in the main component?
zones_in_main = [n for n in range(1,1001) if n in main_comp]
zones_not_in_main = [n for n in range(1,1001) if n not in main_comp and n in all_nodes]
zones_not_in_network = [n for n in range(1,1001) if n not in all_nodes]

print(f"\nZone nodes (1-1000) in main component: {len(zones_in_main)}")
print(f"Zone nodes (1-1000) NOT in main component: {len(zones_not_in_main)}")
print(f"Zone nodes (1-1000) NOT in network at all: {len(zones_not_in_network)}")
if zones_not_in_main:
    print(f"  Disconnected zone IDs: {zones_not_in_main[:30]}")
if zones_not_in_network:
    print(f"  Missing zone IDs: {zones_not_in_network[:30]}")

# Check node 996 specifically (it appeared in nodes.txt out of order)
print(f"\nNode 996 in network? {996 in all_nodes}")
print(f"Node 996 neighbors: {adj.get(996, set())}")
print(f"Node 996 in main component? {996 in main_comp}")
