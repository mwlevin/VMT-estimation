import os
import pickle
import numpy as np

from src import Node
from src import Link
from src import Path
from src import Zone
from src import Bush
from src import Params
from src import PASList
from src import Heap

class Network:

    # construct this Network with the name; read files associated with network name
    def __init__(self,name,ins,B_prop,scal_time,scal_flow,inflate_trips,data_dir=None):
        self.nodes = []
        self.links = []
        self.zones = []
        self.origins = []
        
        self.links2 = []
        self.type = 'UE'
        self.TD = 0
        self.TC = 0 # total cost
        self.params = Params.Params()
        
        self.ins = ins
        
        self.allPAS = PASList.PASList()
        
        self.inf = 1e+9
        self.tol = 1e-2
        
        if len(ins) == 0:
            ins = "net"
        
        # Use custom data_dir if provided, otherwise default to data/{name}
        if data_dir is not None:
            base = data_dir
        else:
            base = "data/" + name
        
        if name == "SiouxFalls2":
            self.readNetwork2(base+"/net2.txt",scal_time,scal_flow)
            self.readTrips2(base+"/demand2.txt",scal_time,scal_flow,inflate_trips)
        else:
            self.readNetwork(base+"/"+ins+".txt",scal_time,scal_flow)
            self.readTrips(base+"/trips.txt",scal_time,scal_flow,inflate_trips)
        
        self.links1 = []
        
        for a in self.links:
            if a not in self.links2:
                self.links1.append(a)
        
        self.B = self.TC * B_prop # budget
        
        # Build link lookup by ID for fast cache restoration
        self.link_by_id = {}
        for a in self.links:
            self.link_by_id[a.id] = a
        
        # Dijkstra cache: per-origin files saved to a directory.
        # Each origin's SP tree is stored as two numpy arrays (costs + pred_link_ids)
        # so we never hold more than one origin's data in memory at a time.
        self._cache_dir = base + "/dijkstra_cache"
        self._num_nodes = len(self.nodes)
        self._init_dijkstra_cache()
        
        #print('Total scaled demand %.1f' % self.TD)
        #print('Total cost %.1f - Budget %.1f' % (self.TC, self.B))
        
    def _init_dijkstra_cache(self):
        """Check how many cached origin files exist."""
        if os.path.isdir(self._cache_dir):
            cached = [f for f in os.listdir(self._cache_dir) if f.endswith('.npz')]
            print(f"  Dijkstra cache found: {len(cached)} origins cached in {self._cache_dir}/")
        else:
            print(f"  No Dijkstra cache found (will compute fresh)")
    
    def _cache_path(self, origin_id):
        """Return the file path for a specific origin's cache."""
        return os.path.join(self._cache_dir, f"origin_{origin_id}.npz")
    
    def has_dijkstra_cache(self, origin):
        """Check if we have a cached file for this origin."""
        return os.path.exists(self._cache_path(origin.id))
    
    def cache_dijkstra_result(self, origin):
        """Save this origin's Dijkstra result to disk immediately as numpy arrays.
        
        NOTE: This cache is ONLY valid for the initial bush building (loadDemand),
        where all link flows = 0 and travel times = free-flow times.
        During iterations, getSPTT() and other methods call dijkstras() directly
        with current congested travel times — they do NOT use this cache.
        
        Stores two arrays indexed by node position (0..N-1):
          costs[i]    = shortest path cost to node i+1
          pred_ids[i] = predecessor link ID for node i+1 (-1 if none)
        Each file is ~16 MB for 1.3M nodes. Only one is in memory at a time.
        """
        N = self._num_nodes
        costs = np.empty(N, dtype=np.float64)
        pred_ids = np.empty(N, dtype=np.int32)
        
        for i, n in enumerate(self.nodes):
            costs[i] = n.cost
            pred_ids[i] = n.pred.id if n.pred is not None else -1
        
        os.makedirs(self._cache_dir, exist_ok=True)
        np.savez(self._cache_path(origin.id), costs=costs, pred_ids=pred_ids)
    
    def restore_dijkstra_from_cache(self, origin):
        """Load cached Dijkstra result from disk and set node costs/preds. Returns True on success."""
        cache_file = self._cache_path(origin.id)
        if not os.path.exists(cache_file):
            return False
        
        try:
            data = np.load(cache_file)
            costs = data['costs']
            pred_ids = data['pred_ids']
            
            for i, n in enumerate(self.nodes):
                n.cost = float(costs[i])
                pid = int(pred_ids[i])
                if pid >= 0 and pid in self.link_by_id:
                    n.pred = self.link_by_id[pid]
                else:
                    n.pred = None
            
            return True
        except Exception as e:
            print(f"  Warning: Could not load cache for origin {origin.id}: {e}")
            return False

    def setType(self, type):
        self.type = type
        
    # read file "/net.txt"
    def readNetwork2(self,netFile,scal_time,scal_flow):
        
        firstThruNode = 1
        numZones = 0
        numNodes = 0
        numLinks = 0
        newLinks = 0
        file = open(netFile, "r")

        line = ""
        
        numZones = 24
        numNodes = 24
        numLinks = 76
        
        for i in range(0, numZones):
            self.zones.append(Zone.Zone(i + 1))

        for i in range(0, numNodes):
            if i < numZones:
                self.nodes.append(self.zones[i])
                
                if i + 1 < firstThruNode:
                    self.zones[i].setThruNode(False)

            else:
                self.nodes.append(Node.Node(i + 1))

        line = ""
        id = 0
        
        print(numLinks)
        while len(line) == 0:
            line = file.readline().strip()

        for i in range(0, numLinks + newLinks):
            line = file.readline().split()
            if len(line) == 0:
                continue
            start = self.nodes[int(line[1]) - 1]
            end = self.nodes[int(line[2]) - 1]
            C = float(line[5])
            


            t_ff = float(line[3]) 
            alpha = float(line[4])
            beta = float(line[6])
            
   
            if len(line) > 7:
                cost = float(line[7])
                print(cost)
            else:
                cost = 0
                
            #print(start, end, cost, line)
            
            self.TC += cost
            
            link = Link.Link(id, start ,end, t_ff, C, alpha, beta, cost)
            id = id +1
            #print(start,end)
            self.links.append(link)
            #print(self.links)
            
            if i >= numLinks:
                self.links2.append(link)
            
        file.close()


    def readTrips2(self,tripsFile,scal_time,scal_flow,inflate_trips):
        
        file = open(tripsFile, "r")
        
        lines = file.readlines()
        
        line_idx = 0

            
        r = None
        
        idx = 0
        
        
        #print(splitted)
        
        for i in range(1, 24):
            splitted = lines[line_idx].split()
            next = splitted[i-1]
            
            r = self.zones[i-1]

            for j in range(1, 24):
                s = self.zones[j-1]
                d = float(splitted[j-1])
                
                
                r.addDemand(s, d)
                self.TD += d

        file.close()
        
        for r in self.zones:
            if r.getProductions() > self.params.flow_epsilon:
                self.origins.append(r)

    def getLinks(self):
        return self.links
    
    def getNodes(self):
        return self.nodes
    
    def getZones(self):
        return self.zones

    # find the node with the given id
    def findNode(self, id):
        if id <= 0 or id > len(self.nodes):
            return None
        return self.nodes[id - 1]

    # find the link with the given start and end nodes
    def findLink(self, i, j):
        if i is None or j is None:
            return None

        for link in i.outgoing:
            if link.end == j:
                return link

        return None
    

    def dijkstras(self, origin, type):
        
            for n in self.nodes:
                n.cost = Params.INFTY
                n.pred = None

            origin.cost = 0.0

            Q = Heap.Heap()
            Q.insert(origin)
            
            #if type == 'RC':
            #    print('origin',origin)

            while Q.size() > 0:

                u = Q.removeMin()
                
                #if type == 'RC':
                #    print('u',u)

                for uv in u.outgoing:
                    v = uv.end
                    tt = uv.getTravelTime(uv.x, type)

                    #if u.cost + tt < v.cost:
                    if u.cost + tt < v.cost and v.cost - u.cost - tt >= self.params.SP_tol:
                        v.cost = u.cost + tt
                        v.pred = uv
                        
                        #if type == 'RC':
                        #    print('v',v,v.pred,v.cost)

                        if v.isThruNode():
                            Q.insert(v)
            

    def trace(self, r, s):
        curr = s

        output = Path.Path()
        output.r = r
        output.s = s
        
        while curr != r and curr is not None:
            ij = curr.pred

            if ij is not None:
                output.add(ij)
                curr = curr.pred.start
              
        #print('trace',r,s,output)
              
        return output
        
    def traceTree(self, tree, r, s):
            curr = s

            output = []

            while curr != r and curr is not None:
                ij = tree[curr]

                if ij is not None:
                    output.append(ij)
                    curr = ij.start
            
            return output

    def getSPTree(self, r):
        self.dijkstras(r, self.type)
        output = {}
        for n in self.nodes:
            if n != r and n.cost < Params.INFTY:
                output[n] = n.pred
        
        return output
    
    # returns the total system travel time
    def getTSTT(self, type):
        output = 0.0
        for ij in self.links:
            #if ij.y == 1 or ij.x > len(self.origins) * self.params.flow_epsilon:
            if ij.y == 1:
                tt = ij.getTravelTime(ij.x, type)
                output += ij.x * tt
                
        return output
    
    def validateLinkFlows(self):
        output = True
        for ij in self.links:
            totbushflow = 0
            
            for r in self.origins:
                totbushflow += r.bush.getFlow(ij)
            
            if abs(ij.x - totbushflow) > self.params.flow_epsilon:
                #print(ij, ij.x, totbushflow, ij.x-totbushflow, ij.getTravelTime(ij.x, self.type))
                output = False
        return output
            
    
    # returns the total system travel time if all demand is on the shortest path
    # NOTE: This always runs fresh Dijkstra with CURRENT link travel times (congested),
    # NOT from cache. The cache is only for initial bush building (free-flow).
    def getSPTT(self, type):
        output = 0.0

        for r in self.origins:
            self.dijkstras(r, type)  # Fresh Dijkstra with current flows, NOT from cache

            for s in self.zones:
                if r.getDemand(s) > 0:
                    if s.cost >= Params.INFTY:
                        continue  # Skip unreachable destinations
                    output += r.getDemand(s) * s.cost

        return output

    # returns the total number of trips in the network
    def getTotalTrips(self):
        output = 0.0

        for r in self.origins:
            output += r.getProductions()

        return output

    # returns the average excess cost
    def getAEC(self):
        return (self.getTSTT() - self.getSPTT()) / self.getTotalTrips()
    
    # returns the UE TAP objective function value
    def getBeckmannOFV(self):
        output = 0.0
        for a in self.links:
            if a.y == 1:
                output += a.getPrimitiveTravelTime(a.x)
                
        return output

    # find the step size for the given iteration number
    def calculateStepsize(self, iteration):
        return 1.0 / iteration
        #print(1.0 / iteration)


    # calculate the new X for all links based on the given step size
    def calculateNewX(self, stepsize):
        for ij in self.links:
            ij.calculateNewX(stepsize)


    # calculate the all-or-nothing assignment
    def calculateAON(self):
        for r in self.origins:
            self.dijkstras(r, self.type)

            for s in self.zones:
                if r.getDemand(s) > 0:
                    pi_star = self.trace(r,s)
                    pi_star.addHstar(r.getDemand(s))
                    
    def setAON(self, type):
        self.setType(type)

        for r in self.origins:
            self.dijkstras(r, type)

            for s in self.zones:
                if r.getDemand(s) > 0:
                    pi_star = self.trace(r,s)
                    pi_star.addHstar(r.getDemand(s))
                    
        for ij in self.links:
            ij.calculateNewX(1)
            
        return self.getTSTT('UE')

    # ITAPAS_CHANGE_2026_06_13_F_ITAPAS_ALGORITHM:
    # iTAPAS is implemented beside TAPAS. It uses single-origin PAS ownership,
    # maximum-flow-first PAS identification in Bush.checkITAPAS(), shift-first
    # PAS storage, dynamic PAS shift skipping, and postposes proportionality.
    def itapas(self, type):
        
        if not self.params.warmstart:
            self.resetTapas()
            
        self.setType(type)
        self.params.pas_use_dynamic_cost_epsilon = True
        
        iter = 1
        max_iter = self.params.tapas_max_iter
        
        if type == 'UE' or type == 'SO':
            min_gap = self.params.min_gap
        elif type == 'SO_OA_cuts':
            min_gap = self.params.min_gap_SO_OA_cuts
        
        if self.params.PRINT_TAP_ITER:
            print("Iteration\tTSTT\tSPTT\tgap\tAEC")
            
        last_iter_gap = 1
        
        import time as _time
        import sys as _sys
        num_origins = len(self.origins)
        print(f"\nBuilding initial bushes for {num_origins} origins...")
        _bush_start = _time.time()
        _cache_hits = 0
        _cache_misses = 0
        for oi, r in enumerate(self.origins):
            if r.bush == None:
                was_cached = self.has_dijkstra_cache(r)
                r.bush = Bush.Bush(self, r)
                if was_cached:
                    _cache_hits += 1
                else:
                    _cache_misses += 1
            if (oi + 1) % 10 == 0 or oi == 0 or (oi + 1) == num_origins:
                elapsed = _time.time() - _bush_start
                rate = (oi + 1) / elapsed if elapsed > 0 else 0
                remaining = (num_origins - oi - 1) / rate if rate > 0 else 0
                _sys.stdout.write(f"\r  Bush {oi+1}/{num_origins} ({(oi+1)/num_origins*100:.1f}%) | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining | cache: {_cache_hits} hits, {_cache_misses} computed")
                _sys.stdout.flush()
        print(f"\n  All bushes built in {_time.time() - _bush_start:.1f}s (cache: {_cache_hits} hits, {_cache_misses} computed)")
        
        if hasattr(self, 'unreachable_pairs') and self.unreachable_pairs:
            unique = set((p['origin'], p['destination']) for p in self.unreachable_pairs)
            lost = sum(p['demand'] for p in self.unreachable_pairs)
            print(f"\n  ⚠ Summary: {len(unique)} unreachable O-D pairs, {lost:.2f} lost demand ({lost/self.TD*100:.1f}% of total)")
        
        while iter <= max_iter:
            self.params.good_pas_flow_mu = 0
            self.params.good_pas_cost_mu = 0
            self.params.good_bush_gap = 0
            self.params.good_pas_cost_epsilon = 0
            self.params.current_gap = last_iter_gap
            self.params.current_iter = iter
            
            _iter_start = _time.time()
            for ri, r in enumerate(self.origins):
                if (ri + 1) % 50 == 0 or ri == 0:
                    _sys.stdout.write(f"\r  iTAPAS iter {iter}: processing origin {ri+1}/{num_origins}...")
                    _sys.stdout.flush()
            
                if self.params.itapas_remove_cycles:
                    r.bush.removeCycles()
                else:
                    r.bush.topologicalSort()
                
                r.bush.checkITAPAS()

            if self.params.PRINT_TAP_ITER:
                print(f"\r  iTAPAS iter {iter}: scanned {num_origins} origins; active PAS={self.allPAS.count_unique()}")
                               
            modified = False
            for shiftIter in range(0, self.params.itapas_equilibrate_iter):
                self.removeITAPAS(iter)
                modified = self.equilibrateITAPAS(iter)
                            
                if not modified:
                    break
                            
            tstt = self.getTSTT(type)
            sptt = self.getSPTT(type)
            gap = (tstt - sptt)/tstt
            aec = (tstt - sptt)/self.TD
            self.params.current_gap = gap
                            
            _iter_time = _time.time() - _iter_start
            if self.params.PRINT_TAP_ITER:
                print(f"\r  iTAPAS iter {iter} done in {_iter_time:.1f}s")
                print(f"  TSTT={tstt:.2f}  SPTT={sptt:.2f}  Gap={gap:.6f}  AEC={aec:.6f}")

            if gap < min_gap:
                break

            if (last_iter_gap - gap) < min_gap:
                self.params.line_search_gap = max(self.params.line_search_gap/10, self.params.min_line_search_gap)
                self.params.itapas_min_flow = max(self.params.itapas_min_flow/10, self.params.flow_epsilon)
                self.params.itapas_min_reduced_cost = max(self.params.itapas_min_reduced_cost/10, 1e-12)
            
            last_iter_gap = gap
            iter += 1
        
        for a in self.links:
            a.x = round(a.x,self.params.rd)
            
        return self.getTSTT('UE')
    
    def getFlowMap(self):
        output = {}
        
        for a in self.links:
            output[a] = a.x
        
        return output
        
    def setFlows(self, flowmap):
        for a in self.links:
            a.x = flowmap[a]
        

    def msa(self, type):
        self.setType(type)
        
        max_iteration = self.params.msa_max_iter
        min_gap = self.params.min_gap
        
       
        if self.params.PRINT_TAP_ITER:
            print("Iteration\tTSTT\tSPTT\tgap\tAEC")
        
        
        start_iteration = getattr(self.params, 'msa_start_iteration', 1)
        for iteration in range(start_iteration, max_iteration + 1):
            self.calculateAON()
            stepsize = self.calculateStepsize(iteration)
            
            self.calculateNewX(stepsize)
            
            tstt = self.getTSTT(self.type)
            sptt = self.getSPTT(self.type)
            gap = (tstt - sptt)/tstt
            aec = (tstt - sptt)/self.TD
            
            if self.params.PRINT_TAP_ITER:
                print(str(iteration)+"\t"+str(tstt)+"\t"+str(sptt)+"\t"+str(gap)+"\t"+str(aec))
                
            if gap < min_gap:
                break
        
        return self.getTSTT('UE')
        
    def resetTapas(self):
        for r in self.origins:
            r.bush = None
            
        for a in self.links:
            a.x = 0
                
        self.params = Params.Params()
        
        self.allPAS = PASList.PASList()
    
    def tapas(self, type):
        
        if not self.params.warmstart:
            self.resetTapas()
            
        self.setType(type)
        
        iter = 1
        max_iter = self.params.tapas_max_iter
        
        if type == 'UE' or type == 'SO':
            min_gap = self.params.min_gap
        elif type == 'SO_OA_cuts':
            min_gap = self.params.min_gap_SO_OA_cuts
        
        #self.params.line_search_gap = pow(10, math.floor(math.log10(self.TD) - 6))
        
        if self.params.PRINT_TAP_ITER:
            print("Iteration\tTSTT\tSPTT\tgap\tAEC")
            
        last_iter_gap = 1
        
        import time as _time
        import sys as _sys
        num_origins = len(self.origins)
        print(f"\nBuilding initial bushes for {num_origins} origins...")
        _bush_start = _time.time()
        _cache_hits = 0
        _cache_misses = 0
        for oi, r in enumerate(self.origins):
            if r.bush == None:
                was_cached = self.has_dijkstra_cache(r)
                r.bush = Bush.Bush(self, r)
                if was_cached:
                    _cache_hits += 1
                else:
                    _cache_misses += 1
            if (oi + 1) % 10 == 0 or oi == 0 or (oi + 1) == num_origins:
                elapsed = _time.time() - _bush_start
                rate = (oi + 1) / elapsed if elapsed > 0 else 0
                remaining = (num_origins - oi - 1) / rate if rate > 0 else 0
                _sys.stdout.write(f"\r  Bush {oi+1}/{num_origins} ({(oi+1)/num_origins*100:.1f}%) | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining | cache: {_cache_hits} hits, {_cache_misses} computed")
                _sys.stdout.flush()
        print(f"\n  All bushes built in {_time.time() - _bush_start:.1f}s (cache: {_cache_hits} hits, {_cache_misses} computed)")
        
        # No bulk save needed — each origin is saved to disk immediately after computing
        
        # Print unreachable summary after all bushes built
        if hasattr(self, 'unreachable_pairs') and self.unreachable_pairs:
            unique = set((p['origin'], p['destination']) for p in self.unreachable_pairs)
            lost = sum(p['demand'] for p in self.unreachable_pairs)
            print(f"\n  ⚠ Summary: {len(unique)} unreachable O-D pairs, {lost:.2f} lost demand ({lost/self.TD*100:.1f}% of total)")
        
        while iter <= max_iter:
                        
            #custom_x = {link: link.x for link in self.links}
            #print(custom_x)
            
            self.params.good_pas_flow_mu = 0
            self.params.good_pas_cost_mu = 0
            self.params.good_bush_gap = 0
            self.params.good_pas_cost_epsilon = 0
            
            # for every origin
            _iter_start = _time.time()
            for ri, r in enumerate(self.origins):
                if (ri + 1) % 50 == 0 or ri == 0:
                    _sys.stdout.write(f"\r  Iter {iter}: processing origin {ri+1}/{num_origins}...")
                    _sys.stdout.flush()
            
                # remove all cyclic flows and topological sort
                if self.params.PRINT_TAPAS_INFO:
                    print("removing cycles", r)
                    
                r.bush.removeCycles()
                # find tree of least cost routes
                            
                if self.params.PRINT_TAPAS_INFO:
                    print("checking for PAS", r)
                                
                r.bush.checkPAS()
                
                if self.params.PRINT_PAS_INFO:
                    print("num PAS", r, r.bush.relevantPAS.count_unique())
                # for every link used by the origin which is not part of the tree
                    # if there is an existing effective PAS
                        # make sure the origin is listed as relevant
                    # else
                        # construct a new PAS
                                    
                # choose a random subset of active PASs
                # shift flow within each chosen PAS
                    
                r.bush.branchShifts()

                # TAPAS_CHANGE_2026_06_13_B_DISABLE_DUPLICATE_ORIGIN_PAS_SHIFTS:
                # The original implementation did the following for every
                # origin:
                #
                #   printed = False
                #   for a in r.bush.relevantPAS.forward:
                #       for p in r.bush.relevantPAS.forward[a]:
                #           if self.params.PRINT_TAPAS_INFO and not printed:
                #               print("initial flow shifts", r)
                #               printed = True
                #           p.flowShift(self.type, self.params)
                #
                # A PAS.flowShift() already considers all relevant origins for
                # that PAS. Running it inside every origin loop can therefore
                # shift the same PAS many times in a single outer iteration.
                # Bar-Gera's TAPAS structure scans origins for PASs, then shifts
                # each active PAS several times. Keep this flag for exact
                # rollback/reproduction of the previous behavior.
                if self.params.tapas_shift_relevant_pas_during_origin_scan:
                    printed = False
                    for a in r.bush.relevantPAS.forward:
                        for p in r.bush.relevantPAS.forward[a]:
                            if self.params.PRINT_TAPAS_INFO and not printed:
                                print("initial flow shifts", r)
                                printed = True
                            p.flowShift(self.type, self.params)
                        
                        # for every active PAS
             
            if self.params.PRINT_TAPAS_INFO:
                print("general flow shifts")

            if self.params.PRINT_TAP_ITER:
                print(f"\r  Iter {iter}: scanned {num_origins} origins; active PAS={self.allPAS.count_unique()}")
                               
            modified = False
            for shiftIter in range(0, self.params.tapas_equilibrate_iter):
                # check if it should be eliminated
                self.removePAS(iter)
                # perform flow shift to equilibrate costs
                modified = self.equilibratePAS(iter)
                # redistribute flows between origins by the proportionality condition
                            
                # in the case that no flow shifting occurred, do not try to equilibrate more
                if not modified:
                    break
                            
            tstt = self.getTSTT(type)
            sptt = self.getSPTT(type)
            gap = (tstt - sptt)/tstt
            aec = (tstt - sptt)/self.TD

            #print(iter, sptt)
                            
            _iter_time = _time.time() - _iter_start
            if self.params.PRINT_TAP_ITER:
                print(f"\r  Iter {iter} done in {_iter_time:.1f}s")
                print(f"  TSTT={tstt:.2f}  SPTT={sptt:.2f}  Gap={gap:.6f}  AEC={aec:.6f}")
                
                #printLinkFlows();

            if gap < min_gap:
                break
                
            # there's an issue where PAS are labeled as not cost effective because the difference in cost is small, less than 5% of the reduced cost
            # for low network gaps, this is causing PAS to not flow shift
            # when the gap is low, increase the flow shift sensitivity
            '''
            if (last_iter_gap - gap) / gap < 0.01:
                print('TAPAS check 1')
                
                #for r in self.origins:
                #    r.bush.algBShift()
                
                self.params.bush_gap = max(self.params.bush_gap/10, 1e-6)
                self.params.pas_cost_mu = max(self.params.pas_cost_mu/10, 1e-6)
                self.params.pas_flow_mu = max(self.params.pas_flow_mu/10, 1e-6)
                self.params.flow_epsilon = max(self.params.flow_epsilon/10, 1e-6)
               
                self.params.line_search_gap = max(self.params.line_search_gap/10, self.params.min_line_search_gap)
                    
                if self.params.PRINT_TAPAS_INFO:
                    print("Adjusting parameters due to small gap "+str(self.params.pas_cost_mu)+" "+str(self.params.line_search_gap))
            '''
            
            
            if (last_iter_gap - gap) < min_gap:
                
                
                self.params.line_search_gap = max(self.params.line_search_gap/10, self.params.min_line_search_gap)
                
                #if self.params.good_pas_cost_mu <= 10 and self.params.pas_cost_mu > 5e-5:
                self.params.pas_cost_mu = max(self.params.pas_cost_mu/10, 1e-6)
                    
                #elif self.params.good_pas_flow_mu <= 10 and self.params.pas_flow_mu > 5e-5:
                self.params.pas_flow_mu = max(self.params.pas_flow_mu/10, 1e-6)
                   
                #elif self.params.good_pas_cost_epsilon <= 10 and self.params.pas_cost_epsilon > 5e-6:
                self.params.pas_cost_epsilon = max(self.params.pas_cost_epsilon/10, 1e-6)
                    
                #elif self.params.bush_gap > 5e-6:
                self.params.bush_gap = max(self.params.bush_gap/10, 1e-6)
                #self.params.resetPAS()
                
                '''
                if self.params.pas_flow_mu > 5e-5:
                    self.params.pas_flow_mu = max(self.params.pas_flow_mu/10, 1e-6)
                    
                elif self.params.pas_cost_epsilon > 5e-6:
                    self.params.pas_cost_epsilon = max(self.params.pas_cost_epsilon/10, 1e-6)
                    
                    self.params.pas_cost_mu = max(self.params.pas_cost_mu/10, 1e-6)
                    
                elif self.params.bush_gap > 5e-6:
                    self.params.bush_gap = max(self.params.bush_gap/10, 1e-6)
                    self.params.resetPAS()
                '''
                if self.params.PRINT_PARAM_ADJ:
                    print('TAPAS gap check', self.params.bush_gap, self.params.pas_cost_mu, self.params.pas_flow_mu, self.params.pas_cost_epsilon)
                    print("\t", self.params.good_bush_gap, self.params.good_pas_cost_mu, self.params.good_pas_flow_mu, self.params.good_pas_cost_epsilon)
                
            last_iter_gap = gap
            iter += 1
        
        #---rounding link flows for numerical stability
        for a in self.links:
            a.x = round(a.x,self.params.rd)
            #print(a.start.id,a.end.id,a.x)
            
        return self.getTSTT('UE')
        
    def findPAS(self, ij, bush):
        
        if not self.allPAS.containsKey(ij):
            return None
        
        #best = None
        #max = self.params.bush_gap
        
        if ij in self.allPAS.backward:
            for p in self.allPAS.backward[ij]:
                '''
                bwdcost = p.getBackwardCost(self.type)
                fwdcost = p.getForwardCost(self.type)
                
                if fwdcost > bwdcost * (1 + self.params.pas_cost_mu):
                    bwdflow = p.maxBackwardBushFlowShift(bush)
                    fwdflow = p.maxForwardBushFlowShift(bush)
                    if fwdflow > max:
                        best = p
                elif bwdcost > fwdcost * (1 + self.params.pas_cost_mu):
                    bwdflow = p.maxBackwardBushFlowShift(bush)
                    fwdflow = p.maxForwardBushFlowShift(bush)
                    if bwdflow > max:
                        best = p
                
                '''
                if p.isEffective(self.type, bush, self.params.pas_cost_mu, bush.origin.getProductions()*self.params.pas_flow_mu, self.params):
                    return p
                        
        if ij in self.allPAS.forward:
            for p in self.allPAS.forward[ij]:
                if p.isEffective(self.type, bush, self.params.pas_cost_mu, bush.origin.getProductions()*self.params.pas_flow_mu, self.params):
                    return p
          
        return None

    # ITAPAS_CHANGE_2026_06_13_G_SINGLE_OWNER_PAS_MATCH:
    # iTAPAS may use an existing PAS ending at the same merge node, but it
    # shifts that PAS only at its owner origin. It never adds the current origin
    # to the PAS relevant-origin set during UE convergence.
    def shiftMatchedITAPAS(self, ij, current_bush):
        reduced_cost = max(ij.getReducedCost(self.type), self.params.flow_epsilon)
        current_link_flow = max(current_bush.getFlow(ij), self.params.flow_epsilon)

        for p in self.allPAS.iter_head(ij.end):
            if p.owner is None or p.owner.bush is None:
                continue

            owner_bush = p.owner.bush
            forwardcost = p.getForwardCost(self.type)
            backwardcost = p.getBackwardCost(self.type)
            matched = False

            if ij in p.backwardlinks:
                matched = (
                    backwardcost - forwardcost > self.params.itapas_pas_cost_mu * reduced_cost
                    and p.maxBackwardBushFlowShift(owner_bush) > self.params.itapas_pas_flow_mu * current_link_flow
                )
            elif ij in p.forwardlinks:
                matched = (
                    forwardcost - backwardcost > self.params.itapas_pas_cost_mu * reduced_cost
                    and p.maxForwardBushFlowShift(owner_bush) > self.params.itapas_pas_flow_mu * current_link_flow
                )

            if matched and p.flowShiftSingleOrigin(owner_bush, self.type, self.params):
                p.lastIterFlowShift = getattr(self.params, 'current_iter', 0)
                return True

        return False

    def equilibrateITAPAS(self, iter):
        output = False

        for p in list(self.allPAS.unique_pas()):
            if p.owner is None or p.owner.bush is None:
                continue

            if p.flowShiftSingleOrigin(p.owner.bush, self.type, self.params):
                output = True
                p.lastIterFlowShift = iter
            elif self.reassignITAPASOwner(p):
                if p.flowShiftSingleOrigin(p.owner.bush, self.type, self.params):
                    output = True
                    p.lastIterFlowShift = iter

        return output

    def reassignITAPASOwner(self, p):
        if p.owner is None or len(self.origins) == 0:
            return False

        try:
            start_idx = self.origins.index(p.owner)
        except ValueError:
            start_idx = 0

        old_owner = p.owner
        window = min(self.params.itapas_reassign_origin_window, len(self.origins) - 1)

        for offset in range(1, window + 1):
            candidate = self.origins[(start_idx + offset) % len(self.origins)]

            if candidate.bush is None:
                continue

            if (
                p.maxBackwardBushFlowShift(candidate.bush) > self.params.flow_epsilon
                or p.maxForwardBushFlowShift(candidate.bush) > self.params.flow_epsilon
            ):
                if old_owner is not None and old_owner.bush is not None:
                    old_owner.bush.relevantPAS.remove(p)
                    p.relevant.discard(old_owner)

                p.setOwnerOrigin(candidate)
                candidate.bush.relevantPAS.add(p)
                return True

        return False

    def removeITAPAS(self, iter):
        removed = []

        for p in list(self.allPAS.unique_pas()):
            if p.owner is None or p.owner.bush is None:
                removed.append(p)
                continue

            owner_bush = p.owner.bush
            no_owner_flow = (
                p.maxForwardBushFlowShift(owner_bush) <= self.params.flow_epsilon
                and p.maxBackwardBushFlowShift(owner_bush) <= self.params.flow_epsilon
            )

            if no_owner_flow and self.reassignITAPASOwner(p):
                continue

            if no_owner_flow or p.lastIterFlowShift < iter-2:
                removed.append(p)

        for p in removed:
            self.removeAPAS(p)
        
    def equilibratePAS(self, iter):
        output = False
        
        # TAPAS_CHANGE_2026_06_13_C_UNIQUE_ACTIVE_PAS_EQUILIBRATION:
        # Shift each active PAS once per equilibration pass. This matches the
        # paper's "several flow shifts for every active PAS" structure and
        # avoids accidental repeated work from dictionary/list bookkeeping.
        for p in list(self.allPAS.unique_pas()):
            if p.flowShift(self.type, self.params):
                output = True
                p.lastIterFlowShift = iter

        return output
        
    def removeAPAS(self, p):
        self.allPAS.remove(p)
            
        for r in p.relevant:
            r.bush.relevantPAS.remove(p)
    
    def findUsedPaths(self):
        output = dict()
        
        for r in self.origins:
            for s in self.zones:
                if r.getDemand(s) > 0:
                    output[(r,s)] = list()
            
        for r in self.origins:
            r.bush.getUsedPaths(output)
        
        return output
    	
    def removePAS(self, iter):
        removed = []
        
        # TAPAS_CHANGE_2026_06_13_D_UNIQUE_ACTIVE_PAS_REMOVAL:
        # Remove from a snapshot of unique PAS objects, because removeAPAS()
        # mutates both PAS indexes.
        for p in list(self.allPAS.unique_pas()):
            if p.lastIterFlowShift < iter-2:
                removed.append(p)
        
        for p in removed:
            self.removeAPAS(p)
            
            
    
    def readNetwork(self,netFile,scal_time,scal_flow):
        
        firstThruNode = 1
        numZones = 0
        numNodes = 0
        numLinks = 0
        newLinks = 0
        file = open(netFile, "r")

        line = ""
        
        while line.strip() != "<END OF METADATA>":
            line = file.readline()
            if "<NUMBER OF ZONES>" in line:
                numZones = int(line[line.index('>') + 1:].strip())
            elif "<NUMBER OF NODES>" in line:
                numNodes = int(line[line.index('>') + 1:].strip())
            elif "<NUMBER OF LINKS>" in line:
                numLinks = int(line[line.index('>') + 1:].strip())
            elif "<NUMBER OF NEW LINKS>" in line:
                newLinks = int(line[line.index('>') + 1:].strip())
            elif "<FIRST THRU NODE>" in line:
                firstThruNode = int(line[line.index('>') + 1:].strip())

        for i in range(0, numZones):
            self.zones.append(Zone.Zone(i + 1))

        for i in range(0, numNodes):
            if i < numZones:
                self.nodes.append(self.zones[i])
                
                if i + 1 < firstThruNode:
                    self.zones[i].setThruNode(False)

            else:
                self.nodes.append(Node.Node(i + 1))

        line = ""
        id = 0
        
        print(numLinks)
        
        # --- Read ALL link lines first to find the true max node ID ---
        while len(line) == 0:
            line = file.readline().strip()
        # 'line' is now the column header line (e.g. "~ init_node term_node ...")
        
        link_lines = []
        trueMaxNode = numNodes
        for i in range(0, numLinks + newLinks):
            raw = file.readline().split()
            if len(raw) == 0:
                continue
            link_lines.append(raw)
            nid1 = int(raw[0])
            nid2 = int(raw[1])
            if nid1 > trueMaxNode:
                trueMaxNode = nid1
            if nid2 > trueMaxNode:
                trueMaxNode = nid2
        
        # If link data references node IDs beyond numNodes, extend the nodes list
        if trueMaxNode > numNodes:
            print(f"  Extending nodes list from {numNodes} to {trueMaxNode} (links reference higher IDs)")
            for i in range(numNodes, trueMaxNode):
                self.nodes.append(Node.Node(i + 1))
            numNodes = trueMaxNode
        
        for link_idx, raw in enumerate(link_lines):
            line = raw
            start = self.nodes[int(line[0]) - 1]
            end = self.nodes[int(line[1]) - 1]
            C = float(line[2]) * scal_flow
            length = float(line[3])  # Link length from column 4


            t_ff = float(line[4]) * scal_time
            alpha = float(line[5])
            beta = float(line[6])
            
   
            try:
                cost = float(line[10])
            except ValueError:
                cost = 0
                
            #print(start, end, cost, line)
            
            self.TC += cost
            
            link = Link.Link(id, start, end, t_ff, C, alpha, beta, cost, length)
            id = id +1
            #print(start,end)
            self.links.append(link)
            #print(self.links)
            
            if link_idx >= numLinks:
                self.links2.append(link)
            
        file.close()


    def readTrips(self,tripsFile,scal_time,scal_flow,inflate_trips):
        
        file = open(tripsFile, "r")
        
        lines = file.readlines()
        
        line_idx = 0
        
        while lines[line_idx].strip() != "<END OF METADATA>":
            line_idx += 1
            
        line_idx += 1
        
        while lines[line_idx].strip() == "":
            line_idx += 1
            
        r = None
        
        idx = 0
        
        splitted = lines[line_idx].split()
        #print(splitted)
        
        while len(lines) < line_idx or idx < len(splitted):

            next = splitted[idx]

            if next == "Origin":
                
                idx += 1
                r = self.zones[int(splitted[idx]) - 1]

            else:
                s = self.zones[int(splitted[idx]) - 1]

                #print(s)
                idx += 2
                next = splitted[idx]
                d = float(next[0:len(next) - 1]) * scal_flow
                d = d * inflate_trips
                
                
                
                r.addDemand(s, d)
                self.TD += d

            idx += 1

            if idx >= len(splitted):
                line_idx += 1
                while line_idx < len(lines) and lines[line_idx].strip() == "":
                    line_idx += 1
                    
                if line_idx < len(lines):
                    line = lines[line_idx].strip()
                    splitted = line.split()
                    idx = 0
            
        file.close()
        
        for r in self.zones:
            if r.getProductions() > self.params.flow_epsilon:
                self.origins.append(r)
