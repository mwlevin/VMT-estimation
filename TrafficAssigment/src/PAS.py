# Created on : Mar 27, 2024, 4:37:50 PM
# Author     : michaellevin


# This is a sample Python script.

# Press Shift+F6 to execute it or replace it with your code.
from src import Params
import contextlib
from src import Network





class PAS:
    next_id = 0
    def __init__(self):
        self.forwardlinks = []
        self.backwardlinks = []
        
        self.id = PAS.next_id
        PAS.next_id += 1
        
        self.relevant = set()
        self.owner = None
        self.lastIterFlowShift = 0
        self.start = None
        self.end = None
    
    def addRelevantOrigin(self, r):
        self.relevant.add(r)
        #print(r)

    def setOwnerOrigin(self, r):
        self.owner = r
        self.addRelevantOrigin(r)

    def signature(self):
        return (
            tuple(l.id for l in self.forwardlinks),
            tuple(l.id for l in self.backwardlinks)
        )
        
    
    def getTT(self, topshift, type, params):
        fwdtt = 0
        bwdtt = 0
        #print("\t\t\tsuppose fwd +", topshift, "bwd +", -topshift)
        #with open('getTT5.txt', 'a') as file, contextlib.redirect_stdout(file):
        for l in self.forwardlinks:
            #print(f"self.forwardlinks {self.forwardlinks}")
            fwdtt += l.getTravelTime(l.x + topshift, type)
            #output += l.getTravelTime(l.x, type)
            #print(f"output is {output}----")
            
        for l in self.backwardlinks:
            bwdtt += l.getTravelTime(l.x - topshift, type)
            #print(output)
        
        if  params.PRINT_PAS_DEBUG:
            print("\t\t\tthen fwd cost is ", fwdtt, "bwd cost is ", bwdtt)
        return fwdtt - bwdtt
        
    def isBackwards(self, type):
        forwardcost = self.getForwardCost(type)
        backwardcost = self.getBackwardCost(type)
        
        return backwardcost < forwardcost
        
    def getEndLinkFwd(self):
        return self.forwardlinks[0]
    
    def getEndLinkBwd(self):
        #with open('result175.txt', 'a') as file, contextlib.redirect_stdout(file):
            answer = self.backwardlinks[-1]
            #print(self.backwardlinks)
            #print(answer)
            return self.backwardlinks[-1]
        
    
    def isEffective(self, type, bush, cost_mu, minflow, params):
        forwardcost = self.getForwardCost(type)
        backwardcost = self.getBackwardCost(type)

        
        if forwardcost > backwardcost:
            maxFwdShift = self.maxForwardBushFlowShift(bush)
            
            if forwardcost > backwardcost * (1 + cost_mu):
                params.good_pas_cost_mu += 1
                
                if maxFwdShift > minflow:
                    params.good_pas_flow_mu += 1
                    
                    return True
                else:
                    return False
            elif maxFwdShift > minflow:
                #params.good_pas_flow_mu += 1
                return False
        else:
            maxBwdShift = self.maxBackwardBushFlowShift(bush)
            
            if backwardcost > forwardcost * (1 + cost_mu):
                params.good_pas_cost_mu += 1
                
                if maxBwdShift > minflow:
                    params.good_pas_flow_mu += 1
                    
                    return True
                else:
                    return False
            elif maxBwdShift > minflow:
                #params.good_pas_flow_mu += 1
                return False
         
        '''   
        if forwardcost > backwardcost * (1 + cost_mu):
            return maxFwdShift > minflow
        elif backwardcost > forwardcost * (1 + cost_mu):
            return maxBwdShift > minflow
        '''
        return False
    
    def isCostEffective(self, type, cost_mu):
        
        forwardcost = self.getForwardCost(type)
        backwardcost = self.getBackwardCost(type)
        
        costdiff = backwardcost - forwardcost
        
        # maybe the forward and backward costs will be reversed sometimes
        output = costdiff > cost_mu * forwardcost or -costdiff > cost_mu * backwardcost
        
        #print("cost eff?" , forwardcost, backwardcost, costdiff, output)
        return output
    
    def isCostEffectiveForLink(self, a, type, cost_mu):
        forwardcost = self.getForwardCost(type)
        backwardcost = self.getBackwardCost(type)
        
        costdiff = 0
        
        
        if a == self.getEndLinkBwd():
            #print(a)
            costdiff = backwardcost - forwardcost
            return costdiff > cost_mu * forwardcost
        elif a == self.getEndLinkFwd():
            costdiff = forwardcost - backwardcost
            return costdiff > cost_mu * backwardcost

        return False
    
    def getForwardCost(self, type):
        forwardcost = 0
        
        for l in self.forwardlinks:
            forwardcost += l.getTravelTime(l.x, type)
        
        
        return forwardcost
    
    
    def getBackwardCost(self, type):
        backwardcost = 0
        
        for l in self.backwardlinks:
            backwardcost += l.getTravelTime(l.x, type)
        
        
        return backwardcost

    def getForwardDerivative(self):
        return sum(l.getDerivativeTravelTime(l.x) for l in self.forwardlinks)

    def getBackwardDerivative(self):
        return sum(l.getDerivativeTravelTime(l.x) for l in self.backwardlinks)

    def _pasCostEpsilon(self, params):
        epsilon = params.pas_cost_epsilon
        current_gap = getattr(params, 'current_gap', None)

        if getattr(params, 'pas_use_dynamic_cost_epsilon', False) and current_gap is not None:
            epsilon = max(epsilon, current_gap / params.pas_dynamic_gap_divisor)

        return epsilon

    def maxForwardOwnerFlowShift(self):
        if self.owner is None or self.owner.bush is None:
            return 0

        return self.maxForwardBushFlowShift(self.owner.bush)

    def maxBackwardOwnerFlowShift(self):
        if self.owner is None or self.owner.bush is None:
            return 0

        return self.maxBackwardBushFlowShift(self.owner.bush)

    def isEffectiveSingleOrigin(self, type, bush, link, cost_mu, flow_mu, params):
        forwardcost = self.getForwardCost(type)
        backwardcost = self.getBackwardCost(type)
        reduced_cost = max(link.getReducedCost(type), params.flow_epsilon)
        link_flow = max(bush.getFlow(link), params.flow_epsilon)

        if link in self.backwardlinks:
            return (
                backwardcost - forwardcost > cost_mu * reduced_cost
                and self.maxBackwardBushFlowShift(bush) > flow_mu * link_flow
            )

        if link in self.forwardlinks:
            return (
                forwardcost - backwardcost > cost_mu * reduced_cost
                and self.maxForwardBushFlowShift(bush) > flow_mu * link_flow
            )

        return False

    def isWorthStoringForSingleOrigin(self, type, bush, params):
        forwardcost = self.getForwardCost(type)
        backwardcost = self.getBackwardCost(type)
        cost_epsilon = self._pasCostEpsilon(params)

        if backwardcost > forwardcost:
            return (
                backwardcost - forwardcost > cost_epsilon * max(forwardcost, params.flow_epsilon)
                and self.maxBackwardBushFlowShift(bush) > params.flow_epsilon
            )

        return (
            forwardcost - backwardcost > cost_epsilon * max(backwardcost, params.flow_epsilon)
            and self.maxForwardBushFlowShift(bush) > params.flow_epsilon
        )
    
    
    def isFlowEffective(self, flow_mu, minflow, type):
        # min flow of high cost segment
        # high cost segment is backwards links
        maxshift = 1e9; 
        flowlastsegment = 0
        
        lookat = self.backwardlinks
        
        if self.isBackwards(type):
            lookat = self.forwardlinks
            
        for l in lookat:
            # only look at high cost segment
            totalFlow = 0
            for r in self.relevant:
                totalFlow += r.bush.getFlow(l)
                
            #print("\t", l, totalFlow)

            maxshift = min(maxshift, totalFlow)
            if l.end == self.end:
                flowlastsegment = totalFlow
            
        output = maxshift > flow_mu * flowlastsegment and maxshift > minflow
        #print("flow eff? ", maxshift, flow_mu, minflow, flowlastsegment, output, self.relevant)
        
        return output
        
    
        
    def maxForwardBushFlowShift(self, bush):
        max = 1E9
        #with open('forwardBush4.txt', 'a') as file, contextlib.redirect_stdout(file):
        for l in self.forwardlinks:
            #print(self.forwardlinks)
            # check flow on link if l in backwards direction
            #print(max)
            max = min(max, bush.getFlow(l))
            #print(bush.getFlow(l))
            #print(f"The result is {max}------------------")

        
        return max
    
    
    def maxBackwardBushFlowShift(self, bush):
        max = 1E9
        #print(self.backwardlinks)
    #with open('maxB2.txt', 'a') as file, contextlib.redirect_stdout(file):
        for l in self.backwardlinks:

            # check flow on link if l in backwards direction
            max = min(max, bush.getFlow(l))
            #print(f"max {max}-----")
            #print(l.id, bush.getFlow(l), bush.origin)
            

        
        return max
        
    def maxNumForwardFlowShift(self):
        maxBush = self.maxForwardFlowShift()
        
        return sum(maxBush[r.bush] for r in self.relevant)
        
    def maxNumBackwardFlowShift(self):
        maxBush = self.maxBackwardFlowShift()
        
        
        return sum(maxBush[r.bush] for r in self.relevant)
    
    def maxForwardFlowShift(self):
        
        maxFlowPerBush = {}
        
        #for r in self.relevant:
        for r in self.relevant:
            maxFlowPerBush[r.bush] = self.maxForwardBushFlowShift(r.bush)
        
        
        return maxFlowPerBush
    
    
    
    def maxBackwardFlowShift(self):
        #with open('result139.txt', 'a') as file, contextlib.redirect_stdout(file):
            maxFlowPerBush = {}
            #sorted_relevant = sorted(self.relevant, key=lambda r: r.id)
            
            #print(f"before sorting {self.relevant}")
            #print(self.relevant)
            for r in self.relevant:
                #print(self.forwardlinks, self.backwardlinks, sorted(self.relevant, key=lambda r: r.id))
                #print(f"r.id is {r.id}")
                #print(f"r is {r}")
                maxFlowPerBush[r.bush] = self.maxBackwardBushFlowShift(r.bush)
                #print(maxFlowPerBush[r.bush])

            return maxFlowPerBush
    
    def zeroBackwardFlow(self, bush):
        maxFlowShift = self.maxBackwardBushFlowShift(bush)
        
        for l in self.forwardlinks:
            # proportion allocated to bush is bush max shift / total max shift
            bush.addFlow(l, maxFlowShift)
            #print(f"maxFlowShift[r.bush] {maxFlowShift[r.bush]}")
            #print(f"overallMaxShift is {overallMaxShift}")
            #print(f"The bot is {bot}")
            #print(f"The backwards is {backwards}")
        
        
        
        for l in self.backwardlinks:
            # proportion allocated to bush is bush max shift / total max shift
            #print(-maxFlowShift[r.bush] / overallMaxShift * bot * backwards)
            #print(f'-maxFlowShift[r.bush] {-maxFlowShift[r.bush]}, overallMaxShift {overallMaxShift},bot{bot},backwards{backwards}')
            bush.addFlow(l, -maxFlowShift)
            
        
        
    def flowShift(self, type, params):
    
        if params.PRINT_PAS_INFO:
            print("\tflow shift", self.id)
        
        forwardcost = self.getForwardCost(type)
        backwardcost = self.getBackwardCost(type)
        
        costdiff = backwardcost - forwardcost

        
        backwards = 1
        if backwardcost < forwardcost:
            backwards = -1
            
        #print("flow shift? " +str(costdiff)+" "+str(forwardcost)+" "+str(backwardcost)+" "+str(cost_mu)+" "+str(backwards))
        
        cost_epsilon = params.pas_cost_epsilon
        #cost_epsilon = 1e-6
        
        # maybe the forward and backward costs will be reversed sometimes
        if backwards == 1:
            if costdiff < cost_epsilon * forwardcost:
                if params.PRINT_PAS_INFO:
                    print("\t\tcostdiff is low", costdiff)
                return False
            #else:
                #params.good_pas_cost_epsilon += 1
        
        elif backwards == -1: 
            if -costdiff < cost_epsilon * backwardcost:
                #print("\t", "costdiff is low", costdiff)
                if params.PRINT_PAS_INFO:
                    print("\t\tcostdiff is low", costdiff)
                return False
            #else:
                #params.good_pas_cost_epsilon += 1
        
        #print(backwards)

        
        overallMaxShift = 0
        
        maxFlowShift = {}
        
        if backwards > 0:
            #print(backwards)
            maxFlowShift = self.maxBackwardFlowShift()
        
        else:
            maxFlowShift = self.maxForwardFlowShift()
        
        
        for b in maxFlowShift:
            overallMaxShift += maxFlowShift[b]
        
            
        if overallMaxShift < params.pas_flow_mu:
            if params.PRINT_PAS_INFO:
                print("\t\tmax shift is low", overallMaxShift)
                print("\t\tmax shift fwd", self.maxNumForwardFlowShift(), "bwd", self.maxNumBackwardFlowShift())
                print("\t\tcostdiff fwd", self.getForwardCost(type), "bwd", self.getBackwardCost(type))
                print("\t\trelevant bush", self.relevant)
            return False
        

        #print("max shift "+str(overallMaxShift)+" "+str(backwards)+" "+str(backwardcost)+" "+str(forwardcost))
        
        #print("backwards")
        #for l in self.backwardlinks:
        #    for r in self.relevant:
        #        print("\t"+str(l.start)+" "+str(l.end)+" "+str(r)+" "+str(r.bush.getFlow(l)))
        #print("forwards")
        #for l in self.forwardlinks:
        #    for r in self.relevant:
        #        print("\t"+str(l.start)+" "+str(l.end)+" "+str(r)+" "+str(r.bush.getFlow(l)))

        #for r in self.relevant:
        #    print(str(r)+" "+str(maxFlowShift[r.bush]))
        
        stop = params.line_search_gap
        bot = 0
        top = overallMaxShift
        
        if params.PRINT_PAS_DEBUG:
            print("start shift fwd ", forwardcost, "bwd", backwardcost)
            
            
        #stop = 1e-6
        #with open('flowShift3.txt', 'a') as file, contextlib.redirect_stdout(file):
        while top - bot > overallMaxShift * stop:
            #print(line_search_gap)
            mid = (top + bot)/2
            
            if top < params.pas_flow_mu:
                if params.PRINT_PAS_INFO:
                    print("\t\tshift is low during line search")
                return False
                
            
            check = self.getTT(mid * backwards, type, params)
            #print(mid * backwards)
            if params.PRINT_PAS_DEBUG:
                print("\t\tcheck this "+str(bot)+" "+str(top)+" "+str(mid)+" "+str(check))
            
            if check*backwards < 0:
                bot = mid
            
            else:
                top = mid

        shift = bot

    #with open('result132.txt', 'a') as file, contextlib.redirect_stdout(file):
        for l in self.forwardlinks:

            for r in self.relevant:
                # proportion allocated to bush is bush max shift / total max shift
                r.bush.addFlow(l, maxFlowShift[r.bush] / overallMaxShift * shift * backwards)
                #print(f"maxFlowShift[r.bush] {maxFlowShift[r.bush]}")
                #print(f"overallMaxShift is {overallMaxShift}")
                #print(f"The bot is {bot}")
                #print(f"The backwards is {backwards}")
        
        
        
        for l in self.backwardlinks:
            for r in self.relevant:
                # proportion allocated to bush is bush max shift / total max shift
                #print(-maxFlowShift[r.bush] / overallMaxShift * bot * backwards)
                #print(f'-maxFlowShift[r.bush] {-maxFlowShift[r.bush]}, overallMaxShift {overallMaxShift},bot{bot},backwards{backwards}')
                r.bush.addFlow(l, -maxFlowShift[r.bush] / overallMaxShift * shift * backwards)
        

        params.good_pas_cost_epsilon += 1
        
        if params.PRINT_PAS_DEBUG:
            print("\tshift", shift, backwards, " up to ", overallMaxShift)
            print("\t\tmax shift fwd", self.maxNumForwardFlowShift(), "bwd", self.maxNumBackwardFlowShift())
            print("\t\tcostdiff fwd", self.getForwardCost(type), "bwd", self.getBackwardCost(type))
            #), overallMaxShift, (-costdiff), (self.getForwardCost(type)-self.getBackwardCost(type)))
        #print(bot, overallMaxShift)
        
        #if self.id == 2799:
        #    print("after shift "+str(bot)+" "+str(overallMaxShift)+" "+ str(bot / overallMaxShift*100)+" "+str(self.getTT(0, type)))
        
        return True

    # ITAPAS_CHANGE_2026_06_13_B_SINGLE_ORIGIN_PAS_SHIFT:
    # iTAPAS associates each PAS with one owner origin during UE solving.
    # This method shifts only that owner bush, while the original flowShift()
    # above remains the multi-origin TAPAS operation.
    def flowShiftSingleOrigin(self, bush, type, params):
        if params.PRINT_PAS_INFO:
            print("\titapas single-origin flow shift", self.id, bush.origin)

        forwardcost = self.getForwardCost(type)
        backwardcost = self.getBackwardCost(type)
        costdiff = backwardcost - forwardcost

        if backwardcost > forwardcost:
            highlinks = self.backwardlinks
            lowlinks = self.forwardlinks
            maxshift = self.maxBackwardBushFlowShift(bush)
            highcost = backwardcost
            lowcost = forwardcost
            direction = 1
        else:
            highlinks = self.forwardlinks
            lowlinks = self.backwardlinks
            maxshift = self.maxForwardBushFlowShift(bush)
            highcost = forwardcost
            lowcost = backwardcost
            direction = -1

        if highcost - lowcost < self._pasCostEpsilon(params) * max(lowcost, params.flow_epsilon):
            return False

        if maxshift < params.flow_epsilon:
            return False

        if getattr(params, 'itapas_use_newton_shift', True):
            derivative = self.getForwardDerivative() + self.getBackwardDerivative()
            if derivative > params.flow_epsilon:
                shift = min((highcost - lowcost) / derivative, maxshift)
            else:
                shift = maxshift
        else:
            bot = 0
            top = maxshift
            stop = params.line_search_gap

            while top - bot > maxshift * stop:
                mid = (top + bot) / 2
                check = self.getTT(mid * direction, type, params)

                if check * direction < 0:
                    bot = mid
                else:
                    top = mid

            shift = bot

        if shift < params.flow_epsilon:
            return False

        for l in lowlinks:
            bush.addFlow(l, shift)

        for l in highlinks:
            bush.addFlow(l, -shift)

        params.good_pas_cost_epsilon += 1
        return True
        
    def __str__(self):
        return str(self.id)+" "+str(self.forwardlinks)+", "+str(self.getForwardCost("UE"))+" - "+str(self.backwardlinks)+", "+str(self.getBackwardCost("UE"))
