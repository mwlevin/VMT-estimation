# Created on : Mar 27, 2024, 10:28:15 PM
# Author     : michael
import contextlib

class PASList:
    def __init__(self):
        self.forward = {}
        self.backward = {}
        self.by_head = {}
        self.by_signature = {}
    
    def containsKey(self, ij):
        return ij in self.forward or ij in self.backward
        
    def size(self):
        return len(self.backward)

    # TAPAS_CHANGE_2026_06_13_A_UNIQUE_PAS_COUNT:
    # size() is kept unchanged for compatibility, but it counts dictionary
    # buckets, not PAS objects. Use these helpers when iterating/diagnosing
    # active PASs so one PAS is handled once per pass.
    def unique_pas(self):
        seen = set()

        for pas_list in self.forward.values():
            for pas in pas_list:
                if pas.id in seen:
                    continue

                seen.add(pas.id)
                yield pas

    def count_unique(self):
        return sum(1 for _ in self.unique_pas())
    
    def add(self, pas):
        #with open('result49.txt', 'a') as file, contextlib.redirect_stdout(file):
        self.by_signature[pas.signature()] = pas

        head = pas.end
        if head not in self.by_head:
            self.by_head[head] = []
        self.by_head[head].append(pas)

        ij = pas.getEndLinkBwd()
        #print(ij)
        if ij not in self.backward:
            self.backward[ij] = []
        #print(ij)
        #print(type(ij))
        self.backward[ij].append(pas)
        
        ij = pas.getEndLinkFwd()
        if ij not in self.forward:
            self.forward[ij] = []
        
        self.forward[ij].append(pas)
        #print(ij)


    def remove(self, pas):
        self.by_signature.pop(pas.signature(), None)

        head = pas.end
        if head in self.by_head and pas in self.by_head[head]:
            self.by_head[head].remove(pas)

            if len(self.by_head[head]) == 0:
                del self.by_head[head]

        ij = pas.getEndLinkBwd()
        
        if ij in self.backward and pas in self.backward[ij]:
            self.backward[ij].remove(pas)
            
            if len(self.backward[ij]) == 0:
                del self.backward[ij]
        
        ij = pas.getEndLinkFwd()
        
        if ij in self.forward and pas in self.forward[ij]:
            self.forward[ij].remove(pas)
            
            if len(self.forward[ij]) == 0:
                del self.forward[ij]

    # ITAPAS_CHANGE_2026_06_13_A_INDEX_PAS_BY_SIGNATURE_AND_HEAD:
    # iTAPAS checks whether a newly identified PAS already exists and searches
    # existing PASs by merge/head node. Keep these helpers separate from the
    # original TAPAS forward/backward endpoint indexes.
    def find_duplicate(self, pas):
        return self.by_signature.get(pas.signature())

    def contains_pas(self, pas):
        return pas.signature() in self.by_signature

    def iter_head(self, head):
        return list(self.by_head.get(head, []))
    
