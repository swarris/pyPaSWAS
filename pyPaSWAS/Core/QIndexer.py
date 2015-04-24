from Bio import  SeqIO
import math
import sys
import os.path
import cPickle
import zlib
import re
import collections

from SWSeqRecord import SWSeqRecord
from Bio.Seq import Seq
from Indexer import Indexer



class QIndexer (Indexer):
    DNA = ['A', 'T', 'C', 'G']
    
    def __init__(self, settings, logger, stepFactor = 0.1, reads= [], qgram=1):
        Indexer.__init__(self, settings, logger, stepFactor, reads)
        self.qgram = qgram
        self.character_list = None
        self.generate_character_list(qgram)
        self.character_index = {}
        index = 1
        for c in self.character_list:
            self.character_index[c] = index
            index += 1

    def generate_character_list(self, level=0):
        if level != 0:
            if self.character_list == None:
                self.character_list = [y for y in self.DNA]
            else:
                self.character_list = [y+x for x in self.DNA for y in self.character_list]
            self.generate_character_list(level-1)

    def count(self, seq, window, start_index, end_index):
        results = [0]*(len(self.character_list)+1)
        results[0] = window
        
        if len(seq) > 0 and end_index - start_index > 0:
            n = seq.count("N", start_index, end_index)
            length = float(end_index - start_index - n)
            fraction = self.compositionScale / float(length-self.qgram+1)
            if length > 0 :
                #results = [int(self.compositionScale*seq.count(x, start_index, end_index) / (length-self.qgram+1)) for x in self.character_list]
                #results = [int(self.compositionScale*len(re.findall(r"(?=" + x + ")", str(seq[start_index: end_index]))) / (length-self.qgram+1)) for x in self.character_list]
                for qgram_string in range(start_index, end_index-self.qgram, self.qgram):
                    results[self.character_index[str(seq[qgram_string:qgram_string + self.qgram])]] += fraction 
                results = map(lambda x: int(x), results)

        return( tuple(results) )

    def createIndexAndStore(self, sequence, fileName, retainInMemory=True):
        self.createIndex(sequence, fileName, retainInMemory)

    def pickleName(self, fileName, length):
        return fileName + ".Q" + str(self.qgram) + "." + str(length) + ".index"


    def findIndices(self,seq, start = 0.0, step=False):
        """ finds the seeding locations for the mapping process.
        Structure of locations:
        (hit, window, distance), with hit: (location, reference seq id)
        Full structure:
        ((location, reference seq id), window, distance)
        :param seq: sequence used for comparison
        :param start: minimum distance. Use default unless you're stepping through distance values
        :param step: set this to True when you're stepping through distance values. Hence: start at 0 <= distance < 0.01, then 0.01 <= distance < 0.02, etc  
        """
        hits = {}
        #find smallest window:
        loc = 0
        while loc < len(self.wSize)-1 and self.windowSize(len(seq)) > self.wSize[loc]:
            loc += 1
        comp = self.count(seq.upper(), self.wSize[loc], 0, len(seq))
        if not step:
            validComp = filter(lambda x : x[0] == comp[0] and reduce(lambda x,y: x+y, [(x[i] -comp[i])**2 for i in range(1,len(self.character_list))]) < self.sliceDistance, self.tupleSet.keys())
        else :
            validComp = filter(lambda x : x[0] == comp[0] and ( start <= math.sqrt(reduce(lambda x,y: x+y, [(x[i] -comp[i])**2 for i in range(1,len(self.character_list))]))/self.compositionScale < start + self.distanceStep), self.tupleSet.keys())
        for valid in validComp:
            for hit in self.tupleSet[valid]:
                if hit[1] not in hits:
                    hits[hit[1]] = []
                hits[hit[1]].extend([(hit, self.wSize[loc], math.sqrt(reduce(lambda x,y: x+y, [(valid[i] -comp[i])**2 for i in range(1,len(self.character_list))]))/self.compositionScale)])
        for hit in hits:    
            hits[hit].sort(cmp=(lambda x,y: -1 if x[0][0] < y[0][0] else 1))
        return hits
