import math
import numpy
import scipy
from scipy.sparse import csc_matrix

from Indexer import Indexer


class QIndexer (Indexer):
    DNA = ['A', 'T', 'C', 'G']
    
    def __init__(self, settings, logger, stepFactor = 0.1, reads= [], qgram=1):
        Indexer.__init__(self, settings, logger, stepFactor, reads)
        self.qgram = qgram
        self.character_list = None
        self.generate_character_list(qgram)
        self.character_index = {}
        self.indicesStep = 0
        self.block = 1000 # only for parallel devices
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
        results = numpy.zeros(len(self.character_list)+1)
        results[0] = window
        # make sure end_index never exceeds length of sequence:
        if end_index >= len(seq):
            end_index = len(seq)
        if len(seq) > 0 and end_index - start_index > 0:
            n = seq.count("N", start_index, end_index)
            length = float(end_index - start_index - n)
            if length-self.qgram+1 > 0 :
                fraction = self.compositionScale / float(length-self.qgram+1)
                for qgram_string in range(start_index, end_index-self.qgram):
                    subStr = str(seq[qgram_string:qgram_string + self.qgram])
                    
                    if "N" not in subStr and subStr in self.character_index and len(subStr.strip()) == len(self.character_list[0]):
                        results[self.character_index[subStr]] += fraction 
        r = results.view(int)
        
        r[:] = results
        return(csc_matrix(r, dtype=numpy.int32))

    def createIndexAndStore(self, sequence, fileName, retainInMemory=True):
        self.createIndex(sequence, fileName, retainInMemory)

    def pickleName(self, fileName, length):
        if self.indicesStep == None:
            return fileName + ".Q" + str(self.qgram) + "." + str(length) + ".index"
        else:
            return fileName + ".Q" + str(self.qgram) + "." + str(length) + "." + str(self.indicesStep) + ".index"

    def distance_calc(self,x,y):
        return numpy.linalg.norm(x.toarray() - y.toarray())/self.compositionScale
 
    def findIndices(self,seqs, start = 0.0, step=False):
        """ finds the seeding locations for the mapping process.
        Structure of locations:
        (hit, window, distance), with hit: (location, reference seq id)
        Full structure:
        ((location, reference seq id), window, distance)
        :param seq: sequence used for comparison
        :param start: minimum distance. Use default unless you're stepping through distance values
        :param step: set this to True when you're stepping through distance values. Hence: start at 0 <= distance < 0.01, then 0.01 <= distance < 0.02, etc  
        """
        hits = []
        for seq in seqs:
            hits.append({})
            #find smallest window:
            loc = 0
            while loc < len(self.wSize)-1 and self.windowSize(len(seq)) > self.wSize[loc]:
                loc += 1
    
            comp = self.count(seq.seq.upper(), self.wSize[loc], 0, len(seq))
            keys = self.tupleSet.keys()
            compAll = keys
            
            
            distances = [self.distance_calc(a, comp) for a in compAll] 
            #self.logger.debug("Distances: {}".format(distances))
            validComp = [keys[x] for x in xrange(len(keys)) if keys[x].data[0] == comp.data[0] and distances[x]  < self.sliceDistance]
            self.logger.debug("Found {} relevant locations".format(len(validComp)))
            
            for valid in validComp:
                for hit in self.tupleSet[valid]:
                    if hit[1] not in hits[-1]:
                        hits[-1][hit[1]] = []
                    hits[-1][hit[1]].extend([(hit, self.wSize[loc], self.distance_calc(valid, comp))])
        return hits
    
