from Bio import  SeqIO
import math
import sys
import cPickle
import zlib
from SWSeqRecord import SWSeqRecord
from Bio.Seq import Seq
from atk import Window

class Indexer:

    sliceDistance = 100000.0
    wSize = {}
    dataset = {}
    stepFactor = 0.1    
    slideStep = 1.0
    tupleSet = {}
    windowStep = 1.0
    compositionScale = 1000.0
    distanceStep = 0.01
    
    def __init__(self, settings, logger, stepFactor = 0.1, reads= []):
        self.sliceDistance= float(settings.maximum_distance)
        self.settings = settings
        self.logger = logger
        self.stepFactor = stepFactor
        self.sliceDistance = self.sliceDistance*self.sliceDistance*self.compositionScale*self.compositionScale

        
        self.wSize = list(set(map(lambda x : len(x) , reads)))
        self.wSize = map(lambda x : self.windowSize(x), self.wSize)
        self.wSize = sorted(self.wSize)

    def count(self, seq, window, start_index, end_index):
        if len(seq) > 0 and end_index - start_index > 0:
            n = seq.count("N", start_index, end_index)
            length = float(end_index - start_index - n)
            if length > 0 :
                a = int(self.compositionScale*seq.count("A", start_index, end_index) / length)
                t = int(self.compositionScale*seq.count("T", start_index, end_index) / length)
                c = int(self.compositionScale*seq.count("C", start_index, end_index) / length)
                g = int(self.compositionScale*seq.count("G", start_index, end_index) / length)
            else:
                a = 0
                t = 0
                c = 0
                g = 0
        else:
            a = 0
            t = 0
            c = 0
            g = 0
            
        return( (window,a,t,c,g) )

    def windowSize(self,seqLength):
        return int(math.ceil(math.floor(seqLength*(self.stepFactor+1.0)) / self.windowStep) * self.windowStep)

    def reverseWindowSize(self, window):
        return int(math.floor(window/(self.stepFactor+1.0)-self.windowStep))

    def createIndex(self, sequence, fileName = None):
        totalElements = 0
        currentTupleSet = {}
        for window in self.wSize:
            self.tupleSet = {}
            for seqId in xrange(len(sequence)):
                self.logger.debug("Creating index of {} with window {}".format(sequence[seqId].id, window))
                seq = str(sequence[seqId].seq.upper())
                endIndex = 1 if len(seq)-window < 0 else len(seq)-window+1
                revWindowSize = int(self.reverseWindowSize(window)*self.stepFactor*self.slideStep) 
                for index in xrange(endIndex):
                    if index % revWindowSize == 0:
                        comp = self.count(seq, window,index,index+int(window))
                        if comp not in self.tupleSet:
                            self.tupleSet[comp] = []
                        self.tupleSet[comp].append((index, seqId))
                        totalElements += 1
            if fileName != None:
                self.pickle(fileName, window)
            currentTupleSet.update(self.tupleSet)
        self.tupleSet = currentTupleSet
                    
    def createIndexAndStore(self, sequence, fileName):
        self.createIndex(sequence, fileName)

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
            validComp = filter(lambda x : x[0] == comp[0] and ((x[1] - comp[1])**2 + (x[2] - comp[2])**2 + (x[3] - comp[3])**2 + (x[4] - comp[4])**2 ) < self.sliceDistance, self.tupleSet.keys())
        else :
            validComp = filter(lambda x : x[0] == comp[0] and ( start <= math.sqrt((x[1] - comp[1])**2 + (x[2] - comp[2])**2 + (x[3] - comp[3])**2 + (x[4] - comp[4])**2 )/self.compositionScale < start + self.distanceStep), self.tupleSet.keys())
        for valid in validComp:
            for hit in self.tupleSet[valid]:
                if hit[1] not in hits:
                    hits[hit[1]] = []
                hits[hit[1]].extend([(hit, self.wSize[loc], math.sqrt(((valid[1] - comp[1])**2 + (valid[2] - comp[2])**2 + (valid[3] - comp[3])**2 + (valid[4] - comp[4])**2))/self.compositionScale)])
        for hit in hits:    
            hits[hit].sort(cmp=(lambda x,y: -1 if x[0][0] < y[0][0] else 1))
        return hits
    
    
    def createSWSeqRecords(self, sequences):
        """ create sequences of all windows based on the list of sequence. Very, very inefficient
        :param sequences: list of sequence on which the index is based.
        """
        self.indexedSequences = []
        currentSeq = 0
        for seq in sequences:
            self.indexedSequences.append({})
            for window in self.wSize:
                self.indexedSequences[currentSeq][window] = {}
                endIndex = 1 if len(seq)-window < 0 else len(seq)-window+1
                revWindowSize = int(self.reverseWindowSize(window)*self.stepFactor*self.slideStep) 
                for startIndex in xrange(endIndex):
                    if startIndex % revWindowSize == 0:
                        self.indexedSequences[currentSeq][window][startIndex] = SWSeqRecord(Seq(str(seq.seq[startIndex:startIndex+window]), seq.seq.alphabet), seq.id, startIndex, original_length = len(seq.seq), distance = 0, refID = seq.id)
            currentSeq += 1
            
    def getSWSeqRecord(self, location, sequences):
        """get a new SWSeqRecord based on seeding location and target sequences
        :param location: seeding location
        :param sequences: target sequences on which the index is based
        Structure of location:
            ((location, reference seq id), window, distance)
        """
        startIndex = location[0][0]
        window = location[1]
        seq = sequences[location[0][1]]
        swSeqRecord = SWSeqRecord(Seq(str(seq.seq[startIndex:startIndex+window]), seq.seq.alphabet), seq.id, startIndex, original_length = len(seq.seq), distance = 0, refID = seq.id)
        return swSeqRecord    
            
    def pickleName(self, fileName, length):
        return fileName + "." + str(length) + ".index"

    def pickle(self, fileName, window = None):
        try:
            self.logger.info("Saving index to file: " + self.pickleName(fileName, self.wSize[0] if window == None else window))
            dump = open(self.pickleName(fileName, self.wSize[0] if window == None else window), "w")
            dump.write(zlib.compress(cPickle.dumps(self.tupleSet, cPickle.HIGHEST_PROTOCOL),9))
            dump.close()
        except:
            self.logger.error("Could not open: " + self.pickleName(fileName, self.wSize[0] if window == None else window))
            
    def unpickle(self, fileName):
        self.tupleSet = {}
        for window in self.wSize:
            try:
                dump = open(self.pickleName(fileName, window), "r")
                self.tupleSet.update(cPickle.loads(zlib.decompress(dump.read())))
                dump.close()
            except:
                self.logger.warning("Could not open pickle file: "+ self.pickleName(fileName, window))
                self.logger.warning("Will need to rebuild index")
                self.tupleSet = {}
                return False
        return True
