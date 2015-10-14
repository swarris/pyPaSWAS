from Bio import  SeqIO
import math
import sys
import os.path
import cPickle
import zlib
from SWSeqRecord import SWSeqRecord
from Bio.Seq import Seq

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
    indicesStepSize = 1000
    readsToProcess = 30
    
    def __init__(self, settings, logger, stepFactor = 0.1, reads= []):
        self.sliceDistance= float(settings.maximum_distance)
        self.settings = settings
        self.logger = logger
        self.stepFactor = stepFactor
        self.sliceDistance = self.sliceDistance#*self.sliceDistance*self.compositionScale*self.compositionScale
        self.readsToProcess = int(settings.reads_to_process)
        
        self.wSize = list(set(map(lambda x : len(x) , reads)))
        self.wSize = map(lambda x : self.windowSize(x), self.wSize)
        self.wSize = sorted(self.wSize)
        self.indicesStep = None
        self.indexCount = 0
        
    def count(self, seq, window, start_index, end_index):
        self.logger.error("Indexer.count needs to by implemented by subclass. Maybe use QIndexer?")

    def windowSize(self,seqLength):
        return int(math.ceil(math.floor(seqLength*(self.stepFactor+1.0)) / self.windowStep) * self.windowStep)

    def reverseWindowSize(self, window):
        return int(math.floor(window/(self.stepFactor+1.0)-self.windowStep))

    def setIndicesStep(self, indicesStep):
        self.indicesStep = indicesStep

    def indicesToProcessLeft(self):
        self.logger.info("At indices: {}, {}, {}".format(self.indexCount, self.indicesStep, self.indicesStepSize))
        return self.indexCount == self.indicesStep+1 or (self.indexCount == 0 and self.indicesStep == 0)


    def createIndex(self, sequence, fileName = None, retainInMemory=True):
        currentTupleSet = {}
        self.tupleSet = {}
        self.prevCount = self.indexCount
        self.indexCount = 0

        for window in self.wSize:
            if not os.path.isfile(self.pickleName(fileName, window)): 
                self.tupleSet = {}
                for seqId in xrange(len(sequence)):
                    self.logger.debug("Creating index of {} with window {}".format(sequence[seqId].id, window))
                    seq = str(sequence[seqId].seq.upper())
                    endIndex = window if len(seq)-window < 0 else len(seq)-window+1
                    revWindowSize = int(self.reverseWindowSize(window)*self.stepFactor*self.slideStep) 
                    for index in xrange(endIndex):
                        if index % revWindowSize == 0:
                            if self.indicesStep == None or (self.indicesStep < self.indexCount <= self.indicesStep + self.indicesStepSize) :    
                                comp = self.count(seq, window,index,index+int(window))
                                if comp not in self.tupleSet:
                                    self.tupleSet[comp] = []
                                self.tupleSet[comp].append((index, seqId))
                            self.indexCount += 1
                        if self.indexCount > self.indicesStep + self.indicesStepSize:
                            break
                    if self.indexCount > self.indicesStep + self.indicesStepSize:
                        break
                        
                if fileName != None and len(self.tupleSet) > 0:
                    self.pickle(fileName, window)
            elif retainInMemory:
                self.unpickleWindow(fileName, window)
                
            currentTupleSet.update(self.tupleSet)

        self.indicesStep += self.indicesStepSize
            
        if retainInMemory:
            self.tupleSet = currentTupleSet
        else:
            self.tupleSet = {}
            self.indexCount = self.indicesStep+1 if self.indexCount == 0 else self.indexCount
            
                    
    def createIndexAndStore(self, sequence, fileName, retainInMemory=True):
        self.createIndex(sequence, fileName, retainInMemory)

    def findIndices(self,seqs, start = 0.0, step=False):
        self.logger.error("Indexer.findIndices needs to by implemented by subclass. Maybe use QIndexer?")
        exit()
    
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
            if self.settings.compressed_index.upper() == "T":
                dump.write(zlib.compress(cPickle.dumps(self.tupleSet, cPickle.HIGHEST_PROTOCOL),9))
            else:
                cPickle.dump(self.tupleSet, dump, cPickle.HIGHEST_PROTOCOL)
            dump.close()
                
            self.logger.info("Done saving index to file.")
        except:
            self.logger.error("Could not open: " + self.pickleName(fileName, self.wSize[0] if window == None else window))
            

    def unpickleWindow(self, fileName, selectedWindow):
        self.logger.debug("unpickle file: "+ self.pickleName(fileName, selectedWindow))
        try:
            dump = open(self.pickleName(fileName, selectedWindow), "r")
            if self.settings.compressed_index.upper() == "T":
                tSet = cPickle.loads(zlib.decompress(dump.read()))
            else:
                tSet = cPickle.load(dump)
            for t in tSet:
                self.indexCount+= len(tSet[t])
            self.indexCount += 1 + self.prevCount if self.prevCount == 0 else self.prevCount     
            self.tupleSet.update(tSet)
            dump.close()
        except:
            self.logger.warning("Could not open pickle file: "+ self.pickleName(fileName, selectedWindow))
            self.logger.warning("Error: " +  str(sys.exc_info()[0]))
            return False
        return True

            
    def unpickle(self, fileName):
        self.tupleSet = {}
        allDone = True
        for window in self.wSize:
            allDone = allDone and self.unpickleWindow(fileName, window)
        if allDone:
            self.indicesStep += self.indicesStepSize
        return allDone
