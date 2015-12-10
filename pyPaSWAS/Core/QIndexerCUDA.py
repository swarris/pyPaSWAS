from string import Template

import numpy
import scipy
import math
from scipy.sparse import csc_matrix

import pycuda.driver as driver
from pycuda.compiler import SourceModule

from pyPaSWAS.Core import resource_filename, read_file
from pyPaSWAS.Core.Exceptions import HardwareException, InvalidOptionException

from QIndexer import QIndexer
from numpy import int32

class QIndexerCUDA(QIndexer):

    device_in_use = False

    def __init__(self, settings, logger, stepFactor = 0.1, reads= [], qgram=1, block = None, indicesStepSize= None, nAs = 'N'):
        QIndexer.__init__(self, settings, logger, stepFactor, reads, qgram)
        self.nAs = nAs
        if indicesStepSize != None:
            self.indicesStepSize = indicesStepSize

        if block != None:
            self.block =block

        if not QIndexerCUDA.device_in_use:
            self._initialize_device(self.settings.device_number)
            QIndexerCUDA.device_in_use = True
            self._init_memory_compAll()

    def pop_context(self):
        '''Destructor. Removes the current running context'''
        self.logger.debug('Destructing QIndexerCUDA.')
        if (driver.Context is not None):
            driver.Context.pop()
            QIndexerCUDA.device_in_use = False

    
    def _initialize_device(self, device_number):
        '''
        Initalizes the GPU device and verifies its computational abilities.
        @param device_number: int value representing the device to use
        '''
        self.logger.debug('Initializing device {0}'.format(device_number))
        '''Sets the device number'''
        try:
            self.device = int(device_number)
        except ValueError:
            raise InvalidOptionException('device should be an int but is {0}'.format(device))
        
        try:
            driver.init()  
            self.device = driver.Device(self.device)  
            self.device = self.device.make_context(flags=driver.ctx_flags.MAP_HOST).get_device()  
        except Exception as exception:
            raise HardwareException('Failed to initialize device. '
                                    'The following exception occurred: {0}'.format(str(exception))) 
        
        code = resource_filename(__name__, 'cuda/indexer.cu')
        code_t = Template(read_file(code))
        code = code_t.safe_substitute(size=len(self.character_list), block=self.block, stepSize=self.indicesStepSize)
        self.module = SourceModule(code)
        
        self.calculate_distance_function = self.module.get_function("calculateDistance")
        self.calculate_qgrams_function = self.module.get_function("calculateQgrams")
        self.setToZero_function = self.module.get_function("setToZero")

    def _init_memory(self):
        self.h_comp = driver.pagelocked_empty(( len(self.seqs) * (len(self.character_list)+1), 1), numpy.int32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_comp = numpy.intp(self.h_comp.base.get_device_pointer())

        self.h_distances = driver.pagelocked_empty(( len(self.seqs) * self.indicesStepSize, 1), numpy.float32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_distances = numpy.intp(self.h_distances.base.get_device_pointer())

        self.h_validComps = driver.pagelocked_empty(( len(self.seqs) * self.indicesStepSize, 1), numpy.int32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_validComps = numpy.intp(self.h_validComps.base.get_device_pointer())
        self.h_seqs = driver.pagelocked_empty(( len(self.seqs) * self.indicesStepSize, 1), numpy.int32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_seqs = numpy.intp(self.h_seqs.base.get_device_pointer())


    def _init_memory_compAll(self):
        self.h_compAll = driver.pagelocked_empty(( self.indicesStepSize * (len(self.character_list)+1), 1), numpy.int32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_compAll = numpy.intp(self.h_compAll.base.get_device_pointer())
        self.h_compAll_index = driver.pagelocked_empty(( self.indicesStepSize * (len(self.character_list)+1), 1), numpy.float32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_compAll_index = numpy.intp(self.h_compAll_index.base.get_device_pointer())
        #self.d_compAll = driver.mem_alloc(( self.indicesStepSize * (len(self.character_list)+1)*4))
    
    def _copy_index(self, compAll):
        driver.memcpy_htod(self.d_compAll, numpy.concatenate(compAll))

    def indicesToProcessLeft(self):
        self.logger.info("At indices: {}, {}, {}".format(self.indexCount, self.indicesStep, self.indicesStepSize))
        return self.indexCount <= self.indicesStep or (self.indexCount == 0 and self.indicesStep == 0)

    def createIndex(self, sequence, fileName = None, retainInMemory=True):
        #QIndexer.createIndex(self, sequence, fileName, retainInMemory)
        currentTupleSet = {}
        self.tupleSet = {}
        self.prevCount = self.indexCount
        self.indexCount = 0
        numberOfWindowsToCalculate = 0
        seqCompleted = False
        
        for window in self.wSize:
            self.tupleSet = {}
            seqId = 0
            #self.logger.debug("if window: {} < {} + {}".format(self.indexCount, self.indicesStep,  self.indicesStepSize))
            while not seqCompleted and seqId < len(sequence) and self.indexCount < self.indicesStep + self.indicesStepSize:
                # get sequence
                seq = str(sequence[seqId].seq)
                # calculate step through genome 
                revWindowSize = int(self.reverseWindowSize(window)*self.stepFactor*self.slideStep) 
                # see how many windows fit in this sequence
                numberOfWindows = int(math.ceil(len(seq) / revWindowSize))
                #self.logger.debug("Number of windows in {}: {}".format(sequence[seqId].id, numberOfWindows))
                #self.logger.debug("if: {} + {} <= {}".format(self.indexCount, numberOfWindows, self.indicesStep))
                if self.indexCount + numberOfWindows <= self.indicesStep:
                    # sequence already completely processed
                    self.indexCount += numberOfWindows
                    seqId += 1
                else:
                    # see how many windows can be calculate
                    seqCompleted = True
                    startWindow = self.prevCount -self.indexCount
                    #self.logger.debug("windows: {} {} {}".format(startWindow, self.indexCount, self.prevCount))
                    numberOfWindowsToCalculate = numberOfWindows - startWindow  
                    if numberOfWindowsToCalculate > self.indicesStepSize:
                        numberOfWindowsToCalculate = self.indicesStepSize

                    self.indexCount += startWindow + numberOfWindowsToCalculate
                    self.logger.debug("Creating index of {} with window {}".format(sequence[seqId].id, window))
                    self.logger.debug("Will process #windows: {}".format(numberOfWindowsToCalculate))
                    startIndex = startWindow * revWindowSize
                    endIndex = numberOfWindowsToCalculate * revWindowSize + startIndex + window
                    seqToIndex = str(sequence[seqId].seq[startIndex:endIndex]).upper() 
                    self.logger.debug("Indices: {}, {}. Len: {}".format(startIndex, endIndex, len(seqToIndex))) 
                    # memory on gpu for count should already be enough, so copy sequence to gpu
                    seqHost = numpy.array(seqToIndex, dtype=numpy.character)
                    seqMem = driver.pagelocked_empty((len(seqToIndex), 1), numpy.byte, mem_flags=driver.host_alloc_flags.DEVICEMAP) 
                    seqGPU = numpy.intp(seqMem.base.get_device_pointer()) #driver.mem_alloc(len(seqToIndex)) #
                    driver.memcpy_htod(seqGPU, seqHost)
                        
                    # set comps to zero
                    dim_grid = (self.indicesStepSize/self.block, self.block,1)
                    dim_block = (len(self.character_list), 1,1)
                    self.setToZero_function(self.d_compAll_index, block=dim_block, grid=dim_grid)

                    driver.Context.synchronize() 
                    # perform count on gpu 
                    dim_grid = (int(math.ceil(len(seqToIndex)/float(len(self.character_list)))), 1,1)
                    dim_block = (len(self.character_list), 1,1)
                    self.logger.debug("Calculating qgram per location in sequence. q={}, length={}, block={}, grid={}".format(
                                self.qgram, len(seqToIndex), dim_block, dim_grid))
                    self.calculate_qgrams_function(seqGPU, numpy.int32(self.qgram), numpy.int32(len(seqToIndex)), self.d_compAll_index,
                                    numpy.float32(window), numpy.float32(revWindowSize), numpy.float32(self.compositionScale / float(window-self.qgram+1)), 
                                     numpy.uint8(ord(self.nAs)),
                                     block=dim_block, 
                                     grid=dim_grid)
                    driver.Context.synchronize() 
                    comps = numpy.ndarray(buffer=self.h_compAll_index, dtype=numpy.float32, shape=(1,len(self.h_compAll_index)))[0].tolist()
                    # add comps to tuple set
                    for w in xrange(numberOfWindowsToCalculate):
                        count = tuple(comps[w*(len(self.character_list)+1):(w+1)*(len(self.character_list)+1)])
                        if count not in self.tupleSet:
                            self.tupleSet[count] = [] 
                        self.tupleSet[count].append((startIndex+ w*revWindowSize, seqId))
                    
                
                    currentTupleSet.update(self.tupleSet)
                    if endIndex >= len(sequence[seqId]):
                        seqId += 1

            if seqId == len(sequence):
                self.indexCount = self.indicesStep + self.indicesStepSize+1
        self.indicesStep = self.indexCount if self.indexCount <= self.indicesStep +self.indicesStepSize else self.indicesStep
                    
        self.tupleSet = currentTupleSet
        keys = self.tupleSet.keys()
        self.logger.info("Preparing index for GPU, size: {}".format(len(keys)))
        compAll = [numpy.array(k, dtype=numpy.int32) for k in keys]
        self._copy_index(compAll)
        #self._copy_index(keys)

    
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
        self.seqs = seqs
        keys = self.tupleSet.keys()
        
        # setup device parameters
        self.dim_grid = (self.indicesStepSize/self.block, self.block*len(seqs),1)
        self.dim_block = (len(self.character_list), 1,1)
        #init memory
        self._init_memory()

        # create index to see how many compositions are found:
        self.d_index_increment = driver.mem_alloc(4)
        index = numpy.zeros((1), dtype=numpy.int32)
        driver.memcpy_htod(self.d_index_increment, index)

        #find smallest window:
        loc = 0
        while loc < len(self.wSize)-1 and self.windowSize(len(seq)) > self.wSize[loc]:
            loc += 1

        # create numpy array with sequence compositions:
        comp = numpy.array([], dtype=numpy.int32)
        for seq in seqs:
            comp = numpy.append(comp,self.count(seq.seq.upper(), self.wSize[loc], 0, len(seq)).toarray())
        self.logger.debug("Copying {} compositions of reads to device".format(len(comp)))
        driver.memcpy_htod(self.d_comp, comp)
      
        self.logger.debug("Calculating distances")
        self.calculate_distance_function(self.d_compAll, self.d_comp, self.d_distances, self.d_validComps, self.d_seqs, self.d_index_increment, numpy.float32(self.compositionScale),
                                         numpy.int32(len(seqs)), numpy.int32(len(keys)), numpy.float32(self.sliceDistance), 
                                     block=self.dim_block, 
                                     grid=self.dim_grid)

        driver.Context.synchronize() 

        self.logger.debug("Getting distances")

        distances = numpy.ndarray(buffer=self.h_distances, dtype=numpy.float32, shape=(1,len(self.h_distances)))[0]
        
        numberOfValidComps = numpy.zeros((1), dtype=numpy.int32)
        driver.memcpy_dtoh(numberOfValidComps, self.d_index_increment)
        numberOfValidComps = numberOfValidComps[0]
        
        validComps = numpy.ndarray(buffer=self.h_validComps, dtype=numpy.int32, shape=(1,len(self.h_validComps)))[0]
        validSeqs = numpy.ndarray(buffer=self.h_seqs, dtype=numpy.int32, shape=(1,len(self.h_seqs)))[0]
        self.logger.debug("Process {} valid compositions".format(numberOfValidComps))
        hits = []
        for s in xrange(len(seqs)):
            hits.append({})
        for s in xrange(numberOfValidComps):

            valid = keys[validComps[s]]
            for hit in self.tupleSet[valid]:
                if hit[1] not in hits[validSeqs[s]]:
                    hits[validSeqs[s]][hit[1]] = []
                hits[validSeqs[s]][hit[1]].extend([(hit, self.wSize[loc], distances[s])])

        return hits

class GenomePlotter(QIndexerCUDA):

    def __init__(self, qindexer, reads = [], block = None,indicesStepSize= None, nAs = 'N'):
        QIndexerCUDA.__init__(self, qindexer.settings, qindexer.logger, qindexer.stepFactor, reads, qindexer.qgram, block, indicesStepSize, nAs)
        
        self.calculate_distance_function = qindexer.calculate_distance_function
        self.calculate_qgrams_function = qindexer.calculate_qgrams_function
        self.setToZero_function = qindexer.setToZero_function

        self.device = qindexer.device
        
        self._init_memory_compAll()

    
    def findDistances(self, qindexer, start = 0.0, step=False):
        self.seqs = qindexer.tupleSet.keys()
        keys = self.tupleSet.keys()
        
        # setup device parameters
        self.dim_grid = (self.indicesStepSize/self.block, self.block*len(self.seqs),1)
        self.dim_block = (len(self.character_list), 1,1)
        #init memory
        self._init_memory()

        # create index to see how many compositions are found:
        self.d_index_increment = driver.mem_alloc(4)
        index = numpy.zeros((1), dtype=numpy.int32)
        driver.memcpy_htod(self.d_index_increment, index)

      
        self.logger.debug("Calculating distances")
        self.calculate_distance_function(self.d_compAll, qindexer.d_compAll, self.d_distances, self.d_validComps, self.d_seqs, self.d_index_increment, numpy.float32(self.compositionScale),
                                         numpy.int32(len(self.seqs)), numpy.int32(len(keys)), numpy.float32(self.sliceDistance), 
                                     block=self.dim_block, 
                                     grid=self.dim_grid)

        driver.Context.synchronize() 

        self.logger.debug("Getting distances")

        distances = numpy.ndarray(buffer=self.h_distances, dtype=numpy.float32, shape=(1,len(self.h_distances)))[0]
        
        numberOfValidComps = numpy.zeros((1), dtype=numpy.int32)
        driver.memcpy_dtoh(numberOfValidComps, self.d_index_increment)
        numberOfValidComps = numberOfValidComps[0]
        
        validComps = numpy.ndarray(buffer=self.h_validComps, dtype=numpy.int32, shape=(1,len(self.h_validComps)))[0]
        validSeqs = numpy.ndarray(buffer=self.h_seqs, dtype=numpy.int32, shape=(1,len(self.h_seqs)))[0]
        self.logger.debug("Process {} valid compositions".format(numberOfValidComps))
        hits = []
        for s in xrange(len(self.seqs)):
            hits.append({})

        for s in xrange(numberOfValidComps):
            valid = keys[validComps[s]]
            for hit in self.tupleSet[valid]:
                if hit[1] not in hits[validSeqs[s]]:
                    hits[validSeqs[s]][hit[1]] = []
                for hitQindexer in qindexer.tupleSet[self.seqs[validSeqs[s]]]:
                    hits[validSeqs[s]][hit[1]].extend([(hit, hitQindexer , distances[s])])
        return hits


