from string import Template

import numpy
import scipy
from scipy.sparse import csc_matrix

import pycuda.driver as driver
from pycuda.compiler import SourceModule

from pyPaSWAS.Core import resource_filename, read_file
from pyPaSWAS.Core.Exceptions import HardwareException, InvalidOptionException

from QIndexer import QIndexer
from numpy import int32

class QIndexerCUDA(QIndexer):
    
    def __init__(self, settings, logger, stepFactor = 0.1, reads= [], qgram=1):
        QIndexer.__init__(self, settings, logger, stepFactor, reads, qgram)

        self._initialize_device(self.settings.device_number)
        self._init_memory_compAll()
    
    def pop_context(self):
        '''Destructor. Removes the current running context'''
        self.logger.debug('Destructing QIndexerCUDA.')
        if (driver.Context is not None): 
            driver.Context.pop()        
    
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
        #self.d_compAll = driver.mem_alloc(( self.indicesStepSize * (len(self.character_list)+1)*4))
    
    def _copy_index(self, compAll):
        driver.memcpy_htod(self.d_compAll, numpy.concatenate(compAll))

    def createIndex(self, sequence, fileName = None, retainInMemory=True):
        QIndexer.createIndex(self, sequence, fileName, retainInMemory)
        self.logger.info("Preparing index for GPU")
        keys = self.tupleSet.keys()
        compAll = [k.toarray() for k in keys]
        self._copy_index(compAll)
        
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
        self.logger.debug("Copying compositions of reads to device")
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
