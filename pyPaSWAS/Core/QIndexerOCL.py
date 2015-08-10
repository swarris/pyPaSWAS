import math
import numpy
import scipy
from scipy.sparse import csc_matrix
import pyopencl as cl
from string import Template

from QIndexer import QIndexer
from pyPaSWAS.Core import resource_filename, read_file
from pyPaSWAS.Core.Exceptions import HardwareException, InvalidOptionException

class QIndexerOCL(QIndexer):
    
    def __init__(self, settings, logger, stepFactor = 0.1, reads= [], qgram=1):
        QIndexer.__init__(self, settings, logger, stepFactor, reads, qgram)
        
        self.block = 10000
        
        self.device_type = 0
        self._set_device_type(self.settings.device_type)
        self._set_platform(self.settings.platform_name)
        
        self._initialize_device(int(self.settings.device_number))
        self._init_memory()
    
    def _set_device_type(self, device_type):
        '''Sets the device type'''
        if device_type.upper() == 'ACCELERATOR':
            self.device_type = cl.device_type.ACCELERATOR
        elif device_type.upper() == 'GPU':
            self.device_type = cl.device_type.GPU
        elif device_type.upper() == 'CPU':
            self.device_type = cl.device_type.CPU
        else:
            self.logger.warning("Warning: device type is set to default: GPU")
            self.device_type = cl.device_type.GPU
    
    def _set_platform(self, platform_name):
        found_platform = False
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                if (platform_name.upper() in str(platform).upper() 
                    and device.get_info(cl.device_info.TYPE) == self.device_type):
                    self.platform = platform
                    found_platform = True
                    break
            if(found_platform):
                self.logger.debug("Found platform {}".format(str(self.platform))) 
                break
        
        if not (self.platform):    
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    if (device.get_info(cl.device_info.TYPE) == self.device_type):
                        self.platform = platform
                        found_platform = True
                        break
                if(found_platform):
                    self.logger.debug('Found platform {}, however this is not the platform indicated by the user'.format(str(self.platform))) 
                    break
        
        if not (self.platform):
            raise RuntimeError('Failed to find platform')

    def _initialize_device(self, device_number):
        self.logger.debug('Initializing device {0}'.format(device_number))
        
        self.device = self.platform.get_devices(device_type=self.device_type)[device_number]
        self.ctx = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.ctx)
        
        code = resource_filename(__name__, 'ocl/indexer.cl')
        code_t = Template(read_file(code))
        code = code_t.safe_substitute(size=len(self.character_list), block=self.block)
        self.program = cl.Program(self.ctx, code).build()#options=['-D', 'NVIDIA'])        
        self.dim_grid = (len(self.character_list) *self.indicesStepSize/self.block, self.block)
        self.dim_block = (len(self.character_list), 1)


    def _init_memory(self):
        self.h_distances = numpy.empty(self.indicesStepSize).astype(numpy.float32)
        self.d_distances = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY| cl.mem_flags.ALLOC_HOST_PTR| cl.mem_flags.COPY_HOST_PTR, hostbuf=self.h_distances)

    def _copy_index(self, compAll):
        compAll = numpy.concatenate(compAll)
        self.d_compAll = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR| cl.mem_flags.COPY_HOST_PTR, hostbuf=compAll) 
        

    def createIndex(self, sequence, fileName = None, retainInMemory=True):
        QIndexer.createIndex(self, sequence, fileName, retainInMemory)
        self.logger.info("Preparing index for OpenCL Device")
        keys = self.tupleSet.keys()
        compAll = [k.toarray() for k in keys]
        self._copy_index(compAll)


    def findIndices(self,seq, start = 0.0, step=False):
        pass
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
        keys = self.tupleSet.keys()

        self.d_comp = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR| cl.mem_flags.COPY_HOST_PTR, hostbuf=comp.toarray())
        self.logger.debug("Starting device")
        self.program.calculateDistance(self.queue, self.dim_grid,self.dim_block, self.d_compAll, self.d_comp, self.d_distances, numpy.float32(self.compositionScale))
        self.logger.debug("Getting results")
        self.h_distances = cl.enqueue_map_buffer(self.queue, self.d_distances, cl.map_flags.READ, 0, (self.indicesStepSize*4, 1), dtype=numpy.byte)[0]
        
        #self.h_distances = numpy.array([0]*self.indicesStepSize, dtype=numpy.float32) 
        #cl.enqueue_copy(self.queue, self.d_distances, self.h_distances)
        distances = self.h_distances
        self.logger.debug("distance: ".format(len(self.h_distances)))
        validComp = [keys[x] for x in xrange(len(keys)) if keys[x].data[0] == comp.data[0] and distances[x]  < self.sliceDistance]
        exit()
        
        for valid in validComp:
            for hit in self.tupleSet[valid]:
                if hit[1] not in hits:
                    hits[hit[1]] = []
                hits[hit[1]].extend([(hit, self.wSize[loc], self.distance_calc(valid, comp))])

        return hits
