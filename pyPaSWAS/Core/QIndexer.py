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
        
        if len(seq) > 0 and end_index - start_index > 0:
            n = seq.count("N", start_index, end_index)
            length = float(end_index - start_index - n)

            if length-self.qgram+1 > 0 :
                fraction = self.compositionScale / float(length-self.qgram+1)
                for qgram_string in range(start_index, end_index-self.qgram):
                    subStr = str(seq[qgram_string:qgram_string + self.qgram])
                    
                    if "N" not in subStr and len(subStr.strip()) > 0:
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
        keys = self.tupleSet.keys()
        compAll = keys
        
        
        distances = [self.distance_calc(a, comp) for a in compAll] 

        validComp = [keys[x] for x in xrange(len(keys)) if keys[x].data[0] == comp.data[0] and distances[x]  < self.sliceDistance]
        
        for valid in validComp:
            for hit in self.tupleSet[valid]:
                if hit[1] not in hits:
                    hits[hit[1]] = []
                hits[hit[1]].extend([(hit, self.wSize[loc], self.distance_calc(valid, comp))])
        return hits
    

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
        self.program = cl.Program(self.ctx, code).build()        
        self.dim_grid = (self.indicesStepSize/self.block, self.block)
        self.dim_block = (len(self.character_list), 1,1)


    def _init_memory(self):
        pass
        self.h_comp = driver.pagelocked_empty(((len(self.character_list)+1), 1), numpy.int32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_comp = numpy.intp(self.h_comp.base.get_device_pointer())

        self.h_compAll = driver.pagelocked_empty(( self.indicesStepSize * (len(self.character_list)+1), 1), numpy.int32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_compAll = numpy.intp(self.h_compAll.base.get_device_pointer())

        self.h_distances = driver.pagelocked_empty(( self.indicesStepSize, 1), numpy.float32, mem_flags=driver.host_alloc_flags.DEVICEMAP)
        self.d_distances = numpy.intp(self.h_distances.base.get_device_pointer())

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
        compAll = [k.toarray() for k in keys]
        
        driver.memcpy_htod(self.d_comp, comp.toarray())
        driver.memcpy_htod(self.d_compAll, numpy.concatenate(compAll))
      
        self.calculate_distance_function(self.d_compAll, self.d_comp, self.d_distances, numpy.float32(self.compositionScale), 
                                     block=self.dim_block, 
                                     grid=self.dim_grid)
        
        driver.Context.synchronize() 

        distances = cl.enqueue_map_buffer(self.queue, self.d_distances, cl.map_flags.READ, 0, (len(self.h_distances), 1), dtype=numpy.float32)[0]

        validComp = [keys[x] for x in xrange(len(keys)) if keys[x].data[0] == comp.data[0] and distances[x]  < self.sliceDistance]
        
        for valid in validComp:
            for hit in self.tupleSet[valid]:
                if hit[1] not in hits:
                    hits[hit[1]] = []
                hits[hit[1]].extend([(hit, self.wSize[loc], self.distance_calc(valid, comp))])

        return hits
