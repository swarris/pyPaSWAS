import pyopencl as cl
import numpy

from pyPaSWAS.Core.SmithWaterman import SmithWaterman
from pyPaSWAS.Core import STOP_DIRECTION, LEFT_DIRECTION, NO_DIRECTION, UPPER_DIRECTION, UPPER_LEFT_DIRECTION
from pyPaSWAS.Core.PaSWAS import CPUcode
from pyPaSWAS.Core.PaSWAS import GPUcode
from pyPaSWAS.Core.StartingPoint import StartingPoint

class SmithWatermanOcl(SmithWaterman):
    '''
    classdocs
    '''


    def __init__(self, logger, score, settings):
        '''
        Constructor
        '''
        SmithWaterman.__init__(self, logger, score, settings)
        
        #self.oclcode = OCLcode(self.logger)
        
        # platforms: A single ICD on a computer
        self.platform = None
        # device: device which will perform computation (for example a CPU or GPU)
        self.device =  None
        # context: manages a command-queue, memory, program and kernel objects
        self.ctx = None
        # queue: stores instructions for the device
        self.queue = None
        # program: the compiled kernel program
        self.program = None
        
        # device_type: type of device to run computations on
        self.device_type = 0
        self._set_device_type(self.settings.device_type)
        self._set_platform(self.settings.platform_name)
        self._initialize_device(int(self.settings.device_number))
    
    def _init_oclcode(self):
                # Compiling part of the CUDA code in advance
        self.oclcode.set_shared_xy_code(self.shared_x, self.shared_y)
        self.oclcode.set_direction_code(NO_DIRECTION, UPPER_LEFT_DIRECTION,
                                         UPPER_DIRECTION, LEFT_DIRECTION,
                                         STOP_DIRECTION)

    def _execute_calculate_score_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the calculate score kernel'''
        pass
        
    def _execute_traceback_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the traceback kernel'''
        pass
    
    def _get_direction_byte_array(self):
        '''
        Get the resulting directions
        @return gives the resulting direction array as byte array
        '''
        pass

    
    def __del__(self):
        '''Destructor. Removes the current running context'''
        del self.program
        del self.queue
        del self.ctx
        del self.device
        del self.platform
        
        self.device_type = 0

        
    def _set_device_type(self, device_type):
        '''Sets the device type'''
        if device_type.upper() == 'ACCELERATOR':
            self.device_type = cl.device_type.ACCELERATOR
        elif device_type.upper() == 'GPU':
            self.device_type = cl.device_type.GPU
        elif device_type.upper() == 'CPU':
            self.device_type = cl.device_type.CPU
        else:
            self.logger.debug("Warning: device type is set to default: CPU")
            self.device_type = cl.device_type.CPU
    
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
        '''
        Initalizes a device and verifies its computational abilities.
        @param device_number: int value representing the device to use
        '''
        self.logger.debug('Initializing device {0}'.format(device_number))
        
        self.device = self.platform.get_devices(device_type=self.device_type)[device_number]
        self.ctx = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.ctx)
        #self.logger.debug("context:{}".format(self.ctx) )
    
    def _device_global_mem_size(self):
        #return clCharacterize.usable_local_mem_size(self.device)
        #  GLOBAL_MEM_SIZE
        return self.device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    
    
    def _clear_memory(self):
        '''Clears the claimed memory on the device.'''
        self.logger.debug('Clearing device memory.')
        self._clear_normal_memory()
        self._clear_zero_copy_memory()
        
    def _clear_normal_memory(self):
        self.logger.debug('Clearing normal device memory.')
        if (self.d_sequences is not None):
            self.d_sequences.release() 
        if (self.d_targets is not None):
            self.d_targets.release()
        if (self.d_matrix is not None):
            self.d_matrix.release()
        if (self.d_global_maxima is not None):
            self.d_global_maxima.release()
            
    def _clear_zero_copy_memory(self):
        self.logger.debug('Clearing zero-copy device memory.')
        if (self.d_starting_points_zero_copy is not None):
            self.d_starting_points_zero_copy.release()
        if (self.d_global_direction_zero_copy is not None):
            self.d_global_direction_zero_copy.release()
        if (self.d_max_possible_score_zero_copy is not None):
            self.d_max_possible_score_zero_copy.release()

    def _init_normal_memory(self):
        '''
        #_init_memory will initialize all required memory on the device based on the current settings.
        Make sure to initialize these values!
        '''
        # Sequence device memory
        self.logger.debug('Initializing normal device memory.')
        memory = self.length_of_x_sequences * self.number_of_sequences
        self.d_sequences = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=memory)
        mem_size = memory
        
        # Target device memory
        memory = self.length_of_y_sequences * self.number_targets
        self.d_targets = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=memory)
        mem_size += memory
        
        return mem_size
        
    def _init_zero_copy_memory(self):
        
        self.logger.debug('Initializing zero-copy memory.')
        # Starting points host memory allocation and device copy
        memory = (self.size_of_startingpoint * self.maximum_number_starting_points * self.number_of_sequences *
        self.number_targets)
        self.d_starting_points_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=memory)
        mem_size = memory
        
        # Global directions host memory allocation and device copy
        memory = (self.length_of_x_sequences * self.number_of_sequences * self.length_of_y_sequences * self.number_targets)
        self.d_global_direction_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=memory)

        mem_size += memory
        
        # Maximum zero copy memory allocation and device copy
        memory = (self.number_of_sequences * self.number_of_targets * SmithWaterman.float_size)
        self.d_max_possible_score_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=memory)
        mem_size += memory
        
        return mem_size
        
    def _init_memory(self):
        mem_size = self._init_normal_memory()
        mem_size += self._init_zero_copy_memory()
        
        self.logger.debug('Allocated: {}MB of memory'.format(str(mem_size / 1024.0 / 1024.00)))
        

    def _init_zero_copy(self):
        ''' Initializes the index used for the 'zero copy' of the found starting points '''
        self.d_index_increment = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=SmithWaterman.int_size)
        index = numpy.zeros((1), dtype=numpy.int32)
        cl.enqueue_write_buffer(self.queue, self.d_index_increment, index).wait()

    def _compile_code(self):
        """Compile the device code with current settings"""
        self.logger.debug('Compiling OpenCL code.')
        code = self.oclcode.get_code(self.score, self.number_of_sequences, self.number_targets, self.length_of_x_sequences, self.length_of_y_sequences)
        #self.logger.debug('Code: \n{}'.format(code))
        self.program = cl.Program(self.ctx, code).build()
    
    def copy_sequences(self, h_sequences, h_targets):
        '''
        Copy the sequences and targets to the device
        @param h_sequences: the sequences to be copied. Should be a single string containing all sequences
        @param h_targets: the targets to be copied. Should be a single string containing all sequences
        '''
        cl.enqueue_copy(self.queue, self.d_sequences, h_sequences).wait()
        cl.enqueue_copy(self.queue, self.d_targets, h_targets).wait()
        
    def _get_number_of_starting_points(self):
        ''' Returns the number of startingpoints. '''
        self.logger.debug('Getting number of starting points.')
        self.index = numpy.zeros((1), dtype=numpy.int32)
        cl.enqueue_copy(self.queue, self.index, self.d_index_increment)
        return self.index[0]
    
    def _fill_max_possible_score(self, target_index, targets, i, index, records_seqs):
        for tI in range(self.number_of_targets):
            if tI+target_index < len(targets) and i+index < len(records_seqs):
                self.set_minimum_score(tI*self.max_sequences + i, float(self.score.highest_score) * (len(records_seqs[i+index]) 
                                                                                                     if len(records_seqs[i+index]) < len(targets[tI+target_index]) 
                                                                                                     else len(targets[tI+target_index])) * float(self.filter_factor))

    
    def _set_max_possible_score(self, target_index, targets, i, index, records_seqs):
        '''fills the max_possible_score datastructure on the host'''
        self.h_max_possible_score_zero_copy = cl.enqueue_map_buffer(self.queue, self.d_max_possible_score_zero_copy, 
                                                                    cl.map_flags.WRITE, 0, 
                                                                    self.number_of_sequences * self.number_targets , 
                                                                    dtype=numpy.float32)[0]
        self._fill_max_possible_score(target_index, targets, i, index, records_seqs)
        #Unmap memory object
        del self.h_max_possible_score_zero_copy
        
    def _get_starting_point_byte_array(self):
        '''
        Get the resulting starting points
        @return gives the resulting starting point array as byte array
        '''
        self.h_starting_points_zero_copy = cl.enqueue_map_buffer(self.queue, self.d_starting_points_zero_copy, cl.map_flags.READ, 0, 
                                                                 (self.size_of_startingpoint * 
                                                                  self.maximum_number_starting_points * 
                                                                  self.number_of_sequences *
                                                                  self.number_targets, 1), dtype=numpy.byte)[0]
        return self.h_starting_points_zero_copy
            
    def _print_alignments(self, sequences, targets, start_seq, start_target, hit_list=None):
        SmithWaterman._print_alignments(self, sequences, targets, start_seq, start_target, hit_list)
        #unmap memory objects
        del self.h_global_direction_zero_copy
        del self.h_starting_points_zero_copy
                
    
class SmithWatermanCPU(SmithWatermanOcl):
    '''
    classdocs
    '''


    def __init__(self, logger, score, settings):
        '''
        Constructor
        '''
        SmithWatermanOcl.__init__(self, logger, score, settings)
        
        self.oclcode = CPUcode(self.logger)
        self.workload_x = 4
        self.workload_y = 4
        
        self.workgroup_x = self.shared_x // self.workload_x
        self.workgroup_y = self.shared_y // self.workload_y
        
        self.d_semaphores = None
        
        self._init_oclcode()
        
    def _init_normal_memory(self):
        mem_size = SmithWatermanOcl._init_normal_memory(self)
        
        # Input matrix device memory
        memory = (SmithWaterman.float_size * (self.length_of_x_sequences + 1) * self.number_of_sequences *
        (self.length_of_y_sequences + 1) * self.number_targets)
        self.d_matrix = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory
        
        pattern = numpy.zeros((1),dtype=numpy.float32)
        cl.enqueue_fill_buffer(self.queue, self.d_matrix, pattern, 0, size = memory)
        
        # Maximum global device memory
        memory = (SmithWaterman.float_size * self.x_div_shared_x * self.number_of_sequences *
        self.y_div_shared_y * self.number_targets * self.workload_x * self.workload_y)
        self.d_global_maxima = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory
        

        memory = (SmithWaterman.int_size * 
                  self.length_of_x_sequences * 
                  self.number_of_sequences * 
                  self.length_of_y_sequences *
                  self.number_targets)
        
        self.d_semaphores = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        pattern = numpy.zeros((1),dtype=numpy.int32)
        cl.enqueue_fill_buffer(self.queue, self.d_semaphores, pattern, 0, size=memory)

        mem_size += memory
        
        return mem_size
    
    def _get_direction_byte_array(self):
        '''
        Get the resulting directions
        @return gives the resulting direction array as byte array
        '''
        self.h_global_direction_zero_copy = cl.enqueue_map_buffer(self.queue, self.d_global_direction_zero_copy, cl.map_flags.READ, 0, 
                                                                  (self.number_of_sequences,
                                                                   self.number_targets,
                                                                   self.length_of_x_sequences,
                                                                   self.length_of_y_sequences), dtype=numpy.byte)[0]
        return self.h_global_direction_zero_copy
    
    def _get_direction(self, direction_array, sequence, target, block_x, block_y, value_x, value_y):
<<<<<<< HEAD
        #return direction_array[sequence][target][starting_point.value_x][starting_point.value_y]
=======
        #self.logger.debug("{}, {}, {}, {}".format(sequence,target,block_x*self.workload_x + value_x,block_y*self.workload_y + value_y))
        #return direction_array[sequence][target][block_x*self.workload_x + value_x][block_y*self.workload_y + value_y]
>>>>>>> branch 'pyopencl' of https://github.com/swarris/pyPaSWAS.git
        return direction_array[sequence][target][block_x*self.shared_x + value_x][block_y*self.shared_y + value_y]
    
    def _set_direction(self, direction, direction_array, sequence, target, block_x, block_y, value_x, value_y):
<<<<<<< HEAD
        #direction_array[sequence][target][starting_point.value_x][starting_point.value_y] = direction
        direction_array[sequence][target][block_x*self.shared_x + value_x][block_y*self.shared_y + value_y] = direction 
=======
        #direction_array[sequence][target][block_x*self.workload_x + value_x][block_y*self.workload_y + value_y] = direction
        direction_array[sequence][target][block_x*self.shared_x + value_x][block_y*self.shared_y + value_y] = direction

>>>>>>> branch 'pyopencl' of https://github.com/swarris/pyPaSWAS.git
        
    def _execute_calculate_score_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the calculate score kernel'''
        dim_block = (self.workgroup_x, self.workgroup_y)
        dim_grid_sw = (self.number_of_sequences * self.workgroup_x, self.number_targets * number_of_blocks * self.workgroup_y)

        self.program.calculateScore(self.queue, 
                                    dim_grid_sw, 
                                    dim_block, 
                                    self.d_matrix, 
                                    numpy.int32(idx), 
                                    numpy.int32(idy),
                                    numpy.int32(number_of_blocks), 
                                    self.d_sequences, 
                                    self.d_targets,
                                    self.d_global_maxima, 
                                    self.d_global_direction_zero_copy).wait()
    
    def _execute_traceback_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the traceback kernel'''
        dim_block = (self.workgroup_x, self.workgroup_y)
        dim_grid_sw = (self.number_of_sequences * self.workgroup_x, self.number_targets * number_of_blocks * self.workgroup_y)
        self.program.traceback(self.queue, dim_grid_sw, dim_block,
                               self.d_matrix,
                               numpy.int32(idx),
                               numpy.int32(idy),
                               numpy.int32(number_of_blocks),
                               self.d_global_maxima,
                               self.d_global_direction_zero_copy,
                               self.d_index_increment,
                               self.d_starting_points_zero_copy,
                               self.d_max_possible_score_zero_copy,
                               self.d_semaphores).wait()
                               
    def _clear_memory(self):
        SmithWatermanOcl._clear_memory(self)
        if (self.d_semaphores is not None):
            self.d_semaphores.release()


class SmithWatermanGPU(SmithWatermanOcl):
    '''
    classdocs
    '''


    def __init__(self, logger, score, settings):
        '''
        Constructor
        '''
        SmithWatermanOcl.__init__(self, logger, score, settings)
        self.oclcode = GPUcode(self.logger)
        self._init_oclcode()
        
    def _init_normal_memory(self):
        
        mem_size = SmithWatermanOcl._init_normal_memory(self)
        
        # Input matrix device memory
        memory = (SmithWaterman.float_size * self.length_of_x_sequences * self.number_of_sequences *
        self.length_of_y_sequences * self.number_targets)
        self.d_matrix = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory
        
        # Maximum global device memory
        memory = (SmithWaterman.float_size * self.x_div_shared_x * self.number_of_sequences *
        self.y_div_shared_y * self.number_targets)
        self.d_global_maxima = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory

        return mem_size
    
    def _get_direction_byte_array(self):
        '''
        Get the resulting directions
        @return gives the resulting direction array as byte array
        '''
        self.h_global_direction_zero_copy = cl.enqueue_map_buffer(self.queue, self.d_global_direction_zero_copy, cl.map_flags.READ, 0, 
                                                                  (self.number_of_sequences,
                                                                   self.number_targets,
                                                                   self.x_div_shared_x,
                                                                   self.y_div_shared_y,
                                                                   self.shared_x,
                                                                   self.shared_y), dtype=numpy.byte)[0]
        return self.h_global_direction_zero_copy
        

    def _execute_calculate_score_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the calculate score kernel'''
        dim_block = (self.shared_x, self.shared_y)
        dim_grid_sw = (self.number_of_sequences * self.shared_x, self.number_targets * number_of_blocks * self.shared_y)
        
        self.program.calculateScore(self.queue, 
                                    dim_grid_sw, 
                                    dim_block, 
                                    self.d_matrix, 
                                    numpy.int32(idx), 
                                    numpy.int32(idy),
                                    numpy.int32(number_of_blocks), 
                                    self.d_sequences, 
                                    self.d_targets,
                                    self.d_global_maxima, 
                                    self.d_global_direction_zero_copy).wait()
                                    
                                    
    
    def _execute_traceback_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the traceback kernel'''
        dim_block = (self.shared_x, self.shared_y)
        dim_grid_sw = (self.number_of_sequences * self.shared_x, self.number_targets * number_of_blocks * self.shared_y)
        self.program.traceback(self.queue, dim_grid_sw, dim_block,
                               self.d_matrix,
                               numpy.int32(idx),
                               numpy.int32(idy),
                               numpy.int32(number_of_blocks),
                               self.d_global_maxima,
                               self.d_global_direction_zero_copy,
                               self.d_index_increment,
                               self.d_starting_points_zero_copy,
                               self.d_max_possible_score_zero_copy).wait()
                            
    
class SmithWatermanNVIDIA(SmithWatermanGPU):
    '''
    classdocs
    '''


    def __init__(self, logger, score, settings):
        '''
        Constructor
        '''
        SmithWatermanGPU.__init__(self, logger, score, settings)
        self.pinned_starting_points_zero_copy = None
        self.pinned_max_possible_score_zero_copy = None
        self.pinned_global_direction_zero_copy = None
        self._init_oclcode()
        
    def _init_zero_copy_memory(self):
        self.logger.debug('Initializing NVIDIA zero-copy memory.')
        # Starting points host memory allocation and device copy
        memory = (self.size_of_startingpoint * self.maximum_number_starting_points * self.number_of_sequences *
        self.number_targets)
        self.pinned_starting_points_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.ALLOC_HOST_PTR, size=memory)
        self.d_starting_points_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=memory)
        self.h_starting_points_zero_copy = cl.enqueue_map_buffer(self.queue, self.pinned_starting_points_zero_copy, cl.map_flags.READ, 0, 
                                                                (memory, 1), dtype=numpy.byte)[0]
        mem_size = memory

        # Global directions host memory allocation and device copy
        memory = (self.length_of_x_sequences * self.number_of_sequences * self.length_of_y_sequences *
        self.number_targets)
        self.pinned_global_direction_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.ALLOC_HOST_PTR, size=memory)
        self.d_global_direction_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=memory)
        self.h_global_direction_zero_copy = cl.enqueue_map_buffer(self.queue, self.pinned_global_direction_zero_copy, cl.map_flags.READ, 0, 
                                                                    (memory, 1), dtype=numpy.byte)[0] 
        mem_size += memory
        
                
        # Maximum zero copy memory allocation and device copy
        memory = (self.number_of_sequences * self.number_of_targets * SmithWaterman.float_size)
        self.pinned_max_possible_score_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.ALLOC_HOST_PTR, size=memory)
        self.d_max_possible_score_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=memory)
        self.h_max_possible_score_zero_copy = cl.enqueue_map_buffer(self.queue, self.pinned_max_possible_score_zero_copy, cl.map_flags.WRITE, 0, 
                                                                    (self.number_of_sequences * self.number_of_targets, 1), dtype=numpy.float32)[0]
        mem_size += memory
        
        # Zero copy buffers are allocated twice in NVIDIA        
        return 2*mem_size
    
    def _set_max_possible_score(self, target_index, targets, i, index, records_seqs):
        cl.enqueue_copy(self.queue, self.d_max_possible_score_zero_copy, self.h_max_possible_score_zero_copy).wait()
        self._fill_max_possible_score(target_index, targets, i, index, records_seqs)
        
    def _get_direction_byte_array(self):
#        cl.enqueue_copy(self.queue, self.h_global_direction_zero_copy, self.d_global_direction_zero_copy).wait()
        self.h_global_direction_zero_copy = cl.enqueue_map_buffer(self.queue, self.d_global_direction_zero_copy, cl.map_flags.READ, 0, 
                                                                  (self.number_of_sequences,
                                                                   self.number_targets,
                                                                   self.x_div_shared_x,
                                                                   self.y_div_shared_y,
                                                                   self.shared_x,
                                                                   self.shared_y), dtype=numpy.byte)[0]

        return self.h_global_direction_zero_copy
    
    def _get_starting_point_byte_array(self):
        cl.enqueue_copy(self.queue, self.h_starting_points_zero_copy, self.d_starting_points_zero_copy).wait()
        return self.h_starting_points_zero_copy
    
    def _clear_zero_copy_memory(self):
        SmithWatermanGPU._clear_zero_copy_memory(self)
                
        if (self.pinned_starting_points_zero_copy is not None):
            self.pinned_starting_points_zero_copy.release()
        if (self.pinned_global_direction_zero_copy is not None):
            self.pinned_global_direction_zero_copy.release()
        if (self.pinned_max_possible_score_zero_copy is not None):
            self.pinned_max_possible_score_zero_copy.release()
            
    def _compile_ocl_code(self):
        """Compile the OpenCL code with current settings"""
        self.logger.debug('Compiling NVIDIA OpenCL code.')
        code = self.oclcode.get_code(self.score, self.number_of_sequences, self.number_targets, self.length_of_x_sequences, self.length_of_y_sequences)
        self.program = cl.Program(self.ctx, code).build(options=['-D', 'NVIDIA'])
            

        
    
