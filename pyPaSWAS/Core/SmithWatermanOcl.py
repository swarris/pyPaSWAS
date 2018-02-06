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

        self.always_reallocate_memory = False

    def _init_oclcode(self):
                # Compiling part of the OpenCL code in advance
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
        '''
        Initalizes a device and verifies its computational abilities.
        @param device_number: int value representing the device to use
        '''
        self.logger.debug('Initializing device {0}'.format(device_number))
        
        self.device = self.platform.get_devices(device_type=self.device_type)[device_number] 
        if int(self.settings.number_of_compute_units) > 0:
            self.device = self.device.create_sub_devices([cl.device_partition_property.EQUALLY,int(self.settings.number_of_compute_units)])[int(self.settings.sub_device)]

        self.ctx = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.ctx)
        #self.logger.debug("context:{}".format(self.ctx) )
    
    def _device_global_mem_size(self):
        #return clCharacterize.usable_local_mem_size(self.device)
        #  GLOBAL_MEM_SIZE
        return self.device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    
    
    def _clear_memory(self):
        '''Clears the claimed memory on the device.'''
        if not self.always_reallocate_memory:
            return
        self.logger.debug('Clearing device memory.')
        self._clear_normal_memory()
        self._clear_zero_copy_memory()
        try: 
            self.queue.finish()
        except:
            pass
        
    def _clear_normal_memory(self):
        self.logger.debug('Clearing normal device memory.')
        if (self.d_sequences is not None):
            try: 
                self.d_sequences.finish()
            except:
                pass
            self.d_sequences.release() 
        if (self.d_targets is not None):
            try:
                self.d_targets.finish()
            except:
                pass
            self.d_targets.release()
        if (self.d_matrix is not None):
            try:
                self.d_matrix.finish()
            except:
                pass
            self.d_matrix.release()
        if (self.gap_extension and self.d_matrix_i is not None):
            try:
                self.d_matrix_i.finish()
            except:
                pass
            self.d_matrix_i.release()
        if (self.gap_extension and self.d_matrix_j is not None):
            try:
                self.d_matrix_j.finish()
            except:
                pass
            self.d_matrix_j.release()
        if (self.d_global_maxima is not None):
            try: 
                self.d_global_maxima.finish()
            except:
                pass
            self.d_global_maxima.release()
        if (self.d_index_increment is not None):
            try: 
                self.d_index_increment.finish()
            except:
                pass
            self.d_index_increment.release()

    def _clear_zero_copy_memory(self):
        self.logger.debug('Clearing zero-copy device memory.')
        if (self.d_starting_points_zero_copy is not None):
            try:
                self.d_starting_points_zero_copy.finish()
            except:
                pass
            self.d_starting_points_zero_copy.release()
        if (self.d_max_possible_score_zero_copy is not None):
            try:
                self.d_max_possible_score_zero_copy.finish()
            except:
                pass
            self.d_max_possible_score_zero_copy.release()

    def _need_reallocation(self, buffer, size):
        if self.always_reallocate_memory:
            return True
        if buffer is None:
            return True
        if buffer.get_info(cl.mem_info.SIZE) < size:
            try:
                buffer.finish()
            except:
                pass
            buffer.release()
            return True
        return False

    def _init_normal_memory(self):
        '''
        #_init_memory will initialize all required memory on the device based on the current settings.
        Make sure to initialize these values!
        '''
        # Sequence device memory
        self.logger.debug('Initializing normal device memory.')
        memory = self.length_of_x_sequences * self.number_of_sequences
        if self._need_reallocation(self.d_sequences, memory):
            self.d_sequences = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=memory)
        mem_size = memory
        
        # Target device memory
        memory = self.length_of_y_sequences * self.number_targets
        if self._need_reallocation(self.d_targets, memory):
            self.d_targets = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, size=memory)
        mem_size += memory

        if self._need_reallocation(self.d_index_increment, SmithWaterman.int_size):
            self.d_index_increment = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=SmithWaterman.int_size)

        return mem_size
        
    def _init_zero_copy_memory(self):
        
        self.logger.debug('Initializing zero-copy memory.')
        # Starting points host memory allocation and device copy
        memory = (self.size_of_startingpoint * self.maximum_number_starting_points * self.number_of_sequences *
        self.number_targets)
        if self._need_reallocation(self.d_starting_points_zero_copy, memory):
            self.d_starting_points_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=memory)
        mem_size = memory

        # Maximum zero copy memory allocation and device copy
        memory = (self.number_of_sequences * self.number_of_targets * SmithWaterman.float_size)
        #self.d_max_possible_score_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=memory)
        mem_size += memory

        return mem_size
        
    def _init_memory(self):
        mem_size = self._init_normal_memory()
        mem_size += self._init_zero_copy_memory()
        
        self.logger.debug('Allocated: {}MB of memory'.format(str(mem_size / 1024.0 / 1024.00)))
        

    def _init_zero_copy(self):
        ''' Initializes the index used for the 'zero copy' of the found starting points '''
        index = numpy.zeros((1), dtype=numpy.int32)
        cl.enqueue_write_buffer(self.queue, self.d_index_increment, index)

    def _compile_code(self):
        """Compile the device code with current settings"""
        self.logger.debug('Compiling OpenCL code.')
        code = self.oclcode.get_code(self.score, self.number_of_sequences, self.number_targets, self.length_of_x_sequences, self.length_of_y_sequences)
        #self.logger.debug('Code: \n{}'.format(code))
        self.program = cl.Program(self.ctx, code).build()
        self.calculateScoreAffineGap_kernel = self.program.calculateScoreAffineGap
        self.calculateScore_kernel = self.program.calculateScore
        self.tracebackAffineGap_kernel = self.program.tracebackAffineGap
        self.traceback_kernel = self.program.traceback
    
    def copy_sequences(self, h_sequences, h_targets):
        '''
        Copy the sequences and targets to the device
        @param h_sequences: the sequences to be copied. Should be a single string containing all sequences
        @param h_targets: the targets to be copied. Should be a single string containing all sequences
        '''
        cl.enqueue_copy(self.queue, self.d_sequences, h_sequences, is_blocking=False)
        cl.enqueue_copy(self.queue, self.d_targets, h_targets, is_blocking=False)
        
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

    def _copy_min_score(self):
        if self._need_reallocation(self.d_max_possible_score_zero_copy, self.min_score_np.nbytes):
            self.d_max_possible_score_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.min_score_np.nbytes)
        cl.enqueue_copy(self.queue, self.d_max_possible_score_zero_copy, self.min_score_np, is_blocking=False)

        
    def _set_max_possible_score(self, target_index, targets, i, index, records_seqs):
        '''fills the max_possible_score datastructure on the host'''
#        self.h_max_possible_score_zero_copy = cl.enqueue_map_buffer(self.queue, self.d_max_possible_score_zero_copy, 
#                                                                    cl.map_flags.WRITE, 0, 
#                                                                    self.number_of_sequences * self.number_targets , 
#                                                                    dtype=numpy.float32)[0]
        self._fill_max_possible_score(target_index, targets, i, index, records_seqs)
        #Unmap memory object
#        del self.h_max_possible_score_zero_copy
        
    def _get_starting_point_byte_array(self, number_of_starting_points):
        '''
        Get the resulting starting points
        @return gives the resulting starting point array as byte array
        '''
        self.h_starting_points_zero_copy = cl.enqueue_map_buffer(self.queue, self.d_starting_points_zero_copy, cl.map_flags.READ, 0, 
                                                                 (self.size_of_startingpoint * 
                                                                  number_of_starting_points, 1), dtype=numpy.byte)[0]
        return self.h_starting_points_zero_copy


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
        if self._need_reallocation(self.d_matrix, memory):
            self.d_matrix = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory

        pattern = numpy.zeros((1),dtype=numpy.float32)
        cl.enqueue_fill_buffer(self.queue, self.d_matrix, pattern, 0, size = memory)

        if self.gap_extension:
            if self._need_reallocation(self.d_matrix_i, memory):
                self.d_matrix_i = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
            mem_size += memory
            if self._need_reallocation(self.d_matrix_j, memory):
                self.d_matrix_j = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
            mem_size += memory
            pattern = numpy.array([-1E10],dtype=numpy.float32)
            cl.enqueue_fill_buffer(self.queue, self.d_matrix_i, pattern, 0, size = memory)
            cl.enqueue_fill_buffer(self.queue, self.d_matrix_j, pattern, 0, size = memory)

        
        # Maximum global device memory
        memory = (SmithWaterman.float_size * self.x_div_shared_x * self.number_of_sequences *
        self.y_div_shared_y * self.number_targets * self.workload_x * self.workload_y)
        if self._need_reallocation(self.d_global_maxima, memory):
            self.d_global_maxima = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory
        

        memory = (SmithWaterman.int_size * 
                  self.length_of_x_sequences * 
                  self.number_of_sequences * 
                  self.length_of_y_sequences *
                  self.number_targets)
        
        if self._need_reallocation(self.d_semaphores, memory):
            self.d_semaphores = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        pattern = numpy.zeros((1),dtype=numpy.int32)
        cl.enqueue_fill_buffer(self.queue, self.d_semaphores, pattern, 0, size=memory)

        mem_size += memory
        
        return mem_size

    def _init_zero_copy_memory(self):
        mem_size = SmithWatermanOcl._init_zero_copy_memory(self)

        # Global directions host memory allocation and device copy
        memory = (self.length_of_x_sequences * self.number_of_sequences * self.length_of_y_sequences * self.number_targets)
        if self._need_reallocation(self.d_global_direction_zero_copy, memory):
            self.d_global_direction_zero_copy = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=memory)
        mem_size += memory

        return mem_size

    def _clear_normal_memory(self):
        SmithWatermanOcl._clear_normal_memory(self)
        if (self.d_semaphores is not None):
            try:
                self.d_semaphores.finish()
            except:
                pass
            self.d_semaphores.release()

    def _clear_zero_copy_memory(self):
        SmithWatermanOcl._clear_zero_copy_memory(self)
        if (self.d_global_direction_zero_copy is not None):
            try:
                self.d_global_direction_zero_copy.finish()
            except:
                pass
            self.d_global_direction_zero_copy.release()

    def _get_direction_byte_array(self):
        '''
        Get the resulting directions
        @return gives the resulting direction array as byte array
        '''
        h_global_direction_zero_copy = cl.enqueue_map_buffer(self.queue, self.d_global_direction_zero_copy, cl.map_flags.READ, 0, 
                                                             (self.number_of_sequences,
                                                              self.number_targets,
                                                              self.length_of_x_sequences,
                                                              self.length_of_y_sequences), dtype=numpy.byte)[0]
        return h_global_direction_zero_copy

    def _get_direction(self, direction_array, sequence, target, block_x, block_y, value_x, value_y):
        return direction_array[sequence][target][block_x*self.shared_x + value_x][block_y*self.shared_y + value_y]
    
    def _set_direction(self, direction, direction_array, sequence, target, block_x, block_y, value_x, value_y):
        direction_array[sequence][target][block_x*self.shared_x + value_x][block_y*self.shared_y + value_y] = direction

        
    def _execute_calculate_score_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the calculate score kernel'''
        dim_block = (self.workgroup_x, self.workgroup_y)
        dim_grid_sw = (self.number_of_sequences * self.workgroup_x, self.number_targets * number_of_blocks * self.workgroup_y)

        if self.gap_extension:
            self.calculateScoreAffineGap_kernel(self.queue,
                                                dim_grid_sw,
                                                dim_block,
                                                self.d_matrix,
                                                self.d_matrix_i,
                                                self.d_matrix_j,
                                                numpy.int32(idx),
                                                numpy.int32(idy),
                                                numpy.int32(number_of_blocks),
                                                self.d_sequences,
                                                self.d_targets,
                                                self.d_global_maxima,
                                                self.d_global_direction_zero_copy)
        else:
            self.calculateScore_kernel(self.queue, 
                                       dim_grid_sw,
                                       dim_block,
                                       self.d_matrix,
                                       numpy.int32(idx),
                                       numpy.int32(idy),
                                       numpy.int32(number_of_blocks),
                                       self.d_sequences,
                                       self.d_targets,
                                       self.d_global_maxima,
                                       self.d_global_direction_zero_copy)
                   
    def _execute_traceback_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the traceback kernel'''
        dim_block = (self.workgroup_x, self.workgroup_y)
        dim_grid_sw = (self.number_of_sequences * self.workgroup_x, self.number_targets * number_of_blocks * self.workgroup_y)
        if self.gap_extension:
            self.tracebackAffineGap_kernel(self.queue, dim_grid_sw, dim_block,
                                           self.d_matrix,
                                           self.d_matrix_i,
                                           self.d_matrix_j,
                                           numpy.int32(idx),
                                           numpy.int32(idy),
                                           numpy.int32(number_of_blocks),
                                           self.d_global_maxima,
                                           self.d_global_direction_zero_copy,
                                           self.d_index_increment,
                                           self.d_starting_points_zero_copy,
                                           self.d_max_possible_score_zero_copy,
                                           self.d_semaphores)
        else:
            self.traceback_kernel(self.queue, dim_grid_sw, dim_block,
                                  self.d_matrix,
                                  numpy.int32(idx),
                                  numpy.int32(idy),
                                  numpy.int32(number_of_blocks),
                                  self.d_global_maxima,
                                  self.d_global_direction_zero_copy,
                                  self.d_index_increment,
                                  self.d_starting_points_zero_copy,
                                  self.d_max_possible_score_zero_copy,
                                  self.d_semaphores)


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

        self.d_global_direction = None
        self.d_is_traceback_required = None

        self._init_oclcode()
        
    def _init_normal_memory(self):
        
        mem_size = SmithWatermanOcl._init_normal_memory(self)
        
        # Input matrix device memory
        memory = (SmithWaterman.float_size * self.length_of_x_sequences * self.number_of_sequences *
        self.length_of_y_sequences * self.number_targets)
        if self._need_reallocation(self.d_matrix, memory):
            self.d_matrix = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory
        if self.gap_extension:
            if self._need_reallocation(self.d_matrix_i, memory):
                self.d_matrix_i = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
            mem_size += memory
            if self._need_reallocation(self.d_matrix_j, memory):
                self.d_matrix_j = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
            mem_size += memory

        # Maximum global device memory
        memory = (SmithWaterman.float_size * self.x_div_shared_x * self.number_of_sequences *
        self.y_div_shared_y * self.number_targets)
        if self._need_reallocation(self.d_global_maxima, memory):
            self.d_global_maxima = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory

        memory = (self.length_of_x_sequences * self.number_of_sequences * self.length_of_y_sequences * self.number_targets)
        if self._need_reallocation(self.d_global_direction, memory):
            self.d_global_direction = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=memory)
        mem_size += memory

        memory = SmithWaterman.int_size
        if self._need_reallocation(self.d_is_traceback_required, memory):
            self.d_is_traceback_required = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=memory)
            flag = numpy.zeros((1), dtype=numpy.uint32)
            cl.enqueue_fill_buffer(self.queue, self.d_is_traceback_required, flag, 0, size=memory)

        return mem_size

    def _clear_normal_memory(self):
        SmithWatermanOcl._clear_normal_memory(self)
        if (self.d_global_direction is not None):
            try:
                self.d_global_direction.finish()
            except:
                pass
            self.d_global_direction.release()
        if (self.d_is_traceback_required is not None):
            try:
                self.d_is_traceback_required.finish()
            except:
                pass
            self.d_is_traceback_required.release()

    def _compile_code(self):
        """Compile the device code with current settings"""
        if self.program is None:
            self.logger.debug('Compiling OpenCL code.')
            code = self.oclcode.get_code(self.score, self.number_of_sequences, self.number_targets, self.length_of_x_sequences, self.length_of_y_sequences)
            self.program = cl.Program(self.ctx, code).build(options=['-cl-fast-relaxed-math'])
            self.calculateScoreAffineGap_kernel = self.program.calculateScoreAffineGap
            self.calculateScore_kernel = self.program.calculateScore
            self.tracebackAffineGap_kernel = self.program.tracebackAffineGap
            self.traceback_kernel = self.program.traceback

    def _get_direction_byte_array(self):
        '''
        Get the resulting directions
        @return gives the resulting direction array as byte array
        '''
        h_global_direction = cl.enqueue_map_buffer(self.queue, self.d_global_direction, cl.map_flags.READ, 0, 
                                                   (self.number_of_sequences,
                                                    self.number_targets,
                                                    self.x_div_shared_x,
                                                    self.y_div_shared_y,
                                                    self.shared_x,
                                                    self.shared_y), dtype=numpy.byte)[0]
        return h_global_direction

    def _execute_calculate_score_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the calculate score kernel'''
        dim_block = (self.shared_x, self.shared_y, 1)
        dim_grid_sw = (number_of_blocks * self.shared_x, self.number_of_sequences * self.shared_y, self.number_targets)
        
        if self.gap_extension:
            self.calculateScoreAffineGap_kernel(self.queue, dim_grid_sw, dim_block,
                                                numpy.uint32(self.number_of_sequences),
                                                numpy.uint32(self.number_targets),
                                                numpy.uint32(self.x_div_shared_x),
                                                numpy.uint32(self.y_div_shared_y),
                                                self.d_matrix,
                                                self.d_matrix_i,
                                                self.d_matrix_j,
                                                numpy.uint32(idx),
                                                numpy.uint32(idy),
                                                self.d_sequences,
                                                self.d_targets,
                                                self.d_global_maxima,
                                                self.d_global_direction,
                                                self.d_max_possible_score_zero_copy,
                                                self.d_is_traceback_required)
        else:
            self.calculateScore_kernel(self.queue, dim_grid_sw, dim_block,
                                       numpy.uint32(self.number_of_sequences),
                                       numpy.uint32(self.number_targets),
                                       numpy.uint32(self.x_div_shared_x),
                                       numpy.uint32(self.y_div_shared_y),
                                       self.d_matrix,
                                       numpy.uint32(idx),
                                       numpy.uint32(idy),
                                       self.d_sequences,
                                       self.d_targets,
                                       self.d_global_maxima,
                                       self.d_global_direction,
                                       self.d_max_possible_score_zero_copy,
                                       self.d_is_traceback_required)

    def _is_traceback_required(self):
        '''Returns False if it is known after calculating scores that there are no possible
        starting points, hence no need to run traceback.
        '''
        flag = numpy.zeros((1), dtype=numpy.uint32)
        cl.enqueue_copy(self.queue, flag, self.d_is_traceback_required)
        if flag[0]:
            # Clear the flag
            flag[0] = 0
            cl.enqueue_fill_buffer(self.queue, self.d_is_traceback_required, flag, 0, size=SmithWaterman.int_size)
            return True
        else:
            return False
    
    def _execute_traceback_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the traceback kernel'''
        dim_block = (self.shared_x, self.shared_y, 1)
        dim_grid_sw = (number_of_blocks * self.shared_x, self.number_of_sequences * self.shared_y, self.number_targets)

        if self.gap_extension:
            self.tracebackAffineGap_kernel(self.queue, dim_grid_sw, dim_block,
                                           numpy.uint32(self.number_of_sequences),
                                           numpy.uint32(self.number_targets),
                                           numpy.uint32(self.x_div_shared_x),
                                           numpy.uint32(self.y_div_shared_y),
                                           self.d_matrix,
                                           self.d_matrix_i,
                                           self.d_matrix_j,
                                           numpy.uint32(idx),
                                           numpy.uint32(idy),
                                           self.d_global_maxima,
                                           self.d_global_direction,
                                           self.d_index_increment,
                                           self.d_starting_points_zero_copy,
                                           self.d_max_possible_score_zero_copy)
        else:
            self.traceback_kernel(self.queue, dim_grid_sw, dim_block,
                                  numpy.uint32(self.number_of_sequences),
                                  numpy.uint32(self.number_targets),
                                  numpy.uint32(self.x_div_shared_x),
                                  numpy.uint32(self.y_div_shared_y),
                                  self.d_matrix,
                                  numpy.uint32(idx),
                                  numpy.uint32(idy),
                                  self.d_global_maxima,
                                  self.d_global_direction,
                                  self.d_index_increment,
                                  self.d_starting_points_zero_copy,
                                  self.d_max_possible_score_zero_copy)
