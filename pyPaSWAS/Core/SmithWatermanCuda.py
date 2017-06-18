from pyPaSWAS.Core.SmithWaterman import SmithWaterman
from pyPaSWAS.Core.PaSWAS import Cudacode

from pyPaSWAS.Core import STOP_DIRECTION, LEFT_DIRECTION, NO_DIRECTION, UPPER_DIRECTION, UPPER_LEFT_DIRECTION, IN_ALIGNMENT
from pyPaSWAS.Core.Exceptions import HardwareException, InvalidOptionException

import pycuda.driver as driver
from pycuda.compiler import SourceModule
import numpy
import math

class SmithWatermanCuda(SmithWaterman):
    '''
    classdocs
    '''


    def __init__(self, logger, score, settings):
        
        SmithWaterman.__init__(self, logger, score, settings)
        
        self.module = None
        
        # d_global_direction keeps track of the direction the score came from
        self.d_global_direction = None


        self.cudacode = Cudacode(self.logger)
        # Compiling part of the CUDA code in advance
        self.cudacode.set_shared_xy_code(self.shared_x, self.shared_y)
        self.cudacode.set_direction_code(NO_DIRECTION, UPPER_LEFT_DIRECTION,
                                         UPPER_DIRECTION, LEFT_DIRECTION,
                                         STOP_DIRECTION)
        
        # Reference to the GPU device
        self._set_device(self.settings.device_number)
        self.logger.debug('Going to initialize device... with number {0}'.format(self.device))
        self._initialize_device(self.device)
        
    def __del__(self):
        '''Destructor. Removes the current running context'''
        self.logger.debug('Destructing SmithWaterman.')
        if (driver.Context is not None):  #@UndefinedVariable @IgnorePep8
            driver.Context.pop()  #@UndefinedVariable @IgnorePep8

    
    def _initialize_device(self, device_number):
        '''
        Initalizes the GPU device and verifies its computational abilities.
        @param device_number: int value representing the device to use
        '''
        self.logger.debug('Initializing device {0}'.format(device_number))
        try:
            driver.init()  #@UndefinedVariable @IgnorePep8
            self.device = driver.Device(device_number)  #@UndefinedVariable @IgnorePep8
            self.device = self.device.make_context(flags=driver.ctx_flags.MAP_HOST).get_device()  #@UndefinedVariable @IgnorePep8
        except Exception as exception:
            raise HardwareException('Failed to initialize device. '
                                    'The following exception occurred: {0}'.format(str(exception)))  #@UndefinedVariable @IgnorePep8
        compute = self.device.compute_capability()
        if not ((compute[0] == 1 and compute[1] >= 2) or (compute[0] >= 2)):
            raise HardwareException('Failed to initialize device: '
                                    'need compute capability 1.2 or newer!')  #@UndefinedVariable @IgnorePep8
    
    def _device_global_mem_size(self):
        '''  defines maximum available mem on device. '''
        return driver.mem_get_info()[0]
    
  
    def _clear_memory(self):
        '''Clears the claimed memory on the device.'''
        self.logger.debug('Clearing device memory.')
        if (self.d_sequences is not None):
            self.d_sequences.free()
        if (self.d_targets is not None):
            self.d_targets.free()
        if (self.d_matrix is not None):
            self.d_matrix.free()
        if (self.gap_extension and self.d_matrix_i is not None):
            self.d_matrix_i.free()
        if (self.gap_extension and self.d_matrix_j is not None):
            self.d_matrix_j.free()
        if (self.d_global_maxima is not None):
            self.d_global_maxima.free()
        if (self.d_global_direction is not None):
            self.d_global_direction.free()
        if (self.h_starting_points_zero_copy is not None):
            self.h_starting_points_zero_copy.base.free()
            self.d_starting_points_zero_copy = None
        if (self.h_global_direction_zero_copy is not None):
            self.h_global_direction_zero_copy.base.free()
            self.d_global_direction_zero_copy = None
        if (self.h_max_possible_score_zero_copy is not None):
            self.h_max_possible_score_zero_copy.base.free()
            self.d_max_possible_score_zero_copy = None

    def _init_memory(self):
        '''
        #_init_memory will initialize all required memory on the device based on the current settings.
        Make sure to initialize these values!
        '''
        # TODO: document each memory allocation (purpose / target)
        # Query sequence device memory
        #self.logger.debug('Initializing device memory.')
        memory = self.length_of_x_sequences * self.number_of_sequences
        self.d_sequences = driver.mem_alloc(memory)  #@UndefinedVariable @IgnorePep8
        mem_size = memory

        # Target device memory
        memory = self.length_of_y_sequences * self.number_targets
        self.d_targets = driver.mem_alloc(memory)  #@UndefinedVariable @IgnorePep8
        mem_size += memory

        # Input matrix device memory
        memory = (SmithWaterman.float_size * self.length_of_x_sequences * self.number_of_sequences *
                  self.length_of_y_sequences * self.number_targets)
        self.d_matrix = driver.mem_alloc(memory)  #@UndefinedVariable @IgnorePep8
        mem_size += memory
        if self.gap_extension:
            self.d_matrix_i = driver.mem_alloc(memory)  #@UndefinedVariable @IgnorePep8
            self.d_matrix_j = driver.mem_alloc(memory)  #@UndefinedVariable @IgnorePep8
            mem_size += 2*memory

        # Maximum global device memory
        memory = (SmithWaterman.float_size * self.x_div_shared_x * self.number_of_sequences *
                  self.y_div_shared_y * self.number_targets)
        self.d_global_maxima = driver.mem_alloc(memory)  #@UndefinedVariable @IgnorePep8
        mem_size += memory

        # Direction device memory
        memory = (self.length_of_x_sequences * self.number_of_sequences *
                     self.length_of_y_sequences * self.number_targets)
        self.d_global_direction = driver.mem_alloc(memory)  #@UndefinedVariable @IgnorePep8
        mem_size += memory

        # Starting points host memory allocation and device copy
        memory = (self.size_of_startingpoint * self.maximum_number_starting_points * self.number_of_sequences *
                  self.number_targets)
        
        self.h_starting_points_zero_copy = driver.pagelocked_empty((memory, 1), numpy.byte,  #@UndefinedVariable
                                                                   mem_flags=driver.host_alloc_flags.DEVICEMAP)  #@UndefinedVariable @IgnorePep8
        self.d_starting_points_zero_copy = numpy.intp(self.h_starting_points_zero_copy.base.get_device_pointer())
        mem_size += memory

        # Global directions host memory allocation and device copy
        memory = (self.length_of_x_sequences * self.number_of_sequences * self.length_of_y_sequences *
                  self.number_targets)
        self.h_global_direction_zero_copy = driver.pagelocked_empty((memory, 1), numpy.byte,  #@UndefinedVariable
                                                                    mem_flags=driver.host_alloc_flags.DEVICEMAP)  #@UndefinedVariable @IgnorePep8
        self.d_global_direction_zero_copy = numpy.intp(self.h_global_direction_zero_copy.base.get_device_pointer())
        mem_size += memory

        # Maximum zero copy memory allocation and device copy
        self.h_max_possible_score_zero_copy = driver.pagelocked_empty((self.number_of_sequences*self.number_of_targets, 1), numpy.float32,  #@UndefinedVariable
                                                                      mem_flags=driver.host_alloc_flags.DEVICEMAP)  #@UndefinedVariable @IgnorePep8
        self.d_max_possible_score_zero_copy = numpy.intp(self.h_max_possible_score_zero_copy.base.get_device_pointer())
        mem_size += self.number_of_sequences *self.number_of_targets * SmithWaterman.float_size

        self.logger.debug('Allocated: {}MB of memory'.format(str(mem_size / 1024.0 / 1024.00)))
    
    def _init_zero_copy(self):
        ''' Initializes the index used for the 'zero copy' of the found starting points '''
        #self.logger.debug('Initializing zero copy.')
        self.d_index_increment = driver.mem_alloc(SmithWaterman.int_size)  #@UndefinedVariable @IgnorePep8
        index = numpy.zeros((1), dtype=numpy.int32)  #@UndefinedVariable @IgnorePep8
        driver.memcpy_htod(self.d_index_increment, index)  #@UndefinedVariable @IgnorePep8

    def _compile_code(self):
        """Compile the device code with current settings"""
        self.logger.debug('Compiling cuda code.')
        code = self.cudacode.get_code(self.score, self.number_of_sequences, self.number_targets, self.length_of_x_sequences, self.length_of_y_sequences)
        self.module = SourceModule(code)
    
    def copy_sequences(self, h_sequences, h_targets):
        '''
        Copy the sequences and targets to the device
        @param h_sequences: the sequences to be copied. Should be a single string containing all sequences
        @param h_targets: the targets to be copied. Should be a single string containing all sequences
        '''
        #self.logger.debug('Copying sequences to device.')
        driver.memcpy_htod(self.d_sequences, h_sequences)  #@UndefinedVariable @IgnorePep8
        driver.memcpy_htod(self.d_targets, h_targets)  #@UndefinedVariable @IgnorePep8
    
    def _copy_min_score(self):
        driver.memcpy_htod(self.d_max_possible_score_zero_copy, self.min_score_np)
 

    
    def _execute_calculate_score_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the calculate score kernel'''
        dim_grid_sw = (self.number_of_sequences, self.number_targets * number_of_blocks)
        dim_block = (self.shared_x, self.shared_y, 1)

        try:
            if self.gap_extension:
                calculate_score_function = self.module.get_function("calculateScoreAffineGap")
                calculate_score_function(self.d_matrix, 
                                         numpy.int32(idx), 
                                         numpy.int32(idy),
                                         numpy.int32(number_of_blocks), 
                                         self.d_sequences, 
                                         self.d_targets,
                                         self.d_global_maxima, 
                                         self.d_global_direction,
                                         block=dim_block, 
                                         grid=dim_grid_sw)
            else:
                calculate_score_function = self.module.get_function("calculateScore")
                calculate_score_function(self.d_matrix, 
                                         numpy.int32(idx), 
                                         numpy.int32(idy),
                                         numpy.int32(number_of_blocks), 
                                         self.d_sequences, 
                                         self.d_targets,
                                         self.d_global_maxima, 
                                         self.d_global_direction,
                                         block=dim_block, 
                                         grid=dim_grid_sw)
                
            driver.Context.synchronize()  #@UndefinedVariable @IgnorePep8            
        # TODO: catch proper exception
        except Exception as exception:
            self.logger.warning('Warning: {0}\nContinuing calculation...'.format(exception))
        
    
    def _execute_traceback_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the traceback kernel'''
        dim_grid_sw = (self.number_of_sequences, self.number_targets * number_of_blocks)
        dim_block = (self.shared_x, self.shared_y, 1)
        
        try:
            traceback_function = self.module.get_function("traceback")
            traceback_function(self.d_matrix,
                               numpy.int32(idx),
                               numpy.int32(idy),
                               numpy.int32(number_of_blocks),
                               self.d_global_maxima,
                               self.d_global_direction,
                               self.d_global_direction_zero_copy,
                               self.d_index_increment,
                               self.d_starting_points_zero_copy,
                               self.d_max_possible_score_zero_copy,
                               block=dim_block,
                               grid=dim_grid_sw)
            driver.Context.synchronize()  #@UndefinedVariable @IgnorePep8
        except Exception as exception:
            self.logger.error('Something went wrong during traceback: {}...'.format(exception))
            raise exception
    

    def _get_number_of_starting_points(self):
        ''' Returns the number of startingpoints. '''    
        self.logger.debug('Getting number of starting points.')
        self.index = numpy.zeros((1), dtype=numpy.int32)
        driver.memcpy_dtoh(self.index, self.d_index_increment)  #@UndefinedVariable @IgnorePep8
        return self.index[0]
    
    def _set_max_possible_score(self, target_index, targets, i, index, records_seqs):
        '''fills the max_possible_score datastructure on the host'''
        for tI in range(self.number_of_targets):
            if tI+target_index < len(targets) and i+index < len(records_seqs):
                self.set_minimum_score(tI*self.max_sequences + i, float(self.score.highest_score) * (len(records_seqs[i+index]) 
                                                                                                     if len(records_seqs[i+index]) < len(targets[tI+target_index]) 
                                                                                                     else len(targets[tI+target_index])) * float(self.filter_factor))
                
    def _get_starting_point_byte_array(self):
        '''
        Get the resulting starting points
        @return gives the resulting starting point array as byte array
        '''
        #TODO: change this to return of list of startingpoints??
        return (numpy.ndarray(buffer=self.h_starting_points_zero_copy,
                              dtype=numpy.byte, shape=(len(self.h_starting_points_zero_copy), 1)))
        
        
    def _get_direction_byte_array(self):
        '''
        Get the resulting directions
        @return gives the resulting direction array as byte array
        '''
        return (numpy.ndarray(buffer=self.h_global_direction_zero_copy,
                                dtype=numpy.byte, shape=(self.number_of_sequences,
                                                         self.number_targets,
                                                         self.x_div_shared_x,
                                                         self.y_div_shared_y,
                                                         self.shared_x,
                                                         self.shared_y)))

    

