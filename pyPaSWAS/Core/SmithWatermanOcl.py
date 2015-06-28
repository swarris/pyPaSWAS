import pyopencl as cl
import numpy
import math

from pyPaSWAS.Core.SmithWaterman import SmithWaterman
from pyPaSWAS.Core.PaSWAS import OCLcode

class SmithWatermanOcl(SmithWaterman):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.oclcode = OCLcode(self.logger)
        
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
    
    def __del__(self):
        '''Destructor. Removes the current running context'''
        pass

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
                self.logger.debug("Found platform {}", str(self.platform)) 
                break
        
        if not (self.platform):    
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    if (device.get_info(cl.device_info.TYPE) == self.device_type):
                        self.platform = platform
                        found_platform = True
                        break
                if(found_platform):
                    self.logger.debug('Found platform {}, however this is not the platform indicated by the user', str(self.platform)) 
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
        self.ctx = cl.Context(devices=self.device)
        self.queue = cl.CommandQueue(self.ctx)
    
    def _get_max_length_xy(self):
        '''
        _get_max_length_xy gives the maximum length of both X and Y possible based on the total memory.
        @return: int value of the maximum length of both X and Y.
        '''
        return (math.floor(math.sqrt((self.device.global_mem_size * self.mem_fill_factor) / 
                                    self._get_mem_size_basic_matrix()))) 
        
    # TODO: check if driver has been initialized
    # TODO: add correct docstring
    def _get_max_number_sequences(self, length_sequences, length_targets, number_of_targets):
        '''
        Returns the maximum length of all sequences
        :param length_sequences:
        :param length_targets:
        '''
        self.logger.debug("Total memory on Device: {}".format(self.device.global_mem_size/1024/1024))
        value = 1
        try:
            value = math.floor((self.device.global_mem_size * self.mem_fill_factor) / #@UndefinedVariable
                               ((length_sequences * length_targets * (self._get_mem_size_basic_matrix()) +
                                 (length_sequences * length_targets * SmithWaterman.float_size) /
                                 (self.shared_x * self.shared_y)) * number_of_targets)) #@UndefinedVariable @IgnorePep8
        except:
            self.logger.warning("Possibly not enough memory for targets")
            return 1
        else:
            return value if value > 0 else 1

    
    def _clear_memory(self):
        '''Clears the claimed memory on the device.'''
        pass

    def _init_memory(self):
        '''
        #_init_memory will initialize all required memory on the device based on the current settings.
        Make sure to initialize these values!
        '''
        pass

    def _init_zero_copy(self):
        ''' Initializes the index used for the 'zero copy' of the found starting points '''
        self.d_index_increment = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=SmithWaterman.int_size)
        index = numpy.zeros((1), dtype=numpy.int32)
        cl.enqueue_write_buffer(self.queue, self.d_index_increment, index).wait()

    def _compile_code(self):
        """Compile the device code with current settings"""
        self.logger.debug('Compiling OpenCL code.')
        code = self.oclcode.get_code(self.score, self.number_of_sequences, self.number_targets, self.length_of_x_sequences, self.length_of_y_sequences)
        self.program = cl.Program(self.ctx, code).build()
    
    def copy_sequences(self, h_sequences, h_targets):
        '''
        Copy the sequences and targets to the device
        @param h_sequences: the sequences to be copied. Should be a single string containing all sequences
        @param h_targets: the targets to be copied. Should be a single string containing all sequences
        '''
        cl.enqueue_write_buffer(self.queue, self.d_sequences, h_sequences).wait()
        cl.enqueue_write_buffer(self.queue, self.d_targets, h_targets).wait()
    
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
                                    self.d_global_direction).wait()
    
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
                               self.d_global_direction,
                               self.d_global_direction_zero_copy,
                               self.d_index_increment,
                               self.d_starting_points_zero_copy,
                               self.d_max_possible_score_zero_copy).wait()

    def _get_number_of_starting_points(self):
        ''' Returns the number of startingpoints. '''
        self.logger.debug('Getting number of starting points.')
        self.index = numpy.zeros((1), dtype=numpy.int32)
        cl.enqueue_read_buffer(self.queue, self.d_index_increment, self.index)
        return self.index[0]    
    

        