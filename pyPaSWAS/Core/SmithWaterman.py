"""
The Smith-Waterman class handles the communication between the python application and the PaSWAS GPGPU code.
It can be used to initialize the device, allocate the required memory and run the CUDA code.

@author Sven Warris
@version 1.5
"""

#import required modules:
import pycuda.driver as driver
from pycuda.compiler import SourceModule
import numpy
import math

from pyPaSWAS.Core.StartingPoint import StartingPoint
from pyPaSWAS.Core.HitList import HitList
from pyPaSWAS.Core.Hit import Hit
from pyPaSWAS.Core.SWSeq import SWSeq

from pyPaSWAS.Core.cudaPaSWAS import Cudacode
from pyPaSWAS.Core.Exceptions import HardwareException, InvalidOptionException

from pyPaSWAS.Core import STOP_DIRECTION, LEFT_DIRECTION, NO_DIRECTION, UPPER_DIRECTION, UPPER_LEFT_DIRECTION, IN_ALIGNMENT


class SmithWaterman(object):
    """
    Smith-Waterman class for communicating between GPU and CPU.

    Attribute naming: starting with d_ means the data is located on the device (GPU RAM).
    h_ means the data is stored on the host (CPU RAM), usually this is host pinned memory.
    ZeroCopy indicates host pinned memory as well.

    Example usage:

    try:
        # create new device and context:
        smithWaterman = SmithWaterman(0)

        # create two dummy sequences:
        seq= numpy.array(['A','A','A','A','A','A','A','A'], dtype=numpy.character)
        tar= numpy.array(['A','A','A','A','A','A','A','A'], dtype=numpy.character)

        # set parameters for memory etc: _set_parameters(length seq, length tar, number of sequences, number of targets)
        smithWaterman._set_parameters(8,8,1,1)
        # initialize memory:
        smithWaterman._init_memory()
        # compile the module:
        smithWaterman._compile_cuda_code()
        # copy sequence to the GPU:
        smithWaterman.copy_sequences(seq,tar)
        # initialize the index for starting points:
        smithWaterman._init_zero_copy()
        # perform the Smith-Waterman calculations on the GPU:
        smithWaterman._calculate_score()
        # perform the traceback on the GPU:
        smithWaterman._traceback_host(0)

        # get the resulting starting points:
        results = smithWaterman._get_starting_point_byte_array()
        # make new starting point:
        starting_point = StartingPoint()
        # get first hit:
        starting_point.parse_byte_string(results[0:starting_point.size])
        # print it:
        print(starting_point)

    except Exception as exception:
        print(exception)
    """

    # Byte size of a float
    float_size = 4
    # Byte size of an int
    int_size = 4

    GAP_CHAR_SEQ = '-'
    GAP_CHAR_ALIGN = ' '
    MISMATCH_CHAR = '.'
    MATCH_CHAR = '|'

    def __init__(self, logger, score, settings):
        """
        Constructor for a Smith-Waterman object.
        """
        self.logger = logger
        self.logger.debug('Initializing SmithWaterman.')
        self.score = score
        self.settings = settings
        self.number_of_targets = 1

        # Added startingpoint instance
        self.starting_point = StartingPoint(self.logger)

        self.target_block_length = 0
        # d_sequences holds the sequences (X) in GPU RAM
        self.d_sequences = None
        # d_targets holds the target sequences (Y) in GPU RAM
        self.d_targets = None
        # d_matrix is used to store the SM scoring
        self.d_matrix = None
        # d_global_maxima holds the maximum values found during the SM calculations
        self.d_global_maxima = None
        # d_global_direction keeps track of the direction the score came from
        self.d_global_direction = None
        # d_index_increment is the 'auto-increment index' used to store the starting points. Hence, after the
        # traceback this attribute holds the number of alignments
        self.d_index_increment = None
        self.index = 0
        # h_global_direction_zero_copy is the direction matrix on the CPU and holds the new values
        # after the traceback call
        self.h_global_direction_zero_copy = None
        self.d_global_direction_zero_copy = None
        # h_starting_points_zero_copy is used to store the starting point struct values in a long list.
        # Use the StartingPoint python class to get the python object representation
        self.h_starting_points_zero_copy = None
        self.d_starting_points_zero_copy = None
        # h_max_possible_score_zero_copy holds the maximum possible score a sequence (X) can have (for each sequence).
        self.h_max_possible_score_zero_copy = None
        self.d_max_possible_score_zero_copy = None

        self.max_length = None
        self.has_been_compiled = False
        self.max_sequences = 0
        # Length of the sequences on the X axis. Necessary for the CUDA code.
        self.length_of_x_sequences = 0
        # Length of the targets on the Y axis. Necessary for the CUDA code.
        self.length_of_y_sequences = 0
        # Number of sequences. Necessary for the CUDA code.
        self.number_of_sequences = 0
        # Number of target sequences. Necessary for the CUDA code.
        self.number_targets = 0
        # Necessary for the CUDA code.
        self.x_div_shared_x = 0
        # Necessary for the CUDA code.
        self.y_div_shared_y = 0

        # Necessary for the CUDA code.
        self.maximum_number_starting_points =100
        # Changed from ... = StartingPoint.size
        self.size_of_startingpoint = self.starting_point.size

        # Defines the number of elements processed in the X direction.
        # Don't edit unless you really know what you are doing!
        self.shared_x = 8
        # Defines the number of elements processed in the Y direction.
        # Don't edit unless you really know what you are doing!
        self.shared_y = 8

        # TODO: Amount of memory used of totally available memory. User should be able to adjust this.
        self.mem_fill_factor = 0.9

        # Filled in later
        self.target_array = numpy.array([], dtype=numpy.character)
        self.module = None

        self.device = 0

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

        self.filter_factor = 0.7
        self.internal_limit = 64000
        self.max_genome_length = 10000
        self._set_filter_factor(self.settings.filter_factor)
        self._set_max_genome_length(self.settings.max_genome_length)
        self._set_internal_limit(self.internal_limit)

    def __del__(self):
        '''Destructor. Removes the current running context'''
        self.logger.debug('Destructing SmithWaterman.')
        if (driver.Context is not None):  #@UndefinedVariable @IgnorePep8
            driver.Context.pop()  #@UndefinedVariable @IgnorePep8

    def _set_device(self, device):
        '''Sets the device number'''
        try:
            self.device = int(device) if device else self.device
        except ValueError:
            raise InvalidOptionException('device should be an int but is {0}'.format(device))

    def _set_internal_limit(self, limit):
        '''sets the internal limit'''
        try:
            self.internal_limit = int(limit) if limit else self.internal_limit
        except ValueError:
            raise InvalidOptionException('internal_limit should be a int but is {0}'.format(limit))

    def _set_max_genome_length(self, max_genome_length):
        '''sets the maximum length of the genome'''
        try:
            self.max_genome_length = int(max_genome_length) if max_genome_length else self.max_genome_length
        except ValueError:
            raise InvalidOptionException('maxGenomeLength should be a int but is {0}'.format(max_genome_length))

    def _set_filter_factor(self, filterfactor):
        '''sets the filter factor'''
        try:
            if filterfactor:
                self.filter_factor = float(filterfactor) if filterfactor else self.filter_factor
        except ValueError:
            raise InvalidOptionException('Filterfactor should be a float but is {0}'.format(filterfactor))

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

    def _set_target_block_length(self, targets, target_index):
        '''
        Returns the length of a block of targets that will fit into the memory of the gpu
        :param targets:
        :param target_index:
        '''
        # make sure length of target sequences can be divided by SHARED_Y
        self.target_block_length = int(math.ceil(len(targets[target_index].seq) / float(self.shared_y)) * self.shared_y)

    def _set_parameters(self, len_x, len_y, number_sequences, number_targets):
        '''
        _set_parameters is used to initialize all values before memory initialization and running a Smith-Waterman analysis.
        @param len_x: length of each sequence on the len_x axes. len_x % shared_x should be zero
        @param len_y: length of each target on the len_y axes. len_y % SHARED_Y should be zero
        @param number_sequences: the number of sequences on the len_x axes
        @param number_targets: the number of targets on the len_y axes
        '''
        self.logger.debug('Setting parameters len_x: {0}, len_y: {1}, '
                          'number_sequences: {2}, '
                          'number_targets: {3} '.format(len_x, len_y, number_sequences, number_targets))
        if len_x % self.shared_x > 0 or len_y % self.shared_y > 0:
            raise InvalidOptionException("Error in _set_parameters: len_x % " + str(self.shared_x) +
                                         " or len_y % " + str(self.shared_y) + " != 0 !")
        self.length_of_x_sequences = len_x
        self.length_of_y_sequences = len_y
        self.number_of_sequences = number_sequences
        self.number_targets = number_targets
        self.x_div_shared_x = len_x / self.shared_x
        self.y_div_shared_y = len_y / self.shared_y

    def _get_number_of_targets(self):
        '''Calculates the number of targets to process on this device.'''
        # figure out how many targets will fit in memory (leaving room for the other sequences as well of course
#        self.number_of_targets = 1
#        if self.settings.short_sequences == "T":
#            max_number_of_targets = int(math.floor(math.sqrt(self._get_max_number_sequences(self.target_block_length,
#                                                                                        self.target_block_length, len()))))

        max_number_of_targets = int(math.floor(math.sqrt(self._get_max_number_sequences(self.target_block_length,
                                                                                        self.target_block_length, 1))))

        return max_number_of_targets

    def _get_number_of_targets_with_sequences(self, records_seqs):
        '''Calculates the number of targets to process on this device.'''

        max_number_of_targets = int(math.floor((self._get_max_number_sequences(self.target_block_length,
                                                                                        int(math.ceil(len(records_seqs[0].seq) / float(self.shared_x)) * self.shared_x), len(records_seqs)))))

        return max_number_of_targets


    def set_targets(self, targets, target_index, max_length = None, records_seqs=None):
        '''Retrieves a block of targets from the target array and returns the index of the last target that will be processed'''

        if self.max_length == None or target_index == 0:                
            self._set_target_block_length(targets, target_index)
            if records_seqs != None and len(records_seqs) > 0: 
                self.max_number_of_targets = self._get_number_of_targets_with_sequences(records_seqs)
                if self.max_number_of_targets < 1:
                    self.max_number_of_targets = self._get_number_of_targets()
            else:
                self.max_number_of_targets = self._get_number_of_targets() 
       
        if max_length != None and self.settings.recompile == "F" and target_index > 0:
            self.max_length = max_length
        
        if target_index + self.max_number_of_targets < len(targets):
            self.number_of_targets = self.max_number_of_targets
        else:
            self.number_of_targets = len(targets) - target_index

        if self.number_of_targets > len(targets):
            self.number_of_targets = len(targets)

        if self.number_of_targets * len(targets[0]) / self.shared_y > self.internal_limit:
            self.number_of_targets = int(self.internal_limit * self.shared_y / len(targets[0]))
        

        # fill the target array with sequences
        targetStr = []
        self.added_dummy_targets = 0
        for i in range(self.number_of_targets):
            if i+target_index < len(targets):
                targetStr.append(SWSeq.extentToFillGPU(str(targets[i+target_index].seq), self.target_block_length))
            else:
                targetStr.append(SWSeq.extentToFillGPU("", self.target_block_length))
                self.added_dummy_targets += 1
        
        self.target_array = numpy.array(''.join(targetStr), dtype=numpy.character)

        return target_index + self.number_of_targets

    def _set_score(self, score):
        '''Sets the score attribute of the class instance to score.'''
        self.score = score

    def _get_mem_size_basic_matrix(self):
        """_get_mem_size_basic_matrix determines the amount of bytes required for an 1x1 sequence alignment.
            @return: the amount of memory in bytes for the 1x1 alignment
            Calculate GPU memory requirements for 1x1 alignment with 1x1 character.
        """
        # size of each element in a smith waterman matrix (lchar, uchar, luchar, value (is float) and direction)
        mem_size = 1
        mem_size += 1
        mem_size += self.float_size
        mem_size += 1
        mem_size += 1
        return mem_size

    def _get_max_length_xy(self):
        '''
        _get_max_length_xy gives the maximum length of both X and Y possible with
        the currently available total memory.
        @return: int value of the maximum length of both X and Y.
        '''
        return (math.floor(math.sqrt((driver.mem_get_info()[0] * self.mem_fill_factor) /  #@UndefinedVariable
                                     self._get_mem_size_basic_matrix())))  #@UndefinedVariable @IgnorePep8

    # TODO: check if driver has been initialized
    # TODO: add correct docstring
    def _get_max_number_sequences(self, length_sequences, length_targets, number_of_targets):
        '''
        Returns the maximum length of all sequences
        :param length_sequences:
        :param length_targets:
        '''
        self.logger.debug("Available memory on GPU: {}".format(driver.mem_get_info()[0]/1024/1024))
        value = 1
        try:
            value =  math.floor((driver.mem_get_info()[0] * self.mem_fill_factor) /  #@UndefinedVariable
                          ((length_sequences * length_targets * (self._get_mem_size_basic_matrix()) +
                            (length_sequences * length_targets * SmithWaterman.float_size) /
                            (self.shared_x * self.shared_y)) * number_of_targets))  #@UndefinedVariable @IgnorePep8
        except:
            self.logger.warning("Possibly not enough memory for targets")
            return 1
        else:
            return value if value > 0 else 1

    def align_sequences(self, records_seqs, targets, target_index):
        '''Aligns sequences against the targets. Returns the resulting alignments in a hitlist.'''
        # reset values for next set of sequences
        index = 0
        prev_seq_length = 0
        prev_target_length = 0
        
        cont = True
        # step through all the sequences
        max_length = 0
        if len(records_seqs) > 0:
            max_length = len(records_seqs[0])
        
        while index < len(records_seqs) and cont:
            # make sure length of sequences can be divided by shared_x
            # don't reset when no need to recompile:
            if self.settings.recompile == "F" :
                length = int(math.ceil(max_length / float(self.shared_x)) * self.shared_x)
            else: 
                length = int(math.ceil(len(records_seqs[index].seq) / float(self.shared_x)) * self.shared_x)
            # if lengths of sequences or targets differ, reset CUDA code and memory, otherwise use current settings
            # this way there is no need to recompile the code for every run

            if ((length != prev_seq_length or self.target_block_length != prev_target_length) or
                (self.max_sequences + index >= len(records_seqs))) and self.settings.recompile=="T":
                # clear memory
                self._clear_memory()
                if (length != prev_seq_length or self.target_block_length != prev_target_length):
                    # see how many sequences fit in memory
                    self.max_sequences = int(math.floor((self._get_max_number_sequences(length, self.target_block_length, self.number_of_targets))))
                    if self.max_sequences * length / self.shared_x > self.internal_limit:
                        self.max_sequences = int(self.internal_limit / self.shared_x * length)   
                    self.logger.info("Maximum number of seqs in memory: {} {}".format(self.max_sequences, self.number_of_targets))
                    # set parameters for this run
                    if self.max_sequences + index >= len(records_seqs):
                        self.max_sequences = len(records_seqs) - index
                elif self.max_sequences + index >= len(records_seqs):
                    self.max_sequences = len(records_seqs) - index
                    cont = False
                if self.max_sequences > self.internal_limit:
                    self.max_sequences = self.internal_limit
                    cont = True
                self._init_sw(length, self.target_block_length, self.max_sequences, self.number_of_targets)
                self.has_been_compiled = False
            elif self.settings.recompile == "F":
                if not self.has_been_compiled:
                    self._clear_memory()
                    self.max_sequences = int(math.floor((self._get_max_number_sequences(length, self.target_block_length, self.number_of_targets))))
                    if self.max_sequences * length / self.shared_x > self.internal_limit:
                        self.max_sequences = int(self.internal_limit / self.shared_x * length)   
                    self._init_sw(length, self.target_block_length, self.max_sequences, self.number_of_targets)

            # add sequences to the list
            self.added_dummy_seqs = 0

            sequenceStr = []
            self.added_dummy_seqs = 0
             
            for i in range(self.max_sequences):
                if i+index < len(records_seqs):
                    sequenceStr.append(SWSeq.extentToFillGPU(str(records_seqs[i+index].seq), length))
                else:
                    sequenceStr.append(SWSeq.extentToFillGPU("", length))
                    self.added_dummy_seqs += 1

                for tI in range(self.number_of_targets):
                    if tI+target_index < len(targets) and i+index < len(records_seqs):
                        #if self.userFilter.bestHit and (seqs[i+index].id.strip("_RC"), targets[tI+targetIndex].id.strip("_RC")) in self.hitList.realHits:
                        #    self.smithWaterman.setMinimumScore(tI*maxSequences + i, self.hitList.realHits[(seqs[i+index].id.strip("_RC"), targets[tI+targetIndex].id.strip("_RC"))].score)
                        #else: 
                        self.set_minimum_score(tI*self.max_sequences + i, float(self.score.highest_score) * (len(records_seqs[i+index]) if len(records_seqs[i+index]) < len(targets[tI+target_index]) else len(targets[tI+target_index])) * float(self.filter_factor))
            
            # copy sequences and targets to the device
            sequence_array = numpy.array(''.join(sequenceStr), dtype=numpy.character)


            self.logger.debug("At sequence: {0} of {1}, length = {2}".format(index, len(records_seqs), self.max_sequences))

            # set lengths of this run
            prev_seq_length = length
            prev_target_length = self.target_block_length
            if self.settings.recompile == "F":
                self._set_parameters(length, self.target_block_length, self.max_sequences-self.added_dummy_seqs, self.number_of_targets-self.added_dummy_targets)
            # copy sequences and targets to the device
            self.copy_sequences(sequence_array, self.target_array)
            # initialize index for zero copy of starting points
            self._init_zero_copy()
            # calculate scores of alignments
            self._calculate_score()
            # perform the traceback
            self._traceback_host()

            # TODO: change to returning a value, change _print_alignments to getAlignments in SmithWaterman
            # TODO: move _print_alignments to here? This should be a statement to retrieve the results and
            # put them into a Hitlist (?)
            hitlist = self._print_alignments(records_seqs, targets, index, target_index)
            index += self.max_sequences
        return hitlist

    def _clear_memory(self):
        '''Clears the claimed memory on the gpu.'''
        self.logger.debug('Clearing device memory.')
        if (self.d_sequences is not None):
            self.d_sequences.free()
        if (self.d_targets is not None):
            self.d_targets.free()
        if (self.d_matrix is not None):
            self.d_matrix.free()
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
        _init_memory will initialize all required memory on the device based on the current settings.
        Make sure to initialize these values!
        '''
        # TODO: document each memory allocation (purpose / target)
        # Query sequence device memory
        self.logger.debug('Initializing device memory.')
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

    def _init_sw(self, length, target_length, max_sequences, number_of_targets):
        '''
        (Re)Initializes the gpu. This becomes necessary if the length of the sequences
        that will be compared do not correctly fit the memory grid of the gpu.
        :param length:
        :param target_length:
        :param max_sequences:
        :param number_of_targets:
        '''
        if not self.has_been_compiled:
            self.logger.debug("Setting parameters for CUDA: {} {} {} {}".format(length, target_length, max_sequences, number_of_targets))
            self._set_parameters(length, target_length, max_sequences, number_of_targets)
            self._set_score(self.score)
            # compile the code
            self.logger.debug('Compiling CUDA code...')
            self._compile_cuda_code()
            # initialize memory
            self.logger.debug('Initializing memory...')
            self._init_memory()
            self.has_been_compiled = True

    def _init_zero_copy(self):
        ''' Initializes the index used for the 'zero copy' of the found starting points '''
        self.logger.debug('Initializing zero copy.')
        self.d_index_increment = driver.mem_alloc(SmithWaterman.int_size)  #@UndefinedVariable @IgnorePep8
        index = numpy.zeros((1), dtype=numpy.int32)  #@UndefinedVariable @IgnorePep8
        driver.memcpy_htod(self.d_index_increment, index)  #@UndefinedVariable @IgnorePep8

    def _compile_cuda_code(self):
        """Compile the CUDA code with current settings"""
        self.logger.debug('Compiling cuda code.')
        code = self.cudacode.get_code(self.score, self.number_of_sequences, self.number_targets, self.length_of_x_sequences, self.length_of_y_sequences)
        self.module = SourceModule(code)

    def copy_sequences(self, h_sequences, h_targets):
        '''
        Copy the sequences and targets to the device
        @param h_sequences: the sequences to be copied. Should be a single string containing all sequences
        @param h_targets: the targets to be copied. Should be a single string containing all sequences
        '''
        self.logger.debug('Copying sequences to device.')
        driver.memcpy_htod(self.d_sequences, h_sequences)  #@UndefinedVariable @IgnorePep8
        driver.memcpy_htod(self.d_targets, h_targets)  #@UndefinedVariable @IgnorePep8

    def set_minimum_score(self, index, minScore):
        # @TO-DO: this is bugfix for the read mapping algorithm. Should not happen, so fix this where it should be fixed
        if index < len(self.h_max_possible_score_zero_copy):
            self.h_max_possible_score_zero_copy[index] = minScore

    def _calculate_score(self):
        """ Calculates the Smith-Waterman scores on the device """
        self.logger.debug('Calculating scores.')
        max_number_of_blocks = min(self.x_div_shared_x, self.y_div_shared_y)
        start_decrease_at = self.x_div_shared_x + self.y_div_shared_y - max_number_of_blocks
        number_of_blocks = 0
        idx = 0
        idy = 0
        dim_block = (self.shared_x, self.shared_y, 1)

        for i in range(1, self.x_div_shared_x + self.y_div_shared_y):
            if (i <= max_number_of_blocks):
                number_of_blocks = i
            elif (i >= start_decrease_at):
                number_of_blocks = self.x_div_shared_x + self.y_div_shared_y - i
            else:
                number_of_blocks = max_number_of_blocks
            dim_grid_sw = (self.number_of_sequences, self.number_targets * number_of_blocks)

            calculate_score_function = self.module.get_function("calculateScore")

            try:
                calculate_score_function(self.d_matrix, numpy.int32(idx), numpy.int32(idy),
                                         numpy.int32(number_of_blocks), self.d_sequences, self.d_targets,
                                         self.d_global_maxima, self.d_global_direction,
                                         block=dim_block, grid=dim_grid_sw)
                driver.Context.synchronize()  #@UndefinedVariable @IgnorePep8
            # TODO: catch proper exception
            except (driver.MemoryError, driver.LogicError,  #@UndefinedVariable
                    driver.LaunchError, driver.RuntimeError) as exception:  #@UndefinedVariable
                self.logger.warning('Warning: {0}\nContinuing calculation...'.format(exception))
            if (idx == self.x_div_shared_x - 1):
                idy += 1
            if (idx < self.x_div_shared_x - 1):
                idx += 1

    def _traceback_host(self):
        ''' Performs the traceback on the device '''
        self.logger.debug('Performing back trace.')
        max_number_of_blocks = min(self.x_div_shared_x, self.y_div_shared_y)
        start_decrease_at = self.x_div_shared_x + self.y_div_shared_y - max_number_of_blocks
        number_of_blocks = 0
        idx = self.x_div_shared_x - 1
        idy = self.y_div_shared_y - 1
        dim_block = (self.shared_x, self.shared_y, 1)

        for i in range(1, self.x_div_shared_x + self.y_div_shared_y):
            if (i <= max_number_of_blocks):
                number_of_blocks = i
            elif(i >= start_decrease_at):
                number_of_blocks = self.x_div_shared_x + self.y_div_shared_y - i
            else:
                number_of_blocks = max_number_of_blocks

            dim_grid_sw = (self.number_of_sequences, self.number_targets * number_of_blocks)
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
            except (driver.MemoryError, driver.LogicError,  #@UndefinedVariable
                    driver.LaunchError, driver.RuntimeError) as exception:  #@UndefinedVariable
                self.logger.error('Something went wrong during traceback: {}...'.format(exception))
                raise exception

            if (idy == 0):
                idx -= 1
            if (idy > 0):
                idy -= 1

    def _get_starting_point_byte_array(self):
        '''
        Get the resulting starting points
        @return gives the resulting starting point array as byte array
        '''
        #TODO: change this to return of list of startingpoints??
        return (numpy.ndarray(buffer=self.h_starting_points_zero_copy,
                              dtype=numpy.byte, shape=(len(self.h_starting_points_zero_copy), 1)))

    def _get_number_of_starting_points(self):
        ''' Returns the number of startingpoints. '''
        self.logger.debug('Getting number of starting points.')
        self.index = numpy.zeros((1), dtype=numpy.int32)
        driver.memcpy_dtoh(self.index, self.d_index_increment)  #@UndefinedVariable @IgnorePep8
        return self.index[0]

    # TODO: return hitlist!!
    # TODO: finish docstring
    def _print_alignments(self, sequences, targets, start_seq, start_target, hit_list=None):
        '''
        Prints alingments
        :param sequences:
        :param targets:
        :param start_seq:
        :param start_target:
        :param hit_list:
        '''
        if hit_list is None:
            hit_list = HitList(self.logger)
        self.logger.debug('Printing alignments.')
        starting_points = self._get_starting_point_byte_array()
        #starting_point = StartingPoint(self.logger)

        number_of_starting_points = self._get_number_of_starting_points()
        self.logger.debug('Number of starting points is: {0}.'.format(number_of_starting_points))
        if number_of_starting_points >= (self.maximum_number_starting_points * self.number_of_sequences * self.number_targets):
            self.logger.warning("Too many hits returned. Skipping the rest. Please set lower_limit_score higher in config.")
            number_of_starting_points = self.maximum_number_starting_points * self.number_of_sequences * self.number_targets
        
        max_score = 0

        direction_array = numpy.ndarray(buffer=self.h_global_direction_zero_copy,
                                        dtype=numpy.byte, shape=(self.number_of_sequences,
                                                                 self.number_targets,
                                                                 self.x_div_shared_x,
                                                                 self.y_div_shared_y,
                                                                 self.shared_x,
                                                                 self.shared_y))
        starting_points_list = []
        for i in range(0,number_of_starting_points):
            starting_point = StartingPoint(self.logger)
            starting_point.parse_byte_string(starting_points, i)
            starting_points_list.append(starting_point)

        starting_points_list.sort(key=lambda s: s.score)
        
        i = 0
        #while (i < number_of_starting_points):
        for starting_point in starting_points_list:
            # TODO: assign starting_point instance to a variable => create function in
            # StartingPoints to retrieve an item by index
            #starting_point.parse_byte_string(starting_points, i)
            i += 1
            
            alignment_length = 0
            gaps_seq = 0
            gaps_target = 0
            matches = 0
            mismatches = 0

            target = []
            sequence = []
            alignment = []

            max_score = starting_point.score if starting_point.score > max_score else max_score

            target_starting_point = starting_point.target
            sequence_starting_point = starting_point.sequence

            block_x = starting_point.block_x
            block_y = starting_point.block_y
            value_x = starting_point.value_x
            value_y = starting_point.value_y

            local_index = 0
            s_end = block_x * self.shared_x + value_x
            t_end = block_y * self.shared_y + value_y
            
            # @TO-DO: this is bugfix for the read mapping algorithm. Should not happen, so fix this where it should be fixed
            if start_seq + sequence_starting_point >= len(sequences) or start_target + target_starting_point >= len(targets):
                continue
 
            if hasattr(sequences[start_seq + sequence_starting_point], 'start_position'):
                s_end += sequences[start_seq + sequence_starting_point].start_position
            if hasattr(targets[start_target + target_starting_point], 'start_position'):
                t_end += targets[start_target + target_starting_point].start_position
            if not hasattr(sequences[sequence_starting_point + start_seq], 'distance'):
                sequences[sequence_starting_point + start_seq].distance = 0.0
            s_start = s_end + 1
            t_start = t_end + 1

            direction = direction_array[sequence_starting_point][target_starting_point][block_x][block_y][value_x][value_y]
            show = True
            # check in 'all to all' when 1 data set is used to filter out hit X vs X (filtered on identical id):
            if sequences[sequence_starting_point + start_seq].id == targets[target_starting_point + start_target].id:
                direction = STOP_DIRECTION
                show = False

            #self.logger.debug('Score is: {0} vs {1}.'.format(starting_point.score, self.settings.minimum_score))
            if starting_point.score < float(self.settings.minimum_score):
                show = False
            
            while (show and block_x >= 0 and block_y >= 0 and value_x >= 0 and value_y >= 0 and
                   direction != STOP_DIRECTION and direction != NO_DIRECTION):
                direction = direction_array[sequence_starting_point][target_starting_point][block_x][block_y][value_x][value_y]
                direction_array[sequence_starting_point][target_starting_point][block_x][block_y][value_x][value_y] = IN_ALIGNMENT
                
                alignment_length += 1
                if (direction == IN_ALIGNMENT):
                    show = False
                elif (direction == UPPER_LEFT_DIRECTION):
                    target.append(targets[start_target + target_starting_point][block_y * self.shared_y + value_y])
                    sequence.append(sequences[start_seq + sequence_starting_point][block_x * self.shared_x + value_x])
                    alignment.append(SmithWaterman.MATCH_CHAR
                                     if target[local_index].lower() == sequence[local_index].lower()
                                     else SmithWaterman.MISMATCH_CHAR)
                    s_start -= 1
                    t_start -= 1
                    matches += 1 if target[local_index].lower() == sequence[local_index].lower() else 0
                    mismatches += 1 if target[local_index].lower() != sequence[local_index].lower() else 0
                    if (value_x == 0):
                        block_x -= 1
                        value_x = self.shared_x - 1
                    else:
                        value_x -= 1
                    if (value_y == 0):
                        block_y -= 1
                        value_y = self.shared_y - 1
                    else:
                        value_y -= 1
                elif (direction == LEFT_DIRECTION):
                    gaps_target += 1
                    target.append(SmithWaterman.GAP_CHAR_SEQ)
                    sequence.append(sequences[start_seq + sequence_starting_point][block_x * self.shared_x + value_x])
                    alignment.append(SmithWaterman.GAP_CHAR_ALIGN)
                    s_start -= 1
                    if (value_x == 0):
                        block_x -= 1
                        value_x = self.shared_x - 1
                    else:
                        value_x -= 1
                elif (direction == UPPER_DIRECTION):
                    gaps_seq += 1
                    target.append(targets[start_target + target_starting_point][block_y * self.shared_y + value_y])
                    sequence.append(SmithWaterman.GAP_CHAR_SEQ)
                    alignment.append(SmithWaterman.GAP_CHAR_ALIGN)
                    t_start -= 1
                    if (value_y == 0):
                        block_y -= 1
                        value_y = self.shared_y - 1

                    else:
                        value_y -= 1
                elif (direction == STOP_DIRECTION):  # end of alignment
                    target.append(targets[start_target + target_starting_point][block_y * self.shared_y + value_y])
                    sequence.append(sequences[start_seq + sequence_starting_point][block_x * self.shared_x + value_x])
                    alignment.append(SmithWaterman.MATCH_CHAR if target[local_index].lower() ==
                                     sequence[local_index].lower() else SmithWaterman.MISMATCH_CHAR)
                    s_start -= 1
                    t_start -= 1
                    matches += 1 if target[local_index].lower() == sequence[local_index].lower() else 0
                    mismatches += 1 if target[local_index].lower() != sequence[local_index].lower() else 0
                else:
                    self.logger.warning('Warning: wrong value in direction matrix: '
                                        '{0}\n\tContinuing calculation...'.format(direction))
                    block_x = -1
                local_index += 1
            if show:
                hit = Hit(self.logger, sequences[sequence_starting_point + start_seq],
                          targets[target_starting_point + start_target],
                          (s_start, s_end), (t_start, t_end))
                # set the strings that contain the alignment information. Since they have been generated by a trace-BACK, they should be reversed first
                sequence.reverse()
                alignment.reverse()
                target.reverse()
                hit.set_sequence_match(''.join(sequence))
                hit.set_alignment(''.join(alignment))
                hit.set_target_match(''.join(target))

                hit.set_scores(starting_point.score, matches, mismatches)
                if self._filter_hit(hit):
                    hit_list.append(hit)
                else:
                    self.logger.debug("Hit {0} -vs- {1} does not meet filter requirements".format(sequences[sequence_starting_point + start_seq].id, targets[target_starting_point + start_target].id ))
        return hit_list
    
    def _filter_hit(self, hit):
        '''Check to see hit meets filter requirements'''
        return (hit.relative_score >= float(self.settings.relative_score) and 
            hit.query_identity >= float(self.settings.query_identity) and 
            hit.query_coverage >= float(self.settings.query_coverage) and 
            hit.base_score >= float(self.settings.base_score))             
