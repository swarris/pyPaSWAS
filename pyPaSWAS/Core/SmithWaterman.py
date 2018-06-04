"""
The Smith-Waterman class handles the communication between the python application and the PaSWAS GPGPU code.
It can be used to initialize the device, allocate the required memory and run the CUDA code.

@author Sven Warris
@version 1.5
"""

#import required modules:
#import pycuda.driver as driver
#from pycuda.compiler import SourceModule
import numpy
import math
import time
import datetime

from pyPaSWAS.Core.StartingPoint import StartingPoint
from pyPaSWAS.Core.HitList import HitList
from pyPaSWAS.Core.Hit import Hit
from pyPaSWAS.Core.SWSeq import SWSeq

from pyPaSWAS.Core.Exceptions import InvalidOptionException 
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
        smithWaterman._compile_code()
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
        # d_matrix_i is used to store the SM scoring with affine gap
        self.d_matrix_i = None
        # d_matrix_j is used to store the SM scoring with affine gap
        self.d_matrix_j = None
        # d_global_maxima holds the maximum values found during the SM calculations
        self.d_global_maxima = None


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
        self.min_score_np = None
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
        
        # check for gap extension penalty:
        self.gap_extension = score.gap_extension != None
        if self.gap_extension:
            self.logger.info("Gap extension penalty detected: using affine gap scoring algorithm")
        else:       
            self.logger.info("No gap extension penalty detected: using original PaSWAS scoring algorithm")
        # Defines the number of elements processed in the X direction.
        # Don't edit unless you really know what you are doing!
        self.shared_x = 8
        # Defines the number of elements processed in the Y direction.
        # Don't edit unless you really know what you are doing!
        self.shared_y = 8

        # Filled in later
        self.target_array = numpy.array([], dtype=numpy.character)
        
        self.device = 0
        self._set_device(self.settings.device_number)

        self.filter_factor = 0.7
        self.internal_limit = 64000
        self.max_genome_length = 10000
        self._set_filter_factor(self.settings.filter_factor)
        self._set_max_genome_length(self.settings.max_genome_length)
        self._set_internal_limit(self.internal_limit)
        # set memory usage
        try: 
            self.mem_fill_factor = float(settings.maximum_memory_usage)
        except ValueError:
            raise InvalidOptionException('maximux_memory_usage is not a float'.format(settings.maximum_memory_usage))
        if self.mem_fill_factor > 1.0 or self.mem_fill_factor <= 0.0:
            raise InvalidOptionException('maximux_memory_usage is not a float between 0.0 and 1.0'.format(settings.maximum_memory_usage))

        # Attibutes related to reporting of current progress
        self.total_work_size = 0
        self.total_processed = 0
        self.start_time = time.time()

    def __del__(self):
        '''Destructor. Removes the current running context'''
        pass

    
    def _initialize_device(self, device_number):
        '''
        Initalizes the GPU device and verifies its computational abilities.
        @param device_number: int value representing the device to use
        '''
        pass

    def _device_global_mem_size(self):
        '''  defines maximum available mem on device. Should be implemented by subclasses. '''
        pass
    
    def _get_max_number_sequences(self, length_sequences, length_targets, number_of_targets):
        '''
        Returns the maximum length of all sequences
        :param length_sequences:
        :param length_targets:
        '''
        self.logger.debug("Total memory on Device: {}".format(self._device_global_mem_size()/1024.0/1024.0))
        value = 1

        try:
            value = math.floor((self._device_global_mem_size() * self.mem_fill_factor) / #@UndefinedVariable
                               ((length_sequences * length_targets * (self._get_mem_size_basic_matrix()) +
                                 (length_sequences * length_targets * SmithWaterman.float_size) /
                                 (self.shared_x * self.shared_y)) * number_of_targets)) #@UndefinedVariable @IgnorePep8
        except:
            self.logger.warning("Possibly not enough memory for targets")
            self.logger.warning("mem: {}, l seq: {}, l targets: {}, #targets: {}".format(self._device_global_mem_size(), length_sequences, length_targets, number_of_targets))
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
        pass

    def _compile_code(self):
        """Compile the device code with current settings"""
        pass
    
    def copy_sequences(self, h_sequences, h_targets):
        '''
        Copy the sequences and targets to the device
        @param h_sequences: the sequences to be copied. Should be a single string containing all sequences
        @param h_targets: the targets to be copied. Should be a single string containing all sequences
        '''
        pass
    
    def _execute_calculate_score_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the calculate score kernel'''
        pass
    
    def _execute_traceback_kernel(self, number_of_blocks, idx, idy):
        ''' Executes a single run of the traceback kernel'''
        pass

    def _get_number_of_starting_points(self):
        ''' Returns the number of startingpoints. '''
        pass    
    
    def _set_max_possible_score(self, target_index, targets, i, index, records_seqs):
        '''fills the max_possible_score datastructure on the host'''
        pass
    
    def _get_starting_point_byte_array(self, number_of_starting_points):
        '''
        Get the resulting starting points
        @return gives the resulting starting point array as byte array
        '''
        pass
    
    def _get_direction_byte_array(self):
        '''
        Get the resulting directions
        @return gives the resulting direction array as byte array
        '''
        pass
    
    def _get_direction(self, direction_array, sequence, target, block_x, block_y, value_x, value_y):
 
        return direction_array[sequence][target][block_x][block_y][value_x][value_y]
        
    def _set_direction(self, direction, direction_array, sequence, target, block_x, block_y, value_x, value_y):
        direction_array[sequence][target][block_x][block_y][value_x][value_y] = direction

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
        #self.logger.debug('Setting parameters len_x: {0}, len_y: {1}, '
        #                  'number_sequences: {2}, '
        #                  'number_targets: {3} '.format(len_x, len_y, number_sequences, number_targets))
        if len_x % self.shared_x > 0 or len_y % self.shared_y > 0:
            raise InvalidOptionException("Error in _set_parameters: len_x % " + str(self.shared_x) +
                                         " or len_y % " + str(self.shared_y) + " != 0 !")
        self.length_of_x_sequences = len_x
        self.length_of_y_sequences = len_y
        self.number_of_sequences = number_sequences
        self.number_targets = number_targets
        self.x_div_shared_x = int(math.floor((len_x / self.shared_x)))
        self.y_div_shared_y = int(math.floor(len_y / self.shared_y))

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


    def set_targets(self, targets, target_index, max_length=None, records_seqs=None, use_all_records_seqs=True):
        '''Retrieves a block of targets from the target array and returns the index of the last target that will be processed'''
        self.max_number_of_targets = 1
        if self.max_length == None or target_index == 0:                
            self._set_target_block_length(targets, target_index)
            if records_seqs != None and len(records_seqs) > 0:
                if use_all_records_seqs:
                    self.max_number_of_targets = self._get_number_of_targets_with_sequences(records_seqs)
                    if self.max_number_of_targets < 1:
                        self.max_number_of_targets = self._get_number_of_targets()
                else:
                    # Find maximum possible number of targets for cases when number of records_seqs
                    # is small. _get_number_of_targets assumes that many sequences are used, hence
                    # it may return too small number.
                    self.max_number_of_targets = max(self._get_number_of_targets(),
                                                     self._get_number_of_targets_with_sequences(records_seqs))

        if max_length != None and self.settings.recompile == "F" and target_index > 0:
            self.max_length = max_length
        
        if target_index + self.max_number_of_targets < len(targets):
            self.number_of_targets = self.max_number_of_targets
        else:
            self.number_of_targets = len(targets) - target_index

        if self.number_of_targets > len(targets):
            self.number_of_targets = len(targets)

        if self.number_of_targets * len(targets[target_index]) / self.shared_y > self.internal_limit:
            self.number_of_targets = int(self.internal_limit * self.shared_y / len(targets[target_index]))
        

        # fill the target array with sequences
        targetStr = []
        self.added_dummy_targets = 0
        for i in range(self.number_of_targets):
            if i+target_index < len(targets):
                targetStr.append(SWSeq.extentToFillGPU(str(targets[i+target_index].seq), self.target_block_length))
            else:
                targetStr.append(SWSeq.extentToFillGPU(SWSeq.SPECIAL_CHAR*self.target_block_length, self.target_block_length))
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
        gapExtensionFactor = 1
        if self.gap_extension:
            gapExtensionFactor = 3

        # size of each element in a smith waterman matrix (direction (is char) and score (is float))
        mem_size = 1
        mem_size += gapExtensionFactor * self.float_size
        return mem_size

    def set_total_work_size(self, size):
        '''Sets total work size (number of cells) and resets current progress.
        '''
        self.total_work_size = size
        self.total_processed = 0
        self.start_time = time.time()

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
        hitlist=HitList(self.logger)
        while index < len(records_seqs) and cont:
            t0 = time.time()
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
                    #self.logger.info("Maximum number of seqs in memory: {} {}".format(self.max_sequences, self.number_of_targets))
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
                    if self.max_sequences + index >= len(records_seqs):
                        self.max_sequences = len(records_seqs) - index
                    self._init_sw(length, self.target_block_length, self.max_sequences, self.number_of_targets)

            # add sequences to the list
            
            self.added_dummy_seqs = 0

            sequenceStr = []
            self.added_dummy_seqs = 0

            self.min_score_np = numpy.zeros(self.max_sequences* self.number_of_targets, dtype=numpy.float32)
            for i in range(self.max_sequences):
                if i+index < len(records_seqs):
                    sequenceStr.append(SWSeq.extentToFillGPU(str(records_seqs[i+index].seq), length))
                else:
                    sequenceStr.append(SWSeq.extentToFillGPU(SWSeq.SPECIAL_CHAR*length, length))
                    self.added_dummy_seqs += 1

                self._set_max_possible_score(target_index, targets, i, index, records_seqs)
            self._copy_min_score()
            # copy sequences and targets to the device
            sequence_array = numpy.array(''.join(sequenceStr), dtype=numpy.character)

            self.logger.debug("At sequence: {0} of {1}, length = {2}".format(index, len(records_seqs), self.max_sequences))

            # set lengths of this run
            prev_seq_length = length
            prev_target_length = self.target_block_length
            if self.settings.recompile == "F":
                self._set_parameters(length, self.target_block_length, self.max_sequences-self.added_dummy_seqs, self.number_of_targets-self.added_dummy_targets)
                #self._set_parameters(length, self.target_block_length, self.max_sequences, self.number_of_targets)
            # copy sequences and targets to the device
            t = time.time()
            self.copy_sequences(sequence_array, self.target_array)
            # initialize index for zero copy of starting points
            self._init_zero_copy()
            # calculate scores of alignments
            self._calculate_score()

            if self._is_traceback_required():
                # perform the traceback
                self._traceback_host()

                # TODO: change to returning a value, change _print_alignments to getAlignments in SmithWaterman
                # TODO: move _print_alignments to here? This should be a statement to retrieve the results and
                # put them into a Hitlist (?)
                #hitlist = self._print_alignments(records_seqs, targets, index, target_index)
                self._print_alignments(records_seqs, targets, index, target_index, hitlist)

            self.logger.debug("Time spent on Smith-Waterman > {}".format(time.time()-t))

            if self.total_work_size > 0:
                t1 = time.time()
                processed = sum(len(s.seq) for s in targets[target_index:(target_index+self.number_targets)]) * \
                            sum(len(s.seq) for s in records_seqs[index:(index+self.number_of_sequences)])
                self.total_processed += processed
                duration = t1 - t0
                total_duration = t1 - self.start_time
                performance = processed / 1e9 / duration
                avg_performance = self.total_processed / 1e9 / total_duration
                progress = self.total_processed / float(self.total_work_size)
                eta = total_duration / progress * (1.0 - progress)
                self.logger.info("Duration: {:5.3f} | Total: {} | Performance: {:5.2f} GCUPS | Avg: {:5.2f} GCUPS | Progress: {:7.3%} | ETA: {}"
                    .format(duration, datetime.timedelta(seconds=round(total_duration)), performance, avg_performance, progress, datetime.timedelta(seconds=round(eta)))
                )

            index += self.max_sequences
        return hitlist

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
            #self.logger.debug("Setting parameters for CUDA: {} {} {} {}".format(length, target_length, max_sequences, number_of_targets))
            self._set_parameters(length, target_length, max_sequences, number_of_targets)
            self._set_score(self.score)
            # compile the code
            #self.logger.debug('Compiling CUDA code...')
            self._compile_code()
            # initialize memory
            #self.logger.debug('Initializing memory...')
            self._init_memory()
            self.has_been_compiled = True

    def _copy_min_score(self):
        pass

    def set_minimum_score(self, index, minScore):
        # @TO-DO: this is bugfix for the read mapping algorithm. Should not happen, so fix this where it should be fixed
        if index < len(self.min_score_np):
            #self.h_max_possible_score_zero_copy[index] = minScore
            self.min_score_np[index] = minScore

    def _calculate_score(self):
        """ Calculates the Smith-Waterman scores on the device """
        self.logger.debug('Calculating scores.')
        max_number_of_blocks = min(self.x_div_shared_x, self.y_div_shared_y)
        start_decrease_at = self.x_div_shared_x + self.y_div_shared_y - max_number_of_blocks
        number_of_blocks = 0
        idx = 0
        idy = 0

        for i in range(1, self.x_div_shared_x + self.y_div_shared_y):
            if (i <= max_number_of_blocks):
                number_of_blocks = i
            elif (i >= start_decrease_at):
                number_of_blocks = self.x_div_shared_x + self.y_div_shared_y - i
            else:
                number_of_blocks = max_number_of_blocks
            
            self._execute_calculate_score_kernel(number_of_blocks, idx, idy)

            if (idx == self.x_div_shared_x - 1):
                idy += 1
            if (idx < self.x_div_shared_x - 1):
                idx += 1

    def _is_traceback_required(self):
        '''Returns False if it is known after calculating scores that there are no possible
        starting points, hence no need to run traceback.
        '''
        return True

    def _traceback_host(self):
        ''' Performs the traceback on the device '''
        self.logger.debug('Performing back trace.')
        max_number_of_blocks = min(self.x_div_shared_x, self.y_div_shared_y)
        start_decrease_at = self.x_div_shared_x + self.y_div_shared_y - max_number_of_blocks
        number_of_blocks = 0
        idx = self.x_div_shared_x - 1
        idy = self.y_div_shared_y - 1


        for i in range(1, self.x_div_shared_x + self.y_div_shared_y):
            if (i <= max_number_of_blocks):
                number_of_blocks = i
            elif(i >= start_decrease_at):
                number_of_blocks = self.x_div_shared_x + self.y_div_shared_y - i
            else:
                number_of_blocks = max_number_of_blocks
            
            self._execute_traceback_kernel(number_of_blocks, idx, idy)

            if (idy == 0):
                idx -= 1
            if (idy > 0):
                idy -= 1

        #self.logger.debug("{}".format(self._get_direction_byte_array()))
    
    def _is_in_alignment(self, show, block_x, block_y, value_x, value_y, direction):
        ''' Checks to see if printing the alignment should continue. This method should be implemented by subclasses
        when looping through direction matrix has changed (see SmithWatermanCPU (SmithWatermanOCL)) '''
        return show and block_x >= 0 and block_y >= 0 and value_x >= 0 and value_y >= 0 and direction != STOP_DIRECTION and direction != NO_DIRECTION      
    
     
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

        number_of_starting_points = self._get_number_of_starting_points()
        self.logger.debug('Number of starting points is: {0}.'.format(number_of_starting_points))
        if number_of_starting_points == 0:
            # No need to read other data from device
            return hit_list
        if number_of_starting_points >= (self.maximum_number_starting_points * self.number_of_sequences * self.number_targets):
            self.logger.warning("Too many hits returned. Skipping the rest. Please set lower_limit_score higher in config.")
            number_of_starting_points = self.maximum_number_starting_points * self.number_of_sequences * self.number_targets

        starting_points = self._get_starting_point_byte_array(number_of_starting_points)

        max_score = 0

        direction_array = self._get_direction_byte_array()
#        self.logger.debug(direction_array)
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

            block_x = int(starting_point.block_x)
            block_y = int(starting_point.block_y)
            value_x = int(starting_point.value_x)
            value_y = int(starting_point.value_y)

            local_index = 0
            s_end = block_x * self.shared_x + value_x
            t_end = block_y * self.shared_y + value_y

            
            # @TO-DO: this is bugfix for the read mapping algorithm. Should not happen, so fix this where it should be fixed
            if start_seq + sequence_starting_point >= len(sequences) or start_target + target_starting_point >= len(targets):
                self.logger.debug("Starting points in hit incorrect. Skipping")
                continue
 
            if hasattr(sequences[start_seq + sequence_starting_point], 'start_position'):
                s_end += sequences[start_seq + sequence_starting_point].start_position
            if hasattr(targets[start_target + target_starting_point], 'start_position'):
                t_end += targets[start_target + target_starting_point].start_position
            if not hasattr(sequences[sequence_starting_point + start_seq], 'distance'):
                sequences[sequence_starting_point + start_seq].distance = 0.0
            s_start = s_end + 1
            t_start = t_end + 1

            #direction = direction_array[sequence_starting_point][target_starting_point][block_x][block_y][value_x][value_y]
            direction = self._get_direction(direction_array, sequence_starting_point,target_starting_point,block_x,block_y,value_x,value_y)
            show = True
            # check in 'all to all' when 1 data set is used to filter out hit X vs X (filtered on identical id):
            if sequences[sequence_starting_point + start_seq].id == targets[target_starting_point + start_target].id:
                direction = STOP_DIRECTION
                self.logger.debug("Found same ID sequence -> target. Skipping")
                show = False

            #self.logger.debug('Score is: {0} vs {1}.'.format(starting_point.score, self.settings.minimum_score))
            if starting_point.score < float(self.settings.minimum_score):
                show = False
            
            while (self._is_in_alignment(show, block_x, block_y, value_x, value_y, direction)):
                direction = self._get_direction(direction_array,sequence_starting_point,target_starting_point,block_x,block_y,value_x,value_y)
                self._set_direction(IN_ALIGNMENT,direction_array,sequence_starting_point,target_starting_point,block_x,block_y,value_x,value_y)
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
            #if not show:
            #    self.logger.debug("Hit {0} -vs- {1} not shown".format(sequences[sequence_starting_point + start_seq].id, targets[target_starting_point + start_target].id ))
        return hit_list
    
    def _filter_hit(self, hit):
        '''Check to see hit meets filter requirements'''
        return (hit.relative_score >= float(self.settings.relative_score) and 
            hit.query_identity >= float(self.settings.query_identity) and 
            hit.query_coverage >= float(self.settings.query_coverage) and 
            hit.base_score >= float(self.settings.base_score))             
