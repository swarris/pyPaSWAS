from string import Template
from pyPaSWAS.Core import resource_filename, read_file


class Code(object):
    '''
    Initializes the CUDA code by setting configuration parameters using the CUDA
    templates located in Core/cuda
    '''
    def __init__(self, logger):
        self.logger = logger
        self.shared_xy_code = ''
        self.directions = ''
        self.score_part = ''
        self.variable_part = ''
        self.variable_source = ''
        self.direction_source = ''
        self.score_source = ''
        self.main_source = ''
    
    def set_shared_xy_code(self, sharedx=8, sharedy=8):
        '''
        Sets the horizontal and the vertical sizes of the smallest alignment matrices in shared memory
        :param sharedx:
        :param sharedy:
        '''
        #self.logger.debug('Setting sharedx to {0}, sharedy to {1}'.format(sharedx, sharedy))
        code_t = Template(read_file(self.main_source))
        self.shared_xy_code = code_t.safe_substitute(SHARED_X=sharedx, SHARED_Y=sharedy)

    def set_direction_code(self, no_direction=0, up_left=1, up=2, left=3, stop=4):
        '''
        TODO: Docstring
        :param no_direction:
        :param up_left:
        :param up:
        :param left:
        :param stop:
        '''
        #self.logger.debug('Setting directions:\n\tno = {0}\n\tup_left = {1}\n\tup = {2}\n\tleft = {3}\n\t'
        #                  'stop = {3}'.format(no_direction, up_left, up, left, stop))
        direction_t = Template(read_file(self.direction_source))
        self.directions = direction_t.safe_substitute(NO_DIRECTION=no_direction,
                                                      UP_LEFT_DIRECTION=up_left,
                                                      UP_DIRECTION=up,
                                                      LEFT_DIRECTION=left,
                                                      STOP_DIRECTION=stop)

    def set_score_code(self, score):
        '''Formats information contained in a score.
        '''
        #self.logger.debug('Sourcing the scorepart of the cuda code')
        score_part_t = Template(read_file(self.score_source))
        gap_extension = 0.0
        if score.gap_extension != None:
            gap_extension = score.gap_extension
            
        self.score_part = score_part_t.safe_substitute(SCORE_TYPE=score.score_type,
                                                       LOWER_LIMIT=score.lower_limit_score,
                                                       MINIMUM_SCORE=score.minimum_score,
                                                       MAX_SCORE=score.lower_limit_max_score,
                                                       GAP_SCORE=score.gap_score,
                                                       GAP_EXTENSION=gap_extension,
                                                       HIGHEST_SCORE=score.highest_score,
                                                       MATRIX=score.__str__(),
                                                       DIMENSION=score.dimensions)

    def set_variable_code(self, number_sequences, number_targets, x_val, y_val, char_offset):
        '''Sets the variable part of the code'''
        #self.logger.debug('Setting the variable part of the cuda code\n\t(using: n_seq: {}, n_targets: {}, '
        #                  'x_val: {}, y_val: {})'.format(number_sequences, number_targets, x_val, y_val))
        variable_t = Template(read_file(self.variable_source))
        self.variable_part = variable_t.safe_substitute(N_SEQUENCES=number_sequences,
                                                        N_TARGETS=number_targets,
                                                        X=x_val,
                                                        Y=y_val,
                                                        CHAR_OFFSET=char_offset)

    def get_code(self, score, number_sequences, number_targets, x_sequence_length, y_sequence_length):
        '''Retrieves the source for the cuda program'''
        #self.logger.debug('Formatting the cuda source code...')
        self.set_score_code(score)
        self.set_variable_code(number_sequences, number_targets, x_sequence_length, y_sequence_length, score.char_offset)
        #self.logger.debug('Formatting the cuda source code OK.')
        return self.variable_part + self.directions + self.score_part + self.shared_xy_code


class Cudacode(Code):
    '''
    Initializes the CUDA code by setting configuration parameters using the CUDA
    templates located in Core/cuda
    '''
    def __init__(self, logger):
        Code.__init__(self, logger)
        self.variable_source = resource_filename(__name__, 'cuda/default_variable.cu')
        self.direction_source = resource_filename(__name__, 'cuda/default_direction.cu')
        self.score_source = resource_filename(__name__, 'cuda/default_score.cu')
        self.main_source = resource_filename(__name__, 'cuda/default_main.cu')

class OCLcode(Code):
    '''
    Initializes the OpenCL code by setting configuration parameters using the OpenCL
    templates located in Core/ocl
    '''
    def __init__(self, logger):
        Code.__init__(self, logger)
        self.variable_source = resource_filename(__name__, 'ocl/default_variable.cl')
        self.direction_source = resource_filename(__name__, 'ocl/default_direction.cl')
        self.score_source = resource_filename(__name__, 'ocl/default_score.cl')
        
class GPUcode(OCLcode):
    '''
    Initializes the GPU OpenCL code by setting configuration parameters using the OpenCL
    templates located in Core/ocl
    '''
    def __init__(self, logger):
        OCLcode.__init__(self, logger)
        self.main_source = resource_filename(__name__, 'ocl/default_main_gpu.cl')
        
class CPUcode(OCLcode):
    '''
    Initializes the GPU OpenCL code by setting configuration parameters using the OpenCL
    templates located in Core/ocl
    '''
    def __init__(self, logger):
        OCLcode.__init__(self, logger)
        self.main_source = resource_filename(__name__, 'ocl/default_main_cpu.cl')
        
    def set_shared_xy_code(self, sharedx=8, sharedy=8, workloadx=4, workloady=4):
        '''
        Sets the horizontal and the vertical sizes of the smallest alignment matrices in shared memory
        :param sharedx:
        :param sharedy:
        '''
        #self.logger.debug('Setting sharedx to {0}, sharedy to {1}'.format(sharedx, sharedy))
        code_t = Template(read_file(self.main_source))
        self.shared_xy_code = code_t.safe_substitute(SHARED_X=sharedx, SHARED_Y=sharedy, WORKLOAD_X=workloadx, WORKLOAD_Y=workloady)