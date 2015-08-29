import numpy
#TODO: Document Startingpoint


class StartingPoint(object):
    """Based on:
        typedef struct {
        unsigned int sequence;
        unsigned int target;
        unsigned int block_x;
        unsigned int block_y;
        unsigned int value_x;
        unsigned int value_y;
        float score;
      float max_score;
      float pos_score;
      } StartingPoint;
    """
    def __init__(self, logger):
        self.target_index = 0
        self.target = 0
        self.value_x = 0
        self.value_y = 0
        self.score = 0
        self.max_score = 0
        self.pos_score = 0
        self.block_x = 0
        self.block_y = 0

        self.sequence_index = 0
        self.size = 9 * 4
        self.sequence = ''
        self.logger = logger

    
    def parse_byte_string(self, byte_string, location):
        """parse_byte_string sets the member variables based on the bytes in the byte string.
        @param byte_string: the byte string with all the starting points
        @param location: which starting point to get
        """
        #self.logger.debug('Parsing byte string...')
        index = location * self.size
        self.sequence = numpy.fromstring(byte_string[index:index + 4], dtype=numpy.int32)
        if (len(self.sequence) > 0):
            self.sequence = self.sequence[0]
        else:
            #TODO: what sequence number?
            self.logger.warning("Warning: could not get sequence number: ")
            self.sequence = 0

        self.target = numpy.fromstring(byte_string[index + 4:index + 8], dtype=numpy.int32)
        if (len(self.target) > 0):
            self.target = self.target[0]
        else:
            #TODO: what sequence number?
            self.logger.warning("Warning: could not get target number")
            self.target = 0

        self.block_x = numpy.fromstring(byte_string[index + 8:index + 12], dtype=numpy.int32)
        if (len(self.block_x) > 0):
            self.block_x = self.block_x[0]
        else:
            self.block_x = 0

        self.block_y = numpy.fromstring(byte_string[index + 12:index + 16], dtype=numpy.int32)
        if (len(self.block_y) > 0):
            self.block_y = self.block_y[0]
        else:
            self.block_y = 0

        self.value_x = numpy.fromstring(byte_string[index + 16:index + 20], dtype=numpy.int32)[0]
        self.value_y = numpy.fromstring(byte_string[index + 20:index + 24], dtype=numpy.int32)[0]
        self.score = numpy.asscalar(numpy.fromstring(byte_string[index + 24:index + 28], dtype=numpy.float32,)[0])
        self.max_score = numpy.fromstring(byte_string[index + 28:index + 32], dtype=numpy.float32)[0]
        self.pos_score = numpy.fromstring(byte_string[index + 32:index + 36], dtype=numpy.float32)[0]
        #self.logger.debug('Parsing byte string OK.')

    @staticmethod
    def _byte_string_to_list(byte_string, length, sequence_index, target_index):
        '''
        TODO: docstring
        :param byte_string:
        :param length:
        :param sequence_index:
        :param target_index:
        '''
        location = 0
        result_list = []
        while location < length:
            starting_point = StartingPoint()
            starting_point.parse_byte_string(byte_string, location)
            starting_point.sequence_index = sequence_index
            starting_point.target_index = target_index
            result_list.append(starting_point)
            location += 1
        return result_list

    def __str__(self):
        """toString method
            @return a string containing the information about the starting point
        """
        return ("Seq: {0}\nTar: {1}\nblockX: {2}\nblockY:{3}\nvalueX: {4}\nvalueY: {5}\nscore: {6}\nmaxScore: {7}"
                "\nposScore: {8}".format(self.sequence, self.target, self.block_x, self.block_y, self.value_x,
                                         self.value_y, self.score, self.max_score, self.pos_score))

