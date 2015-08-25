'''TODO: Document SWSeqRecord'''
from Bio.SeqRecord import SeqRecord
import math


class SWSeqRecord(SeqRecord):
    '''
    TODO: Docstring
    '''

    start_position = 0
    distance = 0
    original_length = 0

    def __init__(self, seq, identifier, start_position=0, original_length = None, distance = 0, refID = None):
        super(SWSeqRecord, self).__init__(seq.upper(), identifier)
        self.start_position = start_position
        if original_length ==None:
            self.original_length = len(seq)
        else:
            self.original_length = original_length
        self.distance = distance 
        self.refID = refID

    def _compare(self, other):
        '''
        TODO: Docstring
        :param other:
        '''
        self.distance = math.sqrt((self.count[0] - other.count[0]) ** 2 + (self.count[1] - other.count[1]) ** 2 +
                                  (self.count[2] - other.count[2]) ** 2 + (self.count[3] - other.count[3]) ** 2 +
                                  (self.count[4] - other.count[4]) ** 2)
        other.distance = self.distance
        return self.distance
