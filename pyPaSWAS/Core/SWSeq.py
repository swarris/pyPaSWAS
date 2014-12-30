'''TODO: Document SWSeq'''
import numpy
from Bio.Seq import Seq


class SWSeq(Seq):
    SPECIAL_CHAR = 'x'

    def __init__(self, string, alphabet):
        super(SWSeq, self).__init__(string, alphabet)

    def to_numpy_array(self, length=None):
        if (length == None):
            return numpy.array(list(str(self.upper())), dtype=numpy.character)
        else:
            if (length < len(self)):
                return numpy.array(list(str(self.upper()))[0:length], dtype=numpy.character)
            else:
                seqList = list(str(self.upper()))
                seqList.extend(list(self.SPECIAL_CHAR * (length - len(self))))
                return numpy.array(seqList, dtype=numpy.character)
    @staticmethod
    def extentToFillGPU(seq, length = None):
        if (length == None):
            return seq.upper()
        else :
            if (length < len(seq)):
                return seq.upper()[0:length]
            else :
                return seq.upper() + SWSeq.SPECIAL_CHAR*(length-len(seq))
