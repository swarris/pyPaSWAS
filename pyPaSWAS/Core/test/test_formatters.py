'''
Changed on May 13, 2013

@author: arne poortinga
'''
import unittest
import logging
import tempfile
import os

from pyPaSWAS.Core import Formatters
from pyPaSWAS.Core.Hit import Hit
from pyPaSWAS.Core.HitList import HitList


class _SeqId():
    ''' Sets parameters of a new sequence object  '''
    def __init__(self, ident, seq):
        self.id = ident
        self.seq = seq
        self.original_length = len(seq)
        self.distance = None


class FormatterTester(unittest.TestCase):

    def setUp(self):
        ''' Create a HitList object with some hits (all bogus) '''
        self.logger = logging.root
        self.hitlist = HitList(self.logger)
        self.number_of_hits = 10000

        for number in range(self.number_of_hits):
            sequence = _SeqId('sequence-{}'.format(number), "GCTGACTGACTG")
            target = _SeqId('target-{}'.format(number), "ACTGACTGACTG")
            alignment = '-CTGACTGACTG'

            # Create hit
            sequence_location = 0, 1
            target_location = 0, 1
            hit = Hit(self.logger, sequence, target, sequence_location, target_location)
            hit.set_alignment(alignment)
            hit.set_sequence_match(sequence.seq)
            hit.set_target_match(target.seq)

            self.hitlist.append(hit)

        #create temporary file for the output
        self.outputfile = tempfile.mkstemp(suffix='.out', prefix='test_formatters_')[1]

    def testDefaultFormatter(self):
        ''' Run the default formatter '''
        formatter = Formatters.DefaultFormatter(self.logger, self.hitlist, self.outputfile)
        formatter.print_results()
        with open(self.outputfile) as reader:
            lines = reader.readlines()
        self.assertEqual(self.number_of_hits * 4, len(lines))

    def testSamFormatter(self):
        '''Run the SAM output formatter '''
        formatter = Formatters.SamFormatter(self.logger, self.hitlist, self.outputfile)
        formatter.print_results()
        with open(self.outputfile) as reader:
            lines = reader.readlines()
        # Not sure if the number of lines is always equal to the assertion below
        self.assertEqual(self.number_of_hits * 2 + 2, len(lines))

    def tearDown(self):
        os.remove(self.outputfile)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testDefaultFormatter']
    unittest.main()
    import cProfile
    cProfile.run('unittest.main()', 'test_formatters.cProfile')
