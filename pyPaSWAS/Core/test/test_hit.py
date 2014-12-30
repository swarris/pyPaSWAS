'''Unittests for Core.Hit
@author: Arne Poortinga
'''

import logging
import unittest
from pyPaSWAS.Core.Hit import Hit
from pyPaSWAS.Core.Exceptions import CudaException


class _FullSequence():
    '''Models the full sequence_info and target_info objects of a Hit'''
    def __init__(self, ident, seq):
        self.id = ident
        self.seq = seq
        self.original_length = len(seq)
        self.distance = None


class HitTester(unittest.TestCase):
    '''The testcase for Core.Hit'''

    def setUp(self):
        '''For the creation of a hit a logging object is required'''
        self.logger = logging.getLogger()

        self.sequence_id = 'sequence'
        self.target_id = 'target'
        self.sequence_seq = 'GGACTGTAGC'
        self.target_seq = 'GGGGACTGTAGCCC'
        self.sequence_match = 'GGACTGTAGC'
        self.target_match = 'GGACTGTAGC'
        self.alignment = '||||||||||'
        self.sequence_location = (0, 9)
        self.target_location = (2, 11)

        self.score = 100.0
        # score can also be an int
        self.intscore = 100
        self.matches = 42
        self.mismatches = 12

        self.seq_info = _FullSequence(self.sequence_id, self.sequence_seq)
        self.targ_info = _FullSequence(self.target_id, self.target_seq)
        self.hit = Hit(self.logger, self.seq_info, self.targ_info, self.sequence_location, self.target_location)
        self.hit.set_sequence_match(self.sequence_match)
        self.hit.set_target_match(self.target_match)
        self.hit.set_alignment(self.alignment)
        self.hit.set_scores(self.score, self.matches, self.mismatches)
        # get the sam-line and split it on its separator. The resulting list can be used in a number of tests
        self.sam_line_fields = self.hit.get_sam_line().split('\t')

        # a more complicated match including inserts, mismatches, deletions, clipping
        self.rev_sequence_id = 'sequenceRC'
        self.rev_target_id = 'targetRC'
        self.rev_sequence_seq = 'ACTGACTGACTGACTGCC'
        self.rev_target_seq = 'CCGCTGTCCGATTCTGTGTGGG'
        self.rev_sequence_match = 'CTGACTGA--CTGACTG'
        self.rev_target_match = 'CTGTCCGATTCTG--TG'
        self.rev_alignment = '|||.|.||  |||  ||'
        self.rev_sequence_location = (1, 15)
        self.rev_target_location = (3, 17)

        rev_seq_info = _FullSequence(self.rev_sequence_id, self.rev_sequence_seq)
        rev_targ_info = _FullSequence(self.rev_target_id, self.rev_target_seq)
        self.rev_hit = Hit(self.logger, rev_seq_info, rev_targ_info, self.rev_sequence_location, self.rev_target_location)
        self.rev_hit.set_sequence_match(self.rev_sequence_match)
        self.rev_hit.set_target_match(self.rev_target_match)
        self.rev_hit.set_alignment(self.rev_alignment)
        self.rev_hit.set_scores(self.intscore, self.matches, self.mismatches)
        self.rev_sam_line_fields = self.rev_hit.get_sam_line().split('\t')

    def tearDown(self):
        '''Explicitly remove all created objects that have a bigger scope than an individual method'''
        self.hit = None
        self.rev_hit = None
        self.logger = None

    def test_constructor(self):
        '''Test whether initialization is correctly performed or not'''
        seq_info = _FullSequence(self.sequence_id, self.sequence_seq)
        targ_info = _FullSequence(self.target_id, self.target_seq)
        hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, self.target_location)
        self.assertEquals(hit.sequence_match, '')
        self.assertEquals(hit.alignment, '')
        self.assertEquals(hit.target_match, '')
        self.assertAlmostEquals(hit.score, 0.0)
        self.assertEquals(hit.matches, 0)
        self.assertEquals(hit.mismatches, 0)
        self.assertEquals(hit.distance, None)

        self.assertEquals(self.hit.get_seq_id(), self.sequence_id)
        self.assertEquals(self.hit.get_target_id(), self.target_id)

    def test_invalid_seq_locations(self):
        '''A location should contain a start- and a stop position. Both should be integers and be situated within the sequence. It should start before it ends.'''
        seq_info = _FullSequence(self.sequence_id, self.sequence_seq)
        targ_info = _FullSequence(self.target_id, self.target_seq)

        # valid location also tested in test_constructor
        sequence_location = (1, 9)
        hit = Hit(self.logger, seq_info, targ_info, sequence_location, self.target_location)
        self.assertEquals(hit.seq_location, sequence_location)

        # length location < 2
        sequence_location = (1)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, sequence_location, self.target_location)

        # length location > 2
        sequence_location = (1, 1, 9)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, sequence_location, self.target_location)

        # start outside start of sequence
        sequence_location = (-1, 9)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, sequence_location, self.target_location)

        # stop outside sequence
        sequence_location = (1, 10)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, sequence_location, self.target_location)

        # start >= stop
        sequence_location = (9, 9)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, sequence_location, self.target_location)

        # start not int
        sequence_location = ('a', 9)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, sequence_location, self.target_location)

        # stop not int
        sequence_location = (1, 'a')
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, sequence_location, self.target_location)

    def test_invalid_target_locations(self):
        '''A location should contain a start- and a stop position. Both should be integers and be situated within the sequence. It should start before it ends.'''
        seq_info = _FullSequence(self.sequence_id, self.sequence_seq)
        targ_info = _FullSequence(self.target_id, self.target_seq)

        # valid location also tested in test_constructor
        target_location = (2, 11)
        hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, target_location)
        self.assertEquals(hit.target_location, target_location)

        # length location < 2
        target_location = (2)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, target_location)

        # length location > 2
        target_location = (0, 0, 11)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, target_location)

        # start outside start of sequence
        target_location = (-1, 9)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, target_location)

        # stop outside sequence
        target_location = (0, 14)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, target_location)

        # start >= stop
        target_location = (7, 7)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, target_location)

        # start not int
        target_location = ('a', 9)
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, target_location)

        # stop not int
        target_location = (0, 'a')
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, seq_info, targ_info, self.sequence_location, target_location)

    def test_set_scores(self):
        '''score should be a positive float or integer, matches should be a positive int, mismatches should be a positive int'''
        # default valid score, matches and mismatches
        self.assertAlmostEqual(self.hit.score, self.score)
        self.assertEquals(self.hit.matches, self.matches)
        self.assertEquals(self.hit.mismatches, self.mismatches)

        self.assertAlmostEqual(self.rev_hit.score, self.intscore)
        self.assertEquals(self.rev_hit.matches, self.matches)
        self.assertEquals(self.rev_hit.mismatches, self.mismatches)

        # negative score
        with self.assertRaises(CudaException):
            self.hit.set_scores(-self.score, self.matches, self.mismatches)

        # invalid type for score
        with self.assertRaises(CudaException):
            self.hit.set_scores(-self.score, self.matches, self.mismatches)

        # negative matches
        with self.assertRaises(CudaException):
            self.hit.set_scores(self.score, -self.matches, self.mismatches)

        # invalid type for matches
        with self.assertRaises(CudaException):
            self.hit.set_scores(-self.score, 42.0, self.mismatches)

        # negative mismatches
        with self.assertRaises(CudaException):
            self.hit.set_scores(self.score, self.matches, -self.mismatches)

        # invalid type for mismatches
        with self.assertRaises(CudaException):
            self.hit.set_scores(-self.score, self.matches, 12.0)

    def test_keys(self):
        '''the number of keys should be 3'''
        self.assertEquals(len(self.hit.keys()), 3)

    def test_get_seq_id(self):
        '''Should be equal to the id of sequence_info.'''
        self.assertEquals(self.hit.get_seq_id(), self.sequence_id)

    def test_get_target_id(self):
        '''Should be equal to the id of target_info.'''
        self.assertEquals(self.hit.get_target_id(), self.target_id)

    def test_get_euclidian_distance(self):
        '''Not set, should return 0.0'''
        self.assertAlmostEqual(self.hit.get_euclidian_distance(), 0, 0)
        self.hit.distance = 10.0
        self.assertAlmostEqual(self.hit.get_euclidian_distance(), 0, 10.0)

    def test_set_sequence_match(self):
        '''Sets the string that contains match information of the sequence part of the alignment
            This determined outside the Hit class
        '''
        self.assertEquals(self.hit.sequence_match, self.sequence_match)
        sequence_match = 'ACTGT.AGC'
        self.hit.set_sequence_match(sequence_match)
        self.assertEquals(self.hit.sequence_match, sequence_match)

    def test_set_target_match(self):
        '''Sets the string that contains match information of the target part of the alignment
            This determined outside the Hit class
        '''
        self.assertEquals(self.hit.target_match, self.target_match)
        target_match = 'ACTGT.AGC'
        self.hit.set_target_match(target_match)
        self.assertEquals(self.hit.target_match, target_match)

    def test_set_alignment(self):
        '''Sets the string that contains the alignment information of the alignment
            This determined outside the Hit class
        '''
        self.assertEquals(self.hit.alignment, self.alignment)
        alignment = 'ACTGT.AGC'
        self.hit.set_alignment(alignment)
        self.assertEquals(self.hit.alignment, alignment)

    def test_get_sam_line(self):
        '''Creates and returns an alignment line as described in http://samtools.sourceforge.net/SAM1.pdf
           The following mappings are used for record lines:
                QNAME:  self.get_seq_id()
                FLAG:   only bit 0x2 and 0x10 are set since the other values are either false or unknown
                RNAME:  self.get_target_id()
                POS:    hit.target_location[0] + 1 (+1 since SAM is 1 based!)
                MAPQ:  set to 255 to indicate quality score is not available
                CIGAR:  see _get_sam_cigar
                RNEXT:  set to * (not available)
                PNEXT:  set to 0 (not available)
                TLEN:   set to 0 (not available)
                SEQ:    self.sequence_info.sequence
                QUAL:   set to * (not available)
                AS:     self.score
                AD:f:   computed euclidian distance between sequence and target
                RS:f:   computed relative score
        '''

        # each sam line should contain 14 fields separated by tabs
        self.assertEquals(len(self.sam_line_fields), 14)

        # sequence_match missing
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, self.seq_info, self.targ_info, self.sequence_location, self.target_location)
            hit.set_target_match(self.target_match)
            hit.set_alignment(self.alignment)
            hit.set_scores(self.score, self.matches, self.mismatches)
            sam_line = hit.get_sam_line()
            # TODO: assert

            # target_match missing
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, self.seq_info, self.targ_info, self.sequence_location, self.target_location)
            hit.set_sequence_match(self.sequence_match)
            hit.set_alignment(self.alignment)
            hit.set_scores(self.score, self.matches, self.mismatches)
            sam_line = hit.get_sam_line()
            # TODO: assert

        # alignment missing
        with self.assertRaises(CudaException):
            hit = Hit(self.logger, self.seq_info, self.targ_info, self.sequence_location, self.target_location)
            hit.set_sequence_match(self.sequence_match)
            hit.set_target_match(self.target_match)
            hit.set_scores(self.score, self.matches, self.mismatches)
            sam_line = hit.get_sam_line()
            # TODO: assert

    def test_get_sam_line_id(self):
        ''' the first field should contain the id of the sequence'''
        self.assertEquals(self.sam_line_fields[0], self.sequence_id, 'sequence id in sam line not correct')

    def test_get_sam_line_flag(self):
        '''FLAG should be 0 or 16 if working with a match on the reversed complement'''
        self.assertEquals(int(self.sam_line_fields[1]), 0, 'flag in sam line not correct')
        self.assertEquals(int(self.rev_sam_line_fields[1]), 16, 'flag of reversed complement in sam line not correct')

    def test_get_sam_line_target_id(self):
        '''the 3rd field should equal the name of the target reference'''
        self.assertEquals(self.sam_line_fields[2], self.target_id, 'sequence id in sam line not correct')

    def test_get_sam_line_pos(self):
        '''match starts at the 2nd position from the left, should be 3 since SAM is 1 based'''
        self.assertEquals(int(self.sam_line_fields[3]), 3, 'pos in sam line not correct')

    def test_get_sam_line_mapq(self):
        '''no mapq score is available: 255'''
        self.assertEquals(int(self.sam_line_fields[4]), 255, 'mapq in sam line not correct')

    def test_get_sam_line_cigar(self):
        '''CIGAR:  see _get_sam_cigar. Soft clipping, (mis)matches, insertions and deletions are covered '''
        # simple alignment
        self.assertEquals(self.sam_line_fields[5], '10=', 'cigar in sam line not correct.Expected "10="\tFound ' + self.sam_line_fields[5])

        # alignment with soft clipping, insertions, deletions and mismatches
        self.assertEquals(self.rev_sam_line_fields[5], '1S3=1X1=1X2=2D3=2I2=2S', 'cigar in sam line not correct.')

    def test_get_sam_line_rnext(self):
        '''RNEXT:  set to * (not available)'''
        self.assertEquals(self.sam_line_fields[6], '*', 'rnext in sam line not correct')

    def test_get_sam_line_pnext(self):
        '''PNEXT:  set to 0 (not available)'''
        self.assertEquals(int(self.sam_line_fields[7]), 0, 'pnext in sam line not correct')

    def test_get_sam_line_tlen(self):
        '''TLEN:   set to 0 (not available)'''
        self.assertEquals(int(self.sam_line_fields[8]), 0, 'tlen in sam line not correct')

    def test_get_sam_line_(self):
        '''SEQ:    self.sequence_info.sequence'''
        self.assertEquals(self.sam_line_fields[9], self.sequence_seq, 'seq in sam line not correct')

    def test_get_sam_line_qual(self):
        '''QUAL:   set to * (not available)'''
        self.assertEquals(self.sam_line_fields[10], '*', 'qual in sam line not correct')

    def test_get_sam_line_opt_fields(self):
        '''AS:     self.score
           AD:f:   computed euclidian distance between sequence and target
           RS:f:   computed relative score (score/length_of_target)
        '''
        self.assertEquals(self.sam_line_fields[11], 'AS:i:{0}'.format(self.intscore), 'score in sam line not correct')
        self.assertEquals(self.sam_line_fields[12], 'AD:f:{0}'.format(0.0), 'euclidian distance in sam line not correct')
        self.assertEquals(self.sam_line_fields[13], 'RS:f:{0}'.format(self.score / len(self.target_seq)), 'relative score in sam line not correct')

    def test_get_sam_sq_line(self):
        '''the SN field should equal the id of the target, the LN fields the length of th target'''
        sq_line = self.hit.get_sam_sq()
        sq_reference = '@SQ\tSN:{0}\tLN:{1}'.format(self.target_id, len(self.target_seq))
        self.assertEquals(sq_line, sq_reference)
