'''
original version by Sven Warris

@author: Arne Poortinga
'''

from pyPaSWAS.Core.Exceptions import CudaException
from Bio.Seq import Seq

class Hit(object):
    '''This class models an alignment between a sequence and a target.
        It contains information about the exact alignment, its positions in both the sequence and the target,
        the final score and the number of matches and mismatches.
    '''
    def __init__(self, logger, sequence_info, target_info, sequence_location, target_location):
        ''' Constructor
            Parameters
                logger:             logging object
                sequence_info:      SWSeqRecord instance that contains information about the
                                    sequence that is being aligned
                target_info:        SWSeqRecord instance that contains information about the
                                    target against which is being aligned
                sequence_location:  the location (tuple with start- and end positions) of the
                                    matching part of full_sequence
                target_location:    the location (tuple with start- and end positions) of the
                                    matching part of full_target
        '''
        self.logger = logger
        #self.logger.debug('Initializing hit: \n{} \n{} \n{} \n{}'. format(sequence_info.id, target_info.id, sequence_location, target_location))

        self.sequence_info = sequence_info
        self.target_info = target_info

        self.seq_location = ()
        self.target_location = ()
        self._set_seq_location(sequence_location)
        self._set_target_location(target_location)

        # the following strings contain the formatted alignment information
        # (viz. classic BLAST alignment representation)
        self.sequence_match = ''
        self.alignment = ''
        self.target_match = ''

        # set scores
        self.score = 0
        self.matches = 0
        self.mismatches = 0
        self.query_identity = 0
        self.query_coverage = 0
        self.relative_score = 0
        self.distance = None
        self.rc = False

        self.logger.debug('Initializing hit OK.')

    @staticmethod
    def _is_a_valid_location(location, sequence_length):
        ''' Verifies wether or not a location is valid.
            A location should contain two integers which should stand for the start of
            the alignment in a sequence (or target) and its stop position. Both start
            and stop should fall between the start and the end of the sequence or target.
        '''
        return isinstance(location, tuple) and len(location) == 2 and isinstance(location[0], int) and \
            isinstance(location[1], int) and 0 <= location[0] < sequence_length and \
            0 <= location[1] < sequence_length and location[0] < location[1]

    def _set_seq_location(self, location):
        ''' Sets the location of the alignment within the sequence.
            This location should be a tuple containing the positions at which the
            alignment begins and ends, respectively.
            Both values should be integers.
        '''
        if self._is_a_valid_location(location, self.sequence_info.original_length):
            #self.logger.debug('Set hit.seqlocation to {0}'.format(location))
            self.seq_location = location
        else:
            raise CudaException('Invalid sequence location: {0}, length={1}, {2}'.format(location, len(self.sequence_info.seq),self.sequence_info.original_length ))

    def _set_target_location(self, location):
        ''' Sets the location of the alignment within the target.
            This location should be a tuple containing the positions at which the
            alignment begins and ends, respectively. Both values should be integers.
        '''
        if self._is_a_valid_location(location, self.target_info.original_length):
            #self.logger.debug('Set hit.target_location to {0}'.format(location))
            self.target_location = location
        else:
            raise CudaException('Invalid target location: {0}'.format(location))

    def set_scores(self, score, matches, mismatches):
        '''Sets the total score, the number of matches and the number of mismatches'''
        self.logger.debug('Setting scores of hit\n\tscore = {0}'
                          '\n\tmatches = {1}\n\tmismatches = {2}'.format(score, matches, mismatches))
        if (isinstance(score, float) or isinstance(score, int)) and score >= 0.0:
            self.score = score
        else:
            #self.logger.debug('\tscore type: {0}'.format(type(score)))
            raise CudaException('Score should be a float or an integer, not {0}'.format(score))

        if isinstance(matches, int) and matches >= 0:
            self.matches = matches
        else:
            raise CudaException('Matches  should be integers. Passed value is {0}'.format(matches))

        if isinstance(mismatches, int) and mismatches >= 0:
            self.mismatches = mismatches
        else:
            raise CudaException('Mismatches should be integers. Passed value is {0}.'.format(mismatches))
        
        # calculate relative score, based on the shortest sequence
        if (self.sequence_info.original_length > self.target_info.original_length) :
            self.relative_score = float(self.score) / float(self.target_info.original_length)
        else:
            self.relative_score = float(self.score) / float(self.sequence_info.original_length)
        
        self.query_identity = float(matches) / float(self.seq_location[1] -  self.seq_location[0] + 1)
        self.query_coverage = float(self.seq_location[1]-  self.seq_location[0] + 1) / float(self.sequence_info.original_length)
        self.base_score = float(self.score) / float(len(self.alignment))

    def set_sequence_match(self, sequence_match):
        '''Sets the string that contains match information of the sequence part of the alignment
            for example:

            sequence_match: --ACTG
            alignment:               | | . |
            target_match:     CCACGG
        '''
        self.sequence_match = sequence_match

    def set_target_match(self, target_match):
        '''Sets the target of which the current alignment is a part
            for example:

            sequence_match: --ACTG
            alignment:               | | . |
            target_match:     CCACGG
        '''
        self.target_match = target_match

    def set_alignment(self, alignment):
        '''Sets the alignment information as a formatted string
            for example:

            sequence_match: --ACTG
            alignment:               | | . |
            target_match:     CCACGG
        '''
        self.alignment = alignment

    def keys(self):
        '''Returns a list of tuples that will be used as keys for this hit in a hitlist.'''
        if self.get_seq_id is not None and self.get_target_id is not None:
            return [(self.get_seq_id(), self.get_target_id(), self.seq_location[0], self.target_location[0]),
                    (self.get_seq_id(), self.get_target_id(), self.seq_location[1], self.target_location[1]),
                    (self.get_seq_id(), self.get_target_id(), self.seq_location[0], self.target_location[0])]
        else:
            return None

    def _check_rc(self):
        '''
        Returns True if the current hit is a hit on the reversed complement of a
        sequence originally read from a file.
        '''
        return(self.get_seq_id()[-2:] == "RC" and self.get_target_id()[-2:] == "RC")

    def get_seq_id(self):
        '''Returns the ID of the sequence that contains the current alignment.'''
        return self.sequence_info.id

    def get_target_id(self):
        '''Returns the ID of the target that contains the current alignment.'''
        return self.target_info.id

    def get_euclidian_distance(self):
        '''Returns the calculated Euclidian distance between sequence and target'''
        if self.target_info.distance is None:
            return 0.0
        else:
            return self.target_info.distance

    def get_sam_line(self):
        '''Creates and returns an alignment line as described in http://samtools.sourceforge.net/SAM1.pdf
           The following mappings are used for record lines:
                QNAME:  self.get_seq_id()
                FLAG:   only bit 0x2 and 0x10 are set since the other values are either false or unknown
                RNAME:  self.get_target_id()
                POS:    hit.target_location[0] + 1 (+1 since SAM is 1 based!)
                MAPQ:   set to 255 to indicate quality score is not available
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
        rc_seq = False
        rc_target = False

        
        identifier = self.get_seq_id()
        if identifier[-2:] == 'RC':
            identifier = identifier[:-3]
            rc_seq = True
        target_id = self.get_target_id()
        if target_id[-2:] == 'RC':
            target_id = target_id[:-3]
            rc_target = True
            
        #rc = (rc_seq and not rc_target) or (not rc_seq and rc_target)
        self.rc = rc_target or rc_seq 
        
        hit_pos = str(self.target_location[0] + 1)
        sam_sequence = self.sequence_info.seq
        if self.rc:
            self.alignment = self.alignment[::-1]
            self.sequence_match = self.sequence_match[::-1]
            sam_sequence =  Seq(str(self.sequence_info.seq), self.sequence_info.seq.alphabet).reverse_complement()
            hit_pos = str(self.target_info.original_length - self.target_location[1])
        if len(self.sequence_match) == 0 or len(self.target_match) == 0 or len(self.alignment) == 0:
            raise CudaException('sequence_match, target_match and alignment should be set and have lengths > 0.')

        return '\t'.join([identifier, self._get_sam_flag(), target_id,
                          hit_pos, Hit._get_sam_mapq(), self._get_sam_cigar(),
                          '*', str(0), str(0), str(sam_sequence), '*', self._get_sam_alignment_score(),
                          self._get_sam_euclidian_distance(),
                          self._get_sam_relative_score(),
                          self._get_sam_relative_base_score(),
                          self._get_sam_query_coverage(),
                          self._get_sam_query_identity()])
 
    def get_trimmed_line(self):
        identifier = self.get_seq_id()
        if identifier[-2:] == 'RC':
            identifier = identifier[:-3]

        target_id = self.get_target_id()
        if target_id[-2:] == 'RC':
            target_id = target_id[:-3]

        if len(self.sequence_match) == 0 or len(self.target_match) == 0 or len(self.alignment) == 0:
            raise CudaException('sequence_match, target_match and alignment should be set and have lengths > 0.')
        return ''.join(['>', target_id, ' ', self.sequence_info.id, "\n", 
                          str(self.sequence_info.seq[:self.target_location[0]] + self.sequence_info.seq[self.target_location[1]+1:])])



    def get_sam_sq(self):
        '''creates and returns the target (reference) information of the hit formatted
            as a SAM SQ-line. See http://samtools.sourceforge.net/SAM1.pdf
        '''
        return '\t'.join(['@SQ', 'SN:{0}'.format(self.get_target_id() if self.get_target_id()[-2:] != "RC" else self.get_target_id()[:-3]),
                          'LN:{0}'.format(self.target_info.original_length)])

    def _get_sam_alignment_score(self):
        '''Returns the (optional) field containing the alignment score of the hit as an int.
        '''
        return ':'.join(['AS', 'i', str(int(self.score))])

    def _get_sam_flag(self):
        '''Assemble the FLAG field of a hit as described in http://samtools.sourceforge.net/SAM1.pdf.
            If the hit involves the reversed complement of a sequence, bit 16 is set.
            The other bits are set to 0, either because their value is unknown or False.
        '''
        flag = 0
        if self.get_seq_id()[-2:] == 'RC' or self.get_target_id()[-2:] == 'RC':
            flag += 16
        return str(flag)

    @staticmethod
    def _get_sam_mapq():
        '''Determine the MAPQ field as described in http://samtools.sourceforge.net/SAM1.pdf.
            At the moment a fixed value of 255 is returned, indicating that mapping quality is not available.
        '''
        return str(255)

    def _get_sam_cigar(self):
        '''Determines the CIGAR of a hit as described in http://samtools.sourceforge.net/SAM1.pdf.
        '''
        cigar_parts = []

        # add clipping information to cigar_parts
        if not self.rc:
            if self.seq_location[0] > 0:
                cigar_parts.append(str(self.seq_location[0]) + 'S')
        else:
            cigar_parts.append(str(len(self.sequence_info.seq) - self.seq_location[1]-1) + 'S')
            
        current_char = self.alignment[0]
        charcount = 0
        i = 0
        for i in range(0, len(self.alignment)):
            if self.alignment[i] == current_char:
                charcount += 1
            else:
                cigar_parts.append(str(charcount) + self._get_cigar_char(i - 1))
                current_char = self.alignment[i]
                charcount = 1
        # add final matches to list
        cigar_parts.append(str(charcount) + self._get_cigar_char(i))
        # append soft clipping at the end
        number_of_remaining_bases = self.sequence_info.original_length - self.seq_location[1] - 1
        if self.rc:
            number_of_remaining_bases = self.seq_location[0]
        if number_of_remaining_bases > 0:
            cigar_parts.append(str(number_of_remaining_bases) + 'S')
        return ''.join(cigar_parts)

    def _get_sam_euclidian_distance(self):
        '''returns the computed distance as an optional sam field
        '''
        return "AD:f:" + str(self.get_euclidian_distance())

    def _get_sam_relative_score(self):
        '''returns the computed relative score as an optional sam field
        '''
        return "RS:f:" + str(self.relative_score)

    def _get_sam_query_coverage(self):
        '''returns the computed query coverage as an optional sam field
        '''
        return "QC:f:" + str(self.query_coverage)

    def _get_sam_query_identity(self):
        '''returns the computed query identity as an optional sam field
        '''
        return "QI:f:" + str(self.query_identity)

        
    def _get_sam_relative_base_score(self):
        '''returns the computed relative base score as an optional sam field
        '''
        return "BS:f:" + str(self.base_score)


    def _get_cigar_char(self, position):
        '''Determines the extended CIGAR character depending
            on the matching characters in the target and alignment strings.
        '''
        char = ''
        if self.alignment[position] == '|':
            char = '='
        elif self.alignment[position] == '.':
            char = 'X'
        else:  # self.alignment[position] == ' '
            if self.sequence_match[position] == '-':
                char = 'D'
            else:
                char = 'I'
        return char
