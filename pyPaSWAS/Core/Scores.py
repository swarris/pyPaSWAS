''' Scores module '''
from pyPaSWAS.Core.Exceptions import InvalidOptionException


class Score(object):
    '''
    This class implements the generic interface for scoring models.
    '''
    def __init__(self, logger, settings):
        self.logger = logger
        self.logger.debug('Initializing Score...')
        #set default values
        self.matrix = None
        self.score_type = 'Unknown'
        self.allowed_errors = 0
        self.dimensions = 0
        self.gap_score = 0
        self.lower_limit_max_score = 0
        self.lower_limit_score = 0
        self.highest_score = 0
        self.minimum_score = 0
        self.code_string = ''

        self._set_score_type()
        self.set_dimensions(26)
        self.set_gap_score(settings.gap_score)
        self.set_lower_limit_score(settings.lower_limit_score)
        self.set_lower_limit_max_score(settings.lower_limit_score)
        self.set_minimum_score(settings.minimum_score)
        self.code_string = None
        self.logger.debug('Initializing score finished.')

    def _create_matrix(self):
        '''
        This method creates the scoring matrix used in the calculation of alignment scores.
        This should be implemented in classes that inherit from Score.
        '''
        pass

    def set_dimensions(self, dimension):
        '''
        Sets the dimensions for the scoring matrix.
        Due to current implementation of the CUDA code, this value must always be 26
        '''
        self.dimensions = dimension

    def set_gap_score(self, gap_score):
        '''
        :param gap_score:
        '''
        self.gap_score = float(gap_score) if gap_score else None

    def set_lower_limit_max_score(self, lower_limit_max_score):
        '''
        :param lower_limit_max_score:
        '''
        self.lower_limit_max_score = float(lower_limit_max_score)

    def set_lower_limit_score(self, lower_limit_score):
        '''
        Sets the value of the lower limit of the score.
        Partial alignments are considered to be HSP if their alignment score is equal to or greater than this value
        '''
        llscore = float(lower_limit_score)
        if llscore < 1.0:
            self.logger.warning('Lower limit score is set < 1.0 ({}), this can result in too many hits to properly handle.'.format(llscore))
        self.lower_limit_score = float(llscore)

    def set_minimum_score(self, min_score):
        ''' Sets the minimal score that initiates a back trace. '''
        self.minimum_score = float(min_score)

    def _set_score_type(self):
        '''
        Sets the string that identifies the scoring matrix used.
        This should be implemented in classes that inherit from Score.
        '''
        pass

    def __str__(self):
        ''' returns the matrix as a string WITHOUT gap_score! '''
        self.logger.debug('Converting score to string...')
        if self.code_string == None:
            self.code_string = '{'
            for row in range(0, self.dimensions):
                self.code_string += '{'
                for col in range(0, self.dimensions):
                    self.code_string += str(self.matrix[row][col]) + (',' if col < self.dimensions - 1 else '')
                self.code_string += '}' + (',' if row < self.dimensions - 1 else '')
            self.code_string += '}'
        return self.code_string


class BasicScore(Score):
    '''
    Basic Scoring class
    '''

    def __init__(self, logger, settings):
        '''
        Constructor
        In:
            match_score:        score for matching nucleotides in subject and query
            mismatch_score:     score for different nucleotides in subject and query
            gap_score:          score for a missing nucleotide in either subject or query
        '''
        Score.__init__(self, logger, settings)
        self.logger.debug('Initializing BasicScore...')

        self.match_score = settings.match_score
        self.mismatch_score = settings.mismatch_score
        self._create_matrix()
        self.logger.debug('Initializing BasicScore finished.')

    def _set_score_type(self):
        self.score_type = 'BASIC'

    def _create_matrix(self):
        self.logger.debug('Creating matrix with parameters:\n\t'
                          'match_score: {0},\n\t\tmismatch_score: {1},\n\t\tgap_score: {2}'.format(self.match_score,
                                                                                                 self.mismatch_score,
                                                                                                 self.gap_score))
        # Is this just the matrix initialization?
        self.matrix = [[self.mismatch_score for _i in range(self.dimensions)] for _j in range(self.dimensions)]
        self.highest_score = self.match_score

        for row in range(0, self.dimensions):
            for col in range(0, self.dimensions):
                if row == col:
                    self.matrix[row][col] = self.match_score


class Blosum80Score(Score):
    '''
    Blosum80 Scoring class
    '''

    def __init__(self, logger, settings):
        '''
        :param logger:
        :param settings:
        '''
        #raise Exception("Blosum80 matrix not yet correctly implemented")
        Score.__init__(self, logger, settings)
        self.logger.debug('Initializing Blosum80Score...')
        self._create_matrix()
        self.logger.debug('Initializing Blosum80Score finished.')

    def _set_score_type(self):
        self.score_type = 'BLOSUM80'

    def _create_matrix(self):
        self.logger.debug('creating matrix with parameters: gap_score: {0}'.format(self.gap_score))
        self.matrix = [[7,-3,-1,-3,-2,-4,0,-3,-3,-8,-1,-3,-2,-3,-8,-1,-2,-3,2,0,-8,-1,-5,-1,-4,-2],
                        [-3,6,-6,6,1,-6,-2,-1,-6,-8,-1,-7,-5,5,-8,-4,-1,-2,0,-1,-8,-6,-8,-3,-5,0],
                        [-1,-6,13,-7,-7,-4,-6,-7,-2,-8,-6,-3,-3,-5,-8,-6,-5,-6,-2,-2,-8,-2,-5,-4,-5,-7],
                        [-3,6,-7,10,2,-6,-3,-2,-7,-8,-2,-7,-6,2,-8,-3,-1,-3,-1,-2,-8,-6,-8,-3,-6,1],
                        [-2,1,-7,2,8,-6,-4,0,-6,-8,1,-6,-4,-1,-8,-2,3,-1,-1,-2,-8,-4,-6,-2,-5,6],
                        [-4,-6,-4,-6,-6,10,-6,-2,-1,-8,-5,0,0,-6,-8,-6,-5,-5,-4,-4,-8,-2,0,-3,4,-6],
                        [0,-2,-6,-3,-4,-6,9,-4,-7,-8,-3,-7,-5,-1,-8,-5,-4,-4,-1,-3,-8,-6,-6,-3,-6,-4],
                        [-3,-1,-7,-2,0,-2,-4,12,-6,-8,-1,-5,-4,1,-8,-4,1,0,-2,-3,-8,-5,-4,-2,3,0],
                        [-3,-6,-2,-7,-6,-1,-7,-6,7,-8,-5,2,2,-6,-8,-5,-5,-5,-4,-2,-8,4,-5,-2,-3,-6],
                        [-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8],
                        [-1,-1,-6,-2,1,-5,-3,-1,-5,-8,8,-4,-3,0,-8,-2,2,3,-1,-1,-8,-4,-6,-2,-4,1],
                        [-3,-7,-3,-7,-6,0,-7,-5,2,-8,-4,6,3,-6,-8,-5,-4,-4,-4,-3,-8,1,-4,-2,-2,-5],
                        [-2,-5,-3,-6,-4,0,-5,-4,2,-8,-3,3,9,-4,-8,-4,-1,-3,-3,-1,-8,1,-3,-2,-3,-3],
                        [-3,5,-5,2,-1,-6,-1,1,-6,-8,0,-6,-4,9,-8,-4,0,-1,1,0,-8,-5,-7,-2,-4,-1],
                        [-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8],
                        [-1,-4,-6,-3,-2,-6,-5,-4,-5,-8,-2,-5,-4,-4,-8,12,-3,-3,-2,-3,-8,-4,-7,-3,-6,-2],
                        [-2,-1,-5,-1,3,-5,-4,1,-5,-8,2,-4,-1,0,-8,-3,9,1,-1,-1,-8,-4,-4,-2,-3,5],
                        [-3,-2,-6,-3,-1,-5,-4,0,-5,-8,3,-4,-3,-1,-8,-3,1,9,-2,-2,-8,-4,-5,-2,-4,0],
                        [2,0,-2,-1,-1,-4,-1,-2,-4,-8,-1,-4,-3,1,-8,-2,-1,-2,7,2,-8,-3,-6,-1,-3,-1],
                        [0,-1,-2,-2,-2,-4,-3,-3,-2,-8,-1,-3,-1,0,-8,-3,-1,-2,2,8,-8,0,-5,-1,-3,-2],
                        [-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8],
                        [-1,-6,-2,-6,-4,-2,-6,-5,4,-8,-4,1,1,-5,-8,-4,-4,-4,-3,0,-8,7,-5,-2,-3,-4],
                        [-5,-8,-5,-8,-6,0,-6,-4,-5,-8,-6,-4,-3,-7,-8,-7,-4,-5,-6,-5,-8,-5,16,-5,3,-5],
                        [-1,-3,-4,-3,-2,-3,-3,-2,-2,-8,-2,-2,-2,-2,-8,-3,-2,-2,-1,-1,-8,-2,-5,-2,-3,-1],
                        [-4,-5,-5,-6,-5,4,-6,3,-3,-8,-4,-2,-3,-4,-8,-6,-3,-4,-3,-3,-8,-3,3,-3,11,-4],
                        [-2,0,-7,1,6,-6,-4,0,-6,-8,1,-5,-3,-1,-8,-2,5,0,-1,-2,-8,-4,-5,-1,-4,6]]

        self.highest_score = max(max(self.matrix))

class Blosum62Score(Score):
    '''
    Blosum62 Scoring class
    '''

    def __init__(self, logger, settings):
        '''
        :param logger:
        :param settings:
        '''
        #raise Exception("Blosum80 matrix not yet correctly implemented")
        Score.__init__(self, logger, settings)
        self.logger.debug('Initializing Blosum62Score...')
        self._create_matrix()
        self.logger.debug('Initializing Blosum62Score finished.')

    def _set_score_type(self):
        self.score_type = 'BLOSUM62'

    def _create_matrix(self):
        self.logger.debug('creating matrix with parameters: gap_score: {0}'.format(self.gap_score))
        self.matrix = [[4,-2,0,-2,-1,-2,0,-2,-1,-1,-1,-1,-1,-2,-4,-1,-1,-1,1,0,-4,0,-3,-1,-2,-1],
                        [-2,4,-3,4,1,-3,-1,0,-3,-3,0,-4,-3,4,-4,-2,0,-1,0,-1,-4,-3,-4,-1,-3,0],
                        [0,-3,9,-3,-4,-2,-3,-3,-1,-1,-3,-1,-1,-3,-4,-3,-3,-3,-1,-1,-4,-1,-2,-1,-2,-3],
                        [-2,4,-3,6,2,-3,-1,-1,-3,-3,-1,-4,-3,1,-4,-1,0,-2,0,-1,-4,-3,-4,-1,-3,1],
                        [-1,1,-4,2,5,-3,-2,0,-3,-3,1,-3,-2,0,-4,-1,2,0,0,-1,-4,-2,-3,-1,-2,4],
                        [-2,-3,-2,-3,-3,6,-3,-1,0,0,-3,0,0,-3,-4,-4,-3,-3,-2,-2,-4,-1,1,-1,3,-3],
                        [0,-1,-3,-1,-2,-3,6,-2,-4,-4,-2,-4,-3,0,-4,-2,-2,-2,0,-2,-4,-3,-2,-1,-3,-2],
                        [-2,0,-3,-1,0,-1,-2,8,-3,-3,-1,-3,-2,1,-4,-2,0,0,-1,-2,-4,-3,-2,-1,2,0],
                        [-1,-3,-1,-3,-3,0,-4,-3,4,3,-3,2,1,-3,-4,-3,-3,-3,-2,-1,-4,3,-3,-1,-1,-3],
                        [-1,-3,-1,-3,-3,0,-4,-3,3,3,-3,3,2,-3,-4,-3,-2,-2,-2,-1,-4,2,-2,-1,-1,-3],
                        [-1,0,-3,-1,1,-3,-2,-1,-3,-3,5,-2,-1,0,-4,-1,1,2,0,-1,-4,-2,-3,-1,-2,1],
                        [-1,-4,-1,-4,-3,0,-4,-3,2,3,-2,4,2,-3,-4,-3,-2,-2,-2,-1,-4,1,-2,-1,-1,-3],
                        [-1,-3,-1,-3,-2,0,-3,-2,1,2,-1,2,5,-2,-4,-2,0,-1,-1,-1,-4,1,-1,-1,-1,-1],
                        [-2,4,-3,1,0,-3,0,1,-3,-3,0,-3,-2,6,-4,-2,0,0,1,0,-4,-3,-4,-1,-2,0],
                        [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4],
                        [-1,-2,-3,-1,-1,-4,-2,-2,-3,-3,-1,-3,-2,-2,-4,7,-1,-2,-1,-1,-4,-2,-4,-1,-3,-1],
                        [-1,0,-3,0,2,-3,-2,0,-3,-2,1,-2,0,0,-4,-1,5,1,0,-1,-4,-2,-2,-1,-1,4],
                        [-1,-1,-3,-2,0,-3,-2,0,-3,-2,2,-2,-1,0,-4,-2,1,5,-1,-1,-4,-3,-3,-1,-2,0],
                        [1,0,-1,0,0,-2,0,-1,-2,-2,0,-2,-1,1,-4,-1,0,-1,4,1,-4,-2,-3,-1,-2,0],
                        [0,-1,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,-1,0,-4,-1,-1,-1,1,5,-4,0,-2,-1,-2,-1],
                        [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4],
                        [0,-3,-1,-3,-2,-1,-3,-3,3,2,-2,1,1,-3,-4,-2,-2,-3,-2,0,-4,4,-3,-1,-1,-2],
                        [-3,-4,-2,-4,-3,1,-2,-2,-3,-2,-3,-2,-1,-4,-4,-4,-2,-3,-3,-2,-4,-3,11,-1,2,-2],
                        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-4,-1,-1,-1,-1,-1,-4,-1,-1,-1,-1,-1],
                        [-2,-3,-2,-3,-2,3,-3,2,-1,-1,-2,-1,-1,-2,-4,-3,-1,-2,-2,-2,-4,-1,2,-1,7,-2],
                        [-1,0,-3,1,4,-3,-2,0,-3,-3,1,-3,-1,0,-4,-1,4,0,0,-1,-4,-2,-2,-1,-2,4]]
        
        self.highest_score = max(max(self.matrix))



class CustomScore(Score):
    ''' This class allows the user to use a custom matrix in pyPaSWAS. '''
    def __init__(self, logger, settings):
        '''
        Constructor
            Parameters:
            matrix:     the custom scoring matrix. Due to the current implementation of pyPaSWAS,
                        this should be a Python list consisting of 26 lists. Each of these inner
                        lists should contain 26 integers. Each integer represents the score for
                        matching nucleotides or amino acids represented by a certain character
                        in subject and query. E.g. the integer at matrix[0][0] contains the score
                        when both the subject and the query contain the character 'A' at the
                        current position, matrix[0][1] when at the current position an 'A' is
                        aligned with a 'B' etc.
            gap_score:   score for a missing nucleotide or amino acid in either subject or query
        '''
        Score.__init__(self, logger, settings)
        self.logger.debug('Initializing CustomScore...')
        #check/set defaults for arguments
        matrix = settings.custom_matrix
        if matrix is None:
            raise InvalidOptionException('Required argument {0} is missing'.format('matrix'))
        self._create_matrix(matrix)
        self.logger.debug('Initializing CustomScore finished.')

    def _set_score_type(self):
        self.score_type = 'CUSTOM'

    def _create_matrix(self, matrix):
        self.logger.debug('Creating matrix with parameters:\n\t\tmatrix: '
                          '{0},\n\t\tgap_score: {1}'.format(matrix, self.gap_score))
        if len(matrix) != self.dimensions or (len(matrix) > 0 and len(matrix[0]) != self.dimensions):
            raise InvalidOptionException('A matrix should consist of a list containing 26 lists of '
                                         '26 integers each.')
        else:
            self.matrix = matrix
            self.highest_score = max(max(self.matrix))


class DnaRnaScore(Score):
    '''
    DNA / RNA Scorings class
    '''
    def __init__(self, logger, settings):
        '''
        Constructor
        In settings object (from optparse module) with:
            match_score:        score for matching nucleotides in subject and query
            mismatch_score:     score for different nucleotides in subject and query
            gap_score:          score for a missing nucleotide in either subject or query
            other_score:        score for a character which is neither in the nucleotide list ('ACGTU'), nor equal to the anyNucleotide character ('N')
            any_score:          score if the anyNucleotide character ('N') is present in either query or subject
        '''
        Score.__init__(self, logger, settings)
        self.logger.debug('Initializing DnaRnaScore...')

        self.match_score = settings.match_score
        self.mismatch_score = settings.mismatch_score
        self.other_score = settings.other_score
        self.any_score = settings.any_score
        self._create_matrix()
        self.logger.debug('Initializing DnaRnaScore finished.')

    def _set_score_type(self):
        self.score_type = 'DNA_RNA'

    def _create_matrix(self):
        self.logger.debug('Creating matrix with parameters:\n\t\tmatch_score: {0},\n\t\tmismatch_score: {1},\n\t\t'
                          'gap_score: {2},\n\t\tother_score: {3},\n\t\tany_score: {4}'.format(self.match_score,
                                                                                              self.mismatch_score,
                                                                                              self.gap_score,
                                                                                              self.other_score,
                                                                                              self.any_score))
        dna_rna_ord = [ord('A') - ord('A'), ord('T') - ord('A'), ord('C') - ord('A'),
                       ord('G') - ord('A'), ord('U') - ord('A')]
        any_ord = ord('N') - ord('A')
        self.matrix = [[self.other_score for _i in range(self.dimensions)] for _j in range(self.dimensions)]
        self.highest_score = self.match_score

        for row in range(0, self.dimensions):
            for col in range(0, self.dimensions):
                if row == col:
                    self.matrix[row][col] = self.match_score
                elif row == any_ord or col == any_ord:
                    self.matrix[row][col] = self.any_score
                elif row in dna_rna_ord or col in dna_rna_ord:
                    self.matrix[row][col] = self.mismatch_score


class IrysScore(Score):
    '''
    DNA / RNA Scorings class
    '''
    def __init__(self, logger, settings):
        '''
        Constructor
        In settings object (from optparse module) with:
            match_score:        score for matching nucleotides in subject and query
            mismatch_score:     score for different nucleotides in subject and query
            gap_score:          score for a missing nucleotide in either subject or query
            other_score:        score for a character which is neither in the nucleotide list ('ACGTU'), nor equal to the anyNucleotide character ('N')
            any_score:          score if the anyNucleotide character ('N') is present in either query or subject
        '''
        Score.__init__(self, logger, settings)
        self.logger.debug('Initializing IrysScore...')

        self.match_score = settings.match_score
        self.mismatch_score = settings.mismatch_score
        self.other_score = settings.other_score
        self.any_score = settings.any_score
        self._create_matrix()
        self.logger.debug('Initializing IrysScore finished.')

    def _set_score_type(self):
        self.score_type = 'IRYS'

    def _create_matrix(self):

        self.matrix = [[self.other_score for _i in range(self.dimensions)] for _j in range(self.dimensions)]
        self.highest_score = self.match_score

        for row in range(0, self.dimensions):
            for col in range(0, self.dimensions):
                if row == col:
                    self.matrix[row][col] = self.match_score
                else:
                    self.matrix[row][col] = str(float(self.mismatch_score) * abs(row-col))
