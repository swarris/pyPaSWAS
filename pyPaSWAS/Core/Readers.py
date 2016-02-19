'''This module contains the different readers used to parse both the
target and the query records from files.'''
import os.path

from itertools import islice
from Bio import  SeqIO
from Bio.Seq import Seq
from pyPaSWAS.Core.SWSeqRecord import SWSeqRecord
from pyPaSWAS.Core.Exceptions import InvalidOptionException
from Exceptions import ReaderException

class Reader(object):
    '''The generic reader from which other readers inherit some common functionalities.
    Each reader should implement '''
    def __init__(self, logger, path, filetype, limitlength=5000):
        '''path: the absolute path to the file that contains the records
           limitlength: the maximimum length of the records that will be used for the alignment '''
        self.logger = logger
        self.logger.debug('Initializing reader\n\tpath = {0}\n\tlimitlength = {1}...'.format(path, limitlength))
        self.rc_string = "_RC"
        self.path = ''
        self.filetype = filetype
        self.records = None
        self.limitlength = limitlength

        self._set_path(path)
        self._set_limit_length(limitlength)
        self.logger.debug('Initializing reader finished.')

    def sort_records(self, reverse=True):
        '''sorts the records'''
        self.logger.debug('Sorting records on length...')
        self.records.sort(key=lambda seqIO: len(seqIO.seq), reverse=reverse)

    @staticmethod
    def _is_a_readable_file(path):
        '''Checks whether or not a file is writeable.'''
        if os.path.isfile(path) and os.access(path, os.R_OK):
            return True
        else:
            return False

    def _set_path(self, path):
        '''Sets the path to the file which contains the records.'''
        if self._is_a_readable_file(path):
            self.path = path
        else:
            raise InvalidOptionException('{0} is not a file or the program is not allowed to access it.'.format(path))

    def _set_filetype(self):
        '''This method sets the type of the file that will be parsed.
            Should be overridden in each reader class.
        '''
        self.filetype = 'Unknown'

    def _set_limit_length(self, limit_length):
        '''sets the limit of the number of sequences that are to be compared at one time)'''
        try:
            self.limitlength = int(limit_length)
        except ValueError:
            raise InvalidOptionException('Limitlength should be an int but is {0}'.format(limit_length))

    def get_records(self):
        '''Getter for the parsed records'''
        #self.logger.debug('Returning records...')
        return self.records

    def complement_records(self):
        '''Appends the reverse complements to the parsed records '''
        #self.logger.debug('Creating complement sequences...')
        seqIO = lambda seqIO: SWSeqRecord(Seq(str(seqIO.seq.reverse_complement()), seqIO.seq.alphabet),
                                          identifier=(str(seqIO.id) + self.rc_string))
        self.records.extend([seqIO(record) for record in self.records])

    def complement_records_only(self):
        '''Creates the reverse complements to the parsed records '''
        seqIO = lambda seqIO: SWSeqRecord(Seq(str(seqIO.seq.reverse_complement()), seqIO.seq.alphabet),
                                          identifier=(str(seqIO.id) + self.rc_string))
        self.records = [seqIO(record) for record in self.records]


class BioPythonReader(Reader):
    '''This class parses input files'''

    def read_records(self, start=0, end=None):
        '''Parses the records from a supported input file'''
        self.logger.debug('Reading from {} file...'.format(self.filetype))
        file_elements = open(self.path, "rU")
        self.records = list(islice(SeqIO.parse(file_elements, self.filetype), start, end))
        file_elements.close()
        if len(self.records) == 0:
            raise ReaderException('No (more) sequence data found in input file ({}), of file type {}.'.format(self.path, self.filetype))

        if self.limitlength > 0:
            nrecords = len(self.records)
            #self.logger.debug('Checking sequences length..')
            self.records = [SWSeqRecord(Seq(str(record.seq), record.seq.alphabet),
                                        identifier=record.id) for record in self.records
                            if len(record.seq) <= self.limitlength and len(record.seq) > 0]
            diff = nrecords - len(self.records)
            if diff > 0:
                self.logger.info('{} sequence(s) removed with length > limit_length ({}bp)'.format(diff, self.limitlength))
            if end==None and len(self.records) == 0:
                self.logger.info('No sequences remaining after filtering on length for {}.'
                                           ' Please adjust using the limit_length parameter.'.format(self.path))
        else:
            self.records = [SWSeqRecord(Seq(str(record.seq), record.seq.alphabet),
                                        identifier=record.id) for record in self.records]

        self.logger.debug('\t{} sequences read.'.format(len(self.records)))
