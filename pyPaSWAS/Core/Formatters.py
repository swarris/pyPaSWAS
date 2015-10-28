'''This module contains the output formatters for pyPaSWAS'''


class DefaultFormatter(object):
    '''This is the default formatter for pyPasWas.
        All available formatters inherit from this formatter.
        The results are parsed into a temporary file, which can be used by the main
        program for permanent storage, printing etc.
    '''
    def __init__(self, logger, hitlist, outputfile):
        self.name = ''
        self.logger = logger
        self.logger.debug('Initializing formatter...')
        self.hitlist = hitlist
        self.outputfile = outputfile

        self._set_name()
        self.logger.debug('Initialized {0}'.format(self.name))

    def _format_hit(self, hit):
        '''This method may be overruled to enable other formats for printed results.'''
        self.logger.debug('Formatting hit {0}'.format(hit.get_seq_id()))
        formatted_hit = ', '.join([hit.get_seq_id(), hit.get_target_id(), str(hit.seq_location[0]), str(hit.seq_location[1]),
                                  str(hit.target_location[0]), str(hit.target_location[1]), str(hit.score), str(hit.matches),
                                  str(hit.mismatches), str(len(hit.alignment) - hit.matches - hit.mismatches),
                                  str(len(hit.alignment)), str(hit.score / len(hit.alignment)),
                                  str(hit.sequence_info.original_length), str(hit.target_info.original_length),
                                  str(hit.score / hit.sequence_info.original_length),
                                  str(hit.score / hit.target_info.original_length), str(hit.distance)])
        formatted_hit = '\n'.join([formatted_hit, hit.sequence_match, hit.alignment, hit.target_match])
        return formatted_hit

    def _set_name(self):
        '''Name of the formatter. Used for logging'''
        self.name = 'defaultformatter'

    def print_results(self):
        '''sets, formats and prints the results to a file.'''
        self.logger.debug('printing results...')
        output = open(self.outputfile, 'w')
        for hit in self.hitlist.real_hits.itervalues():
            formatted_hit = self._format_hit(hit)
            output.write(formatted_hit + "\n")
        self.logger.debug('finished printing results')


class SamFormatter(DefaultFormatter):
    '''This Formatter is used to create SAM output
        See http://samtools.sourceforge.net/SAM1.pdf
    '''

    def __init__(self, logger, hitlist, outputfile):
        '''Since the header contains information about the target sequences and must be
            present before alignment lines, formatted lines are stored before printing.
        '''
        DefaultFormatter.__init__(self, logger, hitlist, outputfile)
        self.sq_lines = {}
        self.record_lines = []

    def _set_name(self):
        '''Name of the formatter. Used for logging'''
        self.name = 'SAM formatter'

    def _format_hit(self, hit):
        '''Adds a header line to self.sq_lines and an alignment line to self.record_lines.
            The following mappings are used for header lines:
                SN: hit.get_target_id()
                LN: hit.full_target.original_length
        '''
        self.logger.debug('Formatting hit {0}'.format(hit.get_seq_id()))
        #add a header line for the target id if not already present
        if hit.get_target_id() not in self.sq_lines:
            if hit.get_target_id()[-2:] != 'RC':
                self.sq_lines[hit.get_target_id()] = hit.get_sam_sq()
            else:
                self.sq_lines[hit.get_target_id()[:-3]] = hit.get_sam_sq()
                
        #add a line for the hit
        self.record_lines.append(hit.get_sam_line())

    def print_results(self):
        '''sets, formats and prints the results to a file.'''
        self.logger.info('formatting results...')
        #format header and hit lines
        for hit in self.hitlist.real_hits.itervalues():
            self._format_hit(hit)

        self.logger.debug('printing results...')
        output = open(self.outputfile, 'w')

        #write the header lines to the file
        header_string = '@HD\tVN:1.4\tSO:unknown'
        output.write(header_string + '\n')
        for header_line in self.sq_lines:
            output.write(self.sq_lines[header_line] + '\n')
        #program information header line
        output.write('@PG\tID:0\tPN:paswas\tVN:3.0\n')
        #write the hit lines to the output file
        for line in self.record_lines:
            output.write(line + '\n')
        output.close()
        self.logger.debug('finished printing results')

class TrimmerFormatter(DefaultFormatter):
    '''This Formatter is used to create SAM output
        See http://samtools.sourceforge.net/SAM1.pdf
    '''

    def __init__(self, logger, hitlist, outputfile):
        '''Since the header contains information about the target sequences and must be
            present before alignment lines, formatted lines are stored before printing.
        '''
        DefaultFormatter.__init__(self, logger, hitlist, outputfile)
        self.sq_lines = {}
        self.record_lines = []

    def _set_name(self):
        '''Name of the formatter. Used for logging'''
        self.name = 'SAM formatter'

    def _format_hit(self, hit):
        '''Adds a header line to self.sq_lines and an alignment line to self.record_lines.
            The following mappings are used for header lines:
                SN: hit.get_target_id()
                LN: hit.full_target.original_length
        '''
        self.logger.debug('Formatting hit {0}'.format(hit.get_seq_id()))
        self.record_lines.append(hit.get_trimmed_line())

    def print_results(self):
        '''sets, formats and prints the results to a file.'''
        self.logger.info('formatting results...')
        #format header and hit lines
        for hit in self.hitlist.real_hits.itervalues():
            self._format_hit(hit)

        self.logger.debug('printing results...')
        output = open(self.outputfile, 'w')

        #write the hit lines to the output file
        for line in self.record_lines:
            output.write(line + '\n')
        output.close()
        self.logger.debug('finished printing results')

