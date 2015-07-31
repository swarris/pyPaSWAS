'''TODO: Docstring'''


class HitList(object):
    ''' This class contains a list of all valid alignments found by the Smith-Waterman algorithm.
        If back tracing yields multiple hits starting at the same nucleotide, only the longest is kept in
        real_hits. The other (shorter) hits are rejected.
    '''

    def __init__(self, logger):
        '''Constructor'''
        self.logger = logger
        self.logger.debug('Initializing hitlist...')
        self.hits = {}
        self.real_hits = {}
        self.logger.debug('Initializing hitlist OK.')

    def extend(self, hitList):
        self.hits.update(hitList.hits)
        self.real_hits.update(hitList.real_hits)

    def append(self, hit):
        '''
        Appends a hit to the hit list
        :param hit: hit to add
        '''
        keys = hit.keys()
        if keys is not None:
            keys0 = keys[0] in self.hits and self.hits[keys[0]].score < hit.score
            keys1 = keys[1] in self.hits and self.hits[keys[1]].score < hit.score
            neither = keys[0] not in self.hits and keys[1] not in self.hits
            
            if keys0 or keys1 or neither:
                if keys1 and self.hits[keys[1]].keys()[2] in self.real_hits:
                    # hit found with same end location, so delete previous hit based on its start location
                    del self.real_hits[self.hits[keys[1]].keys()[2]]
                self.hits[keys[0]] = hit
                self.hits[keys[1]] = hit
                self.real_hits[keys[2]] = hit
                self.logger.info('Added hit (query ID: {}, target ID: {}) to hitlist'.format(hit.get_seq_id(),
                                                                                   hit.get_target_id()))
                                           
            else:
                self.logger.debug('Rejected hit (query ID: {}, target ID: {}) from'
                                  ' addition to hitlist'.format(hit.get_seq_id(),
                                                                hit.get_target_id()))
