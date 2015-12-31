''' This module contains the programs from the pyPaSWAS suite '''
from pyPaSWAS.Core.HitList import HitList
from operator import itemgetter
from SWSeqRecord import SWSeqRecord
from Bio.Seq import Seq
from Hit import Distance

class Aligner(object):
    '''
    Common functionality shared by all aligning programs from the pyPaSWAS suite is encapsulated in this class.
    '''
    # TODO: expand - and update - docstrings!!

    def __init__(self, logger, score, settings):
        '''
        Constructor.

        @param logger: an instance of logging.logger
        @param score: a score object. None by default
        @param settings: an instance of a settings object as returned by optparse. These settings should include:
            filterFactor: Defaults to 0.7
            device: Defaults to 0 i.e. the first card in the array
            #limitLength. Defaults to 5000. Sets a limit to the maximal length of items to be be compared at one time
            maxGenomeLength. Sets the maximum length of (the part of) the sequence that will be processed at one time.
                Should the length of the sequence exceed this setting, the sequence will be divided and the resulting parts
                will be processed separately. Defaults to 200000.
        @raise InvalidOptionException: If either verifySettings of verifyArguments fail.
        '''
        logger.debug('Initializing aligner...')
        self.logger = logger
        self.score = score
        self.hitlist = HitList(self.logger)
        logger.debug('Setting SW...')
        self.settings = settings
        self.qindexerCUDA = False
        self.qindexerOCL = False
        if (self.settings.framework.upper() == 'OPENCL'):
            self.qindexerOCL = True
            if(self.settings.device_type.upper() == 'GPU'):
                if(self.settings.platform_name.upper() == 'NVIDIA'):
                    self.logger.debug('Using OpenCL NVIDIA implementation')
                    from pyPaSWAS.Core.SmithWatermanOcl import SmithWatermanNVIDIA
                    self.smith_waterman = SmithWatermanNVIDIA(self.logger, self.score, settings)
                else:
                    self.logger.debug('Using OpenCL GPU implementation')
                    from pyPaSWAS.Core.SmithWatermanOcl import SmithWatermanGPU
                    self.smith_waterman = SmithWatermanGPU(self.logger, self.score, settings)
            elif(self.settings.device_type.upper() == 'CPU'):
                self.logger.debug('Using OpenCL CPU implementation')
                from pyPaSWAS.Core.SmithWatermanOcl import SmithWatermanCPU
                self.smith_waterman = SmithWatermanCPU(self.logger, self.score, settings)
            elif(self.settings.device_type.upper() == 'ACCELERATOR'):
                self.logger.debug('Using OpenCL Accelerator implementation')
                from pyPaSWAS.Core.SmithWatermanOcl import SmithWatermanGPU
                self.smith_waterman = SmithWatermanGPU(self.logger, self.score, settings)
            else:
                self.logger.debug('Unknown settings for device. Using default OpenCL GPU implementation')
                from pyPaSWAS.Core.SmithWatermanOcl import SmithWatermanGPU
                self.smith_waterman = SmithWatermanGPU(self.logger, self.score, settings)
        elif self.settings.framework.upper() == 'CUDA':
            self.logger.debug('Using CUDA implementation')
            from pyPaSWAS.Core.SmithWatermanCuda import SmithWatermanCuda
            self.smith_waterman = SmithWatermanCuda(self.logger, self.score, settings)
            self.qindexerCUDA = True
        else:
            self.logger.info('Unknown settings for framework. Using OpenCL GPU implementation as default')
            from pyPaSWAS.Core.SmithWatermanOcl import SmithWatermanGPU
            self.smith_waterman = SmithWatermanGPU(self.logger, self.score, settings)
            
        self.logger.debug('Aligner initialized.')

    def process(self, records_seqs, targets, pypaswas):
        '''This methods sends the target- and query sequences to the SmithWaterman instance
        and receives the resulting hitlist.
        '''
        # step through the targets
        self.logger.debug('Aligner processing...')
        target_index = 0

        while target_index < len(targets):
            self.logger.debug('At target: {0} of {1}'.format(target_index, len(targets)))


            last_target_index = self.smith_waterman.set_targets(targets, target_index)
            # results should be a Hitlist()
            results = self.smith_waterman.align_sequences(records_seqs, targets, target_index)
            self.hitlist.extend(results)
            target_index = last_target_index
        self.logger.debug('Aligner processing OK, returning hitlist ({} + {}).'.format(len(self.hitlist.real_hits), len(results.real_hits)))
        return self.hitlist
    
        

class Trimmer(Aligner):
    
    def __init__(self, logger, score, settings):
        Aligner.__init__(self, logger, score, settings)

    def process(self, records_seqs, targets, pypaswas):
        '''This methods sends the target- and query sequences to the SmithWaterman instance
        and receives the resulting hitlist.
        '''
        # step through the targets
        self.logger.debug('Aligner processing...')
        target_index = 0
        if len(targets) > 0 and self.settings.recompile == "F":
            max_length = len(targets[0])
        else:
            max_length = None
            
        while target_index < len(targets):
            self.logger.debug('At target: {0} of {1}'.format(target_index, len(targets)))

            last_target_index = self.smith_waterman.set_targets(targets, target_index, max_length, records_seqs)
            # results should be a Hitlist()
            results = self.smith_waterman.align_sequences(records_seqs, targets, target_index)
            self.hitlist.extend(results)
            target_index = last_target_index
        self.logger.debug('Aligner processing OK, returning hitlist.')
        return self.hitlist

class ComBaRMapper(Aligner):
    
    def __init__(self, logger, score, settings, arguments):
        Aligner.__init__(self, logger, score, settings)
        self.arguments = arguments

    def process(self, records_seqs, targets, pypaswas):
        '''This methods creates index files for targets based on the length of the records.
        '''

        
        # step through the targets                                                                                                                                                                           
        self.logger.debug('ComBaR mapping...')

        indexer = None
        while len(records_seqs) > 0:
            prevLength = len(records_seqs[0])
            #create indexer
            
            if self.qindexerOCL:
                from pyPaSWAS.Core.QIndexerOCL import QIndexerOCL
                indexer = QIndexerOCL(self.settings, self.logger, 0.1, records_seqs[0:1], int(self.settings.qgram))
            elif self.qindexerCUDA:
                from pyPaSWAS.Core.QIndexerCUDA import QIndexerCUDA
                indexer = QIndexerCUDA(self.settings, self.logger, 0.1, records_seqs[0:1], int(self.settings.qgram))                
            else:
                from pyPaSWAS.Core.QIndexer import QIndexer
                indexer = QIndexer(self.settings, self.logger, 0.1, records_seqs[0:1], int(self.settings.qgram))
            """
            indexer = QIndexer(self.settings, self.logger, 0.1, records_seqs[0:1], int(self.settings.qgram))
            """
            #while indices to process, process all reads with same length
            # when done, remove these reads
            while indexer.indicesToProcessLeft():
                currentRead = 0
                # only create index with first read of same length 
                if currentRead == 0:
                    indexer.createIndexAndStore(targets, self.arguments[1])
                 
                while currentRead < len(records_seqs) and len(records_seqs[currentRead]) == prevLength:
                    currentBlockOfReads = currentRead
                    
                    while currentRead - currentBlockOfReads < indexer.readsToProcess and currentRead < len(records_seqs) and len(records_seqs[currentRead]) == prevLength:
                        currentRead += 1


                    allLocations = indexer.findIndices(records_seqs[currentBlockOfReads:currentRead])

                    if len(allLocations) > 0 :
                        for read in xrange(currentBlockOfReads, currentRead):
                            
                            firstRead = records_seqs[read]
                            filteredRecordsSeqs = [firstRead]
            
                            locations = allLocations[read-currentBlockOfReads]
                            locs = []
                            if (len(locations) > 0):
                                self.logger.info("Processing seq: " + records_seqs[read].id)
                                for value in locations.itervalues():
                                    locs.extend(value)

                                if len(locs) > float(self.settings.fraction_of_seeds) * indexer.indicesStepSize:
                                    self.logger.warning("Too many locations for fast processing. Sorting and slicing now with fraction_of_seeds = {}.".format(self.settings.fraction_of_seeds))
                                    locs.sort(key = lambda x:x[2]) # sort on distance
                                    locs = locs[:int(float(self.settings.fraction_of_seeds)*len(locs))]

                                splittedTargets = []
        
                                for loc in locs:
                                    swSeqRecord = indexer.getSWSeqRecord(loc, targets)
                                    #if len(swSeqRecord) > 0:
                                    swSeqRecord.distance = loc[2]
                                    swSeqRecord.id = targets[loc[0][1]].id
                                    swSeqRecord.refID = loc[0][1]
                                    splittedTargets.append(swSeqRecord)
        
                                if (len(splittedTargets) > 0 and len(filteredRecordsSeqs) > 0):
                                    splittedTargets.sort(key=lambda seqIO : len(seqIO.seq), reverse=True)
                                    target_index = 0
                                    # process of the seeds:                                                                                                                                                          
                                    while target_index < len(splittedTargets):
                                        last_target_index = self.smith_waterman.set_targets(splittedTargets, target_index, None, filteredRecordsSeqs)
                                        self.logger.debug('At target: {0} of {1}, processing up to {2}'.format(target_index, len(splittedTargets), str(last_target_index)))
                                        results = self.smith_waterman.align_sequences(filteredRecordsSeqs, splittedTargets, target_index)
                                        self.hitlist.extend(results)
                                        target_index = last_target_index
            
            #filter out reads already processed:
            currentRead = 0
            while currentRead < len(records_seqs) and len(records_seqs[currentRead]) == prevLength:
                currentRead += 1
            records_seqs = records_seqs[currentRead:]
        
        if indexer != None and self.qindexerCUDA:
            indexer.pop_context()                             
        self.logger.debug('ComBaR mapping finished.')
        return self.hitlist

class ComBaRIndexer(Aligner):
    
    def __init__(self, logger, score, settings, arguments):
        logger.debug('Initializing indexer...')
        self.logger = logger
        self.score = score
        self.hitlist = HitList(self.logger)
        self.settings = settings
        self.arguments = arguments

    def process(self, records_seqs, targets, pypaswas):
        '''This methods creates index files for targets based on the length of the records.
        '''
        from pyPaSWAS.Core.QIndexer import QIndexer
        
        # step through the targets                                                                                                                                                                           
        self.logger.debug('ComBaR indexer...')
        from pyPaSWAS.Core.QIndexer import QIndexer

        keepRecords = []
        
        while len(records_seqs) > 0:
            prevLength = len(records_seqs[0])
            #create indexer
            indexer = QIndexer(self.settings, self.logger, 0.1, records_seqs[0:1], int(self.settings.qgram))

            #while indices to process, process all reads with same length
            # when done, remove these reads
            while indexer.indicesToProcessLeft():
                currentRead = 0
                # only create index with first read of same length 
                if currentRead == 0:
                    indexer.createIndexAndStore(targets, self.arguments[1], False)

                while currentRead < len(records_seqs) and len(records_seqs[currentRead]) == prevLength:
                    self.logger.info("Processing seq: " + records_seqs[currentRead].id)
                    currentRead += 1
            
            #filter out reads already processed:
            currentRead = 0
            while currentRead < len(records_seqs) and len(records_seqs[currentRead]) == prevLength:
                currentRead += 1
                keepRecords.append(records_seqs[0])
            records_seqs = records_seqs[currentRead:]
                                     
        self.logger.debug('ComBaR indexer finished.')
        return self.hitlist

class GenomePlotter(Aligner):
    
    def __init__(self, logger, score, settings, arguments):
        Aligner.__init__(self, logger, score, settings)
        self.arguments = arguments

    def process(self, records_seqs, targets, pypaswas):
        '''This methods creates indices and determines differences between sequences.
        '''

        
        # step through the targets                                                                                                                                                                           
        # step through the targets                                                                                                                                                                          
        self.logger.debug('Genome plotter...')
        self.logger.info("Clearing output file")
        formatter = pypaswas._get_formatter(self.hitlist)
        formatter.open()
        indexer = None
        plotter = None

        window = int(self.settings.window_length)
        stepSize = int(0.1 * window)
        block = 100
        indexStepSize = 1000

        seq = records_seqs[0]
        dummySeq = [SWSeqRecord(Seq(str(seq.seq[0:window]), seq.seq.alphabet), seq.id, 0, original_length = len(seq.seq), distance = 0, refID = seq.id)]

        if self.qindexerOCL:
            from pyPaSWAS.Core.QIndexerOCL import QIndexerOCL, GenomePlotter
            indexer = QIndexerOCL(self.settings, self.logger, 0.1, dummySeq, int(self.settings.qgram), block, indexStepSize, nAs='A')
            #plotter = GenomePlotter(indexer, dummySeq, block, indexStepSize)                                                                                                                            
        elif self.qindexerCUDA:
            from pyPaSWAS.Core.QIndexerCUDA import QIndexerCUDA, GenomePlotter
            block = 10
            indexStepSize=1000
            indexer = QIndexerCUDA(self.settings, self.logger, 0.1, dummySeq, int(self.settings.qgram), block, indexStepSize, nAs='A')
            #plotter = GenomePlotter(indexer, dummySeq, block, indexStepSize)                                                                                                                            
        else:
            from pyPaSWAS.Core.QIndexer import QIndexer
            indexer = QIndexer(self.settings, self.logger, 0.1, dummySeq, int(self.settings.qgram))

        self.logger.debug("Starting indexer on targets")
        while indexer.indicesToProcessLeft():
            indexer.createIndexAndStore(targets, self.arguments[1])
            plotter = GenomePlotter(indexer, dummySeq, block, indexStepSize, nAs='C')
            self.logger.debug("Starting indexer on sequences")
            while plotter.indicesToProcessLeft():
                plotter.createIndexAndStore(records_seqs, self.arguments[0])
    
                allLocations = plotter.findDistances(indexer)

                for a in xrange(len(allLocations)):
                    locations = allLocations[a]
                    locs = []
                    if (len(locations) > 0):
                        for value in locations.itervalues():
                            locs.extend(value)
                        splittedTargets = []

                        for loc in locs:
                            #self.logger.debug("Loc: {}, len: {}".format(loc, len(records_seqs)))                                                                                                        
                            distance = loc[2]
                            targetID = targets[loc[1][1]].id
                            targetStartIndex = loc[1][0]
                            refID = loc[1][1]
                            seqRefID = loc[0][1]
                            seqID = records_seqs[loc[0][1]].id
                            seqStartIndex = loc[0][0]
                            t = SWSeqRecord(Seq("", seq.seq.alphabet), targetID, targetStartIndex, original_length = 0, distance = distance, refID = targetID)
                            s = SWSeqRecord(Seq("", seq.seq.alphabet), seqID, seqStartIndex, original_length = 0, distance = distance, refID = seqID)
                            self.hitlist.add(Distance(self.logger, s, t, (seqStartIndex,seqStartIndex+window), (targetStartIndex, targetStartIndex+window)))
                            #self.logger.debug("Target {}, read {}, distance {}".format(t, s, distance))                                                                                                 
                formatter.print_results(self.hitlist)

                self.hitlist = HitList(self.logger)

        if indexer != None and self.qindexerCUDA:
            indexer.pop_context()
        
        formatter.close()
        self.logger.debug('Genome plotter finished.')
        return self.hitlist
        
        
        
