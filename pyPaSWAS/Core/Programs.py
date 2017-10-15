''' This module contains the programs from the pyPaSWAS suite '''
from pyPaSWAS.Core.HitList import HitList
from operator import itemgetter,attrgetter

from pyPaSWAS.Core.SWSeqRecord import SWSeqRecord
from Bio.Seq import Seq
from pyPaSWAS.Core.Hit import Hit

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
        if (self.settings.framework.upper() == 'OPENCL'):
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

        


        
