'''
Created on March 13, 2013

@author: tbeek
'''
import sys
import unittest
import tempfile
import shutil
from pkg_resources import resource_filename  #@UnresolvedImport
from pyPaSWAS.Core.Exceptions import InvalidOptionException
from pyPaSWAS import pypaswasall


class Test(unittest.TestCase):
    ''' Runs the application as the end-user would, testing for correct exception handling
        as well as final output checks comparing generated output to reference (curated)
        output. '''

    def setUp(self):
        # Create pyPaSWAS instance
        self.instance = pypaswasall.Pypaswas()
        # Input files
        self.input_faa_1 = resource_filename(__name__, 'data/query_fasta.faa')
        self.input_faa_2 = resource_filename(__name__, 'data/target_fasta.faa')
        self.input_gb_1 = resource_filename(__name__, 'data/query_genbank.gb')
        self.input_gb_2 = resource_filename(__name__, 'data/target_genbank.gb')
        self.outdir = tempfile.mkdtemp(prefix='test_pypaswas_aligner')

    def test_pypaswas_aligner_invalid_options(self):
        ''' This test checks the raised exception when arguments are missing or incorrect '''
        # Missing arguments, this should raise an InvalidOptionException
        sys.argv = [__name__]
        self.assertRaises(InvalidOptionException, self.instance.run)

        # Trying to get output using the unsupported BAM output format
        sys.argv = [__name__,
                    self.input_faa_1,
                    self.input_faa_2,
                    '--outputformat=BAM']
        self.assertRaises(InvalidOptionException, self.instance.run)

    def _defunct_test_pypaswas_aligner_basic(self):
        ''' Input two FASTA files and align them using all default settings.
            Compares the output alignment with the included reference file. '''
        # Expected output
        reference = resource_filename(__name__, 'data/reference/aligner_basic_output.txt')
        # Most basic alignment (default settings, default alignment output format)
        outfile = '{}/basic_alignment.txt'.format(self.outdir)
        sys.argv = [__name__,
                    self.input_faa_1,
                    self.input_faa_2,
                    '-o', outfile]
        # Start pyPaSWAS using defined arguments in sys.argv
        self.instance.run()
        # Read output / reference file
        expected = _read_file(reference)
        actual = _read_file(outfile)
        # Compare output with reference
        self.assertEqual(actual, expected)

    def test_pypaswas_aligner_basic_genbank(self):
        ''' Input two Genbank files and align them using all default settings.
            Compares the output alignment with the included reference file. '''
        # Expected output
        reference = resource_filename(__name__, 'data/reference/aligner_basic_genbank_output.txt')
        # Most basic alignment (default settings, default alignment output format)
        outfile = '{}/basic_alignment_genbank.txt'.format(self.outdir)
        sys.argv = [__name__,
                    self.input_gb_1,
                    self.input_gb_2,
                    # Set correct file types
                    '--filetype1', 'genbank',
                    '--filetype2', 'genbank',
                    '-o', outfile]

        # Start pyPaSWAS using defined arguments in sys.argv
        self.instance.run()
        # Read output / reference file
        expected = _read_file(reference)
        actual = _read_file(outfile)
        # Compare output with reference
        self.assertEqual(actual, expected)

    def test_pypaswas_aligner_indel(self):
        pass

    def test_pypaswas_aligner_sam(self):
        ''' Single test function to check complete SAM output format. The given input and
            parameters include all possible alignment types which is compared to the reference
            SAM file. '''
        pass

    def tearDown(self):
        # Cleanup
        shutil.rmtree(self.outdir)


def _read_file(filename):
    ''' Helper method to quickly read a file
        @param filename: file to read '''
    with open(filename) as handle:
        return handle.read()
