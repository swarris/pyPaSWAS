pyPaSWAS
========

extented python version of PaSWAS

PaSWAS was developed in C and CUDA. This version only uses the CUDA code and integrates the sequence alignment software with Python.

Installation
------------
In most cases it is enough to clone the repository. Make sure CUDA, pyCuda, numpy and bioPython are installed.

The application also contains a Python script setup.py. This script only requires to have Python installed. The automated procedure will scan the computer for the other packages and install them if needed. When automatic installation fails, the script will report a possible solution.


Running the software
-------------------- 

The key command line options are given in Table 1. The two input files are mandatory. Through the options the user can specify the file types of the input files (default: fasta), an output file and a log file. When requested, PyPaSWAS will terminate if the output file already exists.

Add the source dir to your python path to run the application from any location:
_export $PYTHONPATH=$PYTHONPATH:/main/source/dir_

Run it by calling:
- *python -m pyPaSWAS/pypaswasall |options| file1 file2*

Help file:
- *python -m pyPaSWAS/pypaswasall -help*

Examples
--------
See the github wiki at https://github.com/swarris/pyPaSWAS/wiki for more examples.

Use a fastq-file:
- *python -m pyPaSWAS/pypaswasall testSample.fastq adapters.fa -1 fastq -o out.txt --loglevel=DEBUG*

Output results in SAM format:
- *python -m pyPaSWAS/pypaswasall testSample.fastq adapters.fa -1 fastq -o out.sam --outputformat=SAM --loglevel=DEBUG*

Remove all matches from file 1. Useful from trimming sequences. Sequences with no hits will not be in output
- *python -m pyPaSWAS/pypaswasall testSample.fastq adapters.fa -1 fastq -o out.fa --outputformat=trimmedFasta -p trimmer -L /tmp/log.txt --loglevel=DEBUG*

Align protein sequences:
- *python -m pyPaSWAS/pypaswasall myAA.faa product.faa -M BLOSUM62 -o hits.txt -L /tmp/log.txt --loglevel=DEBUG*



Table 1. Key command line options

| Option	| Long version	| Description|
| --------- | ------------- | ---------- |
| -h| --help| This help|  
|-L	| --logfile	| Path to the log file| 
|	| --loglevel	| Specify the log level for log file output. Valid options are DEBUG, INFO, WARNING, ERROR and CRITICAL| 
|-o	| --output	| Path to the output file. Default ./output| 
|-O	| --overrideOutput	| When output file exists, override it (T/F). Default T (true) | 
|-1	| --filetype1	| File type of the first input file. See bioPython IO for available options. Default fasta| 
|-2	| --filetype2	| File type of the second input file. See bioPython IO for available options. Default fasta| 
|-G	| 	| Float value for the gap penalty. Default -5| 
|-q	| 	| Float value for a mismatch. Default -3| 
|-r	| 	| Float value for a match. Default 1| 
|	| --any	| Float value for an ambiguous nucleotide. Default 1| 
|	| --other	| Float value for an ambiguous nucleotide. Default 1| 
|	| --device	| Integer value indicating the device to use. Default 0 for the first device. | 
|-c	| 	| Option followed by the name of a configuration file. This option allows for short command line calls and administration of used settings. | 

For questions, e-mail s.warris@gmail.com