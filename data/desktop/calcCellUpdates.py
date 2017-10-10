from Bio import SeqIO
import sys

refCount = 0

for s in SeqIO.parse(open(sys.argv[1], "r"), "fasta"):
    refCount += len(s.seq)


for index in range(0,350, 10):
    targetCount = 0
    for s in SeqIO.parse(open(sys.argv[1] + "_{}.fasta".format(index), "r"), "fasta"):
        targetCount += len(s.seq)
    print("{}\t{}".format(index, refCount*targetCount))
