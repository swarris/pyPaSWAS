from Bio import SeqIO
import sys
from collections import defaultdict

hits = defaultdict(list)

boundaryFraction = 0.02
minBoundary = 100
fileType= sys.argv[2].split(".")[-1]
notAssigned = "notAssigned." + fileType


for l in open(sys.argv[1], "r"):
    if ',' in l:
        l = [x.strip() for x in l.strip().split(",")]
        if "_RC" in l[1]:
            l[1] = l[1][:-3]
        h = hits[l[1]]
        boundary = boundaryFraction * float(l[13])
        if boundary < minBoundary:
            boundary = minBoundary
        if int(l[13]) - int(l[4]) < boundary or int(l[5]) < boundary: # interesting hit
            if len(h) == 0 or float(l[6]) > float(h[6]):
                hits[l[1]] = l

for s in SeqIO.parse(sys.argv[2], fileType):
    if len(hits[s.id]) > 0:
        boundary = int(boundaryFraction * float(hits[s.id][13]))
        if boundary < minBoundary:
            boundary = minBoundary
        s.seq = s.seq[boundary:-boundary]
        SeqIO.write(s, open("{}.{}".format(hits[s.id][0],fileType), "a"), fileType)
    else:
        SeqIO.write(s, open(notAssigned, "a"), fileType)
