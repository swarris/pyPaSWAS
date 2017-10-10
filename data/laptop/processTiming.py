import sys

for f in sys.argv[1:]:
    total = 0
    for l in open(f,"r"):
        total += float(l)
    print("{}\t{}".format(f.split("_")[1].split(".")[0], total))
