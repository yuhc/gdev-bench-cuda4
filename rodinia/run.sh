#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/cuda

# 10 in total
#  and  does not work
bm="backprop heartwall bfs gaussian hotspot \
    lud nn pathfinder needle srad/srad_v1 srad/srad_v2"

OUTDIR=$DIR/results
mkdir $OUTDIR &>/dev/null

cd $OCLDIR
exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    for idx in `seq 1 10`; do
        #exe sudo -E perf stat -A -a -e instructions,cache-misses,cache-references,cycles \
        #    ./run
        exe /usr/bin/time ./run
        exe echo
    done
    cd $OCLDIR
    exe echo
    echo
done
