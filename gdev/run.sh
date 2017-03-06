#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/user

# 11 in total
# heartwall does not work
bm="idle loop madd mmul fmadd fmmul \
    memcpy memcpy_pinned memcpy_async shm"

OUTDIR=$DIR/results
mkdir $OUTDIR &>/dev/null
mkdir $OUTDIR/srad &>/dev/null

cd $OCLDIR
exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    make
    for idx in `seq 1 10`; do
        #exe sudo /usr/bin/time ./run
        exe sudo ./user_test
        exe echo
	sleep 2
    done
    cd $OCLDIR
    exe echo
    echo
done
