#!/bin/sh

set -e
set -x

CI_ROOT="$1"
HYDRA_ROOT=$CI_ROOT/hydra

if [ ! -d "$HYDRA_ROOT" ]; then
    cd $CI_ROOT
    wget --no-check-certificate -q http://www.mpich.org/static/downloads/3.2/hydra-3.2.tar.gz
    tar -xzf hydra-3.2.tar.gz
    cd hydra-3.2
    ./configure CC=cc --prefix=$HYDRA_ROOT
    make && make install
else
    echo "MPICH Hydra installed..."
    find $HYDRA_ROOT -name pmi.h
fi
