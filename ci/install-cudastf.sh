#!/bin/bash

CI_ROOT="$1"

cd $CI_ROOT

TARGETDIR=$CI_ROOT/stf/

mkdir -p $TARGETDIR

git clone https://github.com/NVIDIA/cccl.git $TARGETDIR/cccl/
