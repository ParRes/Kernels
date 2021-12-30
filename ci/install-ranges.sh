#!/bin/sh

set -e
set -x

CI_ROOT="$1"
os=`uname`

if [ "$os" = "Darwin" ] ; then
    git clone --depth 1 https://github.com/ericniebler/range-v3.git $CI_ROOT/range-v3
else
    sh ./ci/install-boost.sh $CI_ROOT
fi
